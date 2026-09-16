"""Model Training and Trial Pipeline CLI (Phase 6).

Executes experimental model improvement trials on the leak-free Phase 5 dataset split (v2):
1. Re-runs Phase 1 CUDA hardware smoke test prior to training launch.
2. Formally validates zero train/val spatial leakage before proceeding.
3. Trains the variant on NVIDIA RTX 4060 with mixed precision and CosineAnnealing schedule.
4. Evaluates all held-out validation patches, computing full distribution statistics.
5. Logs per-patch metrics to logs/<exp_id>_eval_distribution.csv.
6. Appends validated experiment record to experiments/experiments_log.jsonl.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project root setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.cuda_smoke_test import run_cuda_smoke_test
from src.data.dataset import SentinelSRDataset
from src.data.degrade import bicubic_upsample, evaluate_reconstruction_fidelity
from src.data.tiling import load_sentinel2_stack
from src.evaluation.edge_check import compute_edge_consistency
from src.evaluation.spectral_check import compute_spectral_consistency
from src.models.loss import CompoundSRLoss
from src.models.model import ResidualSRNet, count_parameters
from src.utils.config import get_project_root, load_config
from src.utils.experiment_tracker import (
    compute_distribution_summary,
    get_git_commit_hash,
    log_experiment_record,
)


def train_trial(
    exp_id: str,
    description: str = "",
    split_mode: str = "v2",
    buffer_pixels: int = 0,
    augment: bool = False,
    num_blocks: int = 4,
    num_features: int = 48,
    grad_weight: float = 0.10,
    lr: float = 0.0005,
    epochs: int = 15,
    batch_size: int = 16,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute a single GPU training trial with strict pre-conditions and full evaluation."""
    root = get_project_root()
    config = load_config()

    print("=" * 82)
    print(f"      GeoFUSE SentinelGuard — Phase 6 Trial Training: {exp_id}")
    print("=" * 82)
    print(f"Description           : {description or 'Experimental training trial'}")
    print(f"Split Mode            : {split_mode} (Buffer: {buffer_pixels}px)")
    print(f"Data Augmentation     : {augment}")
    print(f"Architecture          : ResidualSRNet ({num_blocks} blocks, {num_features} channels)")
    print(f"Loss Configuration    : CompoundSRLoss (L1 + {grad_weight:.2f} * SobelGradient)")
    print(f"Optimizer / LR        : AdamW (lr={lr}, CosineAnnealing)")
    print("-" * 82)

    # 1. GPU Pre-condition: Re-run Phase 1 CUDA smoke test
    print("\n[PRE-CHECK 1/2] Running Phase 1 CUDA Hardware Smoke Test...")
    smoke_passed = run_cuda_smoke_test()
    if not smoke_passed:
        raise RuntimeError(f"[ABORT] CUDA smoke test failed! Cannot launch trial {exp_id}.")
    print("[PRE-CHECK 1/2 PASSED] CUDA hardware verified.\n")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    torch_version = torch.__version__

    # 2. Data Loading & Deterministic Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    stack, _ = load_sentinel2_stack(raw_dir)

    patch_size_hr = 128
    stride = 32
    val_quadrant = (256, 512, 256, 512)

    train_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=patch_size_hr,
        stride=stride,
        split="train",
        val_quadrant=val_quadrant,
        seed=seed,
        split_mode=split_mode,
        buffer_pixels=buffer_pixels,
        augment=augment,
    )
    val_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=patch_size_hr,
        stride=stride,
        split="val",
        val_quadrant=val_quadrant,
        seed=seed + 100,
        split_mode=split_mode,
        buffer_pixels=buffer_pixels,
        augment=False,
    )

    # 3. Leakage Pre-condition: Formally verify zero spatial overlap
    print("[PRE-CHECK 2/2] Verifying zero spatial train/val overlap...")
    overlap_count = SentinelSRDataset.count_overlapping_pairs(train_ds, val_ds)
    print(f"      Train Patches: {len(train_ds)}, Val Patches: {len(val_ds)}, Overlapping Pairs: {overlap_count}")
    if overlap_count != 0:
        raise RuntimeError(
            f"[ABORT] Spatial data leakage detected! {overlap_count} overlapping patch pairs found. "
            f"Trial {exp_id} cannot be trusted."
        )
    print("[PRE-CHECK 2/2 PASSED] Zero spatial leakage confirmed.\n")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 4. Model, Loss, Optimizer & Scaler
    model = ResidualSRNet(
        in_channels=4,
        out_channels=4,
        num_features=num_features,
        num_blocks=num_blocks,
        scale_factor=2,
    ).to(device)

    criterion = CompoundSRLoss(channels=4, grad_weight=grad_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    param_count = count_parameters(model)
    print(f"Model Parameters      : {param_count:,}")

    # Output paths
    ckpt_dir = root / "checkpoints" / exp_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_train_log = logs_dir / f"{exp_id}_metrics.csv"
    csv_eval_log = logs_dir / f"{exp_id}_eval_distribution.csv"

    # CSV Header
    with open(csv_train_log, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_l1", "train_grad", "train_total",
            "val_l1", "val_grad", "val_total", "val_psnr", "val_ssim",
            "lr", "epoch_time_s", "peak_vram_mib"
        ])

    torch.cuda.reset_peak_memory_stats(device)
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics: Dict[str, Any] = {}
    train_start_time = time.time()

    print(f"\n{'Epoch':<6} | {'Train Total':<11} | {'Val Total':<10} | {'Val PSNR':<9} | {'Val SSIM':<8} | {'VRAM (MiB)':<10} | {'Time (s)':<8}")
    print("-" * 76)

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train one epoch
        model.train()
        train_l1, train_grad, train_total, nb_train = 0.0, 0.0, 0.0, 0
        for lr_b, hr_b, _ in train_loader:
            lr_b = lr_b.to(device, non_blocking=True)
            hr_b = hr_b.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=True):
                sr_b = model(lr_b)
                loss, l_dict = criterion(sr_b, hr_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_l1 += l_dict["l1"]
            train_grad += l_dict["grad"]
            train_total += l_dict["total"]
            nb_train += 1

        avg_train_total = train_total / max(1, nb_train)
        avg_train_l1 = train_l1 / max(1, nb_train)
        avg_train_grad = train_grad / max(1, nb_train)

        # Validation loop
        model.eval()
        val_l1, val_grad, val_total, nb_val = 0.0, 0.0, 0.0, 0
        val_psnrs, val_ssims = [], []

        with torch.no_grad():
            for lr_b, hr_b, _ in val_loader:
                lr_b = lr_b.to(device, non_blocking=True)
                hr_b = hr_b.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=True):
                    sr_b = model(lr_b)
                    _, l_dict = criterion(sr_b, hr_b)

                val_l1 += l_dict["l1"]
                val_grad += l_dict["grad"]
                val_total += l_dict["total"]
                nb_val += 1

                sr_np = sr_b.cpu().numpy()
                hr_np = hr_b.cpu().numpy()
                for b_i in range(sr_np.shape[0]):
                    p_tile = np.transpose(sr_np[b_i], (1, 2, 0))
                    g_tile = np.transpose(hr_np[b_i], (1, 2, 0))
                    m = evaluate_reconstruction_fidelity(g_tile, p_tile, data_range=1.0)
                    val_psnrs.append(m["psnr_db"])
                    val_ssims.append(m["ssim"])

        avg_val_total = val_total / max(1, nb_val)
        avg_val_l1 = val_l1 / max(1, nb_val)
        avg_val_grad = val_grad / max(1, nb_val)
        avg_val_psnr = float(np.mean(val_psnrs))
        avg_val_ssim = float(np.mean(val_ssims))

        scheduler.step()
        epoch_time = time.time() - epoch_start
        cur_lr = scheduler.get_last_lr()[0]
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # Log row
        with open(csv_train_log, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                f"{avg_train_l1:.5f}", f"{avg_train_grad:.5f}", f"{avg_train_total:.5f}",
                f"{avg_val_l1:.5f}", f"{avg_val_grad:.5f}", f"{avg_val_total:.5f}",
                f"{avg_val_psnr:.2f}", f"{avg_val_ssim:.4f}",
                f"{cur_lr:.6f}", f"{epoch_time:.2f}", f"{peak_vram:.2f}"
            ])

        is_best = avg_val_total < best_val_loss
        if is_best:
            best_val_loss = avg_val_total
            best_epoch = epoch
            best_metrics = {
                "val_total": avg_val_total,
                "val_psnr": avg_val_psnr,
                "val_ssim": avg_val_ssim,
            }
            torch.save({
                "exp_id": exp_id,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_total,
                "val_psnr": avg_val_psnr,
                "val_ssim": avg_val_ssim,
                "num_blocks": num_blocks,
                "num_features": num_features,
            }, ckpt_dir / "best_model.pth")

        flag = " [BEST]" if is_best else ""
        print(f"{epoch:<6} | {avg_train_total:<11.5f} | {avg_val_total:<10.5f} | {avg_val_psnr:<9.2f} | {avg_val_ssim:<8.4f} | {peak_vram:<10.2f} | {epoch_time:<8.2f}{flag}")

    total_duration = time.time() - train_start_time
    peak_vram_final = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Save latest checkpoint
    torch.save({
        "exp_id": exp_id,
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": avg_val_total,
        "val_psnr": avg_val_psnr,
        "val_ssim": avg_val_ssim,
    }, ckpt_dir / "latest_model.pth")

    print("-" * 76)
    print(f"Training Complete! Duration: {total_duration:.2f}s, Peak VRAM: {peak_vram_final:.2f} MiB, Best Epoch: #{best_epoch}")

    # 6. Full Quantitative Evaluation on Held-Out Validation Patches
    print(f"\nEvaluating Best Checkpoint across all {len(val_ds)} validation patches...")
    best_ckpt = torch.load(ckpt_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # GPU Warmup
    dummy_in = torch.randn(1, 4, 64, 64, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_in)
    torch.cuda.synchronize(device)

    patch_records = []
    latencies_ms = []

    for idx in range(len(val_ds)):
        lr_t, hr_t, meta = val_ds[idx]
        y, x = meta["y"], meta["x"]

        lr_in = lr_t.unsqueeze(0).to(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            sr_t = model(lr_in)
        torch.cuda.synchronize(device)
        t_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(t_ms)

        pred_hr = sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float64)
        gt_hr = hr_t.permute(1, 2, 0).numpy().astype(np.float64)
        lr_np = lr_t.permute(1, 2, 0).numpy().astype(np.float64)
        bicubic_hr = bicubic_upsample(lr_np, scale_factor=2).astype(np.float64)

        m_sr = evaluate_reconstruction_fidelity(gt_hr, pred_hr, data_range=1.0)
        m_bic = evaluate_reconstruction_fidelity(gt_hr, bicubic_hr, data_range=1.0)
        spec_m = compute_spectral_consistency(gt_hr, pred_hr, ndvi_threshold=0.05)
        edge_m = compute_edge_consistency(gt_hr, pred_hr, canny_low=40, canny_high=120)

        patch_records.append({
            "patch_idx": idx,
            "y": y,
            "x": x,
            "sr_psnr_db": m_sr["psnr_db"],
            "sr_ssim": m_sr["ssim"],
            "sr_mae": m_sr["mae"],
            "bicubic_psnr_db": m_bic["psnr_db"],
            "bicubic_ssim": m_bic["ssim"],
            "bicubic_mae": m_bic["mae"],
            "psnr_gain_db": round(m_sr["psnr_db"] - m_bic["psnr_db"], 2),
            "ssim_gain": round(m_sr["ssim"] - m_bic["ssim"], 4),
            "mean_delta_ndvi": spec_m["mean_delta_ndvi"],
            "max_delta_ndvi": spec_m["max_delta_ndvi"],
            "pct_inconsistent_pixels": spec_m["pct_inconsistent_pixels"],
            "gradient_correlation": edge_m["gradient_correlation"],
            "edge_iou": edge_m["edge_iou"],
            "edge_f1": edge_m["edge_f1"],
            "inference_time_ms": round(t_ms, 2),
        })

    # Save per-patch distribution CSV
    fieldnames = list(patch_records[0].keys())
    with open(csv_eval_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(patch_records)

    # 7. Compute Summary Distributions
    metrics_summary = {
        "sr_psnr_db": compute_distribution_summary([r["sr_psnr_db"] for r in patch_records], 2),
        "sr_ssim": compute_distribution_summary([r["sr_ssim"] for r in patch_records], 4),
        "sr_mae": compute_distribution_summary([r["sr_mae"] for r in patch_records], 5),
        "bicubic_psnr_db": compute_distribution_summary([r["bicubic_psnr_db"] for r in patch_records], 2),
        "bicubic_ssim": compute_distribution_summary([r["bicubic_ssim"] for r in patch_records], 4),
        "bicubic_mae": compute_distribution_summary([r["bicubic_mae"] for r in patch_records], 5),
        "psnr_gain_db": compute_distribution_summary([r["psnr_gain_db"] for r in patch_records], 2),
        "ssim_gain": compute_distribution_summary([r["ssim_gain"] for r in patch_records], 4),
        "delta_ndvi": compute_distribution_summary([r["mean_delta_ndvi"] for r in patch_records], 5),
        "pct_inconsistent_pixels": compute_distribution_summary([r["pct_inconsistent_pixels"] for r in patch_records], 2),
        "gradient_correlation": compute_distribution_summary([r["gradient_correlation"] for r in patch_records], 4),
        "edge_iou": compute_distribution_summary([r["edge_iou"] for r in patch_records], 4),
        "edge_f1": compute_distribution_summary([r["edge_f1"] for r in patch_records], 4),
    }

    lat_sum = compute_distribution_summary(latencies_ms, 2)
    mean_lat = lat_sum["mean"]
    throughput = round(1000.0 / mean_lat, 1) if mean_lat > 0 else 0.0

    inference_benchmark = {
        "device": str(device),
        "mean_latency_ms_per_patch": lat_sum["mean"],
        "std_latency_ms": lat_sum["std"],
        "min_latency_ms": lat_sum["min"],
        "max_latency_ms": lat_sum["max"],
        "throughput_patches_per_sec": throughput,
    }

    # 8. Append Experiment Record to JSONL
    exp_record = {
        "experiment_id": exp_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit_hash(root),
        "random_seed": seed,
        "evaluation_type": "synthetic degrade-and-recover validation",
        "dataset": {
            "name": "Sentinel-2 L2A 4-band",
            "channels": ["B02", "B03", "B04", "B08"],
            "patch_size_hr": 128,
            "stride": stride,
            "split_mode": split_mode,
            "buffer_pixels": buffer_pixels,
            "val_quadrant": list(val_quadrant),
            "num_val_patches": len(val_ds),
            "num_train_patches": len(train_ds),
            "augmentation": augment,
        },
        "degradation_config": {
            "optical_psf_blur": {"kernel_size": 3, "sigma": 0.5},
            "downsample_factor": 2,
            "downsample_method": "bicubic",
            "sensor_noise_std": 0.01,
        },
        "architecture_config": {
            "model_name": "ResidualSRNet",
            "num_channels": 4,
            "num_features": num_features,
            "num_residual_blocks": num_blocks,
            "upsample_method": "PixelShuffle",
            "scale_factor": 2,
        },
        "param_count": param_count,
        "training_config": {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": 1e-4,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR (T_max=15, eta_min=1e-6)",
            "loss": f"CompoundSRLoss (L1 + {grad_weight:.2f} * SobelGradient)",
            "mixed_precision": True,
            "description": description,
        },
        "training_results": {
            "best_epoch": best_epoch,
            "train_loss_final": round(avg_train_total, 5),
            "val_loss_best": round(best_val_loss, 5),
            "total_duration_s": round(total_duration, 2),
            "peak_vram_mib": round(peak_vram_final, 2),
        },
        "hardware_telemetry": {
            "device": str(device),
            "gpu_name": gpu_name,
            "cuda_version": cuda_version,
            "torch_version": torch_version,
        },
        "inference_benchmark": inference_benchmark,
        "metrics_summary": metrics_summary,
    }

    log_file = root / "experiments" / "experiments_log.jsonl"
    log_experiment_record(exp_record, log_file=log_file)
    print(f"[OK] Appended trial record to: {log_file.relative_to(root)}")

    # 9. Summary Telemetry
    print("\n" + "=" * 82)
    print(f"      EVALUATION SUMMARY — {exp_id}")
    print("=" * 82)
    print(f"SR PSNR (Fidelity)    : {metrics_summary['sr_psnr_db']['mean']:.2f} ± {metrics_summary['sr_psnr_db']['std']:.2f} dB (Min: {metrics_summary['sr_psnr_db']['min']:.2f}, Max: {metrics_summary['sr_psnr_db']['max']:.2f})")
    print(f"Bicubic Baseline PSNR : {metrics_summary['bicubic_psnr_db']['mean']:.2f} ± {metrics_summary['bicubic_psnr_db']['std']:.2f} dB")
    print(f"PSNR Gain over Bicubic: {metrics_summary['psnr_gain_db']['mean']:+.2f} ± {metrics_summary['psnr_gain_db']['std']:.2f} dB")
    print(f"SR SSIM (Structural)  : {metrics_summary['sr_ssim']['mean']:.4f} ± {metrics_summary['sr_ssim']['std']:.4f}")
    print(f"SSIM Gain over Bicubic: {metrics_summary['ssim_gain']['mean']:+.4f} ± {metrics_summary['ssim_gain']['std']:.4f}")
    print(f"Delta-NDVI (Spectral) : {metrics_summary['delta_ndvi']['mean']:.5f} ± {metrics_summary['delta_ndvi']['std']:.5f}")
    print(f"Inconsistent Pixels   : {metrics_summary['pct_inconsistent_pixels']['mean']:.2f} ± {metrics_summary['pct_inconsistent_pixels']['std']:.2f} %")
    print(f"Gradient Correlation r: {metrics_summary['gradient_correlation']['mean']:.4f} ± {metrics_summary['gradient_correlation']['std']:.4f}")
    print(f"Canny Edge IoU / F1   : {metrics_summary['edge_iou']['mean']:.4f} / {metrics_summary['edge_f1']['mean']:.4f}")
    print(f"Inference Latency     : {mean_lat:.2f} ms ({throughput:.1f} patches/sec)")
    print("=" * 82)

    return exp_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Phase 6 training trial on RTX 4060.")
    parser.add_argument("--exp-id", type=str, required=True, help="Unique experiment identifier (e.g. exp002_baseline_v2)")
    parser.add_argument("--description", type=str, default="", help="Description of trial rationale")
    parser.add_argument("--split-mode", type=str, default="v2", choices=["v1", "v2"], help="Split mode ('v2' leak-free)")
    parser.add_argument("--buffer-pixels", type=int, default=0, help="Guard buffer pixels (e.g. 0 or 32)")
    parser.add_argument("--augment", action="store_true", help="Enable synchronous D4 geometric augmentations")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of residual blocks (default 4)")
    parser.add_argument("--num-features", type=int, default=48, help="Number of feature channels (default 48)")
    parser.add_argument("--grad-weight", type=float, default=0.10, help="Weight of Sobel gradient loss (default 0.10)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Base learning rate (default 0.0005)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default 15)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default 16)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()

    try:
        train_trial(
            exp_id=args.exp_id,
            description=args.description,
            split_mode=args.split_mode,
            buffer_pixels=args.buffer_pixels,
            augment=args.augment,
            num_blocks=args.num_blocks,
            num_features=args.num_features,
            grad_weight=args.grad_weight,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Trial execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
