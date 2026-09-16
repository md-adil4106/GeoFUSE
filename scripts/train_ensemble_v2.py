"""Ensemble Training, Diversity Verification, and Informativeness Audit (Phase 7).

Retrains 3 ensemble members using the Phase 6 best configuration:
- Architecture: ResidualSRNet (6 residual blocks, 48 features, 356k params)
- Loss: CompoundSRLoss (L1 + 0.20 * SobelGradient)
- Split: Leak-free Phase 5 Split v2 (105 train / 25 held-out val)
- Seeds: 42 (Member 1), 101 (Member 2), 2024 (Member 3)

Analyzes whether ensemble disagreement genuinely correlates with structural difficulty
(Sobel gradients and Canny edges) vs. uniform random noise.
"""

import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Project root setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.cuda_smoke_test import run_cuda_smoke_test
from src.data.dataset import SentinelSRDataset
from src.data.degrade import bicubic_upsample, evaluate_reconstruction_fidelity
from src.data.tiling import load_sentinel2_stack
from src.evaluation.edge_check import compute_edge_consistency, compute_gradient_magnitude, extract_canny_edges
from src.evaluation.spectral_check import compute_spectral_consistency
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.models.loss import CompoundSRLoss
from src.models.model import ResidualSRNet, count_parameters
from src.utils.config import get_project_root, load_config
from src.utils.experiment_tracker import compute_distribution_summary, get_git_commit_hash, log_experiment_record


def train_single_member(
    member_idx: int,
    seed: int,
    output_dir: Path,
    device: torch.device,
    config: Dict[str, Any],
    stack: np.ndarray,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 0.0005,
    num_blocks: int = 6,
    num_features: int = 48,
    grad_weight: float = 0.20,
) -> Dict[str, Any]:
    """Train an individual ensemble member with specified random seed."""
    print(f"\n{'='*78}")
    print(f"   Training Ensemble Member #{member_idx} (Seed: {seed}) on {device}")
    print(f"{'='*78}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=128,
        stride=32,
        split="train",
        val_quadrant=(256, 512, 256, 512),
        seed=seed,
        split_mode="v2",
        buffer_pixels=0,
        augment=False,
    )
    val_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=128,
        stride=32,
        split="val",
        val_quadrant=(256, 512, 256, 512),
        seed=seed + 100,
        split_mode="v2",
        buffer_pixels=0,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

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

    best_val_loss = float("inf")
    best_epoch = 0
    best_psnr = 0.0
    best_ssim = 0.0
    start_time = time.time()

    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val PSNR':<9} | {'Val SSIM':<8} | {'VRAM (MiB)':<10}")
    print("-" * 68)

    for epoch in range(1, epochs + 1):
        model.train()
        t_total, nb_t = 0.0, 0
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

            t_total += l_dict["total"]
            nb_t += 1

        avg_t_loss = t_total / max(1, nb_t)

        # Validation
        model.eval()
        v_total, nb_v = 0.0, 0
        psnrs, ssims = [], []
        with torch.no_grad():
            for lr_b, hr_b, _ in val_loader:
                lr_b = lr_b.to(device, non_blocking=True)
                hr_b = hr_b.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=True):
                    sr_b = model(lr_b)
                    _, l_dict = criterion(sr_b, hr_b)

                v_total += l_dict["total"]
                nb_v += 1

                sr_np = sr_b.cpu().numpy()
                hr_np = hr_b.cpu().numpy()
                for b_i in range(sr_np.shape[0]):
                    p_tile = np.transpose(sr_np[b_i], (1, 2, 0))
                    g_tile = np.transpose(hr_np[b_i], (1, 2, 0))
                    m = evaluate_reconstruction_fidelity(g_tile, p_tile, data_range=1.0)
                    psnrs.append(m["psnr_db"])
                    ssims.append(m["ssim"])

        avg_v_loss = v_total / max(1, nb_v)
        avg_psnr = float(np.mean(psnrs))
        avg_ssim = float(np.mean(ssims))
        scheduler.step()
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        is_best = avg_v_loss < best_val_loss
        if is_best:
            best_val_loss = avg_v_loss
            best_epoch = epoch
            best_psnr = avg_psnr
            best_ssim = avg_ssim

            ckpt_payload = {
                "member_idx": member_idx,
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_v_loss,
                "val_psnr": avg_psnr,
                "val_ssim": avg_ssim,
                "num_blocks": num_blocks,
                "num_features": num_features,
            }
            # Save as both .pt and .pth
            torch.save(ckpt_payload, output_dir / f"member_{member_idx}.pt")
            torch.save(ckpt_payload, output_dir / f"member_{member_idx}.pth")

        flag = " [BEST]" if is_best else ""
        print(f"{epoch:<6} | {avg_t_loss:<12.5f} | {avg_v_loss:<10.5f} | {avg_psnr:<9.2f} | {avg_ssim:<8.4f} | {peak_vram:<10.2f}{flag}")

    duration = time.time() - start_time
    print(f"Member #{member_idx} Done: Best Epoch #{best_epoch}, Best PSNR {best_psnr:.2f} dB, Time {duration:.2f}s")

    return {
        "member_idx": member_idx,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_psnr": best_psnr,
        "best_val_ssim": best_ssim,
        "duration_s": duration,
    }


def analyze_ensemble_informativeness(
    output_dir: Path,
    device: torch.device,
    stack: np.ndarray,
) -> Dict[str, Any]:
    """Load the 3 ensemble members, verify diversity, and measure disagreement informativeness."""
    print(f"\n{'='*78}")
    print("   Ensemble Diversity & Disagreement Informativeness Analysis")
    print(f"{'='*78}")

    ckpt_paths = [output_dir / f"member_{i}.pt" for i in (1, 2, 3)]
    models = load_ensemble_members(ckpt_paths, device=device)
    print(f"[OK] Successfully loaded {len(models)} ensemble members onto {device}.")

    val_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=128,
        stride=32,
        split="val",
        val_quadrant=(256, 512, 256, 512),
        seed=142,
        split_mode="v2",
        buffer_pixels=0,
    )
    num_patches = len(val_ds)

    # 1. Output Difference Check (Non-identical check across all members)
    first_lr, _, _ = val_ds[0]
    mean_rec, dis_map, preds = predict_ensemble(models, first_lr)
    
    mse_1_2 = float(np.mean((preds[0] - preds[1]) ** 2))
    mse_1_3 = float(np.mean((preds[0] - preds[2]) ** 2))
    mse_2_3 = float(np.mean((preds[1] - preds[2]) ** 2))
    
    print(f"\nPairwise Prediction MSE on Validation Patch #0:")
    print(f"   Member 1 vs Member 2 MSE: {mse_1_2:.6f}")
    print(f"   Member 1 vs Member 3 MSE: {mse_1_3:.6f}")
    print(f"   Member 2 vs Member 3 MSE: {mse_2_3:.6f}")

    if min(mse_1_2, mse_1_3, mse_2_3) < 1e-8:
        raise RuntimeError("[FAIL] Models produce identical predictions! Diversity requirement failed.")
    print("   [PASS] Verified: All 3 ensemble members produce distinct, non-identical predictions.\n")

    # 2. Informativeness Analysis across all held-out validation patches
    correlations_with_gradient = []
    edge_to_background_ratios = []
    disagreement_means = []
    disagreement_stds = []
    ensemble_psnrs = []
    ensemble_ssims = []
    member1_psnrs = []
    member2_psnrs = []
    member3_psnrs = []
    analysis_records = []

    for idx in range(num_patches):
        lr_t, hr_t, meta = val_ds[idx]
        gt_hr = hr_t.permute(1, 2, 0).numpy().astype(np.float64)

        # Ensemble Prediction
        mean_hr, dis_patch, member_preds = predict_ensemble(models, lr_t)
        mean_hr_64 = mean_hr.astype(np.float64)

        # Fidelity of ensemble mean vs individual members
        m_ens = evaluate_reconstruction_fidelity(gt_hr, mean_hr_64, data_range=1.0)
        m_m1 = evaluate_reconstruction_fidelity(gt_hr, member_preds[0].astype(np.float64), data_range=1.0)
        m_m2 = evaluate_reconstruction_fidelity(gt_hr, member_preds[1].astype(np.float64), data_range=1.0)
        m_m3 = evaluate_reconstruction_fidelity(gt_hr, member_preds[2].astype(np.float64), data_range=1.0)

        ensemble_psnrs.append(m_ens["psnr_db"])
        ensemble_ssims.append(m_ens["ssim"])
        member1_psnrs.append(m_m1["psnr_db"])
        member2_psnrs.append(m_m2["psnr_db"])
        member3_psnrs.append(m_m3["psnr_db"])

        # Ground truth structure: Sobel gradient magnitude & Canny edges
        grad_gt = compute_gradient_magnitude(gt_hr)
        edges_gt = extract_canny_edges(gt_hr, low_thresh=40, high_thresh=120)

        # Flatten for spatial correlation
        dis_flat = dis_patch.flatten()
        grad_flat = grad_gt.flatten()

        # Pearson correlation between Disagreement and High-Frequency Edge Gradients
        if np.std(dis_flat) > 0 and np.std(grad_flat) > 0:
            corr_r = float(np.corrcoef(dis_flat, grad_flat)[0, 1])
        else:
            corr_r = 0.0
        correlations_with_gradient.append(corr_r)

        # Edge vs. Non-Edge Disagreement Ratio
        edge_mask = edges_gt.flatten()
        if np.any(edge_mask) and np.any(~edge_mask):
            dis_on_edges = float(np.mean(dis_flat[edge_mask]))
            dis_on_bg = float(np.mean(dis_flat[~edge_mask]))
            edge_ratio = dis_on_edges / max(1e-8, dis_on_bg)
        else:
            edge_ratio = 1.0
        edge_to_background_ratios.append(edge_ratio)

        disagreement_means.append(float(np.mean(dis_flat)))
        disagreement_stds.append(float(np.std(dis_flat)))

        analysis_records.append({
            "patch_idx": idx,
            "y": meta["y"],
            "x": meta["x"],
            "ensemble_psnr_db": m_ens["psnr_db"],
            "ensemble_ssim": m_ens["ssim"],
            "m1_psnr_db": m_m1["psnr_db"],
            "m2_psnr_db": m_m2["psnr_db"],
            "m3_psnr_db": m_m3["psnr_db"],
            "mean_disagreement": round(float(np.mean(dis_flat)), 6),
            "max_disagreement": round(float(np.max(dis_flat)), 6),
            "corr_disagreement_sobel_gradient": round(corr_r, 4),
            "edge_to_background_disagreement_ratio": round(edge_ratio, 3),
        })

    # Save CSV analysis
    root = get_project_root()
    csv_out = root / "logs" / "ensemble_v2_disagreement_analysis.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(analysis_records[0].keys()))
        writer.writeheader()
        writer.writerows(analysis_records)
    print(f"[OK] Saved per-patch disagreement analysis to: {csv_out.relative_to(root)}")

    # Summary statistics
    r_summary = compute_distribution_summary(correlations_with_gradient, 4)
    ratio_summary = compute_distribution_summary(edge_to_background_ratios, 3)
    ens_psnr_sum = compute_distribution_summary(ensemble_psnrs, 2)
    ens_ssim_sum = compute_distribution_summary(ensemble_ssims, 4)
    m1_psnr_sum = compute_distribution_summary(member1_psnrs, 2)
    m2_psnr_sum = compute_distribution_summary(member2_psnrs, 2)
    m3_psnr_sum = compute_distribution_summary(member3_psnrs, 2)

    print(f"\n{'='*78}")
    print("   DISAGREEMENT INFORMATIVENESS FINDINGS")
    print(f"{'='*78}")
    print(f"Correlation r(Disagreement, Sobel Gradient) : {r_summary['mean']:.4f} ± {r_summary['std']:.4f} (Range: {r_summary['min']:.4f} to {r_summary['max']:.4f})")
    print(f"Edge / Non-Edge Disagreement Ratio         : {ratio_summary['mean']:.3f}x ± {ratio_summary['std']:.3f} (Range: {ratio_summary['min']:.3f}x to {ratio_summary['max']:.3f}x)")
    print(f"Ensemble Mean PSNR                         : {ens_psnr_sum['mean']:.2f} ± {ens_psnr_sum['std']:.2f} dB")
    print(f"Member 1 PSNR                              : {m1_psnr_sum['mean']:.2f} ± {m1_psnr_sum['std']:.2f} dB")
    print(f"Member 2 PSNR                              : {m2_psnr_sum['mean']:.2f} ± {m2_psnr_sum['std']:.2f} dB")
    print(f"Member 3 PSNR                              : {m3_psnr_sum['mean']:.2f} ± {m3_psnr_sum['std']:.2f} dB")
    print(f"Ensemble Mean SSIM                         : {ens_ssim_sum['mean']:.4f} ± {ens_ssim_sum['std']:.4f}")
    print(f"{'='*78}\n")

    return {
        "correlation_summary": r_summary,
        "ratio_summary": ratio_summary,
        "ensemble_psnr": ens_psnr_sum,
        "ensemble_ssim": ens_ssim_sum,
        "member_psnrs": [m1_psnr_sum, m2_psnr_sum, m3_psnr_sum],
        "pairwise_mses": {"1_vs_2": mse_1_2, "1_vs_3": mse_1_3, "2_vs_3": mse_2_3},
    }


def main() -> int:
    root = get_project_root()
    config = load_config()

    # 1. GPU Verification Pre-Check
    print("[PRE-CHECK 1/2] Running Phase 1 CUDA Smoke Test...")
    if not run_cuda_smoke_test():
        print("[FAIL] CUDA smoke test failed! Aborting.", file=sys.stderr)
        return 1

    device = torch.device("cuda:0")

    # 2. Data & Leakage Pre-Check
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    stack, _ = load_sentinel2_stack(raw_dir)

    print("[PRE-CHECK 2/2] Verifying zero spatial train/val overlap on Split v2...")
    t_ds = SentinelSRDataset(stack, split="train", split_mode="v2", buffer_pixels=0)
    v_ds = SentinelSRDataset(stack, split="val", split_mode="v2", buffer_pixels=0)
    if SentinelSRDataset.count_overlapping_pairs(t_ds, v_ds) != 0:
        print("[FAIL] Spatial leakage detected!", file=sys.stderr)
        return 1
    print("[PRE-CHECK 2/2 PASSED] Zero spatial leakage confirmed.\n")

    output_dir = root / "checkpoints" / "ensemble_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Sequential Training of 3 Members
    member_seeds = [(1, 42), (2, 101), (3, 2024)]
    results = []
    for m_idx, s in member_seeds:
        res = train_single_member(
            member_idx=m_idx,
            seed=s,
            output_dir=output_dir,
            device=device,
            config=config,
            stack=stack,
            epochs=15,
            batch_size=16,
            lr=0.0005,
            num_blocks=6,
            num_features=48,
            grad_weight=0.20,
        )
        results.append(res)

    # 4. Diversity & Informativeness Analysis
    info_res = analyze_ensemble_informativeness(output_dir, device, stack)

    print("[SUCCESS] Phase 7 Ensemble Retraining and Analysis Completed Successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
