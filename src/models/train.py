"""Training Pipeline for GeoFUSE SentinelGuard Super-Resolution.

Implements:
1. 2x Super-resolution training with Compound Loss (L1 + Sobel gradient).
2. Mixed precision acceleration (torch.cuda.amp) with automatic CPU fallback.
3. Strict geographic hold-out validation partitioning.
4. Sequential training of ensemble members with distinct random seeds.
5. Automatic OOM recovery (batch size halving).
6. Epoch-by-epoch CSV metrics logging and checkpoint preservation in outputs/checkpoints/.
"""

import csv
import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Project root setup
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.dataset import SentinelSRDataset
from src.data.degrade import evaluate_reconstruction_fidelity
from src.data.tiling import load_sentinel2_stack
from src.models.loss import CompoundSRLoss
from src.models.model import ResidualSRNet, build_model, count_parameters
from src.utils.config import get_device, get_project_root, load_config


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CompoundSRLoss,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    """Execute one full training epoch over the training partition."""
    model.train()
    running_l1 = 0.0
    running_grad = 0.0
    running_total = 0.0
    num_batches = 0

    for lr, hr, _ in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            sr = model(lr)
            loss, loss_dict = criterion(sr, hr)

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Training loss diverged to NaN/Inf!")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_l1 += loss_dict["l1"]
        running_grad += loss_dict["grad"]
        running_total += loss_dict["total"]
        num_batches += 1

    return {
        "train_l1": running_l1 / max(1, num_batches),
        "train_grad": running_grad / max(1, num_batches),
        "train_total": running_total / max(1, num_batches),
    }


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CompoundSRLoss,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    """Evaluate model on the spatially independent geographic hold-out partition."""
    model.eval()
    running_l1 = 0.0
    running_grad = 0.0
    running_total = 0.0
    psnr_scores = []
    ssim_scores = []
    num_batches = 0

    with torch.no_grad():
        for lr, hr, _ in loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                sr = model(lr)
                _, loss_dict = criterion(sr, hr)

            running_l1 += loss_dict["l1"]
            running_grad += loss_dict["grad"]
            running_total += loss_dict["total"]
            num_batches += 1

            # Compute PSNR & SSIM on batch items
            sr_np = sr.cpu().numpy()
            hr_np = hr.cpu().numpy()
            for b_idx in range(sr_np.shape[0]):
                pred_tile = np.transpose(sr_np[b_idx], (1, 2, 0))
                gt_tile = np.transpose(hr_np[b_idx], (1, 2, 0))
                m = evaluate_reconstruction_fidelity(gt_tile, pred_tile, data_range=1.0)
                psnr_scores.append(m["psnr_db"])
                ssim_scores.append(m["ssim"])

    return {
        "val_l1": running_l1 / max(1, num_batches),
        "val_grad": running_grad / max(1, num_batches),
        "val_total": running_total / max(1, num_batches),
        "val_psnr": float(np.mean(psnr_scores)) if psnr_scores else 0.0,
        "val_ssim": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
    }


def train_single_member(
    member_idx: int,
    seed: int,
    stack: np.ndarray,
    config: Dict[str, Any],
    checkpoints_dir: Path,
    device: torch.device,
    val_quadrant: Tuple[int, int, int, int] = (256, 512, 256, 512),
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Train a single ensemble member with a specific random seed and geographic hold-out."""
    # Deterministic seeding for this ensemble member
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    amp_enabled = (device.type == "cuda")
    epochs = epochs or int(config.get("training", {}).get("epochs", 15))
    batch_size = batch_size or int(config.get("training", {}).get("batch_size", 16))
    lr = float(config.get("training", {}).get("learning_rate", 0.0005))
    patch_size_hr = 128
    stride = 32

    # Datasets with member-specific seed for degradation/sampling variation
    train_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=patch_size_hr,
        stride=stride,
        split="train",
        val_quadrant=val_quadrant,
        seed=seed,
    )
    val_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=patch_size_hr,
        stride=stride,
        split="val",
        val_quadrant=val_quadrant,
        seed=seed + 100,
    )

    def create_loaders(bs: int):
        t_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=(device.type == "cuda"))
        v_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=(device.type == "cuda"))
        return t_loader, v_loader

    train_loader, val_loader = create_loaders(batch_size)

    # Initialize model
    model = build_model(config).to(device)
    criterion = CompoundSRLoss(channels=4, grad_weight=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    member_ckpt_path = checkpoints_dir / f"ensemble_member_{member_idx}.pth"
    member_log_path = checkpoints_dir / f"training_log_member_{member_idx}.csv"

    csv_header = [
        "epoch", "train_l1", "train_grad", "train_total",
        "val_l1", "val_grad", "val_total", "val_psnr", "val_ssim", "lr", "epoch_time_s"
    ]
    with open(member_log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_header)

    print(f"\n--- Training Ensemble Member #{member_idx} (Seed: {seed}) ---")
    print(f"{'Epoch':<6} | {'Train Total':<11} | {'Val Total':<10} | {'Val PSNR':<9} | {'Val SSIM':<8} | {'Time (s)':<8}")
    print("-" * 65)

    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        try:
            train_m = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                device=device,
                amp_enabled=amp_enabled,
            )
        except torch.cuda.OutOfMemoryError as oom:
            if batch_size > 4:
                batch_size = max(4, batch_size // 2)
                print(f"[OOM RECOVERY] Reduced batch size to {batch_size} for member #{member_idx}")
                torch.cuda.empty_cache()
                train_loader, val_loader = create_loaders(batch_size)
                continue
            else:
                raise RuntimeError(f"OOM at minimum batch size 4: {oom}")

        val_m = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        cur_lr = scheduler.get_last_lr()[0]

        with open(member_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_m['train_l1']:.5f}", f"{train_m['train_grad']:.5f}", f"{train_m['train_total']:.5f}",
                f"{val_m['val_l1']:.5f}", f"{val_m['val_grad']:.5f}", f"{val_m['val_total']:.5f}",
                f"{val_m['val_psnr']:.2f}", f"{val_m['val_ssim']:.4f}",
                f"{cur_lr:.6f}", f"{epoch_time:.2f}"
            ])

        is_best = val_m["val_total"] < best_val_loss
        if is_best:
            best_val_loss = val_m["val_total"]
            best_epoch = epoch
            torch.save(
                {
                    "member_idx": member_idx,
                    "seed": seed,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_m["val_total"],
                    "val_psnr": val_m["val_psnr"],
                    "val_ssim": val_m["val_ssim"],
                    "config": config,
                },
                member_ckpt_path,
            )

        flag = " [BEST]" if is_best else ""
        print(
            f"{epoch:<6} | "
            f"{train_m['train_total']:<11.5f} | "
            f"{val_m['val_total']:<10.5f} | "
            f"{val_m['val_psnr']:<9.2f} | "
            f"{val_m['val_ssim']:<8.4f} | "
            f"{epoch_time:<8.2f}{flag}"
        )

    duration = time.time() - start_time
    print(f"Member #{member_idx} Complete in {duration:.1f}s. Checkpoint: {member_ckpt_path.name}")

    # Explicit memory cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "member_idx": member_idx,
        "seed": seed,
        "checkpoint_path": member_ckpt_path,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def train_ensemble(
    config: Optional[Dict[str, Any]] = None,
    member_seeds: Optional[List[int]] = None,
    epochs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Sequentially train ensemble members with different seeds."""
    root = get_project_root()
    if config is None:
        config = load_config()

    if member_seeds is None:
        member_seeds = config.get("ensemble", {}).get("member_seeds", [42, 101, 2024])

    device = get_device(config)
    checkpoints_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    print(f"Loading Sentinel-2 stack from {raw_dir}...")
    stack, _ = load_sentinel2_stack(raw_dir)

    val_quadrant = (256, 512, 256, 512)
    epochs = epochs or int(config.get("training", {}).get("epochs", 15))

    print("=" * 76)
    print(f"   GeoFUSE SentinelGuard — Sequential Ensemble Training ({len(member_seeds)} Members)")
    print("=" * 76)

    results = []
    for m_idx, seed in enumerate(member_seeds):
        res = train_single_member(
            member_idx=m_idx,
            seed=seed,
            stack=stack,
            config=config,
            checkpoints_dir=checkpoints_dir,
            device=device,
            val_quadrant=val_quadrant,
            epochs=epochs,
        )
        results.append(res)

    print("\n" + "=" * 76)
    print("   [SUCCESS] All Ensemble Members Successfully Trained Sequentially!")
    print("=" * 76)
    for r in results:
        print(f"  - Member #{r['member_idx']} (Seed {r['seed']}): {r['checkpoint_path'].name} (Val Loss: {r['best_val_loss']:.5f})")

    return results


def train_baseline_experiment(
    config: Optional[Dict[str, Any]] = None,
    exp_id: str = "baseline_exp001",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute reproducible GPU baseline training run (exp001) on NVIDIA RTX 4060.

    Strictly enforces:
    1. GPU execution requirement (raises RuntimeError if CUDA unavailable, never falls back to CPU).
    2. Logs device placement of runtime, model, and first training tensor batch (all cuda:0).
    3. Executes with mixed precision (torch.amp.autocast('cuda') + torch.amp.GradScaler('cuda')).
    4. Records peak VRAM allocated and total training duration.
    5. Saves best_model.pth, latest_model.pth, and config.yaml to checkpoints/{exp_id}/.
    6. Saves metric history to logs/{exp_id}_metrics.csv.
    """
    import yaml

    root = get_project_root()
    if config is None:
        config = load_config()

    # 1. Hard GPU Requirement Check
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[GeoFUSE Error] CUDA is NOT available! Training requires a dedicated GPU (cuda:0). "
            "CPU execution is strictly prohibited for reproducible baseline training."
        )

    device = torch.device("cuda:0")
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
    gpu_name = torch.cuda.get_device_name(0)
    total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

    # 2. Output and Log Directories
    ckpt_dir = root / "checkpoints" / exp_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_log_path = logs_dir / f"{exp_id}_metrics.csv"
    ckpt_csv_log_path = ckpt_dir / "metrics.csv"

    # 3. Data Ingestion
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    stack, _ = load_sentinel2_stack(raw_dir)

    # 4. Hyperparameters
    epochs = epochs or int(config.get("training", {}).get("epochs", 15))
    batch_size = batch_size or int(config.get("training", {}).get("batch_size", 16))
    lr = float(config.get("training", {}).get("learning_rate", 0.0005))
    patch_size_hr = 128
    stride = 32
    val_quadrant = (256, 512, 256, 512)

    # Deterministic Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 5. Datasets & Loaders
    train_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=patch_size_hr,
        stride=stride,
        split="train",
        val_quadrant=val_quadrant,
        seed=seed,
    )
    val_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=patch_size_hr,
        stride=stride,
        split="val",
        val_quadrant=val_quadrant,
        seed=seed + 100,
    )

    def create_loaders(bs: int):
        t_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True)
        v_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True)
        return t_loader, v_loader

    train_loader, val_loader = create_loaders(batch_size)

    # 6. Model, Loss, Optimizer & Scaler
    model = build_model(config).to(device)
    criterion = CompoundSRLoss(channels=4, grad_weight=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # Verify device of model parameters
    model_param_device = next(model.parameters()).device
    if model_param_device.type != "cuda":
        raise RuntimeError(f"[GeoFUSE Error] Model device is {model_param_device}, expected cuda:0!")

    # Verify device of first batch tensor
    first_lr, first_hr, _ = next(iter(train_loader))
    first_lr_dev = first_lr.to(device, non_blocking=True)
    first_hr_dev = first_hr.to(device, non_blocking=True)
    if first_lr_dev.device.type != "cuda":
        raise RuntimeError(f"[GeoFUSE Error] First batch tensor device is {first_lr_dev.device}, expected cuda:0!")

    # Log Training Start Telemetry
    print("=" * 76)
    print(f"   GeoFUSE SentinelGuard — Baseline Training Run ({exp_id})")
    print("=" * 76)
    print(f"PyTorch Version       : {torch_version}")
    print(f"CUDA Version          : {cuda_version}")
    print(f"GPU Name              : {gpu_name}")
    print(f"Total GPU VRAM        : {total_memory_mb:.1f} MiB")
    print(f"Selected Device       : {device}")
    print(f"Model Device          : {model_param_device}")
    print(f"First Batch Device    : {first_lr_dev.device}")
    print(f"Model Parameters      : {count_parameters(model):,}")
    print(f"Training Patches      : {len(train_ds)} (Batch Size: {batch_size})")
    print(f"Validation Patches    : {len(val_ds)}")
    print(f"Epochs                : {epochs}")
    print(f"Mixed Precision       : Enabled (torch.amp.autocast('cuda'))")
    print("-" * 76)

    # Initialize CSV logs
    csv_header = [
        "epoch", "train_l1", "train_grad", "train_total",
        "val_l1", "val_grad", "val_total", "val_psnr", "val_ssim",
        "lr", "epoch_time_s", "peak_vram_mib"
    ]
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_header)

    torch.cuda.reset_peak_memory_stats(device)
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics = {}
    total_start_time = time.time()

    print(f"{'Epoch':<6} | {'Train Total':<11} | {'Val Total':<10} | {'Val PSNR':<9} | {'Val SSIM':<8} | {'VRAM (MiB)':<10} | {'Time (s)':<8}")
    print("-" * 76)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Check for OOM with single allowed reduction as per Stop Condition
        try:
            train_m = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                device=device,
                amp_enabled=True,
            )
        except torch.cuda.OutOfMemoryError as oom:
            if batch_size > 4:
                batch_size = max(4, batch_size // 2)
                print(f"\n[STOP CONDITION TRIGGERED / RECOVERED] OOM at previous batch size. Reduced batch size once to {batch_size}.")
                torch.cuda.empty_cache()
                train_loader, val_loader = create_loaders(batch_size)
                train_m = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    scaler=scaler,
                    device=device,
                    amp_enabled=True,
                )
            else:
                raise RuntimeError(f"[ABORT] OOM at minimum batch size 4: {oom}")

        # Validation
        val_m = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=True,
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        cur_lr = scheduler.get_last_lr()[0]
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # Log row to CSV
        row = [
            epoch,
            f"{train_m['train_l1']:.5f}", f"{train_m['train_grad']:.5f}", f"{train_m['train_total']:.5f}",
            f"{val_m['val_l1']:.5f}", f"{val_m['val_grad']:.5f}", f"{val_m['val_total']:.5f}",
            f"{val_m['val_psnr']:.2f}", f"{val_m['val_ssim']:.4f}",
            f"{cur_lr:.6f}", f"{epoch_time:.2f}", f"{peak_vram:.2f}"
        ]
        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        is_best = val_m["val_total"] < best_val_loss
        if is_best:
            best_val_loss = val_m["val_total"]
            best_epoch = epoch
            best_metrics = {
                "val_total": val_m["val_total"],
                "val_psnr": val_m["val_psnr"],
                "val_ssim": val_m["val_ssim"],
            }
            torch.save(
                {
                    "exp_id": exp_id,
                    "seed": seed,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_m["val_total"],
                    "val_psnr": val_m["val_psnr"],
                    "val_ssim": val_m["val_ssim"],
                    "config": config,
                },
                ckpt_dir / "best_model.pth",
            )

        flag = " [BEST]" if is_best else ""
        print(
            f"{epoch:<6} | "
            f"{train_m['train_total']:<11.5f} | "
            f"{val_m['val_total']:<10.5f} | "
            f"{val_m['val_psnr']:<9.2f} | "
            f"{val_m['val_ssim']:<8.4f} | "
            f"{peak_vram:<10.2f} | "
            f"{epoch_time:<8.2f}{flag}"
        )

    total_duration = time.time() - total_start_time
    peak_vram_final = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Save Latest Model Checkpoint
    torch.save(
        {
            "exp_id": exp_id,
            "seed": seed,
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_m["val_total"],
            "val_psnr": val_m["val_psnr"],
            "val_ssim": val_m["val_ssim"],
            "config": config,
        },
        ckpt_dir / "latest_model.pth",
    )

    # Save accompanying experiment config file
    exp_config = {
        "experiment": {
            "exp_id": exp_id,
            "description": "Reproducible GPU-verified baseline training run on NVIDIA RTX 4060",
            "device": str(device),
            "gpu_name": gpu_name,
            "cuda_version": cuda_version,
            "torch_version": str(torch_version),
            "seed": seed,
            "total_duration_s": round(total_duration, 2),
            "peak_vram_mib": round(peak_vram_final, 2),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_psnr": best_metrics.get("val_psnr"),
            "best_val_ssim": best_metrics.get("val_ssim"),
        },
        "model": {
            "architecture": "ResidualSRNet",
            "scale_factor": 2,
            "num_channels": 4,
            "num_features": 48,
            "num_residual_blocks": 4,
            "parameter_count": count_parameters(model),
        },
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": 1e-4,
            "loss": "CompoundSRLoss (L1 + 0.1 * SobelGradient)",
            "scheduler": "CosineAnnealingLR (T_max=15, eta_min=1e-6)",
            "mixed_precision": True,
            "patch_size_hr": patch_size_hr,
            "stride": stride,
            "val_quadrant": list(val_quadrant),
        }
    }
    with open(ckpt_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(exp_config, f, sort_keys=False)

    # Mirror metrics CSV in checkpoint dir
    import shutil
    shutil.copy2(csv_log_path, ckpt_csv_log_path)

    print("-" * 76)
    print(f"[SUCCESS] Baseline Training Run '{exp_id}' Completed on {device}!")
    print(f"Total Duration        : {total_duration:.2f}s")
    print(f"Peak VRAM Usage       : {peak_vram_final:.2f} MiB")
    print(f"Best Validation Epoch : Epoch #{best_epoch}")
    print(f"Best Val PSNR / SSIM  : {best_metrics.get('val_psnr'):.2f} dB / {best_metrics.get('val_ssim'):.4f}")
    print(f"Artifacts Saved To    : {ckpt_dir.relative_to(root)}")
    print(f"Metrics Log Saved To  : {csv_log_path.relative_to(root)}")
    print("=" * 76)

    return {
        "exp_id": exp_id,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_psnr": best_metrics.get("val_psnr"),
        "best_val_ssim": best_metrics.get("val_ssim"),
        "total_duration_s": total_duration,
        "peak_vram_mib": peak_vram_final,
        "ckpt_dir": ckpt_dir,
        "csv_log_path": csv_log_path,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="GeoFUSE SentinelGuard Training Entry Point")
    parser.add_argument("--baseline", action="store_true", help="Execute reproducible baseline exp001 on GPU")
    parser.add_argument("--ensemble", action="store_true", help="Execute 3-member ensemble training")
    parser.add_argument("--exp-id", type=str, default="baseline_exp001", help="Experiment ID for baseline run")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    args = parser.parse_args()

    # Enforce GPU requirement at entrypoint
    if not torch.cuda.is_available():
        print("[GeoFUSE ERROR] CUDA is NOT available! Training requires a dedicated GPU (cuda:0).", file=sys.stderr)
        print("Aborting training to prevent silent CPU execution.", file=sys.stderr)
        return 1

    try:
        if args.ensemble:
            train_ensemble(epochs=args.epochs)
        else:
            train_baseline_experiment(exp_id=args.exp_id, epochs=args.epochs, batch_size=args.batch_size)
        return 0
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
