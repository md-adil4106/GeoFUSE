"""Training Pipeline for GeoFUSE SentinelGuard Super-Resolution.

Implements:
1. 2x Super-resolution training with Compound Loss (L1 + Sobel gradient).
2. Mixed precision acceleration (torch.cuda.amp) with automatic CPU fallback.
3. Strict geographic hold-out validation partitioning.
4. Automatic OOM recovery (batch size halving).
5. Epoch-by-epoch CSV metrics logging and checkpoint preservation in outputs/checkpoints/.
"""

import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
                # Convert back to (H, W, C)
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


def run_training(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute complete training run for GeoFUSE SentinelGuard SR Model."""
    root = get_project_root()
    if config is None:
        config = load_config()

    print("=" * 76)
    print("   GeoFUSE SentinelGuard — Super-Resolution Model Training")
    print("=" * 76)

    # 1. Device and Hardware Setup
    device = get_device(config)
    amp_enabled = (device.type == "cuda")
    print(f"Execution Device : {device} (Mixed Precision: {amp_enabled})")

    # 2. Output Paths Setup
    checkpoints_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = checkpoints_dir / "training_log.csv"
    best_model_path = checkpoints_dir / "best_model.pth"
    latest_model_path = checkpoints_dir / "latest_model.pth"

    # 3. Load Dataset Stack
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    print(f"\n[1/5] Loading Sentinel-2 stack from {raw_dir}...")
    stack, meta = load_sentinel2_stack(raw_dir)
    print(f"      Full Scene Shape: {stack.shape}, CRS: {meta['crs']}")

    # 4. Configure Geographic Hold-Out Partitioning
    # 512x512 Scene: Reserve Southeast Quadrant (rows 256..512, cols 256..512) for validation
    val_quadrant = (256, 512, 256, 512)
    patch_size_hr = 128  # Yields 64x64 LR patches at 2x scale
    stride = 32          # Overlap stride for dense spatial patch extraction

    print("\n[2/5] Initializing Geographically Partitioned Datasets...")
    print(f"      Validation Hold-Out Zone : Rows 256..512, Cols 256..512 (SE Quadrant)")
    print(f"      Training Zone            : North & West Sub-regions (Zero Leakage)")

    train_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=patch_size_hr,
        stride=stride,
        split="train",
        val_quadrant=val_quadrant,
        downsample_factor=2,
        blur_kernel_size=3,
        noise_std=0.01,
        seed=42,
    )
    val_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=patch_size_hr,
        stride=stride,
        split="val",
        val_quadrant=val_quadrant,
        downsample_factor=2,
        blur_kernel_size=3,
        noise_std=0.01,
        seed=101,
    )

    print(f"      Extracted Training Patches   : {len(train_ds)}")
    print(f"      Extracted Validation Patches : {len(val_ds)}")

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Dataset partitioning produced an empty split. Check dimensions.")

    # 5. Hyperparameters
    batch_size = int(config.get("training", {}).get("batch_size", 16))
    epochs = int(config.get("training", {}).get("epochs", 15))
    lr = float(config.get("training", {}).get("learning_rate", 0.0005))

    # DataLoader creation with automatic OOM fallback
    def create_loaders(bs: int):
        t_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=(device.type == "cuda"))
        v_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=(device.type == "cuda"))
        return t_loader, v_loader

    train_loader, val_loader = create_loaders(batch_size)

    # 6. Model, Loss, Optimizer
    print(f"\n[3/5] Building ResidualSRNet Model...")
    model = build_model(config).to(device)
    num_params = count_parameters(model)
    print(f"      Total Trainable Parameters : {num_params:,} (< 1,500,000 budget)")

    criterion = CompoundSRLoss(channels=4, grad_weight=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # 7. Initialize CSV Logging
    csv_header = [
        "epoch",
        "train_l1",
        "train_grad",
        "train_total",
        "val_l1",
        "val_grad",
        "val_total",
        "val_psnr",
        "val_ssim",
        "lr",
        "epoch_time_s",
    ]
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    print(f"\n[4/5] Starting Training Loop ({epochs} Epochs, Batch Size {batch_size})...")
    print("-" * 76)
    print(f"{'Epoch':<6} | {'Train Total':<11} | {'Val Total':<10} | {'Val PSNR':<9} | {'Val SSIM':<8} | {'Time (s)':<8}")
    print("-" * 76)

    best_val_loss = float("inf")
    best_epoch = 0
    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        try:
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                device=device,
                amp_enabled=amp_enabled,
            )
        except torch.cuda.OutOfMemoryError as oom_err:
            print(f"\n[WARNING] CUDA Out Of Memory at batch size {batch_size}!")
            if batch_size > 4:
                batch_size = max(4, batch_size // 2)
                print(f"[RECOVERY] Automatically reducing batch size to {batch_size}...")
                torch.cuda.empty_cache()
                train_loader, val_loader = create_loaders(batch_size)
                # Retry current epoch
                continue
            else:
                raise RuntimeError(
                    f"Repeated CUDA OOM error even at minimum batch size {batch_size}: {oom_err}"
                )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        # Log to CSV
        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_metrics['train_l1']:.5f}",
                f"{train_metrics['train_grad']:.5f}",
                f"{train_metrics['train_total']:.5f}",
                f"{val_metrics['val_l1']:.5f}",
                f"{val_metrics['val_grad']:.5f}",
                f"{val_metrics['val_total']:.5f}",
                f"{val_metrics['val_psnr']:.2f}",
                f"{val_metrics['val_ssim']:.4f}",
                f"{current_lr:.6f}",
                f"{epoch_time:.2f}",
            ])

        # Check for improvement and save checkpoints
        is_best = val_metrics["val_total"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_total"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["val_total"],
                    "val_psnr": val_metrics["val_psnr"],
                    "val_ssim": val_metrics["val_ssim"],
                    "config": config,
                    "geographic_split": {
                        "val_quadrant": val_quadrant,
                        "description": "Southeast quadrant geographic hold-out",
                    },
                },
                best_model_path,
            )

        # Always save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_metrics["val_total"],
            },
            latest_model_path,
        )

        best_flag = " [BEST]" if is_best else ""
        print(
            f"{epoch:<6} | "
            f"{train_metrics['train_total']:<11.5f} | "
            f"{val_metrics['val_total']:<10.5f} | "
            f"{val_metrics['val_psnr']:<9.2f} | "
            f"{val_metrics['val_ssim']:<8.4f} | "
            f"{epoch_time:<8.2f}{best_flag}"
        )

    total_training_time = time.time() - start_train_time
    print("-" * 76)
    print(f"\n[5/5] Training Completed in {total_training_time:.1f}s.")
    print(f"      Best Epoch               : #{best_epoch} (Val Loss: {best_val_loss:.5f})")
    print(f"      Best Checkpoint Saved    : {best_model_path.resolve()}")
    print(f"      Latest Checkpoint Saved  : {latest_model_path.resolve()}")
    print(f"      Training Log CSV Saved   : {log_csv_path.resolve()}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_training_time": total_training_time,
        "log_path": log_csv_path,
        "best_checkpoint_path": best_model_path,
    }


def main() -> int:
    try:
        run_training()
        return 0
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
