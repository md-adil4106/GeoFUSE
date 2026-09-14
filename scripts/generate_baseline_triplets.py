"""Generate and visualize Sentinel-2 Pseudo-LR / Bicubic-Baseline / Pseudo-HR triplets.

Demonstrates the synthetic degrade-and-recover pipeline and establishes
the 2x bicubic interpolation baseline to beat in subsequent phases.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Project root setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.degrade import (
    bicubic_upsample,
    evaluate_reconstruction_fidelity,
    synthesize_pseudo_lr,
)
from src.data.tiling import extract_tiles, load_sentinel2_stack
from src.utils.config import get_project_root, load_config


def create_rgb_composite(
    tile_4band: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """Extract RGB bands (B04: Red, B03: Green, B02: Blue) and apply percentile stretch.

    Args:
        tile_4band: Array of shape (H, W, 4) with bands [B02, B03, B04, B08].

    Returns:
        np.ndarray: 8-bit RGB image of shape (H, W, 3).
    """
    # Channel indexing: B02 -> 0, B03 -> 1, B04 -> 2, B08 -> 3
    r = tile_4band[:, :, 2]
    g = tile_4band[:, :, 1]
    b = tile_4band[:, :, 0]

    channels = [r, g, b]
    stretched = []
    for ch in channels:
        valid = ch[ch > 0]
        if valid.size > 0:
            vmin = np.percentile(valid, p_low)
            vmax = np.percentile(valid, p_high)
            if vmax > vmin:
                norm = np.clip((ch - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                norm = np.zeros_like(ch, dtype=np.float32)
        else:
            norm = np.zeros_like(ch, dtype=np.float32)
        stretched.append((norm * 255.0).astype(np.uint8))

    return np.stack(stretched, axis=-1)


def render_triplet_figure(
    lr_tile: np.ndarray,
    bicubic_tile: np.ndarray,
    hr_tile: np.ndarray,
    metrics: dict,
    sample_index: int,
    output_path: Path,
) -> None:
    """Render a side-by-side comparison figure for a single triplet sample."""
    # Convert tiles to true-color RGB
    hr_rgb = create_rgb_composite(hr_tile)
    bicubic_rgb = create_rgb_composite(bicubic_tile)

    # Nearest-neighbor upscale of LR tile to HR dimensions for fair visual comparison
    lr_rgb_native = create_rgb_composite(lr_tile)
    lr_rgb_display = cv2.resize(
        lr_rgb_native, (hr_rgb.shape[1], hr_rgb.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    # Compute absolute error residual heatmap (using green channel or mean across channels)
    diff = np.abs(hr_tile.astype(float) - bicubic_tile.astype(float)).mean(axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)
    fig.patch.set_facecolor("#1e1e1e")

    titles = [
        f"Pseudo-LR (2x Downscaled)\n[{lr_tile.shape[0]}x{lr_tile.shape[1]} px, Nearest Display]",
        f"2x Bicubic Baseline\n[PSNR: {metrics['psnr_db']:.2f} dB | SSIM: {metrics['ssim']:.4f}]",
        f"Pseudo-HR Ground Truth (10m)\n[{hr_tile.shape[0]}x{hr_tile.shape[1]} px, Reference]",
        f"Residual Error Map\n[MAE: {metrics['mae']:.4f}]",
    ]

    images = [lr_rgb_display, bicubic_rgb, hr_rgb, diff]

    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        if i == 3:
            im = ax.imshow(img, cmap="inferno")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        else:
            ax.imshow(img)
        ax.set_title(title, color="white", fontsize=10, pad=8)
        ax.axis("off")

    plt.suptitle(
        f"GeoFUSE SentinelGuard — Synthetic Degrade-and-Recover Triplet (Sample #{sample_index})",
        color="white",
        fontsize=13,
        weight="bold",
        y=0.98,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    print("=" * 72)
    print("   GeoFUSE SentinelGuard — Phase 2: Baseline & Synthetic Degradation")
    print("=" * 72)

    root = get_project_root()
    config = load_config()

    # Degradation parameters from config.yaml
    deg_cfg = config.get("verification", {}).get("synthetic_degradation", {})
    downsample_factor = int(deg_cfg.get("downsample_factor", 2))
    blur_kernel = int(deg_cfg.get("blur_kernel_size", 3))
    noise_std = float(deg_cfg.get("noise_std", 0.01))

    # Tiling parameters (using 128x128 patches for clear visualization)
    patch_size = 128
    stride = 96

    print(f"\nDegradation Settings:")
    print(f"  Scale Factor      : {downsample_factor}x")
    print(f"  Blur Kernel Size  : {blur_kernel}x{blur_kernel} (Gaussian PSF)")
    print(f"  Radiometric Noise : std = {noise_std}")
    print(f"  Patch Dimensions  : {patch_size}x{patch_size} pixels")

    raw_rel = config.get("paths", {}).get("raw_data_dir", "data/raw")
    raw_dir = root / raw_rel
    output_preview_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "previews"

    # 1. Load multi-band Sentinel-2 stack
    print(f"\n[Step 1/4] Loading Sentinel-2 stack from {raw_dir.resolve()}...")
    try:
        stack, meta = load_sentinel2_stack(raw_dir)
        print(f"  [OK] Stack loaded: shape={stack.shape}, dtype={stack.dtype}")
        print(f"       CRS: {meta['crs']}, Resolution: {meta['resolution']}")
    except Exception as e:
        print(f"  [FAIL] Error loading Sentinel-2 data: {e}")
        return 1

    # 2. Extract diverse tiles
    print(f"\n[Step 2/4] Extracting candidate tiles across the scene...")
    all_tiles = extract_tiles(stack, patch_size=patch_size, stride=stride)
    print(f"  Extracted {len(all_tiles)} total candidate tiles.")

    # Select 4 representative diverse sample locations (e.g. corners and center)
    indices = [0, len(all_tiles) // 3, (2 * len(all_tiles)) // 3, len(all_tiles) - 1]
    selected_tiles = [all_tiles[i] for i in indices]

    # 3. Process each sample tile
    print(f"\n[Step 3/4] Synthesizing Pseudo-LR and computing Bicubic Baseline...")
    print("-" * 72)
    print(f"{'Sample':<8} | {'HR Shape':<12} | {'LR Shape':<12} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'MAE':<8}")
    print("-" * 72)

    triplet_results = []
    for idx, t_rec in enumerate(selected_tiles):
        hr_tile = t_rec["data"]  # (patch_size, patch_size, 4)

        # Synthesize pseudo-LR
        lr_tile = synthesize_pseudo_lr(
            hr_tile=hr_tile,
            downsample_factor=downsample_factor,
            blur_kernel_size=blur_kernel,
            noise_std=noise_std,
            seed=42 + idx,
        )

        # 2x Bicubic upsampling baseline
        bicubic_tile = bicubic_upsample(
            image=lr_tile,
            scale_factor=downsample_factor,
            target_shape=(hr_tile.shape[0], hr_tile.shape[1]),
        )

        # Quantitative evaluation
        metrics = evaluate_reconstruction_fidelity(
            ground_truth_hr=hr_tile,
            reconstructed_hr=bicubic_tile,
            data_range=1.0,  # Normalized reflectance range
        )

        # Check Stop Condition: Unrealistic degradation
        if metrics["psnr_db"] > 52.0:
            print(f"\n[STOP CONDITION TRIGGERED] PSNR is suspiciously high ({metrics['psnr_db']:.1f} dB).")
            print("Degraded pseudo-LR is near-identical to pseudo-HR. Check blur and downsampling parameters.")
            return 2
        if metrics["psnr_db"] < 15.0 or np.isnan(metrics["psnr_db"]):
            print(f"\n[STOP CONDITION TRIGGERED] PSNR is destructively low ({metrics['psnr_db']:.1f} dB).")
            print("Degradation destroyed the image. Check noise_std and scaling.")
            return 2

        print(
            f"#{idx:<7} | "
            f"{str(hr_tile.shape):<12} | "
            f"{str(lr_tile.shape):<12} | "
            f"{metrics['psnr_db']:<10.2f} | "
            f"{metrics['ssim']:<8.4f} | "
            f"{metrics['mae']:<8.4f}"
        )

        out_img_path = output_preview_dir / f"triplet_sample_{idx}.png"
        render_triplet_figure(
            lr_tile=lr_tile,
            bicubic_tile=bicubic_tile,
            hr_tile=hr_tile,
            metrics=metrics,
            sample_index=idx,
            output_path=out_img_path,
        )

        triplet_results.append({
            "sample_index": idx,
            "metrics": metrics,
            "preview_path": out_img_path,
        })

    print("-" * 72)

    # 4. Verification and Summary
    print(f"\n[Step 4/4] Saved preview triplet figures to: {output_preview_dir.resolve()}")
    for r in triplet_results:
        print(f"  - Sample #{r['sample_index']}: {r['preview_path'].name} "
              f"(PSNR: {r['metrics']['psnr_db']} dB, SSIM: {r['metrics']['ssim']})")

    avg_psnr = np.mean([r["metrics"]["psnr_db"] for r in triplet_results])
    avg_ssim = np.mean([r["metrics"]["ssim"] for r in triplet_results])
    print(f"\nBaseline Benchmark to Beat (Bicubic 2x Average):")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    print("\n[SUCCESS] Phase 2 baseline implementation and synthetic degradation verified!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
