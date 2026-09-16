"""Phase 12: Final Validation Pass & End-to-End Evidence Pipeline Verification.

Runs a rigorous full validation pass on the final recommended model
(Phase 7 Ensemble v2: 3-member ResidualSRNet-6b, 48 features, L1 + 0.20 * Grad)
across all held-out validation patches on the RTX 4060 GPU.

Verifies end-to-end execution through the complete GeoFUSE evidence pipeline:
1. Ensemble Super-Resolution Inference
2. Epistemic Disagreement Calculation
3. Input Perturbation Stability Testing
4. Spectral NDVI Verification
5. Structural Edge Consistency & Gradient Difference
6. Multi-Criteria Evidence Fusion (Trust & Risk Maps)
7. Downstream Building Footprint Morphological Extraction
8. Cryptographic / Auditable Trust Receipt Generation
"""

from pathlib import Path
import csv
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import SentinelSRDataset
from src.data.degrade import bicubic_upsample, evaluate_reconstruction_fidelity
from src.data.tiling import load_sentinel2_stack
from src.evaluation.downstream_eval import extract_building_footprints, compare_downstream_footprints
from src.evaluation.edge_check import compute_edge_consistency, compute_gradient_magnitude
from src.evaluation.fusion import fuse_trust_risk_maps
from src.evaluation.spectral_check import compute_spectral_consistency
from src.evaluation.stability import compute_stability_map
from src.evaluation.trust_receipt import generate_trust_receipt
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_project_root, load_config
from src.utils.experiment_tracker import compute_distribution_summary


def main():
    print("=" * 84)
    print("   GeoFUSE SentinelGuard — Phase 12: Final Full Validation & Pipeline Pass")
    print("=" * 84)

    root = get_project_root()
    config = load_config(root / "config.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        raise RuntimeError("FATAL: Final validation pass requires CUDA GPU.")

    gpu_name = torch.cuda.get_device_name(device)
    vram_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    print(f"[OK] GPU Hardware Verified: {gpu_name} ({vram_total:.1f} MiB VRAM) on {device}")

    # 1. Load Data
    data_dir = root / "data" / "raw"
    stack, profile = load_sentinel2_stack(data_dir)
    print(f"[OK] Loaded Sentinel-2 tile stack: shape {stack.shape}, dtype {stack.dtype}")

    # 2. Validation Dataset: Clean Phase 5 Split v2 (25 held-out patches, zero spatial leakage)
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
    print(f"[OK] Loaded clean held-out validation dataset: {len(val_ds)} patches (Zero Spatial Leakage)")

    # 3. Load Final Recommended Model: Ensemble v2 (3 members of ResidualSRNet-6b, 48 feat)
    ckpt_paths = [root / "checkpoints" / "ensemble_v2" / f"member_{i}.pt" for i in (1, 2, 3)]
    models = load_ensemble_members(ckpt_paths, device=device)
    print(f"[OK] Successfully loaded {len(models)} ensemble members from checkpoints/ensemble_v2/")

    # 4. Execute End-to-End Validation Pass Across All Held-Out Patches
    print(f"\n--- Executing Full Evidence Pipeline Across all {len(val_ds)} Held-Out Patches on {device} ---")
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    records = []
    latencies = []

    for i in range(len(val_ds)):
        item = val_ds[i]
        lr_t, hr_t = item[0], item[1]
        lr_np = lr_t.permute(1, 2, 0).numpy().astype(np.float32)
        hr_np = hr_t.permute(1, 2, 0).numpy().astype(np.float32)

        # Stage 1: Ensemble Super-Resolution Inference (Timed)
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        sr_np, disag_map, member_preds = predict_ensemble(models, lr_np, device=device)
        torch.cuda.synchronize()
        t_infer_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(t_infer_ms)

        # Stage 2: Bicubic Reference Baseline
        bicubic_np = bicubic_upsample(lr_np, scale_factor=2, target_shape=(hr_np.shape[0], hr_np.shape[1]))

        # Stage 3: Physical Reconstruction Fidelity
        fid_sr = evaluate_reconstruction_fidelity(hr_np, sr_np, data_range=1.0)
        fid_bic = evaluate_reconstruction_fidelity(hr_np, bicubic_np, data_range=1.0)

        # Stage 4: Input Perturbation Stability Map
        stab_map, _, _ = compute_stability_map(
            models,
            lr_np,
            num_trials=2,
            device=device,
            base_seed=42 + i,
        )

        # Stage 5: Spectral NDVI Consistency
        spec_metrics = compute_spectral_consistency(hr_np, sr_np, red_idx=2, nir_idx=3)

        # Stage 6: Structural Edge Consistency & Gradient Difference
        grad_gt = compute_gradient_magnitude(hr_np)
        grad_sr = compute_gradient_magnitude(sr_np)
        struct_diff = np.abs(grad_sr - grad_gt)
        edge_metrics = compute_edge_consistency(hr_np, sr_np)

        # Stage 7: Multi-Criteria Evidence Fusion
        fusion = fuse_trust_risk_maps(
            disagreement_map=disag_map,
            stability_map=stab_map,
            delta_ndvi_map=spec_metrics["delta_ndvi"],
            structural_diff_map=struct_diff,
        )

        # Stage 8: Downstream Building Footprint Analysis
        down_cfg = config.get("downstream", {})
        morph_cfg = down_cfg.get("morphology", {})
        foot_bic = extract_building_footprints(
            bicubic_np,
            tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
            tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
            max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
            min_area=int(morph_cfg.get("min_area", 4)),
            max_area=int(morph_cfg.get("max_area", 600)),
            red_idx=2,
            nir_idx=3,
        )
        foot_sr = extract_building_footprints(
            sr_np,
            tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
            tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
            max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
            min_area=int(morph_cfg.get("min_area", 4)),
            max_area=int(morph_cfg.get("max_area", 600)),
            red_idx=2,
            nir_idx=3,
        )
        foot_ref = extract_building_footprints(
            hr_np,
            tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
            tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
            max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
            min_area=int(morph_cfg.get("min_area", 4)),
            max_area=int(morph_cfg.get("max_area", 600)),
            red_idx=2,
            nir_idx=3,
        )
        down_comp = compare_downstream_footprints(
            mask_bicubic=foot_bic["mask"],
            mask_sr=foot_sr["mask"],
            trust_map=fusion["trust_map"],
            ref_mask=foot_ref["mask"],
            trust_threshold=0.85,
        )

        # Stage 9: Cryptographic Trust Receipt Generation (Sample #0 test)
        if i == 0:
            pipeline_data = {
                "fusion_result": fusion,
                "spectral_metrics": spec_metrics,
                "edge_metrics": edge_metrics,
                "downstream_comp": down_comp,
            }
            receipt = generate_trust_receipt(
                tile_idx=0,
                raw_dir=data_dir,
                config=config,
                pipeline_data=pipeline_data,
                min_trust_threshold=86.5,
            )
            assert "receipt_id" in receipt and "scientific_transparency_mandate" in receipt
            print(f"[OK] Verified Stage 9: Generated valid Trust Receipt {receipt['receipt_id']}")

        record = {
            "patch_idx": i,
            # Fidelity
            "sr_psnr_db": fid_sr["psnr_db"],
            "bicubic_psnr_db": fid_bic["psnr_db"],
            "psnr_gain_db": fid_sr["psnr_db"] - fid_bic["psnr_db"],
            "sr_ssim": fid_sr["ssim"],
            "bicubic_ssim": fid_bic["ssim"],
            "ssim_gain": fid_sr["ssim"] - fid_bic["ssim"],
            "sr_mae": fid_sr["mae"],
            # Spectral
            "delta_ndvi": spec_metrics["mean_delta_ndvi"],
            "pct_inconsistent_pixels": spec_metrics["pct_inconsistent_pixels"],
            # Structural
            "gradient_correlation": edge_metrics["gradient_correlation"],
            "edge_iou": edge_metrics["edge_iou"],
            "edge_f1": edge_metrics["edge_f1"],
            # Evidence Fusion
            "trust_score_pct": fusion["trust_score_pct"],
            "mean_risk": fusion["mean_risk_score"],
            "norm_disagreement": fusion["component_stats"]["disagreement"]["norm_mean"],
            "norm_stability": fusion["component_stats"]["stability"]["norm_mean"],
            "norm_spectral": fusion["component_stats"]["spectral"]["norm_mean"],
            "norm_structural": fusion["component_stats"]["structural"]["norm_mean"],
            # Downstream
            "downstream_bic_vs_sr_iou": down_comp["overall_bic_sr"]["iou"],
            "downstream_sr_vs_ref_iou": down_comp["reference_comparison"]["sr_vs_ref_iou"],
            "high_trust_region_iou": down_comp["high_trust_bic_sr"]["iou"],
            # Compute
            "latency_ms": t_infer_ms,
        }
        records.append(record)

    total_eval_time = time.time() - t0
    peak_vram_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # 5. Save per-patch CSV
    csv_path = root / "logs" / "final_model_validation_distribution.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"[OK] Saved final full validation distribution to: {csv_path.relative_to(root)}")

    # 6. Compute & Display Summary Distributions
    df = pd.DataFrame(records)
    print("\n" + "=" * 84)
    print("   PHASE 12 FINAL MODEL VALIDATION DISTRIBUTIONS (25 HELD-OUT PATCHES)")
    print("=" * 84)

    metrics_to_print = [
        ("sr_psnr_db", "SR Reconstruction PSNR (dB)", 2),
        ("bicubic_psnr_db", "Bicubic Baseline PSNR (dB)", 2),
        ("psnr_gain_db", "PSNR Gain over Bicubic (dB)", 2),
        ("sr_ssim", "SR Structural Similarity (SSIM)", 4),
        ("bicubic_ssim", "Bicubic Baseline SSIM", 4),
        ("ssim_gain", "SSIM Gain over Bicubic", 4),
        ("delta_ndvi", "Spectral Error (Mean Delta-NDVI)", 5),
        ("pct_inconsistent_pixels", "Inconsistent Pixels (Delta > 0.05)", 2),
        ("gradient_correlation", "Sobel Gradient Correlation (r)", 4),
        ("edge_iou", "Canny Edge IoU", 4),
        ("edge_f1", "Canny Edge F1 Score", 4),
        ("trust_score_pct", "Composite Evidence Score (%)", 2),
        ("mean_risk", "Mean Composite Risk", 4),
        ("downstream_bic_vs_sr_iou", "Downstream Footprint Agreement (IoU)", 4),
        ("downstream_sr_vs_ref_iou", "Downstream Footprint Accuracy (vs GT)", 4),
        ("latency_ms", "Ensemble Inference Latency (ms)", 2),
    ]

    for key, label, decimals in metrics_to_print:
        mean_val = df[key].mean()
        std_val = df[key].std()
        min_val = df[key].min()
        max_val = df[key].max()
        print(f"  {label:<38s}: {mean_val:.{decimals}f} ± {std_val:.{decimals}f} (Min: {min_val:.{decimals}f}, Max: {max_val:.{decimals}f})")

    print(f"\n  Peak GPU Memory During Pipeline Pass : {peak_vram_mib:.2f} MiB")
    print(f"  Total Pipeline Execution Time        : {total_eval_time:.2f}s ({len(val_ds)} patches)")
    print(f"  Ensemble Throughput                  : {1000.0 / df['latency_ms'].mean():.1f} patches/sec")

    # Assertions for Scientific Integrity
    assert df["sr_psnr_db"].mean() > df["bicubic_psnr_db"].mean(), "Model must beat bicubic on average PSNR"
    assert df["sr_ssim"].mean() > df["bicubic_ssim"].mean(), "Model must beat bicubic on average SSIM"
    assert df["psnr_gain_db"].mean() > 0.15, "Ensemble mean should provide >= 0.15 dB gain over bicubic"
    print("\n[SUCCESS] Final Model Validation Pass Passed all scientific integrity assertions!")


if __name__ == "__main__":
    main()
