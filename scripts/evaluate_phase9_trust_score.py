"""Evaluation Script for Phase 9 — Genuine Trust Score Improvement.

Evaluates the Composite Evidence Score and its constituent breakdown
before and after the Phase 6/7 model improvements:
- Baseline: Legacy Ensemble (outputs/checkpoints/ensemble_member_{0,1,2}.pth, 4 blocks, 48 feat)
- Improved: Phase 7 Ensemble (checkpoints/ensemble_v2/member_{1,2,3}.pt, 6 blocks, 48 feat, edge-tuned)

Evaluated under identical conditions on the host GPU (RTX 4060).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import csv
import json
import numpy as np
import pandas as pd
import torch

from src.data.dataset import SentinelSRDataset
from src.data.degrade import bicubic_upsample, evaluate_reconstruction_fidelity
from src.data.tiling import load_sentinel2_stack
from src.evaluation.edge_check import compute_edge_consistency, compute_gradient_magnitude
from src.evaluation.fusion import fuse_trust_risk_maps
from src.evaluation.spectral_check import compute_spectral_consistency
from src.evaluation.stability import compute_stability_map
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_project_root, load_config
from src.utils.experiment_tracker import compute_distribution_summary, get_git_commit_hash, log_experiment_record


def evaluate_ensemble_on_split(
    models,
    dataset,
    device,
    split_name: str,
    model_name: str,
    num_stability_trials: int = 2,
):
    print(f"\n--- Evaluating {model_name} on {split_name} ({len(dataset)} patches) on {device} ---")
    results = []

    for i in range(len(dataset)):
        item = dataset[i]
        if len(item) == 3:
            lr_t, hr_t, meta = item
        else:
            lr_t, hr_t = item
            meta = {"patch_idx": i}

        lr_np = lr_t.permute(1, 2, 0).numpy().astype(np.float32)
        hr_np = hr_t.permute(1, 2, 0).numpy().astype(np.float32)

        # 1. Ensemble Inference
        sr_np, disag_map, member_preds = predict_ensemble(models, lr_np, device=device)

        # 2. Input Stability Map
        stab_map, _, _ = compute_stability_map(
            models,
            lr_np,
            num_trials=num_stability_trials,
            device=device,
            base_seed=42 + i,
        )

        # 3. Spectral Consistency
        spec_metrics = compute_spectral_consistency(hr_np, sr_np, red_idx=2, nir_idx=3)

        # 4. Structural Gradient & Edge
        grad_gt = compute_gradient_magnitude(hr_np)
        grad_sr = compute_gradient_magnitude(sr_np)
        struct_diff = np.abs(grad_sr - grad_gt)
        edge_metrics = compute_edge_consistency(hr_np, sr_np)

        # 5. Evidence Fusion (Unchanged formula from config / fusion.py)
        fusion = fuse_trust_risk_maps(
            disagreement_map=disag_map,
            stability_map=stab_map,
            delta_ndvi_map=spec_metrics["delta_ndvi"],
            structural_diff_map=struct_diff,
        )

        # 6. Physical Fidelity Metrics
        fidelity = evaluate_reconstruction_fidelity(hr_np, sr_np, data_range=1.0)
        bicubic_np = bicubic_upsample(lr_np, scale_factor=2)
        bic_fid = evaluate_reconstruction_fidelity(hr_np, bicubic_np, data_range=1.0)

        record = {
            "model_name": model_name,
            "split_name": split_name,
            "patch_idx": i,
            # Evidence Score & Risk
            "trust_score_pct": fusion["trust_score_pct"],
            "mean_risk": fusion["mean_risk_score"],
            "pct_high_risk": fusion["pct_high_risk_pixels"],
            # Normalized components (each weighted by 0.25)
            "norm_disagreement": fusion["component_stats"]["disagreement"]["norm_mean"],
            "norm_stability": fusion["component_stats"]["stability"]["norm_mean"],
            "norm_spectral": fusion["component_stats"]["spectral"]["norm_mean"],
            "norm_structural": fusion["component_stats"]["structural"]["norm_mean"],
            # Raw component values
            "raw_disag_mean": fusion["component_stats"]["disagreement"]["raw_mean"],
            "raw_stab_mean": fusion["component_stats"]["stability"]["raw_mean"],
            "raw_spectral_mean": spec_metrics["mean_delta_ndvi"],
            "raw_struct_mean": float(np.mean(struct_diff)),
            # Physical metrics
            "psnr_db": fidelity["psnr_db"],
            "ssim": fidelity["ssim"],
            "psnr_gain": fidelity["psnr_db"] - bic_fid["psnr_db"],
            "gradient_corr": edge_metrics["gradient_correlation"],
            "edge_iou": edge_metrics["edge_iou"],
            "edge_f1": edge_metrics["edge_f1"],
        }
        results.append(record)

    df = pd.DataFrame(results)
    return df


def main():
    root = get_project_root()
    data_dir = root / "data" / "raw"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Phase 9 GPU evaluation!")

    print(f"[OK] Evaluating on GPU: {torch.cuda.get_device_name(0)}")

    stack, profile = load_sentinel2_stack(data_dir)

    # 1. Dataset splits: Phase 4 baseline split (49 patches, split_mode="v1")
    #    and Phase 5 leak-free split (25 patches, split_mode="v2")
    val_ds_v1 = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=128,
        stride=32,
        split="val",
        val_quadrant=(256, 512, 256, 512),
        seed=142,
        split_mode="v1",
    )

    val_ds_v2 = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=128,
        stride=32,
        split="val",
        val_quadrant=(256, 512, 256, 512),
        seed=142,
        split_mode="v2",
    )

    # 2. Models
    legacy_paths = [
        root / "outputs" / "checkpoints" / f"ensemble_member_{i}.pth" for i in (0, 1, 2)
    ]
    legacy_models = load_ensemble_members(legacy_paths, device=device)

    v2_paths = [root / "checkpoints" / "ensemble_v2" / f"member_{i}.pt" for i in (1, 2, 3)]
    v2_models = load_ensemble_members(v2_paths, device=device)

    # 3. Run evaluations on Phase 4 held-out set (49 patches, split v1)
    df_legacy_v1 = evaluate_ensemble_on_split(
        legacy_models, val_ds_v1, device, "Phase 4 Held-out (Split v1, 49 patches)", "Baseline Legacy Ensemble"
    )
    df_v2_v1 = evaluate_ensemble_on_split(
        v2_models, val_ds_v1, device, "Phase 4 Held-out (Split v1, 49 patches)", "Improved Ensemble v2"
    )

    # 4. Run evaluations on Phase 5 leak-free set (25 patches, split v2)
    df_legacy_v2 = evaluate_ensemble_on_split(
        legacy_models, val_ds_v2, device, "Clean Phase 5 (Split v2, 25 patches)", "Baseline Legacy Ensemble"
    )
    df_v2_v2 = evaluate_ensemble_on_split(
        v2_models, val_ds_v2, device, "Clean Phase 5 (Split v2, 25 patches)", "Improved Ensemble v2"
    )

    # Save detailed CSV
    combined_df = pd.concat([df_legacy_v1, df_v2_v1, df_legacy_v2, df_v2_v2], ignore_index=True)
    csv_out = root / "logs" / "phase9_trust_score_comparison.csv"
    combined_df.to_csv(csv_out, index=False)
    print(f"\n[OK] Saved comprehensive comparison to {csv_out.relative_to(root)}")

    # Print Summary Tables
    print("\n" + "=" * 90)
    print("   PHASE 9 RESULTS SUMMARY: PHASE 4 HELD-OUT SET (49 PATCHES, SPLIT V1)")
    print("=" * 90)
    metrics_to_show = [
        "trust_score_pct", "mean_risk", "norm_disagreement", "norm_stability",
        "norm_spectral", "norm_structural", "raw_disag_mean", "raw_stab_mean",
        "raw_spectral_mean", "raw_struct_mean", "psnr_db", "ssim", "gradient_corr"
    ]
    summary_v1 = pd.DataFrame({
        "Baseline Mean": df_legacy_v1[metrics_to_show].mean(),
        "Baseline Std": df_legacy_v1[metrics_to_show].std(),
        "Improved Mean": df_v2_v1[metrics_to_show].mean(),
        "Improved Std": df_v2_v1[metrics_to_show].std(),
        "Delta (Imp - Base)": df_v2_v1[metrics_to_show].mean() - df_legacy_v1[metrics_to_show].mean(),
    })
    print(summary_v1.round(5).to_string())

    print("\n" + "=" * 90)
    print("   PHASE 9 RESULTS SUMMARY: CLEAN PHASE 5 LEAK-FREE SET (25 PATCHES, SPLIT V2)")
    print("=" * 90)
    summary_v2 = pd.DataFrame({
        "Baseline Mean": df_legacy_v2[metrics_to_show].mean(),
        "Baseline Std": df_legacy_v2[metrics_to_show].std(),
        "Improved Mean": df_v2_v2[metrics_to_show].mean(),
        "Improved Std": df_v2_v2[metrics_to_show].std(),
        "Delta (Imp - Base)": df_v2_v2[metrics_to_show].mean() - df_legacy_v2[metrics_to_show].mean(),
    })
    print(summary_v2.round(5).to_string())


if __name__ == "__main__":
    main()
