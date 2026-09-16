"""GeoFUSE SentinelGuard — Phase 13 End-to-End Final Demo Smoke Test.

Validates:
1. Locked final ensemble checkpoints in checkpoints/final_demo_v1/ load correctly on GPU.
2. Dynamic CPU fallback loads correctly with map_location='cpu'.
3. Full pipeline inference:
   - Synthetic pseudo-LR generation
   - 2x Super-Resolution prediction
   - Pixel-wise Ensemble Disagreement map (Uncertainty Proxy)
   - Perturbation Stability map
   - Spectral Consistency check (Delta-NDVI)
   - Structural Consistency check (Sobel gradient + Canny edges)
   - Heuristic Evidence Fusion (Trust & Risk maps, Composite Evidence Score)
   - Downstream Building Footprint Analysis
   - Auditable Cryptographic Trust Receipt generation & verification
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.data.degrade import bicubic_upsample, synthesize_pseudo_lr
from src.data.tiling import extract_tiles, load_sentinel2_stack
from src.evaluation.downstream_eval import (
    compare_downstream_footprints,
    extract_building_footprints,
)
from src.evaluation.edge_check import compute_edge_consistency, compute_gradient_magnitude
from src.evaluation.fusion import fuse_trust_risk_maps
from src.evaluation.spectral_check import compute_spectral_consistency
from src.evaluation.stability import compute_stability_map
from src.evaluation.trust_receipt import (
    extract_geotiff_metadata,
    generate_trust_receipt,
    save_trust_receipt,
)
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_device, get_project_root, load_config


def main() -> int:
    print("=" * 86)
    print("      GeoFUSE SentinelGuard — Phase 13 Final Demo Model Smoke Test")
    print("=" * 86)

    root = get_project_root()
    config = load_config()
    final_dir = root / "checkpoints" / "final_demo_v1"

    # -------------------------------------------------------------------------
    # 1. Inspect Checkpoint Integrity & Manifest
    # -------------------------------------------------------------------------
    print("\n[Step 1/8] Verifying checkpoints/final_demo_v1/ directory & manifest...")
    manifest_path = final_dir / "model_manifest.json"
    if not manifest_path.exists():
        print(f"[FAIL] Manifest not found at {manifest_path}")
        return 1

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print(f"  Model Name    : {manifest.get('model_name')}")
    print(f"  Tag           : {manifest.get('tag')}")
    print(f"  Architecture  : {manifest.get('architecture')} ({manifest.get('num_blocks')} blocks, {manifest.get('num_features')} features)")
    print(f"  Loss Function : {manifest.get('loss_function')}")
    print(f"  Members Count : {len(manifest.get('members', []))}")

    ckpt_paths = [final_dir / f"ensemble_member_{i}.pth" for i in range(3)]
    for p in ckpt_paths:
        if not p.exists():
            print(f"[FAIL] Checkpoint missing: {p}")
            return 1
        with open(p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        print(f"  [OK] {p.name}: {p.stat().st_size:,} bytes | SHA256: {h[:16]}...")

    # -------------------------------------------------------------------------
    # 2. Test GPU Model Loading
    # -------------------------------------------------------------------------
    device = get_device(config)
    print(f"\n[Step 2/8] Testing GPU model loading on: {device}...")
    if device.type == "cuda":
        print(f"  Device Name   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Total    : {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MiB")

    t0 = time.time()
    gpu_models = load_ensemble_members(ckpt_paths, config=config, device=device)
    load_time = time.time() - t0
    print(f"  [OK] Successfully loaded {len(gpu_models)} members onto {device} in {load_time:.3f}s")
    for i, m in enumerate(gpu_models):
        param_count = sum(p.numel() for p in m.parameters())
        print(f"       Member {i}: {param_count:,} parameters | training={m.training}")

    # -------------------------------------------------------------------------
    # 3. Test Dynamic CPU Fallback Loading (map_location='cpu')
    # -------------------------------------------------------------------------
    print("\n[Step 3/8] Testing dynamic CPU fallback loading (map_location='cpu')...")
    cpu_device = torch.device("cpu")
    cpu_models = load_ensemble_members(ckpt_paths, config=config, device=cpu_device)
    print(f"  [OK] Successfully loaded {len(cpu_models)} members onto CPU fallback.")

    # -------------------------------------------------------------------------
    # 4. Ingest Reference Tile & Degrade to Pseudo-LR
    # -------------------------------------------------------------------------
    print("\n[Step 4/8] Ingesting Sentinel-2 tile and synthesizing pseudo-LR...")
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    stack, profile = load_sentinel2_stack(raw_dir)
    tiles = extract_tiles(stack, patch_size=128, stride=96)
    test_tile = tiles[0]["data"]
    print(f"  Sample Tile #0: shape {test_tile.shape}, min={test_tile.min():.4f}, max={test_tile.max():.4f}")

    lr_tile = synthesize_pseudo_lr(
        test_tile,
        downsample_factor=2,
        blur_kernel_size=3,
        noise_std=0.01,
        seed=42,
    )
    bicubic_tile = bicubic_upsample(lr_tile, scale_factor=2)
    print(f"  Synthesized LR : shape {lr_tile.shape}")
    print(f"  Bicubic Baseline: shape {bicubic_tile.shape}")

    # -------------------------------------------------------------------------
    # 5. Super-Resolution Inference & Disagreement Map
    # -------------------------------------------------------------------------
    print("\n[Step 5/8] Running 2x Ensemble Super-Resolution on GPU...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_inf_start = time.time()
    sr_tile, disagreement_map, member_preds = predict_ensemble(gpu_models, lr_tile, device=device)
    inf_duration = time.time() - t_inf_start
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0

    print(f"  [OK] Inference finished in {inf_duration*1000:.1f}ms | Peak VRAM: {peak_vram:.1f} MiB")
    print(f"  SR Reconstruction: shape {sr_tile.shape}, min={sr_tile.min():.4f}, max={sr_tile.max():.4f}")
    print(f"  Disagreement Map : shape {disagreement_map.shape}, mean={disagreement_map.mean():.6f}, max={disagreement_map.max():.6f}")

    # -------------------------------------------------------------------------
    # 6. Verification Pipeline: Stability, Spectral, Structural, Fusion
    # -------------------------------------------------------------------------
    print("\n[Step 6/8] Executing multi-evidence verification & fusion...")
    pert_cfg = config.get("verification", {}).get("perturbation_test", {})
    noise_levels = pert_cfg.get("noise_levels", [0.01, 0.02, 0.05])
    jitter_std = float(pert_cfg.get("brightness_jitter_std", 0.02))

    stability_map, _, _ = compute_stability_map(
        models=gpu_models,
        lr_tile=lr_tile,
        noise_levels=noise_levels,
        brightness_jitter_std=jitter_std,
        num_trials=2,
        device=device,
    )
    print(f"  Stability Map    : shape {stability_map.shape}, mean variance={stability_map.mean():.6f}")

    bands = config.get("preprocessing", {}).get("bands", ["B02", "B03", "B04", "B08"])
    red_idx = bands.index("B04")
    nir_idx = bands.index("B08")

    spectral_res = compute_spectral_consistency(
        gt_tile=test_tile,
        sr_tile=sr_tile,
        red_idx=red_idx,
        nir_idx=nir_idx,
    )
    print(f"  Spectral Check   : mean Delta-NDVI={spectral_res['mean_delta_ndvi']:.5f}, inconsistent={spectral_res['pct_inconsistent_pixels']:.2f}%")

    grad_gt = compute_gradient_magnitude(test_tile)
    grad_sr = compute_gradient_magnitude(sr_tile)
    structural_diff = np.abs(grad_sr - grad_gt)
    edge_res = compute_edge_consistency(test_tile, sr_tile)
    print(f"  Structural Check : Sobel grad corr={edge_res['gradient_correlation']:.4f}, Canny F1={edge_res['edge_f1']:.4f}")

    fusion_cfg = config.get("verification", {}).get("evidence_fusion", {})
    weights = fusion_cfg.get("weights", {"disagreement": 0.25, "stability": 0.25, "spectral": 0.25, "structural": 0.25})
    high_risk_thresh = float(fusion_cfg.get("high_risk_threshold", 0.65))

    fusion_res = fuse_trust_risk_maps(
        disagreement_map=disagreement_map,
        stability_map=stability_map,
        delta_ndvi_map=spectral_res["delta_ndvi"],
        structural_diff_map=structural_diff,
        weights=weights,
        high_risk_threshold=high_risk_thresh,
    )
    print(f"  Fusion Results   : Trust Score={fusion_res['trust_score_pct']:.2f}%, High-Risk Area={fusion_res['pct_high_risk_pixels']:.2f}%")

    # -------------------------------------------------------------------------
    # 7. Downstream Building Footprint Analysis
    # -------------------------------------------------------------------------
    print("\n[Step 7/8] Executing downstream building footprint extraction...")
    morph_cfg = config.get("downstream", {}).get("morphology", {})
    down_cfg = config.get("downstream", {})
    kernel_size = int(morph_cfg.get("tophat_kernel_size", 7))
    tophat_thresh = int(morph_cfg.get("tophat_threshold", 25))
    max_ndvi = float(morph_cfg.get("max_ndvi", 0.20))
    min_area = int(morph_cfg.get("min_area", 4))
    max_area = int(morph_cfg.get("max_area", 600))
    trust_thresh = float(down_cfg.get("trust_partition_threshold", 0.85))

    foot_bic = extract_building_footprints(
        bicubic_tile,
        tophat_kernel_size=kernel_size,
        tophat_threshold=tophat_thresh,
        max_ndvi=max_ndvi,
        min_area=min_area,
        max_area=max_area,
        red_idx=red_idx,
        nir_idx=nir_idx,
    )
    foot_sr = extract_building_footprints(
        sr_tile,
        tophat_kernel_size=kernel_size,
        tophat_threshold=tophat_thresh,
        max_ndvi=max_ndvi,
        min_area=min_area,
        max_area=max_area,
        red_idx=red_idx,
        nir_idx=nir_idx,
    )
    foot_ref = extract_building_footprints(
        test_tile,
        tophat_kernel_size=kernel_size,
        tophat_threshold=tophat_thresh,
        max_ndvi=max_ndvi,
        min_area=min_area,
        max_area=max_area,
        red_idx=red_idx,
        nir_idx=nir_idx,
    )

    downstream_eval = compare_downstream_footprints(
        mask_bicubic=foot_bic["mask"],
        mask_sr=foot_sr["mask"],
        trust_map=fusion_res["trust_map"],
        ref_mask=foot_ref["mask"],
        trust_threshold=trust_thresh,
    )
    print(f"  Downstream Eval  : SR count={foot_sr['num_components']}, Bicubic count={foot_bic['num_components']}, Ref count={foot_ref['num_components']}")
    print(f"                     Overall BIC-vs-SR IoU={downstream_eval['overall_bic_sr']['iou']:.4f}")

    # -------------------------------------------------------------------------
    # 8. Auditable Trust Receipt Generation & Integrity Verification
    # -------------------------------------------------------------------------
    print("\n[Step 8/8] Generating & verifying Auditable Trust Receipt...")
    pipeline_data = {
        "fusion_result": fusion_res,
        "spectral_metrics": spectral_res,
        "edge_metrics": edge_res,
        "downstream_comp": downstream_eval,
    }

    receipt_cfg = config.get("trust_receipt", {})
    min_trust_threshold = float(receipt_cfg.get("min_trust_score_threshold", 86.0))
    receipts_dir = root / "outputs" / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    out_file = receipts_dir / "smoke_test_trust_receipt.json"

    receipt = generate_trust_receipt(
        tile_idx=0,
        raw_dir=raw_dir,
        config=config,
        pipeline_data=pipeline_data,
        min_trust_threshold=min_trust_threshold,
    )
    save_trust_receipt(receipt, out_file)
    print(f"  Saved receipt to : {out_file}")

    # Verify JSON structure and key contents
    with open(out_file, "r") as f:
        loaded_receipt = json.load(f)

    assert "receipt_id" in loaded_receipt
    assert "evidence_metrics" in loaded_receipt
    assert "model_provenance" in loaded_receipt
    assert "trust_evaluation" in loaded_receipt

    print(f"  Receipt ID       : {loaded_receipt['receipt_id']}")
    print(f"  Trust Score      : {loaded_receipt['evidence_metrics']['fused_trust_score_pct']:.2f}%")
    print(f"  Trust Status     : {loaded_receipt['trust_evaluation']['status']}")
    print(f"  Architecture     : {loaded_receipt['model_provenance']['architecture']}")
    print(f"  Checkpoints      : {loaded_receipt['model_provenance']['checkpoint_ids']}")

    print("\n" + "=" * 86)
    print("  [SUCCESS] PHASE 13 END-TO-END SMOKE TEST PASSED WITH ZERO ERRORS!")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    sys.exit(main())
