"""GeoFUSE SentinelGuard — Phase 14 Live Upload Pipeline Integration Test.

Confirms that:
1. The locked Phase 13 final demo ensemble (checkpoints/final_demo_v1/) seamlessly integrates
   with the live-upload / real-time inference workflow.
2. Two distinct valid uploaded files (a single 4-band GeoTIFF and 4 individual band GeoTIFFs)
   pass all 6 pre-flight integrity checks and execute end-to-end inference on GPU.
3. Live inference produces evidence scores, disagreement maps, trust/risk maps, downstream
   footprint comparisons, and cryptographic Trust Receipts identical in form to the demo path.
4. Real-world imagery honesty constraints are strictly upheld (no ground-truth hallucination,
   has_ground_truth=False, relative agreement reported).
5. Error handling correctly traps intentionally invalid files (corrupt raster, missing bands, missing CRS).
6. Streamlit AppTest simulates user interaction without exceptions.
"""

import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
import torch

from src.dashboard.app import load_cached_models, run_live_pipeline_for_patch
from src.data.tiling import extract_tiles
from src.data.upload import load_uploaded_stack, validate_uploaded_raster
from src.evaluation.trust_receipt import render_trust_receipt_html
from src.utils.config import get_device, get_project_root, load_config


def create_in_memory_geotiff(
    shape=(64, 64),
    bands=4,
    dtype="uint16",
    crs="EPSG:32643",
    res=(10.0, 10.0),
    data_range=(200, 2500),
    filename="synthetic_test.tif",
) -> io.BytesIO:
    """Create in-memory GeoTIFF file-like object."""
    h, w = shape
    data = np.random.randint(data_range[0], data_range[1], (bands, h, w), dtype=np.uint16)
    transform = from_origin(500000.0, 3000000.0, res[0], res[1])

    mem = MemoryFile()
    with mem.open(
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
    
    bio = io.BytesIO(mem.read())
    bio.name = filename
    return bio


def main() -> int:
    print("=" * 86)
    print("      GeoFUSE SentinelGuard — Phase 14 Live Upload Integration Verification")
    print("=" * 86)

    root = get_project_root()
    config = load_config()
    device = get_device(config)

    print(f"\n[Environment Telemetry]")
    print(f"  Compute Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU Name            : {torch.cuda.get_device_name(0)}")
        print(f"  Total VRAM          : {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MiB")

    # -------------------------------------------------------------------------
    # 1. Verify Model Loading from Locked Final Demo Checkpoint
    # -------------------------------------------------------------------------
    print(f"\n[Test 1/6] Loading Phase 13 Final Demo Model from checkpoints/final_demo_v1/...")
    models, loaded_cfg, loaded_dev = load_cached_models()
    if models is None or len(models) != 3:
        print("[FAIL] Could not load 3 ensemble members.")
        return 1

    for i, m in enumerate(models):
        p_count = sum(p.numel() for p in m.parameters())
        n_blocks = len(m.body)
        print(f"  Member {i}: {p_count:,} parameters, {n_blocks} residual blocks, device={loaded_dev}")
        assert n_blocks == 6, f"Expected 6 residual blocks, got {n_blocks}"
        assert p_count == 356836, f"Expected 356,836 parameters, got {p_count}"
    print("  [PASS] Model architecture and parameter counts match Phase 13 locked specification.")

    # -------------------------------------------------------------------------
    # 2. Test Valid File 1: Single 4-Band GeoTIFF (examples/sample_upload/sample_s2_4band_128px.tif)
    # -------------------------------------------------------------------------
    sample_file_1 = root / "examples" / "sample_upload" / "sample_s2_4band_128px.tif"
    print(f"\n[Test 2/6] Testing Valid File 1: Single 4-Band GeoTIFF ({sample_file_1.name})...")
    assert sample_file_1.exists(), f"Sample file not found at {sample_file_1}"

    val_res_1 = validate_uploaded_raster([sample_file_1])
    assert val_res_1["is_valid"] is True, f"Validation failed: {val_res_1.get('error_message')}"
    assert val_res_1["mode"] == "single_file"
    assert len(val_res_1["checks"]) == 6
    assert all(c["passed"] for c in val_res_1["checks"])
    print(f"  [PASS] All 6 pre-flight validation checks passed.")

    stack_1, meta_1 = load_uploaded_stack([sample_file_1], val_res_1)
    assert stack_1.shape == (128, 128, 4)
    assert stack_1.dtype == np.float32
    print(f"  Ingested Raster Stack: shape {stack_1.shape}, range [{stack_1.min():.4f}, {stack_1.max():.4f}]")

    tiles_1 = extract_tiles(stack_1, patch_size=64, stride=64)
    assert len(tiles_1) == 4, f"Expected 4 patches (64x64 non-overlapping in 128x128), got {len(tiles_1)}"
    print(f"  Extracted Patches    : {len(tiles_1)} patches of size 64x64")

    # Run live pipeline for Patch #0 on GPU
    t0 = time.time()
    pipe_out_1 = run_live_pipeline_for_patch(tiles_1[0]["data"], patch_idx=0, meta_dict=meta_1)
    duration_1 = time.time() - t0

    assert pipe_out_1["has_ground_truth"] is False
    assert pipe_out_1["hr_tile"] is None
    assert pipe_out_1["sr_tile"].shape == (128, 128, 4)
    assert pipe_out_1["disagreement_map"].shape == (128, 128)
    assert pipe_out_1["stability_map"].shape == (128, 128)
    assert pipe_out_1["fusion_result"]["trust_map"].shape == (128, 128)

    trust_score_1 = pipe_out_1["fusion_result"]["trust_score_pct"]
    receipt_1 = pipe_out_1["receipt"]
    assert 0.0 <= trust_score_1 <= 100.0
    assert "TR-" in receipt_1["receipt_id"]
    assert receipt_1["model_provenance"]["num_residual_blocks"] == 6
    assert receipt_1["model_provenance"]["parameter_count"] == 356836
    assert "checkpoints/final_demo_v1" in receipt_1["model_provenance"]["checkpoint_ids"][0]

    # Verify HTML receipt rendering
    html_1 = render_trust_receipt_html(receipt_1)
    assert "GeoFUSE Trust Receipt" in html_1
    assert "EPSG:32643" in html_1

    print(f"  [PASS] Live Inference complete in {duration_1*1000:.1f}ms on device: {pipe_out_1.get('device_used', device)}")
    print(f"         Trust Score: {trust_score_1:.2f}% | Receipt ID: {receipt_1['receipt_id']}")

    # -------------------------------------------------------------------------
    # 3. Test Valid File 2: Multi-File Individual Bands (4 files from data/raw/)
    # -------------------------------------------------------------------------
    print(f"\n[Test 3/6] Testing Valid File 2: Multi-File Separate Band GeoTIFFs (4 files)...")
    raw_dir = root / "data" / "raw"
    band_files = sorted(list(raw_dir.glob("S2A_*.tif")))
    assert len(band_files) == 4, f"Expected 4 files in data/raw/, found {len(band_files)}"

    val_res_2 = validate_uploaded_raster(band_files)
    assert val_res_2["is_valid"] is True, f"Validation failed: {val_res_2.get('error_message')}"
    assert val_res_2["mode"] == "multi_file"
    assert all(c["passed"] for c in val_res_2["checks"])
    print(f"  [PASS] All 6 multi-file pre-flight checks passed.")

    stack_2, meta_2 = load_uploaded_stack(band_files, val_res_2)
    assert stack_2.shape == (512, 512, 4)
    print(f"  Ingested Raster Stack: shape {stack_2.shape}, dtype {stack_2.dtype}")

    tiles_2 = extract_tiles(stack_2, patch_size=64, stride=64)
    print(f"  Extracted Patches    : {len(tiles_2)} patches of size 64x64")

    # Run live pipeline for Patch #12 on GPU
    t0 = time.time()
    pipe_out_2 = run_live_pipeline_for_patch(tiles_2[12]["data"], patch_idx=12, meta_dict=meta_2)
    duration_2 = time.time() - t0

    assert pipe_out_2["has_ground_truth"] is False
    assert pipe_out_2["sr_tile"].shape == (128, 128, 4)
    trust_score_2 = pipe_out_2["fusion_result"]["trust_score_pct"]
    receipt_2 = pipe_out_2["receipt"]
    assert receipt_2["model_provenance"]["num_residual_blocks"] == 6
    assert "checkpoints/final_demo_v1" in receipt_2["model_provenance"]["checkpoint_ids"][0]

    print(f"  [PASS] Multi-file live inference complete in {duration_2*1000:.1f}ms on device: {pipe_out_2.get('device_used', device)}")
    print(f"         Trust Score: {trust_score_2:.2f}% | Receipt ID: {receipt_2['receipt_id']}")

    # -------------------------------------------------------------------------
    # 4. Test Invalid Files: Error Handling & Pre-Flight Traps
    # -------------------------------------------------------------------------
    print(f"\n[Test 4/6] Testing Error Handling on Intentionally Invalid Files...")

    # Case A: Corrupt / Unreadable bytes
    corrupt_bio = io.BytesIO(b"NOT_A_VALID_TIFF_RANDOM_CORRUPT_BYTES_DATA")
    corrupt_bio.name = "corrupt_file.tif"
    val_corrupt = validate_uploaded_raster([corrupt_bio])
    assert val_corrupt["is_valid"] is False, "Expected corruption validation failure"
    assert val_corrupt["checks"][0]["passed"] is False
    print(f"  [PASS] Case A (Corrupt bytes): Properly rejected at Check 1: '{val_corrupt['checks'][0]['message']}'")

    # Case B: Missing required spectral band (only B02, B03, B04; missing B08)
    missing_band_files = band_files[:3]
    val_missing = validate_uploaded_raster(missing_band_files)
    assert val_missing["is_valid"] is False, "Expected missing band failure"
    check2 = next(c for c in val_missing["checks"] if "Spectral Bands" in c["name"])
    assert check2["passed"] is False
    assert "B08" in check2["message"]
    print(f"  [PASS] Case B (Missing B08): Properly rejected at Check 2: '{check2['message']}'")

    # Case C: Missing CRS / Geographic coordinate system
    bio_no_crs = create_in_memory_geotiff(shape=(64, 64), bands=4, crs=None, filename="no_crs.tif")
    val_no_crs = validate_uploaded_raster([bio_no_crs])
    assert val_no_crs["is_valid"] is False, "Expected missing CRS failure"
    check5 = next(c for c in val_no_crs["checks"] if "CRS" in c["name"])
    assert check5["passed"] is False
    print(f"  [PASS] Case C (Missing CRS): Properly rejected at Check 5: '{check5['message']}'")

    # -------------------------------------------------------------------------
    # 5. Form Parity: Compare Demo Path vs Live Upload Path Outputs
    # -------------------------------------------------------------------------
    print(f"\n[Test 5/6] Confirming Evidence Score & Receipt Form Parity (Demo vs Live Upload)...")
    from src.dashboard.app import run_cached_pipeline
    demo_data = run_cached_pipeline(0)
    assert demo_data is not None, "Demo pipeline returned None"

    # Both must share identical dictionary keys for rendering
    required_keys = [
        "lr_tile", "bicubic_tile", "sr_tile", "disagreement_map",
        "stability_map", "spectral_metrics", "edge_metrics", "fusion_result",
        "downstream_comp", "receipt"
    ]
    for k in required_keys:
        assert k in pipe_out_1, f"Missing key '{k}' in live upload output"
        assert k in demo_data, f"Missing key '{k}' in demo cache output"

    # Both receipts must conform to trust receipt schema
    for r in (pipe_out_1["receipt"], demo_data["receipt"]):
        assert "$schema" in r
        assert "receipt_id" in r
        assert "project" in r
        assert "scientific_transparency_mandate" in r
        assert "tile_metadata" in r
        assert "model_provenance" in r
        assert "evidence_metrics" in r
        assert "trust_evaluation" in r
    print("  [PASS] Live upload output structure is strictly identical in form to offline demo path.")

    # -------------------------------------------------------------------------
    # 6. Streamlit AppTest End-to-End Simulation
    # -------------------------------------------------------------------------
    print(f"\n[Test 6/6] Running Streamlit AppTest for Live Upload Workflow...")
    from streamlit.testing.v1 import AppTest

    app_path = str(root / "src/dashboard/app.py")
    at = AppTest.from_file(app_path, default_timeout=50)
    at.run()
    assert not at.exception, f"Streamlit startup exception: {at.exception}"

    # Select Upload GeoTIFF mode
    at.sidebar.radio[0].set_value("Upload GeoTIFF")
    at.run()
    assert not at.exception
    assert at.session_state["app_mode"] == "Live Analysis"

    # Enable Bundled Sample
    at.sidebar.checkbox[0].check()
    at.run()
    assert not at.exception

    # Trigger Execute button
    exec_buttons = [b for b in at.button if "Execute" in b.label]
    assert len(exec_buttons) >= 1, "Execute button not found"
    exec_buttons[0].click()
    at.run()
    assert not at.exception, f"Execution failed in AppTest: {at.exception}"

    # Verify rendering of metrics and tabs
    assert len(at.tabs) == 5
    combined_text = " ".join(m.value for m in at.markdown)
    assert "Composite Test Score" in combined_text or "Trust Score" in combined_text
    print(f"  [PASS] Streamlit AppTest passed end-to-end with 5 tabs rendered and zero exceptions.")

    print("\n" + "=" * 86)
    print("  [SUCCESS] PHASE 14 LIVE UPLOAD PIPELINE INTEGRATION FULLY VERIFIED!")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    sys.exit(main())
