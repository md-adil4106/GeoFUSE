"""Unit and integration tests for Phase C: Live Upload Pipeline & Reference-Free Evaluation."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from src.dashboard.app import run_live_pipeline_for_patch
from src.data.tiling import extract_tiles
from src.evaluation.trust_receipt import render_trust_receipt_html
from src.utils.config import get_project_root, load_config


def test_extract_tiles_uploaded_raster():
    """Verify safe patch tiling on a simulated uploaded multi-band raster."""
    # Simulated 128x128 4-band image
    raster = np.random.uniform(0.05, 0.45, size=(128, 128, 4)).astype(np.float32)
    tiles = extract_tiles(raster, patch_size=64, stride=64)

    # In a 128x128 raster with patch_size=64 and stride=64, there should be exactly 4 non-overlapping patches
    assert len(tiles) == 4
    for i, t in enumerate(tiles):
        assert t["tile_id"] == i
        assert t["data"].shape == (64, 64, 4)
        assert t["patch_size"] == 64
        assert t["y"] in (0, 64)
        assert t["x"] in (0, 64)


def test_run_live_pipeline_for_patch_execution():
    """Verify that run_live_pipeline_for_patch runs full ensemble SR, stability, spectral, edge, and fusion."""
    # 64x64 4-band patch with typical Sentinel-2 reflectance values [0.0, 0.6]
    lr_tile = np.random.uniform(0.05, 0.40, size=(64, 64, 4)).astype(np.float32)
    mock_meta = {
        "primary_filename": "test_upload_scene.tif",
        "height": 128,
        "width": 128,
        "resolution": (10.0, 10.0),
        "crs": "EPSG:32643",
        "band_count": 4,
        "acquisition_date": "2024-02-27 05:20:54 UTC",
        "dtype": "uint16",
        "needs_reflectance_scaling": True,
        "bounds": {"left": 500000.0, "bottom": 3000000.0, "right": 501280.0, "top": 3001280.0},
    }

    result = run_live_pipeline_for_patch(lr_tile, patch_idx=0, meta_dict=mock_meta)
    assert result is not None, "Pipeline returned None; ensemble checkpoints might be missing."

    # 1. Verify honest handling of no ground truth
    assert result["hr_tile"] is None
    assert result["has_ground_truth"] is False

    # 2. Verify spatial dimensions
    assert result["lr_tile"].shape == (64, 64, 4)
    assert result["bicubic_tile"].shape == (128, 128, 4)
    assert result["sr_tile"].shape == (128, 128, 4)
    assert result["disagreement_map"].shape == (128, 128)
    assert result["stability_map"].shape == (128, 128)
    assert result["fusion_result"]["trust_map"].shape == (128, 128)

    # 3. Verify trust score & risk bounds
    trust_score = result["fusion_result"]["trust_score_pct"]
    assert 0.0 <= trust_score <= 100.0
    assert 0.0 <= result["fusion_result"]["mean_risk_score"] <= 1.0

    # 4. Verify downstream building footprint results
    assert "downstream_comp" in result
    downstream = result["downstream_comp"]
    assert "overall_bic_sr" in downstream
    assert "iou" in downstream["overall_bic_sr"]
    assert downstream.get("reference_comparison") is None, "Real upload must not produce fake ground-truth comparison"

    # 5. Verify valid Trust Receipt generation with uploaded metadata
    receipt = result["receipt"]
    assert receipt is not None
    assert "TR-" in receipt["receipt_id"]
    assert receipt["tile_metadata"]["source_crs"] == "EPSG:32643"
    assert receipt["tile_metadata"]["spatial_resolution_meters"] == 10.0
    assert receipt["tile_metadata"]["reconstructed_resolution_meters"] == 5.0
    assert receipt["evidence_metrics"]["fused_trust_score_pct"] == trust_score

    # 6. Verify HTML rendering doesn't crash on uploaded receipt
    html = render_trust_receipt_html(receipt)
    assert "GeoFUSE Trust Receipt" in html
    assert "EPSG:32643" in html


def test_upload_dashboard_apptest_e2e():
    """Verify end-to-end user upload workflow in Streamlit AppTest without UI crashes."""
    from streamlit.testing.v1 import AppTest

    root = get_project_root()
    raw_dir = root / "data/raw"
    band_files = [
        raw_dir / "S2A_T43PGQ_20240227T052054_L2A_B02_10m.tif",
        raw_dir / "S2A_T43PGQ_20240227T052054_L2A_B03_10m.tif",
        raw_dir / "S2A_T43PGQ_20240227T052054_L2A_B04_10m.tif",
        raw_dir / "S2A_T43PGQ_20240227T052054_L2A_B08_10m.tif",
    ]

    app_path = str(root / "src/dashboard/app.py")
    at = AppTest.from_file(app_path, default_timeout=50)
    at.run()
    assert not at.exception

    # Switch to user upload mode
    at.sidebar.radio[0].set_value("Upload GeoTIFF")
    at.run()
    assert not at.exception
    assert at.session_state["app_mode"] == "Live Analysis"

    # Upload the 4 band files
    uploader = at.sidebar.file_uploader[0]
    for bf in band_files:
        uploader.upload(bf.name, bf.read_bytes(), mime_type="image/tiff")

    at.run()
    assert not at.exception, f"AppTest exception on upload: {at.exception}"
    assert len(at.metric) >= 9
    assert at.session_state["app_mode"] == "Live Analysis"

