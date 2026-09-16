"""Unit and integration tests for Phase 14: Final Model Integration with Live User Upload Pipeline."""

from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard.app import load_cached_models, run_live_pipeline_for_patch
from src.data.tiling import extract_tiles
from src.data.upload import load_uploaded_stack, validate_uploaded_raster
from src.evaluation.trust_receipt import render_trust_receipt_html
from src.utils.config import get_device, get_project_root, load_config


@pytest.fixture
def sample_upload_dir():
    root = get_project_root()
    return root / "examples" / "sample_upload"


def test_final_model_architecture_and_provenance():
    """Confirm the live pipeline loads the Phase 13 locked 6-block ensemble."""
    models, config, device = load_cached_models()
    assert models is not None
    assert len(models) == 3
    for m in models:
        assert len(m.body) == 6
        assert sum(p.numel() for p in m.parameters()) == 356836


def test_live_upload_valid_file_1_128px(sample_upload_dir):
    """Test full pipeline with Valid File 1 (128x128 4-band float32 GeoTIFF)."""
    file_path = sample_upload_dir / "sample_s2_4band_128px.tif"
    assert file_path.exists()

    val_res = validate_uploaded_raster([file_path])
    assert val_res["is_valid"] is True
    assert val_res["mode"] == "single_file"
    assert all(c["passed"] for c in val_res["checks"])

    stack, meta = load_uploaded_stack([file_path], val_res)
    assert stack.shape == (128, 128, 4)
    tiles = extract_tiles(stack, patch_size=64, stride=64)
    assert len(tiles) == 4

    out = run_live_pipeline_for_patch(tiles[0]["data"], patch_idx=0, meta_dict=meta)
    assert out["has_ground_truth"] is False
    assert out["hr_tile"] is None
    assert out["sr_tile"].shape == (128, 128, 4)
    assert out["disagreement_map"].shape == (128, 128)
    assert out["stability_map"].shape == (128, 128)
    assert out["fusion_result"]["trust_map"].shape == (128, 128)
    assert "device_used" in out

    receipt = out["receipt"]
    assert receipt["model_provenance"]["num_residual_blocks"] == 6
    assert receipt["model_provenance"]["parameter_count"] == 356836
    assert "checkpoints/final_demo_v1" in receipt["model_provenance"]["checkpoint_ids"][0]

    html = render_trust_receipt_html(receipt)
    assert "GeoFUSE Trust Receipt" in html


def test_live_upload_valid_file_2_64px(sample_upload_dir):
    """Test full pipeline with Valid File 2 (64x64 4-band uint16 GeoTIFF)."""
    file_path = sample_upload_dir / "sample_s2_4band_64px_urban.tif"
    assert file_path.exists()

    val_res = validate_uploaded_raster([file_path])
    assert val_res["is_valid"] is True
    assert val_res["mode"] == "single_file"
    assert all(c["passed"] for c in val_res["checks"])

    stack, meta = load_uploaded_stack([file_path], val_res)
    assert stack.shape == (64, 64, 4)
    tiles = extract_tiles(stack, patch_size=64, stride=64)
    assert len(tiles) == 1

    out = run_live_pipeline_for_patch(tiles[0]["data"], patch_idx=0, meta_dict=meta)
    assert out["has_ground_truth"] is False
    assert out["sr_tile"].shape == (128, 128, 4)
    assert 0.0 <= out["fusion_result"]["trust_score_pct"] <= 100.0


def test_live_upload_invalid_corrupted_file(sample_upload_dir):
    """Confirm corrupt file is safely rejected with clear error message."""
    file_path = sample_upload_dir / "sample_invalid_corrupted.tif"
    assert file_path.exists()

    val_res = validate_uploaded_raster([file_path])
    assert val_res["is_valid"] is False
    assert val_res["checks"][0]["passed"] is False


def test_live_upload_invalid_missing_band(sample_upload_dir):
    """Confirm 3-band file lacking NIR (B08) is safely rejected."""
    file_path = sample_upload_dir / "sample_invalid_3band_rgb.tif"
    assert file_path.exists()

    val_res = validate_uploaded_raster([file_path])
    assert val_res["is_valid"] is False
    check2 = next(c for c in val_res["checks"] if "Spectral Bands" in c["name"])
    assert check2["passed"] is False


def test_live_upload_ui_apptest():
    """Verify Streamlit user upload and execution workflow via AppTest."""
    root = get_project_root()
    app_path = str(root / "src/dashboard/app.py")
    sample_file = root / "examples" / "sample_upload" / "sample_s2_4band_128px.tif"

    at = AppTest.from_file(app_path, default_timeout=50)
    at.run()
    assert not at.exception

    # Select Upload GeoTIFF mode
    at.sidebar.radio[0].set_value("Upload GeoTIFF")
    at.run()
    assert not at.exception
    assert at.session_state["app_mode"] == "Live Analysis"

    # Upload file
    uploader = at.sidebar.file_uploader[0]
    uploader.upload("sample_s2_4band_128px.tif", sample_file.read_bytes(), mime_type="image/tiff")
    at.run()
    assert not at.exception

    # Execute
    exec_buttons = [b for b in at.button if "Execute" in b.label]
    assert len(exec_buttons) >= 1
    exec_buttons[0].click()
    at.run()
    assert not at.exception

    # Verify results rendered
    assert len(at.tabs) == 5
    combined_text = " ".join(m.value for m in at.markdown)
    assert "Super-Resolution Output & Composite Test Score" in combined_text
