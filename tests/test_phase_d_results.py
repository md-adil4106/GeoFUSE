"""Tests for Phase D: Result Handling and Evidence Generation for Uploads.

Verifies:
1. Auditable Trust Receipt generation for uploaded imagery uses real computed values only.
2. Missing or unextractable metadata fields are strictly marked 'Not available', never inferred or copied.
3. Downstream footprint comparison runs on arbitrary inputs without independent ground-truth reference masks.
4. Downstream reference IoU is explicitly 'Not available' when reference masks are absent.
5. Download capabilities (JSON and HTML summary card) are properly configured.
"""

import io
import json
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.data.upload import validate_uploaded_raster, load_uploaded_stack
from src.data.tiling import extract_tiles
from src.evaluation.downstream_eval import extract_building_footprints, compare_downstream_footprints
from src.evaluation.trust_receipt import generate_trust_receipt, render_trust_receipt_html
from src.utils.config import load_config


def _create_synthetic_tiff_bytes(
    width: int = 128,
    height: int = 128,
    num_bands: int = 4,
    dtype: str = "uint16",
    tags: dict = None,
) -> bytes:
    """Helper to create an in-memory single-file multi-band GeoTIFF."""
    transform = from_origin(500000.0, 1500000.0, 10.0, 10.0)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": num_bands,
        "dtype": dtype,
        "crs": "EPSG:32643",
        "transform": transform,
    }
    buf = io.BytesIO()
    with rasterio.open(buf, "w", **profile) as dst:
        np.random.seed(42)
        for b in range(1, num_bands + 1):
            if dtype == "uint16":
                data = (np.random.rand(height, width) * 3000 + 500).astype(np.uint16)
            else:
                data = (np.random.rand(height, width) * 0.3 + 0.05).astype(np.float32)
            dst.write(data, b)
        if tags:
            dst.update_tags(**tags)
    return buf.getvalue()


def test_trust_receipt_uploaded_scene_strictly_not_available_fields():
    """Verify that an arbitrary uploaded GeoTIFF with no S2 tags marks unextractable fields as 'Not available'."""
    raw_bytes = _create_synthetic_tiff_bytes(width=128, height=128, num_bands=4)

    class MockUpload:
        def __init__(self, data, name):
            self._data = data
            self.name = name

        def getvalue(self):
            return self._data

    file_obj = MockUpload(raw_bytes, "custom_urban_area_2024.tif")
    val_result = validate_uploaded_raster([file_obj], min_patch_size=64)
    assert val_result["is_valid"] is True
    meta = val_result["metadata"]

    # Verify that metadata extraction strictly marks absent fields as 'Not available'
    assert meta["is_upload"] is True
    assert meta["platform"] == "Not available"
    assert meta["mgrs_tile"] == "Not available"
    assert meta["product_level"] == "Not available"
    assert meta["cloud_cover_percentage"] == "Not available"
    assert meta["sun_elevation_angle_deg"] == "Not available"
    assert meta["sun_azimuth_angle_deg"] == "Not available"
    assert meta["satellite_orbit_number"] == "Not available"
    assert meta["acquisition_datetime"] == "Not available"

    # Build mock pipeline data representing a live session
    mock_pipeline_data = {
        "fusion_result": {
            "trust_score_pct": 89.25,
            "mean_risk_score": 0.1075,
            "pct_high_risk_pixels": 0.0,
            "weights_used": {"disagreement": 0.25, "stability": 0.25, "spectral": 0.25, "structural": 0.25},
            "component_stats": {
                "disagreement": {"raw_mean": 0.0011, "raw_max": 0.007},
                "stability": {"raw_mean": 0.0001, "raw_max": 0.0006},
                "spectral": {"raw_mean": 0.012, "raw_max": 0.08},
                "structural": {"raw_mean": 0.031, "raw_max": 0.18},
            },
        },
        "spectral_metrics": {
            "mean_delta_ndvi": 0.012,
            "max_delta_ndvi": 0.08,
            "pct_inconsistent_pixels": 2.1,
            "is_spectrally_consistent": True,
        },
        "edge_metrics": {
            "gradient_correlation": 0.8450,
            "edge_iou": 0.4620,
            "edge_f1": 0.6310,
        },
        "downstream_comp": {
            "scientific_honesty_label": "Bicubic-vs-SR Footprint Agreement (No Independent GT)",
            "overall_bic_sr": {"iou": 0.9410, "dice": 0.9696},
            "high_trust_bic_sr": {"iou": 0.9480},
            "low_trust_bic_sr": {"iou": 0.9250},
            "high_trust_area_pct": 74.2,
            "reference_comparison": None,  # Upload mode: No ground truth
        },
    }

    config = load_config()
    receipt = generate_trust_receipt(
        tile_idx=0,
        raw_dir=None,
        config=config,
        pipeline_data=mock_pipeline_data,
        geo_meta=meta,
    )

    tile_meta = receipt["tile_metadata"]
    # Check that unextractable fields are strictly 'Not available', NEVER inferred or defaulted
    assert tile_meta["platform"] == "Not available"
    assert tile_meta["mgrs_tile"] == "Not available"
    assert tile_meta["product_level"] == "Not available"
    assert tile_meta["cloud_cover_percentage"] == "Not available"
    assert tile_meta["sun_elevation_angle_deg"] == "Not available"
    assert tile_meta["sun_azimuth_angle_deg"] == "Not available"
    assert tile_meta["satellite_orbit_number"] == "Not available"
    assert tile_meta["acquisition_datetime"] == "Not available"

    # Check relative reference IoU in downstream task evaluation is strictly 'Not available'
    down_eval = receipt["evidence_metrics"]["downstream_task_evaluation"]
    assert down_eval["relative_reference_hr_iou"] == "Not available"

    # Check receipt ID is clean
    assert "UPLOAD" in receipt["receipt_id"]
    assert "PATCH00" in receipt["receipt_id"]

    # Render HTML summary
    html = render_trust_receipt_html(receipt)
    assert "🛰️ GeoFUSE Trust Receipt" in html
    assert "Not available" in html
    assert "NOMINAL -- HIGH TRUST" in html


def test_trust_receipt_uploaded_scene_with_s2_naming_pattern():
    """Verify that when standard Sentinel-2 naming is present, platform/tile/date are parsed while missing tags remain 'Not available'."""
    raw_bytes = _create_synthetic_tiff_bytes(width=128, height=128, num_bands=4)

    class MockUpload:
        def __init__(self, data, name):
            self._data = data
            self.name = name

        def getvalue(self):
            return self._data

    file_obj = MockUpload(raw_bytes, "S2A_T43PGQ_20240227T052054_L2A_Stack.tif")
    val_result = validate_uploaded_raster([file_obj], min_patch_size=64)
    assert val_result["is_valid"] is True
    meta = val_result["metadata"]

    assert meta["platform"] == "Sentinel-2A"
    assert meta["mgrs_tile"] == "43PGQ"
    assert meta["product_level"] == "L2A"
    assert "2024-02-27" in meta["acquisition_datetime"]
    assert meta["cloud_cover_percentage"] == "Not available"

    mock_pipeline_data = {
        "fusion_result": {
            "trust_score_pct": 88.0,
            "mean_risk_score": 0.12,
            "pct_high_risk_pixels": 0.0,
            "weights_used": {"disagreement": 0.25, "stability": 0.25, "spectral": 0.25, "structural": 0.25},
            "component_stats": {
                "disagreement": {"raw_mean": 0.001, "raw_max": 0.008},
                "stability": {"raw_mean": 0.0001, "raw_max": 0.0005},
                "spectral": {"raw_mean": 0.015, "raw_max": 0.09},
                "structural": {"raw_mean": 0.035, "raw_max": 0.20},
            },
        },
        "spectral_metrics": {"mean_delta_ndvi": 0.015, "max_delta_ndvi": 0.09, "pct_inconsistent_pixels": 3.0, "is_spectrally_consistent": True},
        "edge_metrics": {"gradient_correlation": 0.83, "edge_iou": 0.45, "edge_f1": 0.62},
        "downstream_comp": {
            "scientific_honesty_label": "Bicubic-vs-SR Footprint Agreement (No Independent GT)",
            "overall_bic_sr": {"iou": 0.93, "dice": 0.96},
            "high_trust_bic_sr": {"iou": 0.94},
            "low_trust_bic_sr": {"iou": 0.91},
            "high_trust_area_pct": 70.0,
            "reference_comparison": None,
        },
    }

    config = load_config()
    receipt = generate_trust_receipt(
        tile_idx=3,
        raw_dir=None,
        config=config,
        pipeline_data=mock_pipeline_data,
        geo_meta=meta,
    )

    assert receipt["tile_metadata"]["platform"] == "Sentinel-2A"
    assert receipt["tile_metadata"]["mgrs_tile"] == "43PGQ"
    assert receipt["tile_metadata"]["product_level"] == "L2A"
    assert receipt["tile_metadata"]["cloud_cover_percentage"] == "Not available"
    assert "TR-Sentinel-2A-43PGQ-T03-" in receipt["receipt_id"]


def test_downstream_evaluation_on_arbitrary_upload_without_reference():
    """Verify that downstream building footprint comparison runs robustly on arbitrary inputs with ref_mask=None."""
    np.random.seed(42)
    sr_tile = np.random.rand(128, 128, 4).astype(np.float32) * 0.4 + 0.1
    import cv2
    bic_tile = cv2.GaussianBlur(sr_tile, (3, 3), 0.8)

    foot_sr = extract_building_footprints(sr_tile, red_idx=2, nir_idx=3)
    foot_bic = extract_building_footprints(bic_tile, red_idx=2, nir_idx=3)
    trust_map = np.random.rand(128, 128).astype(np.float32) * 0.3 + 0.7

    comp = compare_downstream_footprints(
        mask_bicubic=foot_bic["mask"],
        mask_sr=foot_sr["mask"],
        trust_map=trust_map,
        ref_mask=None,  # Upload mode: None
    )

    assert comp["reference_comparison"] is None
    assert 0.0 <= comp["overall_bic_sr"]["iou"] <= 1.0
    assert 0.0 <= comp["high_trust_bic_sr"]["iou"] <= 1.0
    assert 0.0 <= comp["low_trust_bic_sr"]["iou"] <= 1.0
    assert comp["discrepancy_mask"].shape == (128, 128)
    assert "No Independent GT" in comp["scientific_honesty_label"]


def test_phase_d_apptest_ui_elements():
    """Verify Streamlit dashboard UI renders dual download buttons and ground truth notices on upload."""
    from streamlit.testing.v1 import AppTest
    from src.utils.config import get_project_root

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

    # Upload bands
    uploader = at.sidebar.file_uploader[0]
    for bf in band_files:
        uploader.upload(bf.name, bf.read_bytes(), mime_type="image/tiff")

    at.run()
    assert not at.exception

    # Verify dual download buttons in Tab 2
    dl_labels = [b.label for b in at.download_button]
    assert "📥 Download Trust Receipt (JSON)" in dl_labels
    assert "📄 Download Summary Card (HTML)" in dl_labels

    # Verify downstream ground truth limitation notice
    infos = [info.value for info in at.info]
    assert any("Reference Mask Unavailable for Uploaded Imagery" in val for val in infos)

    # Verify metric displays 'Not available (No GT)'
    ref_metric = next((m for m in at.metric if m.label == "Relative Ref HR IoU"), None)
    assert ref_metric is not None
    assert ref_metric.value == "Not available (No GT)"

