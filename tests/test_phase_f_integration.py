"""Unit and integration tests for Phase F: Integrate Comparison / Evidence / Downstream Views.

Verifies:
1. Redesigned 5-tab structure operates identically for both Demo Mode and Live Upload Mode.
2. Strict 'Not available' rules are enforced for uploaded scenes lacking ground truth across all tabs.
3. Downstream building footprint metrics are consistently labeled as 'agreement' / 'overlap' (IoU), NEVER 'accuracy'.
4. Trust Receipt functions as an audit artifact with complete metadata, model provenance, limitations, and preserved JSON/HTML export buttons.
"""

from pathlib import Path
import json
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import io

from streamlit.testing.v1 import AppTest
from src.utils.config import get_project_root


def _create_synthetic_geotiff_bytes(width: int = 128, height: int = 128) -> bytes:
    """Create in-memory 4-band GeoTIFF bytes for AppTest simulation."""
    data = np.random.uniform(0.05, 0.45, (4, height, width)).astype(np.float32)
    transform = from_origin(500000, 3000000, 10.0, 10.0)
    crs = "EPSG:32643"
    mem_buf = io.BytesIO()
    with rasterio.open(
        mem_buf,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(4):
            dst.write(data[i], i + 1)
    mem_buf.seek(0)
    return mem_buf.getvalue()


def test_phase_f_identical_tab_structure_in_demo_and_upload():
    """Verify that both Demo Mode and Live Upload Mode instantiate the exact same 5 workflow tabs."""
    app_path = str(Path(__file__).resolve().parent.parent / "src/dashboard/app.py")

    # 1. Test Demo Mode tabs
    at_demo = AppTest.from_file(app_path, default_timeout=30)
    at_demo.run()
    assert not at_demo.exception
    demo_tabs = [t.label for t in at_demo.tabs]
    expected_tabs = ["Overview", "Comparison", "Evidence", "Downstream Impact", "Trust Receipt"]
    for tab_name in expected_tabs:
        assert tab_name in demo_tabs, f"Missing tab {tab_name} in Demo Mode"

    # 2. Test Live Upload Mode tabs
    at_upload = AppTest.from_file(app_path, default_timeout=30)
    at_upload.run()
    at_upload.sidebar.radio[0].set_value("Upload GeoTIFF")
    at_upload.run()

    raw_bytes = _create_synthetic_geotiff_bytes(128, 128)
    uploader = at_upload.sidebar.file_uploader[0]
    uploader.upload("S2A_MSIL2A_TEST_B02_B03_B04_B08.tif", raw_bytes, mime_type="image/tiff")
    at_upload.run()

    assert not at_upload.exception, f"AppTest exception on upload: {at_upload.exception}"
    upload_tabs = [t.label for t in at_upload.tabs]
    for tab_name in expected_tabs:
        assert tab_name in upload_tabs, f"Missing tab {tab_name} in Upload Mode"


def test_phase_f_strict_not_available_rules_in_upload_mode():
    """Verify all reference-free panels and unextractable metadata display 'Not available' in Upload Mode."""
    app_path = str(Path(__file__).resolve().parent.parent / "src/dashboard/app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("Upload GeoTIFF")
    at.run()

    raw_bytes = _create_synthetic_geotiff_bytes(128, 128)
    uploader = at.sidebar.file_uploader[0]
    uploader.upload("custom_raster.tif", raw_bytes, mime_type="image/tiff")
    at.run()

    assert not at.exception
    combined_md = "\n".join(m.value for m in at.markdown)

    # Overview tab metadata
    assert "Evaluation Context" in combined_md
    assert "External Upload (Reference-Free)" in combined_md

    # Comparison tab unavailable card
    assert "sci-unavailable-card" in combined_md
    assert "Reference: not available for uploaded imagery." in combined_md

    # Downstream tab ground-truth limitation notice
    infos = [info.value for info in at.info]
    assert any("Reference Mask Unavailable for Uploaded Imagery" in info for info in infos)

    # Downstream relative reference metric
    ref_metric = next((m for m in at.metric if m.label == "Relative Ref HR IoU"), None)
    assert ref_metric is not None
    assert ref_metric.value == "Not available (No GT)"


def test_phase_f_downstream_nomenclature_no_accuracy():
    """Verify downstream metrics and labels strictly use 'agreement' / 'overlap' (IoU), never 'accuracy'."""
    app_path = str(Path(__file__).resolve().parent.parent / "src/dashboard/app.py")

    for mode in ["Use Demo Scene", "Upload GeoTIFF"]:
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()

        if mode == "Upload GeoTIFF":
            at.sidebar.radio[0].set_value("Upload GeoTIFF")
            at.run()
            raw_bytes = _create_synthetic_geotiff_bytes(128, 128)
            uploader = at.sidebar.file_uploader[0]
            uploader.upload("test_scene.tif", raw_bytes, mime_type="image/tiff")
            at.run()

        assert not at.exception

        # Check metric labels in downstream tab
        downstream_metric_labels = [
            m.label for m in at.metric
            if m.label in [
                "Overall Bicubic vs SR IoU",
                "High-Trust Region IoU",
                "Low-Trust Region IoU",
                "Relative Ref HR IoU",
            ]
        ]
        assert "Overall Bicubic vs SR IoU" in downstream_metric_labels
        assert "High-Trust Region IoU" in downstream_metric_labels
        assert "Low-Trust Region IoU" in downstream_metric_labels

        # Ensure the standalone word 'accuracy' is NEVER used as a metric label
        for m in at.metric:
            assert "accuracy" not in m.label.lower(), f"Forbidden word 'accuracy' found in metric label: {m.label}"


def test_phase_f_trust_receipt_audit_artifact_and_dual_export():
    """Verify Trust Receipt audit artifact completeness and dual JSON/HTML export buttons in both modes."""
    app_path = str(Path(__file__).resolve().parent.parent / "src/dashboard/app.py")

    for mode in ["Use Demo Scene", "Upload GeoTIFF"]:
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()

        if mode == "Upload GeoTIFF":
            at.sidebar.radio[0].set_value("Upload GeoTIFF")
            at.run()
            raw_bytes = _create_synthetic_geotiff_bytes(128, 128)
            uploader = at.sidebar.file_uploader[0]
            uploader.upload("test_scene.tif", raw_bytes, mime_type="image/tiff")
            at.run()

        assert not at.exception

        # Verify dual export download buttons are rendered in both modes
        dl_buttons = at.download_button
        assert len(dl_buttons) >= 2
        dl_labels = [b.label for b in dl_buttons]
        assert "📥 Download Trust Receipt (JSON)" in dl_labels
        assert "📄 Download Summary Card (HTML)" in dl_labels

    # Also verify the audit receipt record structure directly
    from src.dashboard.app import run_cached_pipeline
    data = run_cached_pipeline(0)
    assert data is not None
    receipt_dict = data["receipt"]

    # Audit receipt required sections (Requirement 2)
    assert "receipt_id" in receipt_dict
    assert "generation_timestamp" in receipt_dict
    assert "tile_metadata" in receipt_dict
    assert "model_provenance" in receipt_dict
    assert "evidence_metrics" in receipt_dict
    assert "trust_evaluation" in receipt_dict
    assert "scientific_transparency_mandate" in receipt_dict
