"""Unit and integration tests for Phase E: UI Redesign (Scientific / Restrained GIS Monitoring Tool).

Verifies:
1. 5 guided workflow tabs exist and render without exceptions: Overview, Comparison, Evidence, Downstream Impact, Trust Receipt.
2. Top-level evidence banner uses qualified scientific phrasing without pass/fail 'APPROVED' framing.
3. Trust/Risk Map legend and heuristic disclaimer sentence are rendered.
4. Evidence tab contains compact horizontal progress bars with real computed percentage metrics.
5. Five-panel comparison contains clean muted monospace image labels for all categories.
6. Clean rendering in both Demo Mode and Live Upload Mode without exceptions.
"""

from pathlib import Path
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


def test_phase_e_5_guided_tabs_rendered():
    """Verify that all 5 guided workflow tabs exist and render without error in AppTest."""
    app_path = str(Path(__file__).resolve().parent.parent / "src/dashboard/app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()

    assert not at.exception, f"AppTest raised an exception: {at.exception}"

    # Verify tabs exist
    assert len(at.tabs) >= 5
    tab_labels = [t.label for t in at.tabs]
    assert "Overview" in tab_labels
    assert "Comparison" in tab_labels
    assert "Evidence" in tab_labels
    assert "Downstream Impact" in tab_labels
    assert "Trust Receipt" in tab_labels


def test_phase_e_scientific_banner_and_disclaimer():
    """Verify restrained scientific wording in top banner and legend disclaimer."""
    app_path = str(Path(__file__).resolve().parent.parent / "src/dashboard/app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()

    assert not at.exception

    # Check markdown blocks for scientific banner and heuristic disclaimer
    md_texts = [m.value for m in at.markdown]
    combined_md = "\n".join(md_texts)

    # Qualified scientific phrasing (Requirement 2)
    assert "Composite Evidence Score:" in combined_md
    assert "Heuristic multi-criteria reliability indicator" in combined_md
    assert "not a calibrated probability of correctness" in combined_md

    # Check absence of crude marketing certification banner
    assert "HIGH TRUST APPROVED" not in combined_md

    # Trust/Risk map legend & heuristic disclaimer (Requirement 6)
    assert "Lower estimated reconstruction risk" in combined_md
    assert "Moderate" in combined_md
    assert "Higher estimated reconstruction risk" in combined_md
    assert "This map is a heuristic evidence indicator derived from ensemble disagreement" in combined_md
    assert "not a ground-truth error map" in combined_md


def test_phase_e_evidence_horizontal_progress_bars():
    """Verify that Evidence tab renders horizontal progress bars for real computed signals."""
    app_path = str(Path(__file__).resolve().parent.parent / "src/dashboard/app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()

    assert not at.exception

    combined_md = "\n".join(m.value for m in at.markdown)

    # Real computed components (Requirement 7)
    assert "1. Ensemble Agreement" in combined_md
    assert "2. Perturbation Stability" in combined_md
    assert "3. Spectral Consistency" in combined_md
    assert "4. Structural Consistency" in combined_md
    assert "sci-bar-fill" in combined_md
    assert "sci-bar-container" in combined_md


def test_phase_e_five_panel_muted_labels():
    """Verify that the five-panel comparison renders muted labels without decorative badge pills."""
    app_path = str(Path(__file__).resolve().parent.parent / "src/dashboard/app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()

    assert not at.exception

    combined_md = "\n".join(m.value for m in at.markdown)

    # Check muted image labels
    assert "sci-image-label" in combined_md
    assert "1. Actual Model Input" in combined_md
    assert "2. Bicubic Baseline" in combined_md
    assert "3. GeoFUSE SR (Ours)" in combined_md
    assert "4. Clean Reference" in combined_md
    assert "5. Trust / Risk Map" in combined_md


def test_phase_e_upload_mode_visual_hierarchy():
    """Verify that Upload Mode conforms to Phase E visual styling and displays Reference Unavailable card."""
    app_path = str(Path(__file__).resolve().parent.parent / "src/dashboard/app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()

    # Switch to upload mode
    at.sidebar.radio[0].set_value("Upload GeoTIFF")
    at.run()

    raw_bytes = _create_synthetic_geotiff_bytes(128, 128)
    uploader = at.sidebar.file_uploader[0]
    uploader.upload("S2A_MSIL2A_TEST_B02_B03_B04_B08.tif", raw_bytes, mime_type="image/tiff")
    at.run()

    assert not at.exception, f"AppTest exception on upload: {at.exception}"

    combined_md = "\n".join(m.value for m in at.markdown)

    # Reference unavailable styling
    assert "sci-unavailable-card" in combined_md
    assert "Reference: not available for uploaded imagery." in combined_md

    # Mode indicator
    assert "Mode: Live User Upload" in combined_md
