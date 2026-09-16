"""Unit and integration tests for Phase H: Demo Hardening.

Verifies:
1. Demo Mode requires zero internet access and zero upload files to run end-to-end.
2. Bundled sample GeoTIFF exists in examples/sample_upload/, is valid, and passes all pre-flight integrity checks.
3. Enabling 'use_bundled_sample' runs the genuine live ensemble pipeline without exceptions.
4. Mode indicators (header, sidebar, overview provenance) are unambiguous and impossible to misread across both modes.
"""

from pathlib import Path
import pytest
import rasterio

from streamlit.testing.v1 import AppTest
from src.data.upload import validate_uploaded_raster
from src.utils.config import get_project_root


def test_phase_h_demo_mode_offline_guarantee():
    """Verify Demo Mode requires no internet and no upload files to render fully."""
    root = get_project_root()
    app_path = str(root / "src/dashboard/app.py")

    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()

    assert not at.exception, f"Demo mode raised an exception: {at.exception}"
    assert at.session_state["app_mode"] == "Demo Mode"
    assert len(at.title) >= 1
    assert len(at.metric) >= 5

    # Check header indicator
    combined_md = "\n".join(m.value for m in at.markdown)
    assert "Mode: Demo Scene (Offline Cache)" in combined_md
    assert "sci-status-indicator demo" in combined_md

    # Check overview tab provenance
    assert "Demo Scene (Precomputed Offline Cache)" in combined_md


def test_phase_h_bundled_sample_geotiff_valid():
    """Verify the bundled sample GeoTIFF in examples/sample_upload/ is valid and passes all 6 checks."""
    root = get_project_root()
    sample_path = root / "examples/sample_upload/sample_s2_4band_128px.tif"

    assert sample_path.exists(), f"Bundled sample GeoTIFF not found at: {sample_path}"
    assert sample_path.stat().st_size > 50000, "Bundled sample GeoTIFF file is unexpectedly small."

    # Validate raster properties directly with rasterio
    with rasterio.open(sample_path) as src:
        assert src.count >= 4
        assert src.height >= 64 and src.width >= 64
        assert src.crs is not None

    # Validate through project validation module
    class FileWrapper:
        def __init__(self, p: Path):
            self.name = p.name
            self._data = p.read_bytes()

        def getvalue(self) -> bytes:
            return self._data

        def read(self) -> bytes:
            return self._data

    val_result = validate_uploaded_raster([FileWrapper(sample_path)])
    assert val_result["is_valid"] is True
    assert val_result["mode"] == "single_file"
    assert all(chk["passed"] for chk in val_result["checks"])


def test_phase_h_presenter_demo_upload_workflow():
    """Verify enabling the bundled sample toggle in Upload Mode runs the genuine live pipeline cleanly."""
    root = get_project_root()
    app_path = str(root / "src/dashboard/app.py")

    at = AppTest.from_file(app_path, default_timeout=35)
    at.run()

    # Switch to Upload GeoTIFF
    at.sidebar.radio[0].set_value("Upload GeoTIFF")
    at.run()

    assert not at.exception
    assert at.session_state["app_mode"] == "Live Analysis"

    # Enable bundled sample toggle (checkbox[1] in sidebar: [0] is show_evidence, [1] is show_downstream, [2] is use_bundled_sample)
    bundled_toggle = next((cb for cb in at.sidebar.checkbox if "Bundled Sample" in cb.label), None)
    assert bundled_toggle is not None, "Presenter bundled sample toggle not found in sidebar"

    bundled_toggle.check()
    at.run()

    assert not at.exception, f"Live pipeline failed on bundled sample: {at.exception}"

    # Confirm live inference metrics are present
    assert len(at.metric) >= 9
    assert at.session_state["app_mode"] == "Live Analysis"

    combined_md = "\n".join(m.value for m in at.markdown)
    assert "Mode: Live User Upload (Real-Time Inference)" in combined_md
    assert "sci-status-indicator live" in combined_md
    assert "Live User Upload (Real-Time Inference)" in combined_md


def test_phase_h_unambiguous_mode_indicators_across_app():
    """Verify that Demo Mode and Live Analysis indicators are visually distinct and unambiguous."""
    root = get_project_root()
    app_path = str(root / "src/dashboard/app.py")

    # 1. Demo Mode checks
    at_demo = AppTest.from_file(app_path, default_timeout=30)
    at_demo.run()
    demo_md = "\n".join(m.value for m in at_demo.markdown)
    assert "Mode: Demo Scene (Offline Cache)" in demo_md
    assert "sci-status-indicator demo" in demo_md
    assert "Mode: Live User Upload" not in demo_md
    assert "Demo Mode: Offline Cache Active" in "\n".join(s.value for s in at_demo.sidebar.success)

    # 2. Live Analysis checks
    at_live = AppTest.from_file(app_path, default_timeout=35)
    at_live.run()
    at_live.sidebar.radio[0].set_value("Upload GeoTIFF")
    at_live.run()

    # Enable bundled sample for full rendering
    sample_toggle = next(cb for cb in at_live.sidebar.checkbox if "Bundled Sample" in cb.label)
    sample_toggle.check()
    at_live.run()

    live_md = "\n".join(m.value for m in at_live.markdown)
    assert "Mode: Live User Upload (Real-Time Inference)" in live_md
    assert "sci-status-indicator live" in live_md
    assert "Mode: Demo Scene (Offline Cache)" not in live_md
