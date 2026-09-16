"""Unit and integration tests for user upload Execute button and super-resolution scoring workflow."""

import os
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.utils.config import get_project_root


def _create_synthetic_geotiff_bytes(h=128, w=128) -> bytes:
    """Helper to generate a valid 4-band GeoTIFF in memory."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    data = np.random.randint(200, 3000, size=(4, h, w), dtype=np.uint16)
    transform = from_origin(500000.0, 3000000.0, 10.0, 10.0)

    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            height=h,
            width=w,
            count=4,
            dtype="uint16",
            crs="EPSG:32643",
            transform=transform,
        ) as dst:
            dst.write(data)
        return bytes(mem.getbuffer())


def test_upload_execute_button_workflow():
    """Verify that user upload displays the Execute button, and tapping it produces SR image and score."""
    root = get_project_root()
    app_path = str(root / "src/dashboard/app.py")

    at = AppTest.from_file(app_path, default_timeout=40)
    at.run()
    assert not at.exception

    # Switch to user upload mode
    at.sidebar.radio[0].set_value("Upload GeoTIFF")
    at.run()
    assert not at.exception
    assert at.session_state["app_mode"] == "Live Analysis"

    # Upload synthetic 4-band raster
    raw_bytes = _create_synthetic_geotiff_bytes(128, 128)
    uploader = at.sidebar.file_uploader[0]
    uploader.upload("S2A_MSIL2A_TEST_B02_B03_B04_B08.tif", raw_bytes, mime_type="image/tiff")
    at.run()
    assert not at.exception

    # Verify Execute button is present
    exec_buttons = [b for b in at.button if "Execute" in b.label]
    assert len(exec_buttons) >= 1, "Execute Super-Resolution button not found"

    # Click Execute button
    exec_buttons[0].click()
    at.run()
    assert not at.exception, f"Exception during execution: {at.exception}"

    # Verify Quick-Glance Super-Resolution card and Test Score
    combined_md = "\n".join(m.value for m in at.markdown)
    assert "Super-Resolution Output & Composite Test Score" in combined_md
    assert "Composite Test Score" in combined_md
    assert "/ 100" in combined_md

    # Verify 5 tabs are present
    assert len(at.tabs) == 5
    tab_labels = [t.label for t in at.tabs]
    assert "Overview" in tab_labels
    assert "Comparison" in tab_labels
    assert "Evidence" in tab_labels
    assert "Downstream Impact" in tab_labels
    assert "Trust Receipt" in tab_labels
