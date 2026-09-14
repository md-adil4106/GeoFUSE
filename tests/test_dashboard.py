"""Unit tests for Phase 10: Streamlit Dashboard UI and Pipeline Execution."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.dashboard.app import run_cached_pipeline, to_display_rgb
from src.utils.config import get_project_root


def test_to_display_rgb():
    """Verify RGB conversion for natural color and false-color CIR."""
    tile = np.random.uniform(0.1, 0.9, size=(32, 32, 4)).astype(np.float32)

    # Natural RGB
    rgb = to_display_rgb(tile, false_color=False)
    assert rgb.shape == (32, 32, 3)
    assert rgb.dtype == np.uint8
    assert np.min(rgb) >= 0 and np.max(rgb) <= 255

    # False-Color Infrared
    cir = to_display_rgb(tile, false_color=True)
    assert cir.shape == (32, 32, 3)
    assert cir.dtype == np.uint8


def test_run_cached_pipeline():
    """Verify that the dashboard pipeline executes inference, fusion, and footprint extraction."""
    data = run_cached_pipeline(0)

    assert "hr_tile" in data
    assert "sr_tile" in data
    assert "bicubic_tile" in data
    assert "fusion_result" in data
    assert "downstream_comp" in data

    # Check that all four views exist and have identical spatial dimensions
    hr_shape = data["hr_tile"].shape[:2]
    sr_shape = data["sr_tile"].shape[:2]
    bic_shape = data["bicubic_tile"].shape[:2]
    trust_shape = data["fusion_result"]["trust_map"].shape

    assert hr_shape == sr_shape == bic_shape == trust_shape
    assert 0.0 <= data["fusion_result"]["trust_score_pct"] <= 100.0


def test_dashboard_ui_renders_in_isolated_process():
    """Verify that the Streamlit AppTest launches and renders UI widgets without errors in isolation."""
    code = (
        "from streamlit.testing.v1 import AppTest\n"
        "from src.utils.config import get_project_root\n"
        "root = get_project_root()\n"
        "app_path = str(root / 'src/dashboard/app.py')\n"
        "at = AppTest.from_file(app_path, default_timeout=35)\n"
        "at.run()\n"
        "assert not at.exception, f'Exception: {at.exception}'\n"
        "assert len(at.title) >= 1\n"
        "assert len(at.metric) >= 5\n"
        "# Test evidence breakdown toggle\n"
        "at.sidebar.checkbox[0].check()\n"
        "at.run()\n"
        "assert not at.exception\n"
        "print('UI_SUCCESS')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(get_project_root()),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"AppTest failed:\nStdout: {result.stdout}\nStderr: {result.stderr}"
    assert "UI_SUCCESS" in result.stdout
