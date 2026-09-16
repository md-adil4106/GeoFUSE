"""Unit tests for Phase 13: Demo Hardening and Fault-Tolerant Offline Operation."""

import json
import pickle
from pathlib import Path
import pytest
import numpy as np

from src.dashboard.app import (
    get_demo_cache_dir,
    load_demo_manifest,
    load_demo_bundle,
    run_cached_pipeline,
    to_display_rgb,
)
from src.utils.config import get_project_root


def test_demo_cache_files_exist():
    """Verify that all offline demo assets are precomputed and present."""
    cache_dir = get_demo_cache_dir()
    assert cache_dir.exists(), f"Demo cache directory not found: {cache_dir}"

    manifest_file = cache_dir / "manifest.json"
    assert manifest_file.exists(), f"Manifest file missing: {manifest_file}"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "tiles" in manifest
    assert len(manifest["tiles"]) == 4

    tile_indices = [t["tile_idx"] for t in manifest["tiles"]]
    assert 0 in tile_indices
    assert 8 in tile_indices
    assert 16 in tile_indices
    assert 24 in tile_indices

    for t_idx in tile_indices:
        bundle_path = cache_dir / f"demo_tile_{t_idx}.pkl"
        receipt_path = cache_dir / f"demo_tile_{t_idx}_receipt.json"
        assert bundle_path.exists(), f"Missing bundle for tile {t_idx}"
        assert receipt_path.exists(), f"Missing receipt for tile {t_idx}"


def test_confirmed_low_trust_tile_in_demo():
    """Checklist: Verify at least one example tile is deliberately flagged low-trust."""
    manifest = load_demo_manifest()
    assert manifest is not None

    low_trust_tiles = [t for t in manifest["tiles"] if not t["is_trusted"]]
    assert len(low_trust_tiles) >= 1, "Stop Condition Failure: No demo tile was flagged low-trust!"

    # Specifically verify Tile #16 and Tile #24 are low-trust examples
    flagged_indices = [t["tile_idx"] for t in low_trust_tiles]
    assert 16 in flagged_indices or 24 in flagged_indices

    # Verify that the flagged tile has a trust score below threshold
    for t in low_trust_tiles:
        assert t["trust_score_pct"] < manifest["threshold"]
        assert t["status"] == "WARNING_LOW_TRUST"


def test_demo_bundle_loading_offline():
    """Verify that offline demo bundles load cleanly and contain complete pipeline structures."""
    # Test Tile #0 (After Phase 6/7 model, all demo tiles are below 86.5% threshold)
    data_0 = load_demo_bundle(0)
    assert data_0 is not None
    assert "hr_tile" in data_0
    assert "bicubic_tile" in data_0
    assert "sr_tile" in data_0
    assert "fusion_result" in data_0
    assert "downstream_comp" in data_0
    assert "receipt" in data_0
    # All 4 demo tiles are WARNING_LOW_TRUST with Phase 6/7 model (scores 84.87-85.77%)
    assert data_0["receipt"]["trust_evaluation"]["is_trusted"] is False

    # Test Tile #16 (Confirmed Low Trust Warning)
    data_16 = load_demo_bundle(16)
    assert data_16 is not None
    assert data_16["receipt"]["trust_evaluation"]["is_trusted"] is False
    assert len(data_16["receipt"]["trust_evaluation"]["warnings_and_advisories"]) >= 1


def test_dashboard_fault_tolerance_missing_file(tmp_path):
    """Checklist: Deliberately test missing files and edge cases to ensure app never crashes."""
    # 1. to_display_rgb handles null and corrupt inputs safely
    assert to_display_rgb(None).shape == (128, 128, 3)
    assert to_display_rgb(np.zeros((10, 10), dtype=np.float32)).shape == (128, 128, 3)

    # 2. run_cached_pipeline on invalid / non-existent tile index returns None gracefully without crash
    invalid_result = run_cached_pipeline(999999)
    assert invalid_result is None

    # 3. Valid demo tiles always return complete dictionary from cache
    valid_result = run_cached_pipeline(0)
    assert valid_result is not None
    assert "fusion_result" in valid_result
