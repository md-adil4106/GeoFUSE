"""Unit tests for Phase 9: Downstream Building Footprint Evaluation."""

import numpy as np
import pytest

from src.evaluation.downstream_eval import (
    FootprintExtractionError,
    compare_downstream_footprints,
    extract_building_footprints,
)


def test_extract_building_footprints_synthetic():
    H, W = 64, 64
    # Dark, low-NDVI background (soil/pavement: Red=0.2, NIR=0.2, Blue=0.2, Green=0.2 -> NDVI=0.0)
    tile = np.full((H, W, 4), 0.20, dtype=np.float32)

    # Add a bright building rooftop of size 4x4 (16 pixels, within [4, 600])
    tile[20:24, 20:24, :] = 0.85

    res = extract_building_footprints(tile, tophat_kernel_size=7, tophat_threshold=20, max_ndvi=0.20)
    mask = res["mask"]

    assert mask.shape == (H, W)
    assert res["num_components"] == 1
    assert res["footprint_pixels"] == 16
    assert bool(mask[22, 22]) is True
    assert bool(mask[5, 5]) is False


def test_extract_building_footprints_vegetation_rejection():
    H, W = 64, 64
    # Dense vegetation: Red=0.05, Green=0.4, Blue=0.05, NIR=0.6 -> NDVI = 0.55 / 0.65 = 0.85
    tile = np.zeros((H, W, 4), dtype=np.float32)
    tile[:, :, 0] = 0.05
    tile[:, :, 1] = 0.40
    tile[:, :, 2] = 0.05
    tile[:, :, 3] = 0.60

    # Even with high brightness contrast, vegetation should be rejected
    tile[20:24, 20:24, :] += 0.35

    res = extract_building_footprints(tile, tophat_kernel_size=7, max_ndvi=0.20)
    assert res["footprint_pixels"] == 0
    assert res["num_components"] == 0


def test_extract_building_footprints_area_filter():
    H, W = 64, 64
    tile = np.full((H, W, 4), 0.10, dtype=np.float32)

    # Single-pixel noise speck (1 pixel < min_area 4)
    tile[10, 10, :] = 0.90

    # Valid building (4x4 = 16 pixels, within [4, 600])
    tile[30:34, 30:34, :] = 0.90

    res = extract_building_footprints(tile, tophat_kernel_size=7, min_area=4, max_area=600)
    mask = res["mask"]

    assert not mask[10, 10]      # Filtered out
    assert bool(mask[32, 32])    # Retained
    assert res["num_components"] == 1


def test_compare_downstream_footprints_identical():
    H, W = 32, 32
    mask = np.zeros((H, W), dtype=bool)
    mask[10:20, 10:20] = True
    trust_map = np.full((H, W), 0.90, dtype=np.float32)

    stats = compare_downstream_footprints(mask, mask, trust_map)
    assert stats["overall_bic_sr"]["iou"] == 1.0
    assert stats["overall_bic_sr"]["dice"] == 1.0
    assert np.sum(stats["discrepancy_mask"]) == 0
    assert stats["scientific_honesty_label"] == "Bicubic-vs-SR Footprint Agreement (No Independent GT)"


def test_compare_downstream_footprints_trust_stratification():
    H, W = 32, 32
    # High-trust in top half, low-trust in bottom half
    trust_map = np.zeros((H, W), dtype=np.float32)
    trust_map[:16, :] = 0.95
    trust_map[16:, :] = 0.60

    mask_bic = np.zeros((H, W), dtype=bool)
    mask_sr = np.zeros((H, W), dtype=bool)

    # Top half (high trust): perfect agreement
    mask_bic[4:12, 4:12] = True
    mask_sr[4:12, 4:12] = True

    # Bottom half (low trust): divergence/discrepancy
    mask_bic[20:28, 4:12] = True
    mask_sr[22:30, 4:12] = True

    stats = compare_downstream_footprints(mask_bic, mask_sr, trust_map, trust_threshold=0.85)

    assert stats["high_trust_bic_sr"]["iou"] == 1.0
    assert stats["low_trust_bic_sr"]["iou"] < 1.0
    assert stats["high_trust_area_pct"] == 50.0
    assert stats["low_trust_area_pct"] == 50.0


def test_compare_downstream_footprints_shape_mismatch():
    m1 = np.zeros((32, 32), dtype=bool)
    m2 = np.zeros((16, 16), dtype=bool)
    t = np.zeros((32, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_downstream_footprints(m1, m2, t)


def test_footprint_extraction_invalid_channel_dim():
    tile_3band = np.zeros((32, 32, 3), dtype=np.float32)
    with pytest.raises(FootprintExtractionError):
        extract_building_footprints(tile_3band)
