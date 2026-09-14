"""Unit tests for Phase 7: Spectral Consistency (NDVI) and Structural Edge Checks."""

import numpy as np
import pytest

from src.evaluation.edge_check import (
    compute_edge_consistency,
    compute_gradient_magnitude,
    extract_canny_edges,
)
from src.evaluation.spectral_check import (
    SpectralBandError,
    compute_ndvi,
    compute_spectral_consistency,
)


def test_compute_ndvi_valid():
    # Shape (32, 32, 4) with [B02, B03, B04, B08]
    tile = np.zeros((32, 32, 4), dtype=np.float32)
    # Simulate vegetation: Red (idx 2) = 0.05, NIR (idx 3) = 0.45
    tile[:, :, 2] = 0.05
    tile[:, :, 3] = 0.45

    ndvi = compute_ndvi(tile, red_idx=2, nir_idx=3)
    assert ndvi.shape == (32, 32)
    expected = (0.45 - 0.05) / (0.45 + 0.05)  # 0.40 / 0.50 = 0.80
    assert np.allclose(ndvi, expected, atol=1e-3)
    assert np.all(ndvi >= -1.0) and np.all(ndvi <= 1.0)


def test_compute_ndvi_water_or_soil():
    tile = np.zeros((32, 32, 4), dtype=np.float32)
    # Water: Red = 0.20, NIR = 0.05
    tile[:, :, 2] = 0.20
    tile[:, :, 3] = 0.05

    ndvi = compute_ndvi(tile, red_idx=2, nir_idx=3)
    expected = (0.05 - 0.20) / (0.05 + 0.20)  # -0.15 / 0.25 = -0.60
    assert np.allclose(ndvi, expected, atol=1e-3)


def test_spectral_band_error():
    # Less than 4 bands
    tile_3band = np.zeros((32, 32, 3), dtype=np.float32)
    with pytest.raises(SpectralBandError):
        compute_ndvi(tile_3band, red_idx=2, nir_idx=3)

    # Out of bounds index
    tile_4band = np.zeros((32, 32, 4), dtype=np.float32)
    with pytest.raises(SpectralBandError):
        compute_ndvi(tile_4band, red_idx=2, nir_idx=5)


def test_compute_spectral_consistency_identical():
    tile = np.random.uniform(0.1, 0.9, size=(32, 32, 4)).astype(np.float32)
    metrics = compute_spectral_consistency(tile, tile, ndvi_threshold=0.05)

    assert metrics["mean_delta_ndvi"] == 0.0
    assert metrics["pct_inconsistent_pixels"] == 0.0
    assert metrics["is_spectrally_consistent"] is True
    assert metrics["mean_delta_green_red_ratio"] == 0.0


def test_compute_spectral_consistency_mismatch_and_deviation():
    gt = np.random.uniform(0.1, 0.9, size=(32, 32, 4)).astype(np.float32)
    sr = gt.copy()
    # Introduce severe NIR deviation in a corner patch
    sr[:10, :10, 3] += 0.5

    metrics = compute_spectral_consistency(gt, sr, ndvi_threshold=0.05)
    assert metrics["mean_delta_ndvi"] > 0.0
    assert metrics["max_delta_ndvi"] > 0.05
    assert metrics["pct_inconsistent_pixels"] > 0.0

    # Shape mismatch test
    diff_shape = np.zeros((16, 16, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_spectral_consistency(gt, diff_shape)


def test_extract_canny_edges():
    # Constant tile: no edges
    flat = np.full((64, 64, 4), 0.3, dtype=np.float32)
    edges_flat = extract_canny_edges(flat)
    assert edges_flat.shape == (64, 64)
    assert not edges_flat.any()

    # Step boundary tile: has sharp edge
    step = np.full((64, 64, 4), 0.1, dtype=np.float32)
    step[:, 32:, :] = 0.9
    edges_step = extract_canny_edges(step, low_thresh=30, high_thresh=80)
    assert edges_step.shape == (64, 64)
    assert edges_step.any()


def test_compute_gradient_magnitude():
    flat = np.full((32, 32, 4), 0.5, dtype=np.float32)
    grad_flat = compute_gradient_magnitude(flat)
    assert grad_flat.shape == (32, 32)
    assert np.allclose(grad_flat, 0.0)

    step = np.zeros((32, 32, 4), dtype=np.float32)
    step[:, 16:, :] = 1.0
    grad_step = compute_gradient_magnitude(step)
    assert grad_step.shape == (32, 32)
    assert np.max(grad_step) > 0.0


def test_compute_edge_consistency_identical():
    # Image with some distinct features
    img = np.zeros((64, 64, 4), dtype=np.float32)
    img[10:50, 10:50, :] = 0.8

    metrics = compute_edge_consistency(img, img)
    assert metrics["edge_iou"] == 1.0
    assert metrics["edge_f1"] == 1.0
    assert np.isclose(metrics["gradient_correlation"], 1.0, atol=1e-3)


def test_compute_edge_consistency_shifted():
    img_gt = np.zeros((64, 64, 4), dtype=np.float32)
    img_gt[10:50, 10:50, :] = 0.8

    # Shifted feature in SR
    img_sr = np.zeros((64, 64, 4), dtype=np.float32)
    img_sr[15:55, 15:55, :] = 0.8

    metrics = compute_edge_consistency(img_gt, img_sr)
    assert metrics["edge_iou"] < 1.0
    assert metrics["gradient_correlation"] < 1.0
    assert int(np.sum(metrics["edges_fp"])) > 0
    assert int(np.sum(metrics["edges_fn"])) > 0
