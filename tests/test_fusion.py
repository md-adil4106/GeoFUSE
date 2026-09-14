"""Unit tests for Phase 8: Multi-Evidence Trust/Risk Map Fusion."""

import numpy as np
import pytest

from src.evaluation.fusion import fuse_trust_risk_maps, normalize_evidence_map


def test_normalize_evidence_map_bounds():
    # Array spanning arbitrary range
    arr = np.array([[-10.0, 5.0], [20.0, 50.0]], dtype=np.float32)
    norm = normalize_evidence_map(arr)

    assert norm.shape == arr.shape
    assert np.min(norm) == 0.0
    assert np.max(norm) == 1.0
    assert norm[0, 0] == 0.0
    assert norm[1, 1] == 1.0
    assert not np.isnan(norm).any()


def test_normalize_evidence_map_flat():
    # Flat array: zero dynamic range
    flat = np.full((32, 32), 0.42, dtype=np.float32)
    norm = normalize_evidence_map(flat)

    assert norm.shape == flat.shape
    assert not np.isnan(norm).any()
    assert np.all(norm == 0.0)


def test_fuse_trust_risk_maps_complements():
    H, W = 32, 32
    np.random.seed(42)
    disag = np.random.uniform(0.001, 0.015, (H, W)).astype(np.float32)
    stab = np.random.uniform(0.0001, 0.0008, (H, W)).astype(np.float32)
    spec = np.random.uniform(0.01, 0.12, (H, W)).astype(np.float32)
    struct = np.random.uniform(0.0, 1.5, (H, W)).astype(np.float32)

    result = fuse_trust_risk_maps(
        disagreement_map=disag,
        stability_map=stab,
        delta_ndvi_map=spec,
        structural_diff_map=struct,
    )

    risk = result["risk_map"]
    trust = result["trust_map"]

    # Trust and Risk must be exact complements summing to 1.0
    assert risk.shape == (H, W)
    assert trust.shape == (H, W)
    assert np.allclose(trust + risk, 1.0, atol=1e-5)
    assert 0.0 <= result["mean_trust_score"] <= 1.0
    assert 0.0 <= result["mean_risk_score"] <= 1.0
    assert result["has_spatial_variation"] is True
    assert result["std_risk"] > 0.0


def test_fuse_trust_risk_maps_weights_normalization():
    H, W = 16, 16
    disag = np.ones((H, W), dtype=np.float32)
    stab = np.ones((H, W), dtype=np.float32)
    spec = np.ones((H, W), dtype=np.float32)
    struct = np.ones((H, W), dtype=np.float32)

    # Pass non-normalized weights
    custom_weights = {
        "disagreement": 10.0,
        "stability": 20.0,
        "spectral": 30.0,
        "structural": 40.0,
    }

    result = fuse_trust_risk_maps(
        disagreement_map=disag,
        stability_map=stab,
        delta_ndvi_map=spec,
        structural_diff_map=struct,
        weights=custom_weights,
    )

    weights_used = result["weights_used"]
    assert np.isclose(sum(weights_used.values()), 1.0, atol=1e-4)
    assert weights_used["disagreement"] == 0.10
    assert weights_used["stability"] == 0.20
    assert weights_used["spectral"] == 0.30
    assert weights_used["structural"] == 0.40


def test_fuse_trust_risk_maps_scale_invariance_stop_condition():
    """Stop Condition Test: Confirms micro-scale signal is not dominated by macro-scale signal."""
    H, W = 32, 32

    # Tiny signal: disagreement std ~ 0.001
    tiny_signal = np.zeros((H, W), dtype=np.float32)
    tiny_signal[16:, :] = 0.002

    # Huge signal: raw structural error ~ 200.0
    huge_signal = np.zeros((H, W), dtype=np.float32)
    huge_signal[:, 16:] = 200.0

    stab = np.zeros((H, W), dtype=np.float32)
    spec = np.zeros((H, W), dtype=np.float32)

    # Fuse with equal weights (0.25 each)
    result = fuse_trust_risk_maps(
        disagreement_map=tiny_signal,
        stability_map=stab,
        delta_ndvi_map=spec,
        structural_diff_map=huge_signal,
        weights={"disagreement": 0.5, "stability": 0.0, "spectral": 0.0, "structural": 0.5},
    )

    risk = result["risk_map"]
    # If unnormalized, huge_signal (200.0) would make tiny_signal (0.002) contribute ~0.001%
    # With [0, 1] normalization, both signals contribute 0.5 * 1.0 = 0.5 in their active quadrants
    assert np.isclose(risk[20, 5], 0.5, atol=1e-3)   # Only tiny_signal active
    assert np.isclose(risk[5, 20], 0.5, atol=1e-3)   # Only huge_signal active
    assert np.isclose(risk[20, 20], 1.0, atol=1e-3)  # Both active
    assert np.isclose(risk[5, 5], 0.0, atol=1e-3)    # Neither active


def test_fuse_trust_risk_maps_dimension_mismatch():
    disag = np.zeros((32, 32), dtype=np.float32)
    stab = np.zeros((16, 16), dtype=np.float32)  # Mismatched shape
    spec = np.zeros((32, 32), dtype=np.float32)
    struct = np.zeros((32, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        fuse_trust_risk_maps(disag, stab, spec, struct)

    # 3D array test
    disag_3d = np.zeros((32, 32, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 2D"):
        fuse_trust_risk_maps(disag_3d, stab, spec, struct)
