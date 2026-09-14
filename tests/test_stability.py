"""Unit tests for Phase 6: Input-Perturbation Stability Testing."""

import numpy as np
import pytest
import torch

from src.evaluation.stability import (
    apply_controlled_perturbation,
    compare_disagreement_and_stability,
    compute_stability_map,
)
from src.models.model import ResidualSRNet


def test_apply_controlled_perturbation():
    tile = np.full((32, 32, 4), 0.3, dtype=np.float32)

    # Test noise addition
    noisy = apply_controlled_perturbation(tile, noise_std=0.02, brightness_jitter_std=0.0, seed=42)
    assert noisy.shape == tile.shape
    assert not np.array_equal(noisy, tile)
    assert np.all(noisy >= 0.0)
    assert np.all(noisy <= 1.2)

    # Test brightness jitter
    jittered = apply_controlled_perturbation(tile, noise_std=0.0, brightness_jitter_std=0.05, seed=123)
    assert jittered.shape == tile.shape
    assert not np.array_equal(jittered, tile)
    assert np.all(jittered >= 0.0)


def test_compute_stability_map():
    # Setup two small models
    torch.manual_seed(1)
    m1 = ResidualSRNet(in_channels=4, out_channels=4, num_features=32, num_blocks=4, scale_factor=2)
    torch.manual_seed(2)
    m2 = ResidualSRNet(in_channels=4, out_channels=4, num_features=32, num_blocks=4, scale_factor=2)

    lr_tile = np.random.uniform(0.1, 0.8, size=(32, 32, 4)).astype(np.float32)

    stability_map, mean_out, all_outputs = compute_stability_map(
        models=[m1, m2],
        lr_tile=lr_tile,
        noise_levels=[0.01, 0.02],
        brightness_jitter_std=0.02,
        num_trials=2,
    )

    assert stability_map.shape == (64, 64)
    assert mean_out.shape == (64, 64, 4)
    assert len(all_outputs) == 5  # 1 unperturbed + 2 levels * 2 trials = 5
    assert not np.isnan(stability_map).any()
    assert np.all(stability_map >= 0.0)
    # Output variance must exhibit non-zero spatial variation
    assert np.max(stability_map) > 0.0
    assert np.std(stability_map) > 0.0


def test_compare_disagreement_and_stability():
    # Two synthetic related but distinct 2D maps
    x, y = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
    disagreement = (np.sin(3 * x) * np.cos(3 * y) + 1.0) / 2.0
    # Stability has correlated base plus distinct high-frequency component
    stability = disagreement * 0.7 + np.random.normal(0, 0.05, size=(64, 64))
    stability = np.clip(stability, 0.0, 1.0)

    cmp_info = compare_disagreement_and_stability(disagreement, stability)
    assert not cmp_info["is_degenerate"]
    assert not cmp_info["is_identical"]
    assert 0.1 < cmp_info["pearson_r"] < 0.98
    assert cmp_info["checklist_passed"]


def test_degenerate_detection():
    flat_zero = np.zeros((64, 64), dtype=np.float32)
    disag = np.random.uniform(0, 1, (64, 64)).astype(np.float32)

    cmp_info = compare_disagreement_and_stability(disag, flat_zero)
    assert cmp_info["is_degenerate"] is True
    assert cmp_info["checklist_passed"] is False
