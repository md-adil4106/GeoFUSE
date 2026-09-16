"""Tests for Ensemble v2 Retraining, Diversity, and Informativeness (Phase 7)."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.evaluation.edge_check import compute_gradient_magnitude
from src.data.degrade import evaluate_reconstruction_fidelity
from src.utils.config import get_project_root


@pytest.fixture
def ensemble_v2_paths():
    root = get_project_root()
    paths = [root / "checkpoints" / "ensemble_v2" / f"member_{i}.pt" for i in (1, 2, 3)]
    for p in paths:
        assert p.exists(), f"Ensemble checkpoint missing: {p}"
    return paths


@pytest.fixture
def dummy_input():
    rng = np.random.default_rng(42)
    return rng.uniform(0.1, 0.9, size=(64, 64, 4)).astype(np.float32)


def test_ensemble_v2_checkpoints_exist_and_load(ensemble_v2_paths):
    """Confirm all 3 checkpoints load cleanly into memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_ensemble_members(ensemble_v2_paths, device=device)
    assert len(models) == 3
    for m in models:
        assert isinstance(m, torch.nn.Module)
        assert not m.training


def test_ensemble_v2_members_produce_differing_outputs(ensemble_v2_paths, dummy_input):
    """Confirm that the 3 ensemble members produce distinct (differing) predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_ensemble_members(ensemble_v2_paths, device=device)

    mean_recon, disagreement, preds = predict_ensemble(models, dummy_input, device=device)
    assert len(preds) == 3

    # Pairwise MSE must be strictly positive
    mse_1_2 = float(np.mean((preds[0] - preds[1]) ** 2))
    mse_1_3 = float(np.mean((preds[0] - preds[2]) ** 2))
    mse_2_3 = float(np.mean((preds[1] - preds[2]) ** 2))

    assert mse_1_2 > 1e-8, f"Member 1 and 2 produced identical predictions (MSE={mse_1_2})"
    assert mse_1_3 > 1e-8, f"Member 1 and 3 produced identical predictions (MSE={mse_1_3})"
    assert mse_2_3 > 1e-8, f"Member 2 and 3 produced identical predictions (MSE={mse_2_3})"


def test_ensemble_v2_disagreement_properties(ensemble_v2_paths, dummy_input):
    """Verify properties of the empirical disagreement map."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_ensemble_members(ensemble_v2_paths, device=device)

    mean_recon, disagreement, _ = predict_ensemble(models, dummy_input, device=device)

    assert disagreement.shape == (128, 128)
    assert np.all(disagreement >= 0.0), "Disagreement standard deviation must be non-negative"
    assert np.mean(disagreement) > 0.0, "Disagreement cannot be entirely zero across distinct members"
    assert np.max(disagreement) < 1.0, "Disagreement in normalized reflectance [0, 1] cannot exceed 1.0"
