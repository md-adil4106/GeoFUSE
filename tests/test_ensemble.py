"""Unit tests for Phase 5: Ensemble Super-Resolution and Disagreement Map."""

import numpy as np
import pytest
import torch

from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.models.model import ResidualSRNet


def test_predict_ensemble_shapes():
    # 3 models with different initialization seeds
    torch.manual_seed(10)
    m1 = ResidualSRNet(in_channels=4, out_channels=4, num_features=32, num_blocks=4, scale_factor=2)
    torch.manual_seed(20)
    m2 = ResidualSRNet(in_channels=4, out_channels=4, num_features=32, num_blocks=4, scale_factor=2)
    torch.manual_seed(30)
    m3 = ResidualSRNet(in_channels=4, out_channels=4, num_features=32, num_blocks=4, scale_factor=2)

    models = [m1, m2, m3]
    lr_np = np.random.uniform(0.1, 0.8, size=(32, 32, 4)).astype(np.float32)

    mean_recon, disagreement, preds = predict_ensemble(models, lr_np)

    assert mean_recon.shape == (64, 64, 4)
    assert disagreement.shape == (64, 64)
    assert len(preds) == 3
    assert not np.isnan(mean_recon).any()
    assert not np.isnan(disagreement).any()


def test_disagreement_map_variation():
    # Models with different initializations must produce non-zero disagreement with spatial variation
    torch.manual_seed(100)
    m1 = ResidualSRNet(in_channels=4, out_channels=4, num_features=32, num_blocks=4, scale_factor=2)
    torch.manual_seed(200)
    m2 = ResidualSRNet(in_channels=4, out_channels=4, num_features=32, num_blocks=4, scale_factor=2)

    lr_np = np.random.uniform(0.1, 0.8, size=(32, 32, 4)).astype(np.float32)
    _, disagreement, _ = predict_ensemble([m1, m2], lr_np)

    assert np.max(disagreement) > 0.0, "Disagreement map should not be all zeros"
    assert np.std(disagreement) > 0.0, "Disagreement map should exhibit spatial variation"


def test_disagreement_zero_for_cloned_models():
    # Exactly identical models should have 0 disagreement
    m1 = ResidualSRNet(in_channels=4, out_channels=4, num_features=32, num_blocks=4, scale_factor=2)
    m2 = ResidualSRNet(in_channels=4, out_channels=4, num_features=32, num_blocks=4, scale_factor=2)
    m2.load_state_dict(m1.state_dict())

    lr_np = np.random.uniform(0.1, 0.8, size=(32, 32, 4)).astype(np.float32)
    _, disagreement, _ = predict_ensemble([m1, m2], lr_np)

    assert np.allclose(disagreement, 0.0, atol=1e-6)


def test_missing_checkpoint_raises_error():
    with pytest.raises(FileNotFoundError):
        load_ensemble_members(["non_existent_model_checkpoint_path.pth"])
