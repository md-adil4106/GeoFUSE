"""Unit and integration tests for Phase 13 locked final demo ensemble."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_device, get_project_root, load_config


def test_final_demo_checkpoints_exist_and_load():
    root = get_project_root()
    final_dir = root / "checkpoints" / "final_demo_v1"
    manifest_path = final_dir / "model_manifest.json"

    assert final_dir.exists(), f"Directory not found: {final_dir}"
    assert manifest_path.exists(), f"Manifest not found: {manifest_path}"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assert manifest["tag"] == "final_demo_v1"
    assert manifest["num_blocks"] == 6
    assert manifest["num_features"] == 48

    ckpt_paths = [final_dir / f"ensemble_member_{i}.pth" for i in range(3)]
    for p in ckpt_paths:
        assert p.exists(), f"Checkpoint missing: {p}"

    # Verify loading on CPU
    cpu_device = torch.device("cpu")
    cpu_models = load_ensemble_members(ckpt_paths, device=cpu_device)
    assert len(cpu_models) == 3
    for m in cpu_models:
        assert not m.training
        assert sum(p.numel() for p in m.parameters()) == 356836

    # Verify loading on hardware device (GPU if available)
    config = load_config()
    hw_device = get_device(config)
    hw_models = load_ensemble_members(ckpt_paths, device=hw_device)
    assert len(hw_models) == 3


def test_final_demo_ensemble_inference():
    root = get_project_root()
    final_dir = root / "checkpoints" / "final_demo_v1"
    ckpt_paths = [final_dir / f"ensemble_member_{i}.pth" for i in range(3)]

    cpu_device = torch.device("cpu")
    models = load_ensemble_members(ckpt_paths, device=cpu_device)

    # 4-band dummy LR tile (64, 64, 4)
    np.random.seed(42)
    lr_dummy = np.random.uniform(0.1, 0.8, size=(64, 64, 4)).astype(np.float32)

    mean_sr, disag_map, member_preds = predict_ensemble(models, lr_dummy, device=cpu_device)

    assert mean_sr.shape == (128, 128, 4)
    assert disag_map.shape == (128, 128)
    assert len(member_preds) == 3
    assert np.all(np.isfinite(mean_sr))
    assert np.all(np.isfinite(disag_map))
    assert np.all(disag_map >= 0.0)
