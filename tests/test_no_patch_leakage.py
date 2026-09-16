"""Tests for Spatial Leakage Verification and Dataset Partitioning Integrity (Phase 5).

Verifies that the new geographic split (split_mode='v2') achieves strictly zero
patch overlap between training and validation sets, while preserving legacy
(split_mode='v1') backward compatibility.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import SentinelSRDataset
from src.data.tiling import load_sentinel2_stack
from src.utils.config import get_project_root


@pytest.fixture
def dummy_scene():
    """Create a deterministic 512x512x4 test raster."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.05, 0.85, size=(512, 512, 4)).astype(np.float32)


def test_v2_split_zero_patch_overlap(dummy_scene):
    """Verify that split_mode='v2' has strictly zero overlapping train/val patch pairs."""
    val_quadrant = (256, 512, 256, 512)
    patch_size = 128
    stride = 32

    train_ds = SentinelSRDataset(
        full_image=dummy_scene,
        patch_size_hr=patch_size,
        stride=stride,
        split="train",
        val_quadrant=val_quadrant,
        split_mode="v2",
        buffer_pixels=0,
    )
    val_ds = SentinelSRDataset(
        full_image=dummy_scene,
        patch_size_hr=patch_size,
        stride=stride,
        split="val",
        val_quadrant=val_quadrant,
        split_mode="v2",
        buffer_pixels=0,
    )

    assert len(train_ds) == 105, f"Expected 105 train patches, got {len(train_ds)}"
    assert len(val_ds) == 25, f"Expected 25 val patches, got {len(val_ds)}"

    # 1. Check via static dataset method
    overlap_count = SentinelSRDataset.count_overlapping_pairs(train_ds, val_ds)
    assert overlap_count == 0, f"Spatial leakage detected! Found {overlap_count} overlapping patch pairs."

    # 2. Check bounding box containment directly
    for p in val_ds.patches:
        y_min, y_max, x_min, x_max = p["bbox"]
        assert y_min >= 256 and y_max <= 512, f"Val patch {p['bbox']} rows exceed Southeast quadrant [256:512]"
        assert x_min >= 256 and x_max <= 512, f"Val patch {p['bbox']} cols exceed Southeast quadrant [256:512]"

    for p in train_ds.patches:
        y_min, y_max, x_min, x_max = p["bbox"]
        # Train patch must be strictly disjoint from Southeast quadrant
        assert (y_max <= 256 or x_max <= 256), (
            f"Train patch {p['bbox']} encroaches into Southeast quadrant [256:512, 256:512]!"
        )


def test_v2_split_with_guard_buffer(dummy_scene):
    """Verify that a guard buffer creates minimum physical separation between sets."""
    val_quadrant = (256, 512, 256, 512)
    buffer_pixels = 32
    patch_size = 128
    stride = 32

    train_ds = SentinelSRDataset(
        full_image=dummy_scene,
        patch_size_hr=patch_size,
        stride=stride,
        split="train",
        val_quadrant=val_quadrant,
        split_mode="v2",
        buffer_pixels=buffer_pixels,
    )
    val_ds = SentinelSRDataset(
        full_image=dummy_scene,
        patch_size_hr=patch_size,
        stride=stride,
        split="val",
        val_quadrant=val_quadrant,
        split_mode="v2",
        buffer_pixels=buffer_pixels,
    )

    assert len(train_ds) == 88, f"Expected 88 buffered train patches, got {len(train_ds)}"
    assert len(val_ds) == 25, f"Expected 25 val patches, got {len(val_ds)}"

    # Verify zero overlap
    overlap_count = SentinelSRDataset.count_overlapping_pairs(train_ds, val_ds)
    assert overlap_count == 0

    # Calculate minimum spatial distance between bounding boxes
    min_dist = float("inf")
    for t in train_ds.patches:
        ty0, ty1, tx0, tx1 = t["bbox"]
        for v in val_ds.patches:
            vy0, vy1, vx0, vx1 = v["bbox"]
            dy = max(0, max(ty0 - vy1, vy0 - ty1))
            dx = max(0, max(tx0 - vx1, vx0 - tx1))
            dist = max(dy, dx)
            if dist < min_dist:
                min_dist = dist

    assert min_dist >= buffer_pixels, (
        f"Minimum boundary separation {min_dist}px is less than configured buffer {buffer_pixels}px!"
    )


def test_v1_split_backward_compatibility(dummy_scene):
    """Verify that split_mode='v1' reproduces legacy Phase 3/4 baseline counts."""
    val_quadrant = (256, 512, 256, 512)
    patch_size = 128
    stride = 32

    train_ds = SentinelSRDataset(
        full_image=dummy_scene,
        patch_size_hr=patch_size,
        stride=stride,
        split="train",
        val_quadrant=val_quadrant,
        split_mode="v1",
    )
    val_ds = SentinelSRDataset(
        full_image=dummy_scene,
        patch_size_hr=patch_size,
        stride=stride,
        split="val",
        val_quadrant=val_quadrant,
        split_mode="v1",
    )

    # Legacy counts from Phase 3 audit
    assert len(train_ds) == 120
    assert len(val_ds) == 49

    # Legacy center-point overlap is detected
    overlap_count = SentinelSRDataset.count_overlapping_pairs(train_ds, val_ds)
    assert overlap_count > 0, "Legacy split unexpectedly reported zero overlap"


def test_split_config_v2_json_validity():
    """Verify data/split_config_v2.json exists, is valid JSON, and matches dataset constants."""
    root = get_project_root()
    config_path = root / "data" / "split_config_v2.json"
    assert config_path.exists(), f"split_config_v2.json not found at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["split_version"] == "v2.0"
    assert data["partitioning_strategy"]["partition_yields"]["train_patches"] == 105
    assert data["partitioning_strategy"]["partition_yields"]["val_patches"] == 25
    assert data["partitioning_strategy"]["partition_yields"]["overlapping_patch_pairs"] == 0
    assert data["degradation_setup"]["paradigm"] == "synthetic degrade-and-recover validation"


def test_real_sentinel2_stack_leak_free():
    """Run full spatial leakage verification against the actual Sentinel-2 scene stack."""
    root = get_project_root()
    raw_dir = root / "data" / "raw"
    stack, _ = load_sentinel2_stack(raw_dir)

    train_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=128,
        stride=32,
        split="train",
        split_mode="v2",
        buffer_pixels=0,
    )
    val_ds = SentinelSRDataset(
        full_image=stack,
        patch_size_hr=128,
        stride=32,
        split="val",
        split_mode="v2",
        buffer_pixels=0,
    )

    overlap_count = SentinelSRDataset.count_overlapping_pairs(train_ds, val_ds)
    assert overlap_count == 0, f"Real scene stack produced {overlap_count} overlapping pairs!"
