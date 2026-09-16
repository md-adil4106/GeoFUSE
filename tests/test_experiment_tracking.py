"""Tests for Experiment Tracking and JSONL Schema Management."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.utils.experiment_tracker import (
    compute_distribution_summary,
    get_git_commit_hash,
    load_experiment_records,
    log_experiment_record,
    validate_experiment_record,
)


@pytest.fixture
def sample_valid_record():
    return {
        "experiment_id": "test_exp000",
        "timestamp": "2026-09-16T12:00:00Z",
        "git_commit": "abcdef1234567890abcdef1234567890abcdef12",
        "random_seed": 42,
        "evaluation_type": "synthetic degrade-and-recover validation",
        "dataset": {
            "name": "Sentinel-2 L2A 4-band",
            "channels": ["B02", "B03", "B04", "B08"],
            "patch_size_hr": 128,
            "val_quadrant": [256, 512, 256, 512],
            "num_val_patches": 49,
        },
        "degradation_config": {
            "optical_psf_blur": {"kernel_size": 3, "sigma": 0.5},
            "downsample_factor": 2,
            "downsample_method": "bicubic",
            "sensor_noise_std": 0.01,
        },
        "architecture_config": {
            "model_name": "ResidualSRNet",
            "num_channels": 4,
            "num_features": 48,
            "num_residual_blocks": 4,
            "upsample_method": "PixelShuffle",
            "scale_factor": 2,
        },
        "param_count": 273700,
        "training_config": {
            "batch_size": 16,
            "epochs": 15,
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss": "CompoundSRLoss (L1 + 0.1 * SobelGradient)",
            "mixed_precision": True,
        },
        "training_results": {
            "best_epoch": 15,
            "train_loss_final": 0.01770,
            "val_loss_best": 0.01728,
            "total_duration_s": 6.40,
            "peak_vram_mib": 158.18,
        },
        "hardware_telemetry": {
            "device": "cuda:0",
            "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "cuda_version": "12.4",
            "torch_version": "2.6.0+cu124",
        },
        "inference_benchmark": {
            "device": "cuda:0",
            "mean_latency_ms_per_patch": 2.15,
            "std_latency_ms": 0.20,
            "min_latency_ms": 1.95,
            "max_latency_ms": 2.80,
            "throughput_patches_per_sec": 465.1,
        },
        "metrics_summary": {
            "sr_psnr_db": {"mean": 38.67, "std": 1.20, "min": 35.10, "max": 41.30},
            "sr_ssim": {"mean": 0.9272, "std": 0.015, "min": 0.8900, "max": 0.9550},
            "sr_mae": {"mean": 0.0091, "std": 0.0015, "min": 0.0060, "max": 0.0140},
            "bicubic_psnr_db": {"mean": 34.20, "std": 1.10, "min": 31.50, "max": 37.00},
            "bicubic_ssim": {"mean": 0.8650, "std": 0.020, "min": 0.8200, "max": 0.9050},
            "psnr_gain_db": {"mean": 4.47, "std": 0.45, "min": 3.20, "max": 5.60},
            "ssim_gain": {"mean": 0.0622, "std": 0.011, "min": 0.0350, "max": 0.0890},
            "delta_ndvi": {"mean": 0.0125, "std": 0.004, "min": 0.0050, "max": 0.0250},
            "gradient_correlation": {"mean": 0.9420, "std": 0.021, "min": 0.8800, "max": 0.9750},
            "edge_iou": {"mean": 0.6850, "std": 0.045, "min": 0.5800, "max": 0.7700},
            "edge_f1": {"mean": 0.8120, "std": 0.032, "min": 0.7300, "max": 0.8700},
        },
    }


def test_distribution_summary():
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    summary = compute_distribution_summary(data, round_digits=2)
    assert summary["mean"] == 30.0
    assert summary["min"] == 10.0
    assert summary["max"] == 50.0
    assert round(summary["std"], 2) == round(float(np.std(data)), 2)

    empty_summary = compute_distribution_summary([])
    assert empty_summary == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}


def test_validate_record(sample_valid_record):
    # Valid record should not raise
    validate_experiment_record(sample_valid_record)

    # Missing required key should raise ValueError
    invalid_record = sample_valid_record.copy()
    del invalid_record["architecture_config"]
    with pytest.raises(ValueError, match="missing required schema keys"):
        validate_experiment_record(invalid_record)

    # Invalid metric summary structure should raise ValueError
    invalid_metric_record = sample_valid_record.copy()
    invalid_metric_record["metrics_summary"] = {
        "sr_psnr_db": {"mean": 38.67, "std": 1.20}  # missing min and max
    }
    with pytest.raises(ValueError, match="missing required statistical key"):
        validate_experiment_record(invalid_metric_record)


def test_round_trip_jsonl_logging(sample_valid_record, tmp_path):
    log_file = tmp_path / "experiments_test.jsonl"
    
    # 1. Write first record
    written_path = log_experiment_record(sample_valid_record, log_file=log_file)
    assert written_path.exists()

    # 2. Append second record
    record2 = sample_valid_record.copy()
    record2["experiment_id"] = "test_exp002"
    log_experiment_record(record2, log_file=log_file)

    # 3. Read back
    loaded = load_experiment_records(log_file)
    assert len(loaded) == 2
    assert loaded[0]["experiment_id"] == "test_exp000"
    assert loaded[1]["experiment_id"] == "test_exp002"
    assert loaded[0]["metrics_summary"]["sr_psnr_db"]["mean"] == 38.67
    assert loaded[0]["evaluation_type"] == "synthetic degrade-and-recover validation"


def test_git_commit_hash():
    commit = get_git_commit_hash()
    assert isinstance(commit, str)
    assert len(commit) > 0
