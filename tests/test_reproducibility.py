"""Unit tests for Phase 12: End-to-End Reproducibility Pipeline."""

import json
from pathlib import Path
import pytest

from scripts.reproduce_all import run_pipeline
from src.utils.config import get_project_root, load_config


def test_reproducibility_pipeline_execution():
    """Verify that run_pipeline executes cleanly with skip_training=True and skip_tests=True."""
    results = run_pipeline(skip_training=True, skip_tests=True, launch_dashboard=False)

    assert isinstance(results, dict)
    assert "device" in results
    assert "stack_shape" in results
    assert len(results["stack_shape"]) == 3
    assert results["stack_shape"][2] == 4  # 4 spectral bands

    assert "baseline_psnr" in results
    assert results["baseline_psnr"] > 25.0  # Plausible satellite PSNR

    assert "trust_scores" in results
    assert len(results["trust_scores"]) == 4
    for score in results["trust_scores"]:
        assert 0.0 <= score <= 100.0

    assert "receipt_paths" in results
    assert len(results["receipt_paths"]) == 4
    for p_str in results["receipt_paths"]:
        p = Path(p_str)
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "receipt_id" in data
            assert "evidence_metrics" in data
            assert "trust_evaluation" in data
