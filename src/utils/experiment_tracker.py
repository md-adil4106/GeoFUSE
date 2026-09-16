"""Experiment Tracking and Schema Management for GeoFUSE SentinelGuard.

Provides structured append-only logging (JSONL) for reproducible training and
evaluation experiments, capturing complete provenance: git commit, seeds,
dataset, degradation config, model architecture, training telemetry, and
per-patch evaluation distributions.
"""

import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


REQUIRED_SCHEMA_KEYS = [
    "experiment_id",
    "timestamp",
    "git_commit",
    "random_seed",
    "evaluation_type",
    "dataset",
    "degradation_config",
    "architecture_config",
    "param_count",
    "training_config",
    "training_results",
    "hardware_telemetry",
    "inference_benchmark",
    "metrics_summary",
]


def get_git_commit_hash(repo_dir: Optional[Path] = None) -> str:
    """Retrieve the current HEAD commit hash from git.

    Args:
        repo_dir: Optional path to git repository root.

    Returns:
        str: 40-character SHA-1 commit hash, or 'unknown' if git fails.
    """
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        cwd = str(repo_dir) if repo_dir else None
        res = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_distribution_summary(
    values: Union[List[float], np.ndarray],
    round_digits: int = 4,
) -> Dict[str, float]:
    """Calculate mean, standard deviation, minimum, and maximum for an array of numbers.

    Args:
        values: List or 1D array of numerical observations.
        round_digits: Number of decimal places to round results to.

    Returns:
        Dict[str, float]: Dictionary with 'mean', 'std', 'min', 'max'.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": round(float(np.mean(arr)), round_digits),
        "std": round(float(np.std(arr)), round_digits),
        "min": round(float(np.min(arr)), round_digits),
        "max": round(float(np.max(arr)), round_digits),
    }


def validate_experiment_record(record: Dict[str, Any]) -> None:
    """Ensure that the experiment dictionary adheres to the required schema.

    Args:
        record: Experiment record dictionary.

    Raises:
        ValueError: If any mandatory key is missing or improperly typed.
    """
    missing_keys = [k for k in REQUIRED_SCHEMA_KEYS if k not in record]
    if missing_keys:
        raise ValueError(f"Experiment record missing required schema keys: {missing_keys}")

    if not isinstance(record["experiment_id"], str) or not record["experiment_id"]:
        raise ValueError("Field 'experiment_id' must be a non-empty string.")

    if not isinstance(record["metrics_summary"], dict) or not record["metrics_summary"]:
        raise ValueError("Field 'metrics_summary' must be a populated dictionary of metric distributions.")

    # Check that distributions contain mean, std, min, max
    for metric_name, dist in record["metrics_summary"].items():
        if isinstance(dist, dict):
            for sub_k in ["mean", "std", "min", "max"]:
                if sub_k not in dist:
                    raise ValueError(
                        f"Metric distribution '{metric_name}' missing required statistical key '{sub_k}'"
                    )


def log_experiment_record(
    record: Dict[str, Any],
    log_file: Optional[Path] = None,
) -> Path:
    """Validate and append an experiment record to the JSONL log file.

    Args:
        record: Complete experiment dictionary adhering to REQUIRED_SCHEMA_KEYS.
        log_file: Target .jsonl file path. Defaults to <project_root>/experiments/experiments_log.jsonl.

    Returns:
        Path: Path to the written JSONL file.
    """
    if "timestamp" not in record or not record["timestamp"]:
        record["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    validate_experiment_record(record)

    if log_file is None:
        from src.utils.config import get_project_root
        log_file = get_project_root() / "experiments" / "experiments_log.jsonl"

    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        json_line = json.dumps(record, ensure_ascii=False)
        f.write(json_line + "\n")

    return log_file


def load_experiment_records(log_file: Path) -> List[Dict[str, Any]]:
    """Read all experiment records from an append-only JSONL log file.

    Args:
        log_file: Path to the .jsonl log file.

    Returns:
        List[Dict[str, Any]]: List of parsed and validated experiment records.
    """
    if not log_file.exists():
        return []

    records = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                validate_experiment_record(rec)
                records.append(rec)
            except Exception as e:
                raise ValueError(f"Error parsing experiment record at line {line_num} in {log_file}: {e}")

    return records
