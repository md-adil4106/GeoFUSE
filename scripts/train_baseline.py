"""Dedicated CLI script for reproducible GPU baseline training run (exp001)."""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.train import train_baseline_experiment

if __name__ == "__main__":
    train_baseline_experiment(exp_id="baseline_exp001")
