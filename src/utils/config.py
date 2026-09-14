"""Configuration loader and device resolution for GeoFUSE SentinelGuard.

Provides functions to parse config.yaml and resolve execution device with
proper scientific honesty and fallback warnings.
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and parse the project YAML configuration file.

    Args:
        config_path: Optional path to config.yaml. If not provided,
                     defaults to `<project_root>/config.yaml`.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the specified or default config file does not exist.
    """
    if config_path is None:
        root = get_project_root()
        resolved_path = root / "config.yaml"
    else:
        resolved_path = Path(config_path)

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at: {resolved_path.resolve()}. "
            "Please ensure config.yaml exists in the project root."
        )

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_device(config: Optional[Dict[str, Any]] = None) -> torch.device:
    """Determine and return the PyTorch execution device based on config and hardware.

    Honors instructions:
    - If GPU availability cannot be confirmed, prints a clear warning and continues on CPU.
    - Does not silently fail or make unsubstantiated claims.

    Args:
        config: Optional parsed configuration dictionary.

    Returns:
        torch.device: Resolved PyTorch device ('cuda' or 'cpu').
    """
    configured_mode = "auto"
    if config and "device" in config:
        configured_mode = config["device"].get("mode", "auto").lower()

    if configured_mode == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GeoFUSE Info] CUDA acceleration available: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        return torch.device("cuda")

    # GPU requested or auto-selected, but CUDA is not available in torch build or runtime
    warnings.warn(
        "[GeoFUSE Warning] GPU acceleration is not available in the current PyTorch environment. "
        "Falling back to CPU execution. Training and inference may take longer.",
        RuntimeWarning,
        stacklevel=2,
    )
    print(
        "[GeoFUSE Notice] Running on CPU. (To enable GPU, install a CUDA-enabled PyTorch build compatible with your driver)."
    )
    return torch.device("cpu")


def ensure_directories(config: Dict[str, Any]) -> None:
    """Ensure all directories specified in config.yaml exist.

    Args:
        config: Configuration dictionary with 'paths' section.
    """
    root = get_project_root()
    paths = config.get("paths", {})
    for key, rel_path in paths.items():
        if key.endswith("_dir"):
            full_path = root / rel_path
            full_path.mkdir(parents=True, exist_ok=True)
