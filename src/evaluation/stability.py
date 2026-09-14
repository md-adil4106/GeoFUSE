"""Input-Perturbation Stability Testing Module for GeoFUSE SentinelGuard.

Computes the empirical input-sensitivity stability map:
1. Applies controlled radiometric perturbations (additive Gaussian noise and brightness jitter)
   defined strictly in config.yaml.
2. Re-runs ensemble inference on each perturbed tile.
3. Calculates pixel-wise output variance across perturbations as an independent uncertainty signal.
4. Analyzes spatial correlation between the parameter-space Disagreement Map (Phase 5)
   and the input-space Stability Map (Phase 6).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.models.ensemble import predict_ensemble


def apply_controlled_perturbation(
    tile: np.ndarray,
    noise_std: float = 0.01,
    brightness_jitter_std: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply controlled input perturbations strictly from config.yaml specifications.

    Args:
        tile: Input tile array of shape (H, W, C) in normalized reflectance.
        noise_std: Standard deviation of additive zero-mean Gaussian noise.
        brightness_jitter_std: Standard deviation of multiplicative brightness scaling factor.
        seed: Optional random seed for deterministic reproducibility.

    Returns:
        np.ndarray: Perturbed tile array clipped to valid reflectance range [0.0, 1.2].
    """
    rng = np.random.default_rng(seed)
    perturbed = tile.copy().astype(np.float32)

    # 1. Multiplicative brightness jitter (e.g., simulating slight atmospheric/solar angle shifts)
    if brightness_jitter_std > 0.0:
        # Scale factor centered at 1.0
        jitter = 1.0 + rng.normal(loc=0.0, scale=brightness_jitter_std)
        jitter = max(0.85, min(1.15, jitter))  # Guard against extreme outliers
        perturbed = perturbed * jitter

    # 2. Additive Gaussian radiometric noise (simulating sensor noise)
    if noise_std > 0.0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=perturbed.shape).astype(np.float32)
        perturbed = perturbed + noise

    # Clip to physical reflectance bounds
    perturbed = np.clip(perturbed, 0.0, 1.2)
    return perturbed


def compute_stability_map(
    models: List[nn.Module],
    lr_tile: np.ndarray,
    noise_levels: Optional[List[float]] = None,
    brightness_jitter_std: float = 0.02,
    num_trials: int = 3,
    device: Optional[torch.device] = None,
    base_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Generate stability map by measuring output variance across input perturbations.

    Args:
        models: List of evaluation-ready ensemble models.
        lr_tile: Base low-resolution input tile (H, W, C).
        noise_levels: List of additive noise standard deviations from config.yaml.
        brightness_jitter_std: Multiplicative brightness jitter standard deviation.
        num_trials: Repetitions per perturbation setting.
        device: PyTorch compute device.
        base_seed: Base seed for reproducible perturbation testing.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
            - stability_map: 2D array of shape (H_hr, W_hr) representing output variance
                             across perturbations (empirical input sensitivity).
            - mean_perturbed_output: Average output reconstruction across all perturbations (H_hr, W_hr, C).
            - all_perturbed_outputs: List of all individual perturbed outputs.
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.02, 0.05]

    all_outputs: List[np.ndarray] = []
    trial_idx = 0

    # Also include the unperturbed prediction as baseline
    unperturbed_mean, _, _ = predict_ensemble(models, lr_tile, device=device)
    all_outputs.append(unperturbed_mean)

    for noise_std in noise_levels:
        for _ in range(num_trials):
            trial_seed = base_seed + trial_idx
            trial_idx += 1

            # Perturb input
            perturbed_lr = apply_controlled_perturbation(
                tile=lr_tile,
                noise_std=noise_std,
                brightness_jitter_std=brightness_jitter_std,
                seed=trial_seed,
            )

            # Re-run ensemble inference on perturbed input
            perturbed_mean, _, _ = predict_ensemble(models, perturbed_lr, device=device)
            all_outputs.append(perturbed_mean)

    # Stack along perturbation dimension: (K, H_hr, W_hr, C)
    stacked = np.stack(all_outputs, axis=0)

    # Compute pixel-wise variance across perturbations: (H_hr, W_hr, C)
    # Output variance represents the stability map
    variance_per_band = np.var(stacked, axis=0)

    # Aggregate across spectral channels (mean variance) -> (H_hr, W_hr)
    stability_map = np.mean(variance_per_band, axis=-1).astype(np.float32)

    mean_perturbed_output = np.mean(stacked, axis=0).astype(np.float32)

    return stability_map, mean_perturbed_output, all_outputs


def compare_disagreement_and_stability(
    disagreement_map: np.ndarray,
    stability_map: np.ndarray,
) -> Dict[str, Any]:
    """Compute spatial correlation and comparative statistics between disagreement and stability maps.

    Checklist Rule:
    The stability map should correlate somewhat with the disagreement map (capturing
    sensitive image regions), but must NOT be identical or degenerate.

    Args:
        disagreement_map: Phase 5 ensemble disagreement map (H, W).
        stability_map: Phase 6 input perturbation stability map (H, W).

    Returns:
        Dict[str, Any]: Metrics including Pearson correlation, Spearman rank correlation,
                        normalized difference map, and checklist validation flags.
    """
    d_flat = disagreement_map.flatten().astype(np.float64)
    s_flat = stability_map.flatten().astype(np.float64)

    # Check for degenerate maps
    d_std = float(np.std(d_flat))
    s_std = float(np.std(s_flat))

    is_degenerate = (s_std == 0.0) or (np.max(s_flat) == 0.0) or np.isnan(s_flat).any()
    if is_degenerate:
        return {
            "pearson_r": 0.0,
            "spearman_r": 0.0,
            "is_degenerate": True,
            "is_identical": False,
            "checklist_passed": False,
        }

    # Pearson correlation
    corr_matrix = np.corrcoef(d_flat, s_flat)
    pearson_r = float(corr_matrix[0, 1])

    # Min-max normalization for direct visual/numerical comparison
    d_norm = (disagreement_map - np.min(disagreement_map)) / max(1e-6, np.ptp(disagreement_map))
    s_norm = (stability_map - np.min(stability_map)) / max(1e-6, np.ptp(stability_map))

    diff_map = np.abs(d_norm - s_norm).astype(np.float32)
    mean_diff = float(np.mean(diff_map))

    # Identical check: difference is practically zero and r == 1.0
    is_identical = bool(np.allclose(d_norm, s_norm, atol=1e-4) or pearson_r > 0.999)

    # Checklist: Correlates somewhat (e.g. 0.05 <= |r| < 0.99), not degenerate, not identical
    checklist_passed = (
        not is_degenerate
        and not is_identical
        and (0.05 <= abs(pearson_r) < 0.99)
    )

    return {
        "pearson_r": round(pearson_r, 4),
        "mean_absolute_difference": round(mean_diff, 4),
        "is_degenerate": is_degenerate,
        "is_identical": is_identical,
        "checklist_passed": checklist_passed,
        "diff_map": diff_map,
    }
