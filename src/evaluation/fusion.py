"""Evidence Fusion Module for GeoFUSE SentinelGuard.

Combines heterogeneous empirical evidence signals into a unified spatial Trust/Risk Map:
1. Ensemble Disagreement Map (Phase 5) - epistemic uncertainty proxy
2. Input-Perturbation Stability Map (Phase 6) - sensitivity proxy
3. Spectral NDVI Inconsistency Map (Phase 7) - radiometric fidelity check
4. Structural Gradient Error Map (Phase 7) - high-frequency boundary fidelity check

========================================================================================
CRITICAL SCIENTIFIC DISCLAIMER:
This composite trust/risk map is an EMPIRICAL HEURISTIC FUSION, NOT a calibrated
Bayesian posterior probability or conformal prediction guarantee. It provides an
actionable multi-criteria reliability indicator for Earth Observation practitioners.
========================================================================================
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def normalize_evidence_map(map_2d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a 2D evidence/error map strictly to the [0.0, 1.0] interval.

    Uses min-max scaling across the tile extent to ensure no single evidence
    metric dominates the composite fusion due to scale or unit mismatches.

    Args:
        map_2d: 2D array of shape (H, W).
        eps: Small epsilon to prevent division by zero for uniform arrays.

    Returns:
        np.ndarray: Normalized 2D array strictly bounded in [0.0, 1.0].
    """
    clean_map = np.nan_to_num(map_2d, nan=0.0, posinf=1.0, neginf=0.0)
    v_min = float(np.min(clean_map))
    v_max = float(np.max(clean_map))
    rng = v_max - v_min

    if rng > eps:
        norm = (clean_map - v_min) / rng
    else:
        # If the map is completely flat / constant, variance is zero
        norm = np.zeros_like(clean_map, dtype=np.float32)

    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def fuse_trust_risk_maps(
    disagreement_map: np.ndarray,
    stability_map: np.ndarray,
    delta_ndvi_map: np.ndarray,
    structural_diff_map: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
    high_risk_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Fuse multi-source evidence into a single composite Trust/Risk Map.

    NOTE: This is a heuristic multi-criteria fusion, not a calibrated Bayesian probability.

    Args:
        disagreement_map: 2D array (H, W) of ensemble std deviation (Phase 5).
        stability_map: 2D array (H, W) of perturbation variance (Phase 6).
        delta_ndvi_map: 2D array (H, W) of absolute delta-NDVI error (Phase 7).
        structural_diff_map: 2D array (H, W) of gradient magnitude difference (Phase 7).
        weights: Optional dictionary of weights for each evidence stream.
                 Keys: 'disagreement', 'stability', 'spectral', 'structural'.
                 If omitted, equal weights (0.25 each) are used.
        high_risk_threshold: Value in [0, 1] above which risk is flagged as high.

    Returns:
        Dict[str, Any]: Detailed fusion dictionary containing:
            - 'risk_map': 2D float array in [0, 1] (0 = lowest risk, 1 = highest risk)
            - 'trust_map': 2D float array in [0, 1] (1 = highest trust, 0 = lowest trust)
            - 'mean_trust_score': Average trust score across tile [0.0, 1.0]
            - 'trust_score_pct': Average trust score formatted as percentage [0.0, 100.0]
            - 'mean_risk_score': Average risk score across tile [0.0, 1.0]
            - 'pct_high_risk_pixels': Percentage of pixels exceeding high_risk_threshold
            - 'weights_used': Normalized weights dictionary summing to 1.0
            - 'normalized_signals': Dictionary of individually normalized [0, 1] maps
            - 'raw_signals': Dictionary of raw input maps
            - 'has_spatial_variation': Boolean confirming map is not a degenerate flat color

    Raises:
        ValueError: If array dimensions mismatch or arrays are not 2D.
    """
    # 1. Validate spatial dimensions
    shape = disagreement_map.shape
    if disagreement_map.ndim != 2:
        raise ValueError(f"disagreement_map must be 2D, got shape {disagreement_map.shape}")

    for name, arr in [
        ("stability_map", stability_map),
        ("delta_ndvi_map", delta_ndvi_map),
        ("structural_diff_map", structural_diff_map),
    ]:
        if arr.shape != shape:
            raise ValueError(
                f"Dimension mismatch between evidence maps: disagreement_map has {shape}, but {name} has {arr.shape}."
            )

    # 2. Parse and normalize weights (single source of truth from config)
    default_weights = {
        "disagreement": 0.25,
        "stability": 0.25,
        "spectral": 0.25,
        "structural": 0.25,
    }
    raw_weights = weights if weights is not None else default_weights

    w_disag = float(raw_weights.get("disagreement", 0.25))
    w_stab = float(raw_weights.get("stability", 0.25))
    w_spec = float(raw_weights.get("spectral", 0.25))
    w_struct = float(raw_weights.get("structural", 0.25))

    total_w = w_disag + w_stab + w_spec + w_struct
    if total_w <= 0:
        w_disag = w_stab = w_spec = w_struct = 0.25
        total_w = 1.0

    norm_weights = {
        "disagreement": round(w_disag / total_w, 4),
        "stability": round(w_stab / total_w, 4),
        "spectral": round(w_spec / total_w, 4),
        "structural": round(w_struct / total_w, 4),
    }

    # 3. Strictly normalize each evidence map to [0.0, 1.0]
    # This directly fulfills the Stop Condition requirement:
    # "normalize each signal to [0,1] before combining to prevent scale dominance"
    norm_disag = normalize_evidence_map(disagreement_map)
    norm_stab = normalize_evidence_map(stability_map)
    norm_spec = normalize_evidence_map(delta_ndvi_map)
    norm_struct = normalize_evidence_map(structural_diff_map)

    # 4. Compute weighted composite Risk map in [0.0, 1.0]
    risk_map = (
        norm_weights["disagreement"] * norm_disag
        + norm_weights["stability"] * norm_stab
        + norm_weights["spectral"] * norm_spec
        + norm_weights["structural"] * norm_struct
    )
    risk_map = np.clip(risk_map, 0.0, 1.0).astype(np.float32)

    # 5. Complementary Trust map: Trust = 1.0 - Risk
    trust_map = (1.0 - risk_map).astype(np.float32)

    # 6. Summary metrics
    mean_trust = float(np.mean(trust_map))
    mean_risk = float(np.mean(risk_map))
    high_risk_mask = risk_map >= high_risk_threshold
    pct_high_risk = float(np.mean(high_risk_mask) * 100.0)

    # Check for spatial variation (Checklist item: not a flat single color)
    std_risk = float(np.std(risk_map))
    has_spatial_variation = std_risk > 1e-4

    # Per-component statistics for transparency
    component_stats = {
        "disagreement": {
            "raw_mean": float(np.mean(disagreement_map)),
            "raw_max": float(np.max(disagreement_map)),
            "norm_mean": float(np.mean(norm_disag)),
            "weight": norm_weights["disagreement"],
        },
        "stability": {
            "raw_mean": float(np.mean(stability_map)),
            "raw_max": float(np.max(stability_map)),
            "norm_mean": float(np.mean(norm_stab)),
            "weight": norm_weights["stability"],
        },
        "spectral": {
            "raw_mean": float(np.mean(delta_ndvi_map)),
            "raw_max": float(np.max(delta_ndvi_map)),
            "norm_mean": float(np.mean(norm_spec)),
            "weight": norm_weights["spectral"],
        },
        "structural": {
            "raw_mean": float(np.mean(structural_diff_map)),
            "raw_max": float(np.max(structural_diff_map)),
            "norm_mean": float(np.mean(norm_struct)),
            "weight": norm_weights["structural"],
        },
    }

    return {
        "risk_map": risk_map,
        "trust_map": trust_map,
        "mean_trust_score": round(mean_trust, 4),
        "trust_score_pct": round(mean_trust * 100.0, 2),
        "mean_risk_score": round(mean_risk, 4),
        "pct_high_risk_pixels": round(pct_high_risk, 2),
        "high_risk_mask": high_risk_mask,
        "std_risk": round(std_risk, 5),
        "has_spatial_variation": has_spatial_variation,
        "weights_used": norm_weights,
        "component_stats": component_stats,
        "normalized_signals": {
            "disagreement": norm_disag,
            "stability": norm_stab,
            "spectral": norm_spec,
            "structural": norm_struct,
        },
        "raw_signals": {
            "disagreement": disagreement_map,
            "stability": stability_map,
            "spectral": delta_ndvi_map,
            "structural": structural_diff_map,
        },
    }


def render_trust_risk_overlay(
    gt_tile: np.ndarray,
    sr_tile: np.ndarray,
    fusion_result: Dict[str, Any],
    sample_index: int,
    output_path: str,
) -> None:
    """Render comprehensive diagnostic panel with color-coded risk/trust overlays.

    Color scheme:
    - Trust Map: RdYlGn colormap where Green = High Trust (1.0), Red = Low Trust / High Risk (0.0).
    - Risk Map Overlay: SR RGB with semi-transparent red highlights on high-risk pixels.

    Args:
        gt_tile: Ground truth tile (H, W, C).
        sr_tile: Super-Resolution reconstructed tile (H, W, C).
        fusion_result: Output dictionary from fuse_trust_risk_maps.
        sample_index: Index of the evaluated sample.
        output_path: Filepath where the figure PNG will be saved.
    """
    def to_rgb(t):
        r, g, b = t[:, :, 2], t[:, :, 1], t[:, :, 0]
        rgb = np.stack([r, g, b], axis=-1)
        v_min, v_max = np.percentile(rgb, (2, 98))
        if v_max > v_min:
            rgb = np.clip((rgb - v_min) / (v_max - v_min), 0.0, 1.0)
        return (rgb * 255).astype(np.uint8)

    gt_rgb = to_rgb(gt_tile)
    sr_rgb = to_rgb(sr_tile)

    trust_map = fusion_result["trust_map"]
    risk_map = fusion_result["risk_map"]
    norm_signals = fusion_result["normalized_signals"]

    # Create semi-transparent overlay atop SR RGB
    # Color-code using RdYlGn colormap applied to trust_map (1.0=green, 0.0=red)
    cmap = plt.get_cmap("RdYlGn")
    trust_colored = (cmap(trust_map)[:, :, :3] * 255).astype(np.uint8)
    blended_overlay = (0.55 * sr_rgb + 0.45 * trust_colored).astype(np.uint8)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), dpi=150)
    fig.patch.set_facecolor("#161616")

    # Row 1: Visual and Input Evidence Signals
    # Panel (0, 0): Reference GT RGB
    axes[0, 0].imshow(gt_rgb)
    axes[0, 0].set_title("1. Reference GT RGB (10m)\n[Pre-degradation]", color="white", fontsize=9.5, pad=8)

    # Panel (0, 1): Reconstructed SR RGB
    axes[0, 1].imshow(sr_rgb)
    axes[0, 1].set_title("2. Reconstructed SR RGB (2x)\n[Ensemble Super-Resolution]", color="white", fontsize=9.5, pad=8)

    # Panel (0, 2): Normalized Disagreement Map
    im_disag = axes[0, 2].imshow(norm_signals["disagreement"], cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title(
        f"3. Disagreement Proxy [w={fusion_result['weights_used']['disagreement']}]\n(Normalized Ensemble Variance)",
        color="white",
        fontsize=9.5,
        pad=8,
    )
    cbar02 = plt.colorbar(im_disag, ax=axes[0, 2], fraction=0.046, pad=0.04)
    cbar02.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar02.ax.axes, "yticklabels"), color="white")

    # Panel (0, 3): Normalized Stability Map
    im_stab = axes[0, 3].imshow(norm_signals["stability"], cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0, 3].set_title(
        f"4. Perturbation Sensitivity [w={fusion_result['weights_used']['stability']}]\n(Normalized Input Stability)",
        color="white",
        fontsize=9.5,
        pad=8,
    )
    cbar03 = plt.colorbar(im_stab, ax=axes[0, 3], fraction=0.046, pad=0.04)
    cbar03.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar03.ax.axes, "yticklabels"), color="white")

    # Row 2: Verification Signals & Composite Trust/Risk Outputs
    # Panel (1, 0): Normalized Delta-NDVI Map
    im_spec = axes[1, 0].imshow(norm_signals["spectral"], cmap="cividis", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title(
        f"5. Spectral Inconsistency [w={fusion_result['weights_used']['spectral']}]\n(Normalized Delta-NDVI Error)",
        color="white",
        fontsize=9.5,
        pad=8,
    )
    cbar10 = plt.colorbar(im_spec, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar10.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar10.ax.axes, "yticklabels"), color="white")

    # Panel (1, 1): Normalized Structural Gradient Error Map
    im_struct = axes[1, 1].imshow(norm_signals["structural"], cmap="plasma", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title(
        f"6. Structural Edge Error [w={fusion_result['weights_used']['structural']}]\n(Normalized Gradient Difference)",
        color="white",
        fontsize=9.5,
        pad=8,
    )
    cbar11 = plt.colorbar(im_struct, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar11.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar11.ax.axes, "yticklabels"), color="white")

    # Panel (1, 2): Composite Trust Heatmap (RdYlGn: Green=High Trust, Red=Low Trust)
    im_trust = axes[1, 2].imshow(trust_map, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title(
        f"7. Composite Trust Map\n[Mean Trust Score: {fusion_result['trust_score_pct']}%]",
        color="white",
        fontsize=9.5,
        pad=8,
    )
    cbar12 = plt.colorbar(im_trust, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar12.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar12.ax.axes, "yticklabels"), color="white")

    # Panel (1, 3): Blended Risk/Trust Overlay atop SR RGB
    axes[1, 3].imshow(blended_overlay)
    axes[1, 3].set_title(
        f"8. Trust Overlay atop SR RGB\n[Green=High Trust, Red=High Risk ({fusion_result['pct_high_risk_pixels']}% flagged)]",
        color="white",
        fontsize=9.5,
        pad=8,
    )

    for r in range(2):
        for c in range(4):
            axes[r, c].axis("off")

    plt.suptitle(
        f"GeoFUSE SentinelGuard -- Multi-Evidence Trust/Risk Map Fusion (Sample #{sample_index})\n"
        "[Empirical Heuristic Reliability Proxy -- Normalized Scale-Invariant Fusion]",
        color="white",
        fontsize=13,
        weight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
