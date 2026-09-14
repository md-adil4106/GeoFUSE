"""Structural and Edge Consistency Verification Module for GeoFUSE SentinelGuard.

Evaluates high-frequency boundary and structural alignment between the reconstructed
Super-Resolution tile and the pre-degradation reference tile:
1. Canny Edge Alignment: Measures Edge IoU, Precision, Recall, and F1 score.
2. Gradient Correlation: Computes spatial Pearson correlation between Sobel gradient magnitudes.
3. Multi-Color Edge Overlay: Visualizes matched edges (Green), hallucinated edges (Red),
   and missing edges (Cyan) atop the reconstructed scene.
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def extract_canny_edges(
    image: np.ndarray,
    low_thresh: float = 40,
    high_thresh: float = 120,
) -> np.ndarray:
    """Extract binary Canny edge map from an image (single or multi-channel).

    Args:
        image: Array of shape (H, W) or (H, W, C) in range [0, 1] or [0, 255].
        low_thresh: Lower hysteresis threshold for Canny.
        high_thresh: Upper hysteresis threshold for Canny.

    Returns:
        np.ndarray: Binary boolean edge map of shape (H, W).
    """
    if image.ndim == 3:
        # Convert multi-spectral RGB (channels 2, 1, 0) to 8-bit grayscale
        r = image[:, :, 2]
        g = image[:, :, 1]
        b = image[:, :, 0]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = image.copy()

    # Normalize to 0-255 uint8
    g_min, g_max = np.min(gray), np.max(gray)
    if g_max > g_min:
        gray_u8 = ((gray - g_min) / (g_max - g_min) * 255.0).astype(np.uint8)
    else:
        gray_u8 = np.zeros_like(gray, dtype=np.uint8)

    # Optional gentle blur to reduce noise prior to edge detection
    blurred = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    edges = cv2.Canny(blurred, int(low_thresh), int(high_thresh))
    return edges > 0


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute 2D spatial gradient magnitude using Sobel operators."""
    if image.ndim == 3:
        gray = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32)
    dx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(dx**2 + dy**2)
    return mag


def compute_edge_consistency(
    gt_tile: np.ndarray,
    sr_tile: np.ndarray,
    canny_low: float = 40,
    canny_high: float = 120,
) -> Dict[str, Any]:
    """Calculate quantitative structural and edge consistency metrics.

    Args:
        gt_tile: Reference HR tile (H, W, C).
        sr_tile: Reconstructed SR tile (H, W, C).
        canny_low: Canny lower threshold.
        canny_high: Canny upper threshold.

    Returns:
        Dict[str, Any]: Metrics dictionary including Edge IoU, F1 score,
                        gradient correlation, and binary edge masks.
    """
    if gt_tile.shape != sr_tile.shape:
        raise ValueError(f"Shape mismatch: GT {gt_tile.shape} vs SR {sr_tile.shape}")

    # 1. Binary Canny edges
    edges_gt = extract_canny_edges(gt_tile, canny_low, canny_high)
    edges_sr = extract_canny_edges(sr_tile, canny_low, canny_high)

    # Contingency table components
    tp = np.logical_and(edges_gt, edges_sr)
    fp = np.logical_and(np.logical_not(edges_gt), edges_sr)  # False/hallucinated edges
    fn = np.logical_and(edges_gt, np.logical_not(edges_sr))  # Missed edges
    union = np.logical_or(edges_gt, edges_sr)

    count_tp = int(np.sum(tp))
    count_fp = int(np.sum(fp))
    count_fn = int(np.sum(fn))
    count_union = int(np.sum(union))

    edge_iou = float(count_tp / count_union) if count_union > 0 else 1.0
    precision = float(count_tp / (count_tp + count_fp)) if (count_tp + count_fp) > 0 else 1.0
    recall = float(count_tp / (count_tp + count_fn)) if (count_tp + count_fn) > 0 else 1.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 2. Gradient magnitude correlation
    grad_gt = compute_gradient_magnitude(gt_tile)
    grad_sr = compute_gradient_magnitude(sr_tile)

    g_gt_flat = grad_gt.flatten()
    g_sr_flat = grad_sr.flatten()

    if np.std(g_gt_flat) > 0 and np.std(g_sr_flat) > 0:
        grad_corr = float(np.corrcoef(g_gt_flat, g_sr_flat)[0, 1])
    else:
        grad_corr = 0.0

    return {
        "edge_iou": round(edge_iou, 4),
        "edge_precision": round(precision, 4),
        "edge_recall": round(recall, 4),
        "edge_f1": round(f1, 4),
        "gradient_correlation": round(grad_corr, 4),
        "edges_gt": edges_gt,
        "edges_sr": edges_sr,
        "edges_tp": tp,
        "edges_fp": fp,
        "edges_fn": fn,
        "grad_gt": grad_gt,
        "grad_sr": grad_sr,
    }


def render_edge_consistency_overlay(
    gt_tile: np.ndarray,
    sr_tile: np.ndarray,
    edge_metrics: Dict[str, Any],
    sample_index: int,
    output_path: str,
) -> None:
    """Render 5-panel structural figure with multi-color edge agreement overlay."""
    def to_rgb(t):
        r, g, b = t[:, :, 2], t[:, :, 1], t[:, :, 0]
        rgb = np.stack([r, g, b], axis=-1)
        v_min, v_max = np.percentile(rgb, (2, 98))
        if v_max > v_min:
            rgb = np.clip((rgb - v_min) / (v_max - v_min), 0.0, 1.0)
        return (rgb * 255).astype(np.uint8)

    sr_rgb = to_rgb(sr_tile)

    tp = edge_metrics["edges_tp"]
    fp = edge_metrics["edges_fp"]
    fn = edge_metrics["edges_fn"]

    # Multi-color edge overlay on desaturated SR background
    gray_bg = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2GRAY)
    overlay_rgb = np.stack([gray_bg, gray_bg, gray_bg], axis=-1)

    # True Positive (Matched): Vibrant Green
    overlay_rgb[tp] = [0, 255, 60]
    # False Positive (Hallucinated / Displaced): Bright Red
    overlay_rgb[fp] = [255, 30, 30]
    # False Negative (Missed GT edge): Cyan
    overlay_rgb[fn] = [0, 210, 255]

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.5), dpi=150)
    fig.patch.set_facecolor("#181818")

    titles = [
        "GT Canny Edges\n[Reference 10m]",
        "SR Canny Edges\n[Reconstructed 2x]",
        f"GT Gradient Magnitude\n[Sobel Operator]",
        f"SR Gradient Magnitude\n[Sobel Operator, r = {edge_metrics['gradient_correlation']:.3f}]",
        f"Edge Alignment Overlay\n[IoU: {edge_metrics['edge_iou']:.3f} | F1: {edge_metrics['edge_f1']:.3f}]",
    ]

    # Panel 0: GT edges
    axes[0].imshow(edge_metrics["edges_gt"], cmap="gray")
    # Panel 1: SR edges
    axes[1].imshow(edge_metrics["edges_sr"], cmap="gray")
    # Panel 2: GT Gradient
    im2 = axes[2].imshow(edge_metrics["grad_gt"], cmap="magma")
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar2.ax.axes, "yticklabels"), color="white")

    # Panel 3: SR Gradient
    im3 = axes[3].imshow(edge_metrics["grad_sr"], cmap="magma")
    cbar3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    cbar3.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar3.ax.axes, "yticklabels"), color="white")

    # Panel 4: Composite Overlay
    axes[4].imshow(overlay_rgb)

    for ax, title in zip(axes, titles):
        ax.set_title(title, color="white", fontsize=9.5, pad=8)
        ax.axis("off")

    plt.suptitle(
        f"GeoFUSE SentinelGuard — Edge & Structural Consistency (Sample #{sample_index}) "
        "[Green: Matched | Red: Hallucinated/Shifted | Cyan: Missed]",
        color="white",
        fontsize=12,
        weight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
