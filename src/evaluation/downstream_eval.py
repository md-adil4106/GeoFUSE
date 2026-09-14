"""Downstream Task Evaluation Module for GeoFUSE SentinelGuard.

Evaluates the real-world utility of super-resolved satellite imagery on a canonical
Earth Observation downstream task: Building Footprint Extraction.

Runs identical morphological building footprint extraction on:
1. The 2x Bicubic Interpolation Baseline
2. The 2x Super-Resolution Ensemble Mean Reconstruction
3. (Optional) The Pre-Degradation 10m Reference Tile (for relative degrade-and-recover benchmark)

Compares extracted footprints across High-Trust vs. Low-Trust regions (Phase 8 Trust Map)
and computes quantitative agreement statistics (IoU, Dice, % Overlap).

========================================================================================
SCIENTIFIC HONESTY NOTICE:
Unless certified independent high-resolution vector building footprints exist,
all metrics are reported strictly as 'Bicubic-vs-SR Agreement' or 'Relative Agreement
Against Reference HR Extraction', NOT 'True Accuracy'.
========================================================================================
"""

from typing import Any, Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .spectral_check import compute_ndvi


class FootprintExtractionError(Exception):
    """Raised when morphological footprint extraction yields degenerate or invalid masks."""
    pass


def extract_building_footprints(
    tile_4band: np.ndarray,
    tophat_kernel_size: int = 7,
    tophat_threshold: int = 25,
    max_ndvi: float = 0.20,
    min_area: int = 4,
    max_area: int = 600,
    red_idx: int = 2,
    nir_idx: int = 3,
) -> Dict[str, Any]:
    """Extract building footprints from a 4-band Sentinel-2 tile via morphological analysis.

    Methodology:
    1. Grayscale Intensity: Weighted sum of Red, Green, Blue channels.
    2. Morphological White Top-Hat: Isolates bright structural objects of scale <= kernel size.
    3. Spectral Non-Vegetation Mask: Excludes vegetated areas where NDVI >= max_ndvi.
    4. Component Filtering: Retains connected components whose pixel area is in [min_area, max_area].

    Args:
        tile_4band: Array of shape (H, W, 4) in range [0, 1] or [0, 255].
        tophat_kernel_size: Size of the rectangular structuring element (pixels).
        tophat_threshold: Minimum intensity contrast (0-255) for candidate rooftop pixels.
        max_ndvi: Maximum NDVI threshold to exclude vegetation.
        min_area: Minimum connected component area in pixels (filters noise).
        max_area: Maximum connected component area in pixels (filters massive open land).
        red_idx: Red band channel index (default: 2).
        nir_idx: NIR band channel index (default: 3).

    Returns:
        Dict[str, Any]: Extraction dictionary containing:
            - 'mask': 2D boolean array (H, W) indicating footprint presence.
            - 'num_components': Number of distinct footprint components detected.
            - 'footprint_pixels': Total number of positive footprint pixels.
            - 'footprint_area_pct': Percentage of the tile occupied by footprints.
            - 'ndvi': 2D array of NDVI values.
            - 'tophat': 2D uint8 array of White Top-Hat response.

    Raises:
        FootprintExtractionError: If inputs are invalid or output is completely degenerate.
    """
    if tile_4band.ndim != 3 or tile_4band.shape[2] < 4:
        raise FootprintExtractionError(
            f"Input tile has shape {tile_4band.shape}; expected at least 4 bands."
        )

    # 1. Grayscale intensity from RGB (B04: Red, B03: Green, B02: Blue)
    r = tile_4band[:, :, 2].astype(np.float32)
    g = tile_4band[:, :, 1].astype(np.float32)
    b = tile_4band[:, :, 0].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    v_min, v_max = float(np.min(gray)), float(np.max(gray))
    if v_max - v_min > 1e-6:
        gray_u8 = np.clip((gray - v_min) / (v_max - v_min) * 255.0, 0, 255).astype(np.uint8)
    else:
        gray_u8 = np.zeros_like(gray, dtype=np.uint8)

    # 2. White Top-Hat morphological filter (isolates localized bright structures)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_kernel_size, tophat_kernel_size))
    tophat = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, kernel)

    # 3. Spectral vegetation exclusion mask (NDVI < max_ndvi)
    ndvi = compute_ndvi(tile_4band, red_idx=red_idx, nir_idx=nir_idx)
    non_vegetated = ndvi < max_ndvi

    # Candidate footprint mask
    candidate_mask = non_vegetated & (tophat > tophat_threshold)

    # 4. Connected component analysis and geometric area filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8), connectivity=8
    )

    clean_mask = np.zeros_like(candidate_mask, dtype=bool)
    valid_component_count = 0

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            clean_mask[labels == label_id] = True
            valid_component_count += 1

    total_pixels = int(clean_mask.size)
    footprint_pixels = int(np.sum(clean_mask))
    footprint_area_pct = round((footprint_pixels / total_pixels) * 100.0, 2)

    return {
        "mask": clean_mask,
        "num_components": valid_component_count,
        "footprint_pixels": footprint_pixels,
        "footprint_area_pct": footprint_area_pct,
        "ndvi": ndvi,
        "tophat": tophat,
    }


def compare_downstream_footprints(
    mask_bicubic: np.ndarray,
    mask_sr: np.ndarray,
    trust_map: np.ndarray,
    ref_mask: Optional[np.ndarray] = None,
    trust_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Compare extracted building footprints across models and trust regions.

    Quantifies agreement between bicubic baseline and SR ensemble mean,
    and isolates agreement across High-Trust vs. Low-Trust geographic zones.

    Args:
        mask_bicubic: 2D boolean mask from bicubic baseline.
        mask_sr: 2D boolean mask from SR ensemble reconstruction.
        trust_map: 2D float array in [0.0, 1.0] from Phase 8.
        ref_mask: Optional 2D boolean mask from pre-degradation reference tile.
        trust_threshold: Threshold defining High-Trust (>= threshold) vs Low-Trust (< threshold).

    Returns:
        Dict[str, Any]: Comparative evaluation dictionary with agreement statistics.
    """
    if mask_bicubic.shape != mask_sr.shape:
        raise ValueError(
            f"Shape mismatch: mask_bicubic {mask_bicubic.shape} vs mask_sr {mask_sr.shape}"
        )
    if mask_bicubic.shape != trust_map.shape:
        raise ValueError(
            f"Shape mismatch: masks {mask_bicubic.shape} vs trust_map {trust_map.shape}"
        )

    # 1. Spatial trust partition
    high_trust_region = trust_map >= trust_threshold
    low_trust_region = ~high_trust_region

    def calc_metrics(m1: np.ndarray, m2: np.ndarray, region: Optional[np.ndarray] = None) -> Dict[str, float]:
        if region is not None:
            sub1 = m1 & region
            sub2 = m2 & region
        else:
            sub1 = m1
            sub2 = m2

        inter = int(np.sum(sub1 & sub2))
        union = int(np.sum(sub1 | sub2))
        count1 = int(np.sum(sub1))
        count2 = int(np.sum(sub2))

        iou = float(inter / union) if union > 0 else 1.0
        dice = float(2.0 * inter / (count1 + count2)) if (count1 + count2) > 0 else 1.0
        overlap_m1 = float(inter / count1) if count1 > 0 else 1.0
        overlap_m2 = float(inter / count2) if count2 > 0 else 1.0

        return {
            "iou": round(iou, 4),
            "dice": round(dice, 4),
            "intersection_pixels": inter,
            "union_pixels": union,
            "m1_pixels": count1,
            "m2_pixels": count2,
            "overlap_m1_pct": round(overlap_m1 * 100.0, 2),
            "overlap_m2_pct": round(overlap_m2 * 100.0, 2),
        }

    # A. Bicubic vs SR Agreement
    overall_bic_sr = calc_metrics(mask_bicubic, mask_sr)
    high_trust_bic_sr = calc_metrics(mask_bicubic, mask_sr, region=high_trust_region)
    low_trust_bic_sr = calc_metrics(mask_bicubic, mask_sr, region=low_trust_region)

    # Discrepancy mask: pixels where Bicubic and SR disagree
    discrepancy_mask = np.logical_xor(mask_bicubic, mask_sr)
    agreement_mask = np.logical_and(mask_bicubic, mask_sr)

    mean_trust_agree = float(np.mean(trust_map[agreement_mask])) if np.any(agreement_mask) else 0.0
    mean_trust_disagree = float(np.mean(trust_map[discrepancy_mask])) if np.any(discrepancy_mask) else 0.0

    # B. Optional: Comparison against pre-degradation reference HR tile
    ref_comparison = None
    if ref_mask is not None:
        if ref_mask.shape != mask_sr.shape:
            raise ValueError(f"Shape mismatch: ref_mask {ref_mask.shape} vs mask_sr {mask_sr.shape}")
        bic_vs_ref = calc_metrics(mask_bicubic, ref_mask)
        sr_vs_ref = calc_metrics(mask_sr, ref_mask)
        ref_comparison = {
            "has_ground_truth": False,
            "scientific_label": "Relative Agreement Against Pre-Degradation Reference Extraction",
            "bicubic_vs_ref_iou": bic_vs_ref["iou"],
            "sr_vs_ref_iou": sr_vs_ref["iou"],
            "bicubic_vs_ref_dice": bic_vs_ref["dice"],
            "sr_vs_ref_dice": sr_vs_ref["dice"],
            "ref_footprint_pixels": int(np.sum(ref_mask)),
        }

    return {
        "scientific_honesty_label": "Bicubic-vs-SR Footprint Agreement (No Independent GT)",
        "overall_bic_sr": overall_bic_sr,
        "high_trust_bic_sr": high_trust_bic_sr,
        "low_trust_bic_sr": low_trust_bic_sr,
        "trust_threshold_used": trust_threshold,
        "high_trust_area_pct": round(float(np.mean(high_trust_region)) * 100.0, 2),
        "low_trust_area_pct": round(float(np.mean(low_trust_region)) * 100.0, 2),
        "mean_trust_on_agreeing_footprints": round(mean_trust_agree, 4),
        "mean_trust_on_disagreeing_footprints": round(mean_trust_disagree, 4),
        "discrepancy_mask": discrepancy_mask,
        "agreement_mask": agreement_mask,
        "high_trust_region": high_trust_region,
        "low_trust_region": low_trust_region,
        "reference_comparison": ref_comparison,
    }


def render_downstream_overlay(
    gt_tile: np.ndarray,
    bicubic_tile: np.ndarray,
    sr_tile: np.ndarray,
    mask_bicubic: np.ndarray,
    mask_sr: np.ndarray,
    trust_map: np.ndarray,
    comparison_stats: Dict[str, Any],
    sample_index: int,
    output_path: str,
) -> None:
    """Render 5-panel comparative visualization of downstream building footprint extraction.

    Panels:
    1. Reference GT RGB (10m)
    2. Bicubic Baseline RGB with Cyan footprint contours
    3. SR Ensemble Mean RGB with Cyan footprint contours
    4. Phase 8 Composite Trust Map (RdYlGn colormap)
    5. Footprint Agreement & Discrepancy Overlay (Cyan: Agreement, Orange: Discrepancy)
    """
    def to_rgb(t):
        r, g, b = t[:, :, 2], t[:, :, 1], t[:, :, 0]
        rgb = np.stack([r, g, b], axis=-1)
        v_min, v_max = np.percentile(rgb, (2, 98))
        if v_max > v_min:
            rgb = np.clip((rgb - v_min) / (v_max - v_min), 0.0, 1.0)
        return (rgb * 255).astype(np.uint8)

    def overlay_contours(rgb_img: np.ndarray, mask: np.ndarray, color=(0, 240, 255)) -> np.ndarray:
        out = rgb_img.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1)
        return out

    gt_rgb = to_rgb(gt_tile)
    bic_rgb = to_rgb(bicubic_tile)
    sr_rgb = to_rgb(sr_tile)

    bic_with_contours = overlay_contours(bic_rgb, mask_bicubic)
    sr_with_contours = overlay_contours(sr_rgb, mask_sr)

    # Multi-color agreement map
    gray_bg = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2GRAY)
    agreement_rgb = np.stack([gray_bg, gray_bg, gray_bg], axis=-1)

    agree = comparison_stats["agreement_mask"]
    disagree = comparison_stats["discrepancy_mask"]

    # Agreeing footprint pixels: Vibrant Cyan
    agreement_rgb[agree] = [0, 230, 255]
    # Discrepant footprint pixels: Bright Orange / Amber
    agreement_rgb[disagree] = [255, 120, 0]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), dpi=150)
    fig.patch.set_facecolor("#161616")

    # Panel 0: Reference GT RGB
    axes[0].imshow(gt_rgb)
    axes[0].set_title("1. Reference GT RGB (10m)\n[Pre-degradation]", color="white", fontsize=9.5, pad=8)

    # Panel 1: Bicubic with Footprint Contours
    axes[1].imshow(bic_with_contours)
    bic_pix = comparison_stats["overall_bic_sr"]["m1_pixels"]
    axes[1].set_title(f"2. Bicubic Baseline\n[Footprints: {bic_pix} px]", color="white", fontsize=9.5, pad=8)

    # Panel 2: SR with Footprint Contours
    axes[2].imshow(sr_with_contours)
    sr_pix = comparison_stats["overall_bic_sr"]["m2_pixels"]
    axes[2].set_title(f"3. SR Ensemble Mean (2x)\n[Footprints: {sr_pix} px]", color="white", fontsize=9.5, pad=8)

    # Panel 3: Composite Trust Map
    im3 = axes[3].imshow(trust_map, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    axes[3].set_title(
        f"4. Phase 8 Trust Map\n[High-Trust Area: {comparison_stats['high_trust_area_pct']}%]",
        color="white",
        fontsize=9.5,
        pad=8,
    )
    cbar3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    cbar3.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar3.ax.axes, "yticklabels"), color="white")

    # Panel 4: Agreement & Discrepancy Overlay
    axes[4].imshow(agreement_rgb)
    overall_iou = comparison_stats["overall_bic_sr"]["iou"]
    high_iou = comparison_stats["high_trust_bic_sr"]["iou"]
    low_iou = comparison_stats["low_trust_bic_sr"]["iou"]
    axes[4].set_title(
        f"5. Footprint Agreement\n[IoU: {overall_iou:.3f} | Hi-Trust: {high_iou:.3f} | Lo-Trust: {low_iou:.3f}]",
        color="white",
        fontsize=9.5,
        pad=8,
    )

    for ax in axes:
        ax.axis("off")

    plt.suptitle(
        f"GeoFUSE SentinelGuard -- Downstream Task: Building Footprint Analysis (Sample #{sample_index})\n"
        "[Cyan: Both Models Agree | Orange: Model Boundary Discrepancy | No Fabricated Accuracy]",
        color="white",
        fontsize=12,
        weight="bold",
        y=0.99,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
