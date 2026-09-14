"""GeoFUSE SentinelGuard -- Interactive Streamlit Demonstration Dashboard.

Demonstrates trust-aware satellite image super-resolution on real Sentinel-2 L2A imagery:
1. Side-by-side view: Reference 10m, Bicubic baseline (2x), GeoFUSE SR (2x), and Trust/Risk Map.
2. Comprehensive multi-criteria evidence breakdown (disagreement, stability, NDVI, edge gradient).
3. Downstream task evaluation toggle: Building footprint extraction & trust stratification.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Project root setup
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.data.degrade import bicubic_upsample, synthesize_pseudo_lr
from src.data.tiling import extract_tiles, load_sentinel2_stack
from src.evaluation.downstream_eval import (
    compare_downstream_footprints,
    extract_building_footprints,
)
from src.evaluation.edge_check import (
    compute_edge_consistency,
    compute_gradient_magnitude,
)
from src.evaluation.fusion import fuse_trust_risk_maps
from src.evaluation.spectral_check import compute_spectral_consistency
from src.evaluation.stability import compute_stability_map
from src.models.ensemble import load_ensemble_members, predict_ensemble
from src.utils.config import get_device, get_project_root, load_config


# -----------------------------------------------------------------------------
# Configuration & Resource Caching (Prevents live inference timeouts)
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading ensemble checkpoints onto compute device...")
def load_cached_models():
    """Load the 3 trained ensemble members once and cache in memory."""
    root = get_project_root()
    config = load_config()
    device = get_device(config)
    ckpt_dir = root / config.get("paths", {}).get("outputs_dir", "outputs") / "checkpoints"
    ckpt_paths = [
        ckpt_dir / f"ensemble_member_{i}.pth" for i in range(3)
    ]
    models = load_ensemble_members(ckpt_paths, config=config, device=device)
    return models, config, device


@st.cache_data(show_spinner="Loading Sentinel-2 reference scene...")
def load_cached_scene():
    """Load the raw Sentinel-2 4-band scene and pre-slice candidate tiles."""
    root = get_project_root()
    config = load_config()
    raw_dir = root / config.get("paths", {}).get("raw_data_dir", "data/raw")
    stack, meta = load_sentinel2_stack(raw_dir)
    tiles = extract_tiles(stack, patch_size=128, stride=96)
    return stack, meta, tiles


@st.cache_data(show_spinner="Computing super-resolution and evidence verification...")
def run_cached_pipeline(tile_idx: int) -> Dict[str, Any]:
    """Execute complete inference, verification, and fusion pipeline for a tile."""
    models, config, device = load_cached_models()
    _, _, tiles = load_cached_scene()

    hr_tile = tiles[tile_idx]["data"]
    lr_tile = synthesize_pseudo_lr(
        hr_tile,
        downsample_factor=2,
        blur_kernel_size=3,
        noise_std=0.01,
        seed=1000 + tile_idx,
    )

    # 1. Baseline & SR reconstruction
    bicubic_tile = bicubic_upsample(lr_tile, scale_factor=2)
    sr_tile, disagreement_map, _ = predict_ensemble(models, lr_tile, device=device)

    # 2. Perturbation stability
    pert_cfg = config.get("verification", {}).get("perturbation_test", {})
    noise_levels = pert_cfg.get("noise_levels", [0.01, 0.02, 0.05])
    jitter_std = float(pert_cfg.get("brightness_jitter_std", 0.02))
    stability_map, _, _ = compute_stability_map(
        models=models,
        lr_tile=lr_tile,
        noise_levels=noise_levels,
        brightness_jitter_std=jitter_std,
        num_trials=2,
        device=device,
    )

    # 3. Spectral check (Delta-NDVI)
    bands = config.get("preprocessing", {}).get("bands", ["B02", "B03", "B04", "B08"])
    red_idx = bands.index("B04")
    nir_idx = bands.index("B08")
    spectral_metrics = compute_spectral_consistency(
        gt_tile=hr_tile,
        sr_tile=sr_tile,
        red_idx=red_idx,
        nir_idx=nir_idx,
    )

    # 4. Structural gradient check
    grad_gt = compute_gradient_magnitude(hr_tile)
    grad_sr = compute_gradient_magnitude(sr_tile)
    structural_diff = np.abs(grad_sr - grad_gt)
    edge_metrics = compute_edge_consistency(hr_tile, sr_tile)

    # 5. Phase 8 Evidence Fusion
    fusion_cfg = config.get("verification", {}).get("evidence_fusion", {})
    weights = fusion_cfg.get("weights", {
        "disagreement": 0.25,
        "stability": 0.25,
        "spectral": 0.25,
        "structural": 0.25,
    })
    fusion_result = fuse_trust_risk_maps(
        disagreement_map=disagreement_map,
        stability_map=stability_map,
        delta_ndvi_map=spectral_metrics["delta_ndvi"],
        structural_diff_map=structural_diff,
        weights=weights,
    )

    # 6. Downstream building footprint extraction
    down_cfg = config.get("downstream", {})
    morph_cfg = down_cfg.get("morphology", {})
    foot_bic = extract_building_footprints(
        bicubic_tile,
        tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
        tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
        max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
        min_area=int(morph_cfg.get("min_area", 4)),
        max_area=int(morph_cfg.get("max_area", 600)),
        red_idx=red_idx,
        nir_idx=nir_idx,
    )
    foot_sr = extract_building_footprints(
        sr_tile,
        tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
        tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
        max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
        min_area=int(morph_cfg.get("min_area", 4)),
        max_area=int(morph_cfg.get("max_area", 600)),
        red_idx=red_idx,
        nir_idx=nir_idx,
    )
    foot_ref = extract_building_footprints(
        hr_tile,
        tophat_kernel_size=int(morph_cfg.get("tophat_kernel_size", 7)),
        tophat_threshold=int(morph_cfg.get("tophat_threshold", 25)),
        max_ndvi=float(morph_cfg.get("max_ndvi", 0.20)),
        min_area=int(morph_cfg.get("min_area", 4)),
        max_area=int(morph_cfg.get("max_area", 600)),
        red_idx=red_idx,
        nir_idx=nir_idx,
    )
    downstream_comp = compare_downstream_footprints(
        mask_bicubic=foot_bic["mask"],
        mask_sr=foot_sr["mask"],
        trust_map=fusion_result["trust_map"],
        ref_mask=foot_ref["mask"],
        trust_threshold=float(down_cfg.get("trust_partition_threshold", 0.85)),
    )

    return {
        "hr_tile": hr_tile,
        "lr_tile": lr_tile,
        "bicubic_tile": bicubic_tile,
        "sr_tile": sr_tile,
        "disagreement_map": disagreement_map,
        "stability_map": stability_map,
        "spectral_metrics": spectral_metrics,
        "edge_metrics": edge_metrics,
        "structural_diff": structural_diff,
        "fusion_result": fusion_result,
        "foot_bic": foot_bic,
        "foot_sr": foot_sr,
        "foot_ref": foot_ref,
        "downstream_comp": downstream_comp,
    }


def to_display_rgb(tile: np.ndarray, false_color: bool = False) -> np.ndarray:
    """Convert 4-band Sentinel-2 tile to an 8-bit RGB image with percentile stretch."""
    if false_color and tile.shape[2] >= 4:
        # False-Color Infrared: NIR (B08), Red (B04), Green (B03)
        ch_r = tile[:, :, 3]  # NIR
        ch_g = tile[:, :, 2]  # Red
        ch_b = tile[:, :, 1]  # Green
    else:
        # True-Color Natural RGB: Red (B04), Green (B03), Blue (B02)
        ch_r = tile[:, :, 2]  # Red
        ch_g = tile[:, :, 1]  # Green
        ch_b = tile[:, :, 0]  # Blue

    rgb = np.stack([ch_r, ch_g, ch_b], axis=-1).astype(np.float32)
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="GeoFUSE SentinelGuard",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header & Banner
    st.title("🛰️ GeoFUSE SentinelGuard")
    st.markdown(
        "**Trust-Aware Super-Resolution Mapping from Medium-Resolution Sentinel-2 Imagery**  \n"
        "*" "Sharper imagery, with evidence attached." "*"
    )

    st.info(
        "**Scientific Transparency Mandate**: Reconstructions are accompanied by multi-criteria empirical evidence "
        "(ensemble disagreement, perturbation stability, spectral NDVI fidelity, structural gradient checks). "
        "The Trust/Risk Map is an **empirical heuristic proxy**, not a calibrated Bayesian posterior probability. "
        "Downstream metrics report **Bicubic-vs-SR Agreement** in the absence of independent vector ground truth."
    )

    # 1. Sidebar Controls
    st.sidebar.header("🕹️ Controls & Settings")
    _, _, tiles = load_cached_scene()
    num_tiles = len(tiles)

    tile_options = {
        0: "Tile #0 -- Central Settlement Cluster",
        num_tiles // 3: f"Tile #{num_tiles // 3} -- Mixed Agricultural & Roads",
        (2 * num_tiles) // 3: f"Tile #{(2 * num_tiles) // 3} -- Rural Landscape",
        num_tiles - 1: f"Tile #{num_tiles - 1} -- Held-Out Southeast Quadrant",
    }
    selected_idx = st.sidebar.selectbox(
        "Select Sentinel-2 Scene Tile:",
        options=list(tile_options.keys()),
        format_func=lambda x: tile_options.get(x, f"Tile #{x}"),
    )

    color_mode = st.sidebar.radio(
        "Band Visualization Mode:",
        options=["Natural RGB (B04-B03-B02)", "False-Color Infrared (B08-B04-B03)"],
    )
    is_false_color = "False-Color" in color_mode

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Evidence Inspection Toggles")
    show_evidence = st.sidebar.checkbox("Show Detailed Evidence Breakdown (Phase 5-7)", value=False)
    show_downstream = st.sidebar.checkbox("Show Downstream Building Footprint Analysis (Phase 9)", value=True)

    # 2. Run Pipeline (Cached)
    data = run_cached_pipeline(selected_idx)

    hr_tile = data["hr_tile"]
    bicubic_tile = data["bicubic_tile"]
    sr_tile = data["sr_tile"]
    fusion_result = data["fusion_result"]
    trust_map = fusion_result["trust_map"]

    # Prepare Display Images
    hr_rgb = to_display_rgb(hr_tile, false_color=is_false_color)
    bic_rgb = to_display_rgb(bicubic_tile, false_color=is_false_color)
    sr_rgb = to_display_rgb(sr_tile, false_color=is_false_color)

    # Semi-transparent Trust/Risk overlay
    cmap = plt.get_cmap("RdYlGn")
    trust_colored = (cmap(trust_map)[:, :, :3] * 255).astype(np.uint8)
    trust_overlay = (0.55 * sr_rgb + 0.45 * trust_colored).astype(np.uint8)

    # 3. Main Side-by-Side View (4 Columns as requested by prompt)
    st.markdown("### 🖼️ Side-by-Side Super-Resolution & Trust Verification")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("1. Original Reference (10m)")
        st.image(hr_rgb, caption="Pre-degradation Sentinel-2", use_container_width=True)
        st.caption("Ground Sample Distance: 10m")

    with col2:
        st.subheader("2. Bicubic Baseline (2x)")
        st.image(bic_rgb, caption="Standard Interpolation", use_container_width=True)
        st.caption("Soft upsample, no structural synthesis")

    with col3:
        st.subheader("3. GeoFUSE SR (2x)")
        st.image(sr_rgb, caption="Ensemble Mean Reconstruction", use_container_width=True)
        st.caption("Lightweight ResidualSRNet Ensemble (~0.27M params)")

    with col4:
        st.subheader("4. Trust / Risk Map Overlay")
        st.image(trust_overlay, caption="RdYlGn: Green=Trust, Red=Risk", use_container_width=True)
        st.caption(f"Mean Trust Score: **{fusion_result['trust_score_pct']}%**")

    # 4. Summary Reliability Scorecards
    st.markdown("---")
    st.markdown("### 📊 Quantitative Reliability Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trust Score", f"{fusion_result['trust_score_pct']}%", delta=f"{fusion_result['trust_score_pct'] - 50:.1f}%")
    m2.metric("Mean Composite Risk", f"{fusion_result['mean_risk_score']:.4f}")
    m3.metric("Disagreement Std (σ)", f"{fusion_result['component_stats']['disagreement']['raw_mean']:.5f}")
    m4.metric("Mean Delta-NDVI", f"{data['spectral_metrics']['mean_delta_ndvi']:.4f}")
    m5.metric("Edge Grad Corr (r)", f"{data['edge_metrics']['gradient_correlation']:.4f}")

    # 5. Downstream Building Footprint Analysis (Phase 9 Toggle)
    if show_downstream:
        st.markdown("---")
        st.markdown("### 🏢 Downstream Task Evaluation: Building Footprint Extraction (Phase 9)")
        st.caption(
            "Extracts rooftop components using identical morphological top-hat filtering and NDVI vegetation rejection. "
            "Evaluates consistency between Bicubic and SR reconstructions across High-Trust vs. Low-Trust geographic zones."
        )

        foot_bic = data["foot_bic"]
        foot_sr = data["foot_sr"]
        comp = data["downstream_comp"]

        # Contour overlays
        def add_contours(base_rgb, mask, color=(0, 240, 255)):
            out = base_rgb.copy()
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 1)
            return out

        bic_cnt = add_contours(bic_rgb, foot_bic["mask"])
        sr_cnt = add_contours(sr_rgb, foot_sr["mask"])

        # Multi-color agreement map
        gray_bg = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2GRAY)
        agree_rgb = np.stack([gray_bg, gray_bg, gray_bg], axis=-1)
        agree_rgb[comp["agreement_mask"]] = [0, 230, 255]       # Cyan: Matched detections
        agree_rgb[comp["discrepancy_mask"]] = [255, 120, 0]     # Orange: Discrepant detections

        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.markdown(f"**Bicubic Footprints** ({foot_bic['footprint_pixels']} px)")
            st.image(bic_cnt, caption="Cyan Contours: Bicubic Detections", use_container_width=True)

        with d_col2:
            st.markdown(f"**GeoFUSE SR Footprints** ({foot_sr['footprint_pixels']} px)")
            st.image(sr_cnt, caption="Cyan Contours: SR Detections", use_container_width=True)

        with d_col3:
            st.markdown("**Footprint Agreement Map**")
            st.image(agree_rgb, caption="Cyan: Consensus | Orange: Boundary Discrepancy", use_container_width=True)

        # Downstream statistics table
        st.markdown("#### Quantitative Footprint Agreement Breakdown")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Overall Bicubic vs SR IoU", f"{comp['overall_bic_sr']['iou']:.4f}")
        stat_col2.metric("High-Trust Region IoU", f"{comp['high_trust_bic_sr']['iou']:.4f}")
        stat_col3.metric("Low-Trust Region IoU", f"{comp['low_trust_bic_sr']['iou']:.4f}")
        stat_col4.metric("Relative Ref HR IoU", f"{comp['reference_comparison']['sr_vs_ref_iou']:.4f}")

    # 6. Detailed Multi-Criteria Evidence Breakdown (Phase 5-7 Toggle)
    if show_evidence:
        st.markdown("---")
        st.markdown("### 🔬 Multi-Source Evidence Signal Breakdown (Phases 5 -- 7)")
        st.caption("Each signal is min-max normalized to [0, 1] to prevent scale dominance prior to weighted fusion.")

        norm_sig = fusion_result["normalized_signals"]
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)

        with e_col1:
            st.markdown("**Disagreement Proxy (Phase 5)**")
            fig1, ax1 = plt.subplots(figsize=(4, 3.5), dpi=100)
            im1 = ax1.imshow(norm_sig["disagreement"], cmap="magma", vmin=0, vmax=1)
            ax1.axis("off")
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

        with e_col2:
            st.markdown("**Perturbation Stability (Phase 6)**")
            fig2, ax2 = plt.subplots(figsize=(4, 3.5), dpi=100)
            im2 = ax2.imshow(norm_sig["stability"], cmap="inferno", vmin=0, vmax=1)
            ax2.axis("off")
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        with e_col3:
            st.markdown("**Spectral Delta-NDVI (Phase 7)**")
            fig3, ax3 = plt.subplots(figsize=(4, 3.5), dpi=100)
            im3 = ax3.imshow(norm_sig["spectral"], cmap="cividis", vmin=0, vmax=1)
            ax3.axis("off")
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

        with e_col4:
            st.markdown("**Structural Gradient Error (Phase 7)**")
            fig4, ax4 = plt.subplots(figsize=(4, 3.5), dpi=100)
            im4 = ax4.imshow(norm_sig["structural"], cmap="plasma", vmin=0, vmax=1)
            ax4.axis("off")
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)


if __name__ == "__main__":
    main()
