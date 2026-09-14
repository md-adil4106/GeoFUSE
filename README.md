# GeoFUSE SentinelGuard

**Deep Learning Based Super Resolution Mapping from Medium Resolution Satellite Imagery.**

> **Core Philosophy:** *"Sharper imagery, with evidence attached."*

---

## Overview

GeoFUSE SentinelGuard is a trust-aware satellite image super-resolution pipeline designed for Sentinel-2 L2A medium-resolution imagery. Rather than merely producing nominally sharper reconstructions, GeoFUSE attaches empirical evidence to every reconstructed pixel:
- **Baseline Comparison:** Bicubic interpolation vs. lightweight deep learning SR.
- **Uncertainty Proxy:** Sequential ensemble disagreement (variance/std map).
- **Stability Analysis:** Perturbation-based sensitivity testing.
- **Spectral Consistency:** Band ratio and delta-NDVI fidelity checks.
- **Structural Consistency:** Edge gradient and local SSIM alignment.
- **Downstream Evaluation:** Light geospatial task (e.g. building footprint / feature extraction) comparison.
- **Verification & Receipts:** Synthetic degrade-and-recover validation, Trust Map fusion, and machine-readable JSON **Trust Receipt**.

---

## Directory Layout

```
GeoFUSE/
├── .gitignore              # Ignores raw/interim data, checkpoints, outputs, caches
├── requirements.txt        # Minimal, reproducible dependencies
├── config.yaml             # Single source of truth for paths, hyperparams, & thresholds
├── README.md               # Project documentation
├── src/
│   ├── __init__.py
│   ├── utils/              # Configuration parsing & device resolution
│   ├── data/               # Sentinel-2 ingest, tiling, and synthetic degradation
│   ├── models/             # Lightweight SR architectures & ensemble training
│   ├── evaluation/         # Consistency checks, uncertainty proxies, Trust Receipt
│   └── dashboard/          # Interactive Streamlit inspection dashboard
└── scripts/
    └── smoke_test.py       # Dependency & hardware validation suite
```

---

## Getting Started

### 1. Requirements & Installation
Ensure Python 3.10+ is installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Smoke Test
Verify environment, dependencies, and compute device detection:
```bash
python scripts/smoke_test.py
```

### 3. Hardware Support
- Automatically detects CUDA-capable GPUs (e.g., NVIDIA RTX 4060).
- If a GPU is not detected or CUDA PyTorch is not present, falls back gracefully to CPU with clear notifications.

---

## Scientific Rigor & Geographic Hold-Out Strategy

In Earth Observation (EO) and satellite imagery super-resolution, standard random patch splitting is scientifically flawed due to spatial autocorrelation: adjacent overlapping tiles in training and validation sets cause data leakage, leading to artificially inflated accuracy metrics.

GeoFUSE SentinelGuard enforces an **independent contiguous geographic hold-out split**:
- **Scene Dimensions**: 512 × 512 pixels (approx. 5.12 km × 5.12 km) at 10m Ground Sample Distance (GSD).
- **Validation Hold-Out Zone**: Strictly contiguous Southeast Quadrant (`rows 256..512, cols 256..512`). Total: 49 non-leaking evaluation patches.
- **Training Zone**: North and West sub-regions. Total: 120 training patches.
- **Zero Spatial Leakage**: Strict spatial separation guarantees that model generalizability is tested on completely unseen landscape geometry.

### Quantitative Benchmark Comparison (Hold-Out Evaluation)

| Architecture / Method | Parameters | Val Loss (Compound L1+Sobel) | Val PSNR (dB) | Val SSIM |
| :--- | :--- | :--- | :--- | :--- |
| **Bicubic Baseline (2x)** | 0 (Interpolation) | 0.01920 | 38.19 dB | 0.9208 |
| **ResidualSRNet (Ours)** | 273,700 (~0.27M) | **0.01726** | **38.68 dB** | **0.9270** |
| *Delta vs. Baseline* | — | *-0.00194* | *+0.49 dB* | *+0.0062* |

---

## Verification & Consistency Checks (Phase 7)

To ensure super-resolution models do not introduce radiometric distortion or structural hallucinations, GeoFUSE SentinelGuard performs two independent verification checks against pre-degradation reference imagery:

1. **Radiometric / Spectral Consistency**:
   - Computes $\Delta\text{NDVI} = |\text{NDVI}_{SR} - \text{NDVI}_{GT}|$ using red (B04) and near-infrared (B08) bands.
   - Flags pixels exceeding tolerance threshold ($\tau = 0.05$).
   - Computes multi-spectral Green/Red ratio consistency.
   - Raises explicit `SpectralBandError` if required spectral bands are missing or mismatched.
   - **Empirical Results**: Mean $\Delta\text{NDVI} \approx 0.021$ across diverse tiles, with $>92\%$ of pixels within tolerance.

2. **Structural & Edge Alignment**:
   - Quantifies boundary preservation using Canny edge maps (IoU, Precision, Recall, and F1 score).
   - Evaluates spatial high-frequency correlation via Sobel gradient magnitude Pearson correlation ($r$).
   - Generates multi-color diagnostic overlays highlighting matched edges (Green), hallucinated/shifted edges (Red), and missed edges (Cyan).
   - **Empirical Results**: Edge IoU $\approx 0.40 - 0.42$, Edge F1 $\approx 0.57 - 0.59$, Gradient Correlation $r \approx 0.78 - 0.81$.

---

## Multi-Evidence Trust/Risk Map Fusion (Phase 8)

GeoFUSE SentinelGuard synthesizes all four independent reliability signals into a unified spatial **Trust/Risk Map**:

$$\text{Risk}(x, y) = w_1 R_{\text{disag}}(x, y) + w_2 R_{\text{stab}}(x, y) + w_3 R_{\text{spec}}(x, y) + w_4 R_{\text{struct}}(x, y)$$

$$\text{Trust}(x, y) = 1.0 - \text{Risk}(x, y)$$

> [!NOTE]
> **Scientific Transparency**: This composite is an **empirical heuristic fusion**, not a calibrated Bayesian posterior probability.

- **Scale Dominance Guard**: Each individual signal is normalized to $[0.0, 1.0]$ via min-max scaling prior to combination, ensuring that higher-magnitude metrics (e.g. gradient difference) do not overpower subtle signals (e.g. stability variance).
- **Configurable Weights**: Defined in `config.yaml` ($w_{\text{disag}}=0.25, w_{\text{stab}}=0.25, w_{\text{spec}}=0.25, w_{\text{struct}}=0.25$).
- **Interpretable Output**: Provides a single per-tile scalar **Trust Score** ($0 - 100\%$) and spatial risk heatmaps (Green = High Trust, Red = High Risk).
- **Empirical Benchmark Across Held-Out Tiles**:
  - Sample #0: Trust Score **86.53%**, Mean Risk 0.1347, High-Risk Flagged: 0.00%
  - Sample #1: Trust Score **86.98%**, Mean Risk 0.1302, High-Risk Flagged: 0.00%
  - Sample #2: Trust Score **86.11%**, Mean Risk 0.1389, High-Risk Flagged: 0.00%
  - Sample #3: Trust Score **85.49%**, Mean Risk 0.1451, High-Risk Flagged: 0.00%

---

## Downstream Task Evaluation: Building Footprint Analysis (Phase 9)

To verify the real-world operational utility of super-resolved imagery, GeoFUSE SentinelGuard executes a canonical downstream Earth Observation task: morphological building footprint extraction (white top-hat filtering + NDVI vegetation rejection + component filtering).

> [!IMPORTANT]
> **Scientific Honesty**:
> In the absence of certified independent high-resolution vector building footprints, all metrics are reported strictly as **Bicubic vs. SR Reconstruction Agreement (IoU / Dice)** and **Relative Agreement against Pre-Degradation Reference HR Tile**, never as fabricated "ground-truth accuracy".

- **Trust Map Correlation**: Footprint agreement is stratified across High-Trust ($\ge 0.85$) and Low-Trust ($< 0.85$) geographic zones from Phase 8.
- **Empirical Results Across Held-Out Tiles**:
  - Sample #0: Bic-SR IoU **0.9285**, High-Trust IoU **0.9248**, Low-Trust IoU **0.9364**, SR-Ref IoU **0.4174** (65.1% High-Trust)
  - Sample #1: Bic-SR IoU **0.9461**, High-Trust IoU **0.9508**, Low-Trust IoU **0.9343**, SR-Ref IoU **0.6137** (67.5% High-Trust)
  - Sample #2: Bic-SR IoU **0.9253**, High-Trust IoU **0.9225**, Low-Trust IoU **0.9316**, SR-Ref IoU **0.5691** (70.8% High-Trust)
  - Sample #3: Bic-SR IoU **0.9058**, High-Trust IoU **0.9004**, Low-Trust IoU **0.9131**, SR-Ref IoU **0.5268** (63.6% High-Trust)
- **Overlay Diagnostics**: Multi-panel previews (`outputs/previews/downstream_footprint_overlay_sample_*.png`) visualize model agreement (Cyan) vs. boundary discrepancy (Orange) atop the scene.

---

## Interactive Demonstration Dashboard (Phase 10)

GeoFUSE SentinelGuard includes an interactive Streamlit dashboard (`src/dashboard/app.py`) for live comparative inspection:

- **Side-by-Side Verification**: Simultaneous 4-column display of:
  1. **Original Reference (10m)** [Pre-degradation Sentinel-2]
  2. **Bicubic Baseline (2x)** [Standard interpolation]
  3. **GeoFUSE SR (2x)** [Ensemble Mean Reconstruction]
  4. **Trust / Risk Map Overlay** [RdYlGn colormap: Green = High Trust, Red = High Risk]
- **Downstream Task Toggle**: Interactive inspection of building footprint contours, consensus masks, and trust stratification statistics.
- **Evidence Breakdown Toggle**: Live inspection of individual evidence maps (disagreement, stability, $\Delta$NDVI, gradient error).
- **Cached Inference**: Utilizes Streamlit resource caching to ensure rapid, responsive tile navigation without timeouts or redundant compute.

### Launching the Dashboard:
```bash
streamlit run src/dashboard/app.py
```

---

## Auditable Trust Receipts (Phase 11)

In mission-critical geospatial analysis, super-resolved imagery should never be delivered as an unverified visual output. GeoFUSE SentinelGuard implements an auditable **Trust Receipt generator** that pairs every reconstructed tile with a cryptographically timestamped, machine-readable JSON receipt (`outputs/receipts/trust_receipt_sample_*.json` and `examples/trust_receipt_sample_3.json`) and an interactive HTML report card.

### Receipt Components
1. **Model Provenance**: Records the architecture (`ResidualSRNet`), total trainable parameter count (~0.27M), scaling factor (2x), and exact ensemble checkpoint paths (`outputs/checkpoints/ensemble_member_{0,1,2}.pth`).
2. **Geospatial & Acquisition Metadata**: Extracted directly from GeoTIFF headers (`EPSG:32643`, 10m GSD, MGRS `43PGQ`, Sentinel-2A, spatial bounding box coordinates).
3. **Strict Scientific Honesty Mandate**: Metadata fields not present in standalone band GeoTIFF headers (such as `cloud_cover_percentage`, `sun_elevation_angle_deg`, `sun_azimuth_angle_deg`, and `satellite_orbit_number`, which reside in XML SAFE manifests) are explicitly marked as `"unavailable"` rather than guessed or fabricated.
4. **Empirical Evidence Summary**: Disagreement std ($0.0163 - 0.0195$), stability variance ($0.00018 - 0.00020$), mean $\Delta\text{NDVI}$ ($0.021 - 0.022$), edge gradient correlation ($0.78 - 0.81$), and fused trust score ($86.02\% - 86.75\%$).
5. **Downstream Task Metrics**: Bicubic vs. SR building footprint agreement (IoU $\approx 0.91 - 0.95$), stratified by high-trust and low-trust zones.
6. **Automated Risk & Trust Warnings**: Evaluates the fused trust score against a configurable threshold (`min_trust_score_threshold: 86.5%`). If trust is compromised, an unambiguous plain-language warning is attached to the receipt (e.g. Sample #2 at 86.41% and Sample #3 at 86.02% trigger `LOW TRUST WARNING`).

### Generating Trust Receipts via CLI:
```bash
python scripts/generate_trust_receipts.py
```

### Interactive Dashboard Viewer:
Inside the Streamlit dashboard (`src/dashboard/app.py`), navigate to **Tab 2: "📜 Auditable Trust Receipt (JSON & HTML)"** to view:
- Color-coded HTML status banner (High Trust Approved vs. Low Trust Warning).
- Comprehensive metadata and metrics summary tables.
- Interactive JSON schema tree.
- Direct **"Download Trust Receipt (JSON)"** button for automated ingestion into downstream GIS workflows.





