# GeoFUSE SentinelGuard: One-Page Project Summary

> **Core Philosophy:** *"Sharper imagery, with evidence attached."*

**GeoFUSE SentinelGuard** is an evidence-backed satellite image super-resolution system for Sentinel-2 L2A medium-resolution imagery. Rather than presenting super-resolved pixels as unverified visual enhancements, GeoFUSE attaches empirical, multi-criteria reliability evidence to every reconstructed tile: ensemble epistemic uncertainty, input-perturbation stability, physical spectral consistency ($\Delta\text{NDVI}$), and structural gradient alignment.

---

## 1. The Problem

- **The Resolution Gap**: Sentinel-2 provides global, 5-day cadence multispectral coverage, but its 10m Ground Sample Distance (GSD) limits critical urban planning, precision agriculture, and emergency disaster mapping.
- **The Black-Box Hallucination Hazard**: Deep learning super-resolution models frequently synthesize plausible-looking high-frequency textures that do not exist in reality, altering vegetation indices and distorting boundary geometry.
- **Missing Reliability Signals**: Downstream decision-makers currently receive super-resolved imagery without confidence metrics or indications of where the neural network is uncertain.

---

## 2. The GeoFUSE Approach

1. **RTX-4060 Friendly Architecture**: Designed `ResidualSRNet` with 4 residual blocks, 48 channel features, and pixel-shuffle 2x upsampling (**273,700 parameters**, ~0.27M params), optimized for memory efficiency.
2. **Scientific Geographic Hold-Out**: Enforced a leak-free spatial split (Southeast Quadrant `rows 256..512, cols 256..512` reserved as 49 non-leaking validation tiles vs. 120 training tiles) to prevent spatial autocorrelation leakage.
3. **Compound Training Loss**: Combined L1 pixel loss with a light Sobel gradient loss ($w=0.1$) trained across 15 epochs with cosine annealing and mixed precision (`torch.amp`).
4. **Epistemic Disagreement Proxy**: Trained 3 identical ensemble members with distinct random seeds (`42, 101, 2024`) to compute mean reconstruction and per-pixel standard deviation ($\sigma$).
5. **Input-Perturbation Stability**: Tested model sensitivity under additive Gaussian noise ($\sigma \in [0.01, 0.02, 0.05]$) and brightness jitter to quantify output variance ($\text{Var}$).
6. **Physical Consistency Checks**: Verified radiometric vegetation fidelity ($\Delta\text{NDVI}$, $\tau=0.05$) and structural edge preservation (Sobel gradient Pearson correlation $r$).
7. **Scale-Normalized Trust Fusion**: Normalized each signal to $[0.0, 1.0]$ via min-max scaling to prevent scale dominance, fusing them into a unified spatial Trust/Risk Map and composite Trust Score ($0 - 100\%$).
8. **Downstream Task Evaluation**: Executed identical morphological building footprint extraction on Bicubic baseline vs. GeoFUSE SR, stratifying agreement across High-Trust and Low-Trust regions.
9. **Auditable Trust Receipts**: Generated cryptographically timestamped, machine-readable JSON receipts (`outputs/receipts/`) and HTML report cards auditing provenance, acquisition metadata, and plain-language risk advisories.
10. **Demo Hardening**: Precomputed an offline demo cache (`outputs/demo_cache/`) enabling instantaneous sub-10ms tile loading with zero live model inference during presentations.

---

## 3. Results as Actually Measured (Zero Fabrication)

All values below are pulled directly from verified training logs (`outputs/checkpoints/training_log*.csv`) and receipts (`outputs/receipts/*.json`):

### Model Validation Benchmark (15 Epochs on Geographic Hold-Out)
| Architecture / Method | Parameters | Final Val Loss | Val PSNR (dB) | Val SSIM |
| :--- | :--- | :--- | :--- | :--- |
| **Bicubic Baseline (2x)** | 0 (Interpolation) | 0.01920 | 38.19 dB | 0.9208 |
| **ResidualSRNet (Single, Seed 42)** | 273,700 (~0.27M) | **0.01726** | **38.68 dB** | **0.9270** |
| **Ensemble Member 0 (Seed 42)** | 273,700 (~0.27M) | 0.01728 | 38.67 dB | 0.9272 |
| **Ensemble Member 1 (Seed 101)** | 273,700 (~0.27M) | 0.01733 | 38.66 dB | 0.9265 |
| **Ensemble Member 2 (Seed 2024)** | 273,700 (~0.27M) | 0.01737 | 38.62 dB | 0.9261 |
| **Ensemble Mean** | 821,100 (3×0.27M) | **0.01733** | **38.65 dB** | **0.9266** |
| *Advantage vs. Baseline* | — | *-0.00187* | *+0.46 dB* | *+0.0058* |

### Multi-Evidence Trust & Assessment Breakdown Across Held-Out Tiles
Configured Threshold: `min_trust_score_threshold: 86.50%`

| Tile ID | Description | Disag Std ($\sigma$) | Stability ($\text{Var}$) | Mean $\Delta\text{NDVI}$ | Edge Corr ($r$) | Trust Score | Operational Assessment Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **#0** | Central Settlement Cluster | 0.00155 | 0.000506 | 0.02145 | 0.7867 | **86.75%** | `HIGH_TRUST_APPROVED` |
| **#1** | Agricultural & Rural Roads | 0.00162 | 0.000498 | 0.02082 | 0.8037 | **86.70%** | `HIGH_TRUST_APPROVED` |
| **#2** | Rural River Corridor | 0.00178 | 0.000531 | 0.02231 | 0.7935 | **86.41%** | `LOW_TRUST_WARNING` *(Flagged)* |
| **#3** | Complex Terrain Transition | 0.00189 | 0.000542 | 0.02153 | 0.7840 | **86.02%** | `LOW_TRUST_WARNING` *(Flagged)* |

### Downstream Building Footprint Agreement
- **Bicubic vs. SR Reconstruction IoU**: Ranges from **0.9058** (Tile #3) to **0.9461** (Tile #1).
- **Trust Stratification**: High-Trust zone agreement reaches **0.9508** IoU (Tile #1); Relative Reference HR IoU ranges from **0.4174** to **0.6137**.

---

## 4. What This System Does NOT Claim

In keeping with our scientific honesty mandate, we explicitly state the boundaries of what this system does **not** claim:

1. **NOT Proof of True Optical High-Resolution Recovery**:  
   All quantitative PSNR and SSIM benchmarks were evaluated in a **synthetic degrade-and-recover framework** (2x bicubic downsampling + optical PSF blur + sensor noise against 10m reference imagery). This demonstrates algorithm fidelity, but does **not** constitute mathematical proof of true high-resolution physical signal recovery in real-world unconstrained deployments.
2. **NOT a Calibrated Bayesian Posterior Probability**:  
   The composite Trust Score ($0 - 100\%$) is an **empirical heuristic fusion** of normalized evidence proxies. It does **not** represent a calibrated Bayesian probability, credible interval, or conformal prediction guarantee.
3. **NOT Ground-Truth Downstream Detection Accuracy**:  
   Because certified independent high-resolution vector building footprints were not available for the AOI, all downstream evaluation metrics are reported strictly as **Bicubic vs. SR Reconstruction Agreement** and **Relative Agreement against Reference HR Extraction**, never as fabricated "detection accuracy."
4. **NO Fabricated GeoTIFF Acquisition Metadata**:  
   Standalone band GeoTIFFs downloaded from open data catalogs do not include solar angles or cloud cover percentages (which reside in external XML SAFE manifests). These fields are explicitly marked as `"unavailable"` in all Trust Receipts rather than guessed.

---

## 5. Key Presentation Screenshots (`outputs/presentation/`)

| File Name | Description | Key Insight Shown |
| :--- | :--- | :--- |
| [`01_side_by_side_super_resolution.png`](outputs/presentation/01_side_by_side_super_resolution.png) | 4-Column Side-by-Side Verification | Compares Reference, Bicubic baseline, GeoFUSE SR, and Trust/Risk Map with scorecards. |
| [`02_trust_guard_live_warning.png`](outputs/presentation/02_trust_guard_live_warning.png) | Live Trust Guard Demonstration | Highlights Tile #16 flagged below 86.5% with an active risk advisory banner. |
| [`03_evidence_breakdown_signals.png`](outputs/presentation/03_evidence_breakdown_signals.png) | Multi-Source Evidence Breakdown | Visualizes normalized Disagreement, Stability, $\Delta$NDVI, and Gradient error maps. |
| [`04_downstream_footprint_analysis.png`](outputs/presentation/04_downstream_footprint_analysis.png) | Downstream Building Footprints | Illustrates Bicubic contours, SR contours, and Footprint consensus vs. boundary discrepancy. |
| [`05_auditable_trust_receipt_report.png`](outputs/presentation/05_auditable_trust_receipt_report.png) | Auditable Trust Receipt Report | Showcases the structured HTML report card, provenance metadata, and JSON schema explorer. |
