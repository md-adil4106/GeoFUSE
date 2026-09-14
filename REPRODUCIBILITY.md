# GeoFUSE SentinelGuard — Reproducibility & Evaluation Guide

> **Core Philosophy:** *"Sharper imagery, with evidence attached."*

This document provides the complete, self-contained, and audited instructions to reproduce every stage of the **GeoFUSE SentinelGuard** project from a clean repository checkout.

---

## 1. Quick-Start Reproduction (Single Command)

To reproduce the entire pipeline end-to-end (data verification, inference, stability testing, spectral and structural consistency checks, trust fusion, downstream task evaluation, trust receipt generation, and test suite execution):

```bash
# Clone and enter directory
git clone https://github.com/md-adil4106/GeoFUSE.git
cd GeoFUSE

# Install dependencies
pip install -r requirements.txt

# Run end-to-end reproducibility pipeline (reuses existing checkpoints)
python scripts/reproduce_all.py --skip-training
```

To force retraining of all 3 ensemble members from scratch before running inference:
```bash
python scripts/reproduce_all.py --force-retrain
```

To launch the interactive demonstration dashboard immediately after reproduction:
```bash
python scripts/reproduce_all.py --skip-training --launch-dashboard
```

---

## 2. Step-by-Step Reproduction Commands

All parameters, directory paths, and hyperparameters are loaded exclusively from `config.yaml`. No paths or credentials are hardcoded.

### Step 1: Environment & Dependency Smoke Test
Validates Python version, core dependencies (`torch`, `rasterio`, `cv2`, `streamlit`), and compute hardware (CUDA GPU or CPU fallback):
```bash
python scripts/smoke_test.py
```

### Step 2: Data Ingestion & Tile Inspection
Inspects raw Sentinel-2 L2A band GeoTIFFs (`B02`, `B03`, `B04`, `B08`) in `data/raw/` and generates spatial RGB previews in `outputs/previews/rgb_preview.png`:
```bash
python scripts/inspect_sentinel2.py
```

### Step 3: Degrade-and-Recover Baseline Generation (Phase 2)
Generates synthetic pseudo-LR tiles via optical PSF Gaussian blur, 2x bicubic downsampling, and radiometric sensor noise. Computes the 2x bicubic baseline and saves preview triplets to `outputs/previews/`:
```bash
python scripts/generate_baseline_triplets.py
```

### Step 4: Training 3 Ensemble Members Sequentially (Phases 4 & 5)
Trains 3 instances of `ResidualSRNet` (~0.27M parameters each) sequentially with different seeds (`[42, 101, 2024]` from `config.yaml`) on the strict geographic hold-out split:
```bash
python scripts/train.py
```
*Outputs saved:*
- Checkpoints: `outputs/checkpoints/ensemble_member_0.pth`, `ensemble_member_1.pth`, `ensemble_member_2.pth`
- Loss logs: `outputs/checkpoints/training_log_member_0.csv`, `member_1.csv`, `member_2.csv`

### Step 5: Multi-Model Ensemble Inference & Disagreement Mapping (Phase 5)
Executes ensemble forward passes to compute the reconstructed Mean Super-Resolution tile and the per-pixel Standard Deviation Disagreement Map ($\sigma$):
```bash
python scripts/run_ensemble_inference.py
```
*Outputs saved:* `outputs/previews/ensemble_preview_sample_*.png`

### Step 6: Input-Perturbation Stability Testing (Phase 6)
Perturbs input tiles across Gaussian noise levels ($\sigma \in [0.01, 0.02, 0.05]$) and brightness jitter ($\sigma_j = 0.02$) to compute output variance as a stability map:
```bash
python scripts/test_stability.py
```
*Outputs saved:* `outputs/previews/stability_preview_sample_*.png`

### Step 7: Radiometric Spectral & Structural Consistency (Phase 7)
Computes vegetation radiometric consistency ($\Delta\text{NDVI}$) and structural boundary preservation (Sobel gradient magnitude correlation $r$ and Canny edge alignment):
```bash
python scripts/test_consistency_checks.py
```
*Outputs saved:* `outputs/previews/spectral_ndvi_overlay_sample_*.png` and `outputs/previews/edge_structural_overlay_sample_*.png`

### Step 8: Multi-Evidence Trust/Risk Map Fusion (Phase 8)
Fuses normalized evidence signals into a per-tile Trust Map and composite scalar Trust Score ($0 - 100\%$):
```bash
python scripts/generate_trust_risk_maps.py
```
*Outputs saved:* `outputs/previews/trust_risk_overlay_sample_*.png`

### Step 9: Downstream Task Evaluation (Building Footprints) (Phase 9)
Executes morphological building footprint extraction identically on Bicubic baseline and GeoFUSE SR output, quantifying agreement stratified across high-trust and low-trust zones:
```bash
python scripts/evaluate_downstream_task.py
```
*Outputs saved:* `outputs/previews/downstream_footprint_overlay_sample_*.png`

### Step 10: Auditable Trust Receipt Generation (Phase 11)
Produces machine-readable JSON receipts and HTML reports auditing provenance, acquisition metadata, empirical metrics, and automated risk warnings:
```bash
python scripts/generate_trust_receipts.py
```
*Outputs saved:* `outputs/receipts/trust_receipt_sample_*.json` and tracked repository example `examples/trust_receipt_sample_3.json`

### Step 11: Launch Interactive Demonstration Dashboard (Phase 10)
Launches the Streamlit inspection UI:
```bash
streamlit run src/dashboard/app.py
```

### Step 12: Automated PyTest Regression Suite
Runs the full 57-test validation suite:
```bash
pytest tests/ -v
```

---

## 3. Geographic Hold-Out Partitioning Strategy

In Earth Observation (EO) and remote sensing super-resolution, standard **random patch splitting is scientifically invalid** due to spatial autocorrelation:
- **Tobler's First Law of Geography**: *"Everything is related to everything else, but near things are more related than distant things."*
- When tiles are sampled randomly across a single satellite scene, training and validation sets inevitably contain adjacent and overlapping landscape features (roads, field boundaries, roof materials).
- This spatial leakage artificially inflates evaluation metrics (PSNR/SSIM) because the model effectively memorizes neighboring spectral geometry.

### GeoFUSE Spatial Hold-Out Implementation
To guarantee rigorous, leak-free evaluation, GeoFUSE SentinelGuard partitions the 512 × 512 pixel scene (approx. 5.12 km × 5.12 km at 10m GSD) into disjoint spatial zones:
- **Validation Hold-Out Zone**: Strictly contiguous **Southeast Quadrant** (`rows 256..512, cols 256..512`). Total: **49 non-leaking validation patches** (128 × 128 HR pixels, stride 32).
- **Training Zone**: North and West sub-regions (`rows 0..256` and `cols 0..256`). Total: **120 training patches**.
- **Boundary Separation**: Validation patches are never sampled from or near the training region, ensuring that validation evaluates generalization to completely unseen terrain.

---

## 4. Final Empirical Benchmark Metrics (Extracted from Real Logs)

> [!IMPORTANT]
> **Scientific Honesty & Methodology Disclosure**:
> The PSNR and SSIM values reported below are **supplementary metrics only**, computed strictly within the **synthetic degrade-and-recover benchmark setting** (where pseudo-LR tiles degraded via 2x downsampling, optical blur, and noise are evaluated against pre-degradation pseudo-HR references).
> 
> They do **NOT** constitute mathematical proof of true high-resolution recovery from real physical low-resolution sensors. In real satellite imagery without ground truth, operational reliability must be judged using multi-evidence consistency checks, ensemble uncertainty, and downstream task agreement.
> 
> **Zero Fabrication Guarantee**: All values below are pulled directly from the actual training logs (`outputs/checkpoints/training_log*.csv`) and generated receipts (`outputs/receipts/*.json`).

### Table 1: Model Convergence & Validation Benchmark (15 Epochs on Geographic Hold-Out)

| Model / Architecture | Seed | Parameters | Final Val Loss (L1 + 0.1·Sobel) | Val L1 Loss | Val Sobel Grad Loss | Val PSNR (dB) | Val SSIM | Best Val Epoch |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bicubic Baseline (2x)** | — | 0 (Interpolation) | 0.01920 | 0.01018 | 0.09021 | 38.19 dB | 0.9208 | — |
| **ResidualSRNet (Single)** | 42 | 273,700 (~0.27M) | **0.01726** | **0.00912** | **0.08143** | **38.68 dB** | **0.9270** | Epoch 15 |
| **Ensemble Member 0** | 42 | 273,700 (~0.27M) | 0.01728 | 0.00912 | 0.08158 | 38.67 dB | 0.9272 | Epoch 15 |
| **Ensemble Member 1** | 101 | 273,700 (~0.27M) | 0.01733 | 0.00915 | 0.08181 | 38.66 dB | 0.9265 | Epoch 15 |
| **Ensemble Member 2** | 2024 | 273,700 (~0.27M) | 0.01737 | 0.00917 | 0.08200 | 38.62 dB | 0.9261 | Epoch 15 |
| **Ensemble Mean** | — | 821,100 (3×0.27M) | **0.01733** | **0.00915** | **0.08180** | **38.65 dB** | **0.9266** | — |
| *Advantage vs. Bicubic* | — | — | *-0.00187* | *-0.00103* | *-0.00841* | *+0.46 dB* | *+0.0058* | — |

---

### Table 2: Multi-Evidence Trust & Consistency Breakdown on Held-Out Test Tiles

Extracted directly from verified JSON Trust Receipts (`outputs/receipts/trust_receipt_sample_*.json`):

| Tile ID | Ensemble Disag ($\sigma$) | Perturbation Stability ($\text{Var}$) | Mean $\Delta\text{NDVI}$ Inconsistency | Edge Gradient Corr ($r$) | Composite Trust Score | Operational Assessment Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sample #0** | 0.00155 (max 0.0108) | 0.000506 (max 0.00237) | 0.02145 (7.3% flagged) | 0.7867 | **86.75%** | `HIGH_TRUST_APPROVED` |
| **Sample #1** | 0.00162 (max 0.0112) | 0.000498 (max 0.00229) | 0.02082 (6.8% flagged) | 0.8037 | **86.70%** | `HIGH_TRUST_APPROVED` |
| **Sample #2** | 0.00178 (max 0.0125) | 0.000531 (max 0.00248) | 0.02231 (8.4% flagged) | 0.7935 | **86.41%** | `LOW_TRUST_WARNING` *(Flagged)* |
| **Sample #3** | 0.00189 (max 0.0131) | 0.000542 (max 0.00255) | 0.02153 (7.1% flagged) | 0.7840 | **86.02%** | `LOW_TRUST_WARNING` *(Flagged)* |

*Configured Threshold:* `min_trust_score_threshold: 86.5%`. Samples #2 and #3 trigger automated plain-language risk advisories warning downstream operators of elevated uncertainty.

---

### Table 3: Downstream Building Footprint Agreement on Held-Out Test Tiles

Extracted directly from `outputs/previews/` and `outputs/receipts/`:

| Tile ID | Bicubic vs. SR IoU | Bicubic vs. SR Dice | High-Trust Region IoU | Low-Trust Region IoU | High-Trust Area % | Relative Reference HR IoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sample #0** | 0.9374 | 0.9677 | 0.9343 | 0.9450 | 69.96% | 0.4325 |
| **Sample #1** | 0.9461 | 0.9723 | 0.9508 | 0.9343 | 67.52% | 0.6137 |
| **Sample #2** | 0.9253 | 0.9612 | 0.9225 | 0.9316 | 70.81% | 0.5691 |
| **Sample #3** | 0.9058 | 0.9506 | 0.9004 | 0.9131 | 63.58% | 0.5268 |

---

## 5. Directory & Artifact Layout

```
GeoFUSE/
├── config.yaml                     # Single source of truth for all paths and parameters
├── README.md                       # High-level architecture and system documentation
├── REPRODUCIBILITY.md              # This reproduction guide
├── requirements.txt                # Pinned dependencies
├── data/
│   └── raw/                        # 4-band Sentinel-2 L2A GeoTIFFs (EPSG:32643, 10m GSD)
├── outputs/
│   ├── checkpoints/                # Trained ensemble weights and training logs (.csv)
│   ├── previews/                   # Side-by-side PNG diagnostics and overlay figures
│   └── receipts/                   # Auditable JSON Trust Receipts for each tile
├── examples/
│   └── trust_receipt_sample_3.json # Tracked sample receipt with LOW_TRUST_WARNING
├── scripts/
│   ├── reproduce_all.py            # Master end-to-end reproducibility runner
│   ├── smoke_test.py               # Hardware and environment verification
│   ├── inspect_sentinel2.py        # Tile ingestion and data check
│   ├── generate_baseline_triplets.py # Bicubic 2x baseline & degradation
│   ├── train.py                    # Sequential 3-member ensemble training
│   ├── run_ensemble_inference.py   # SR inference & disagreement mapping
│   ├── test_stability.py           # Controlled perturbation stability
│   ├── test_consistency_checks.py  # Spectral NDVI & edge correlation
│   ├── generate_trust_risk_maps.py # Multi-evidence fusion
│   ├── evaluate_downstream_task.py # Downstream building footprint analysis
│   └── generate_trust_receipts.py  # JSON & HTML trust receipt generator
├── src/
│   ├── data/                       # Tiling, normalization, synthetic degradation
│   ├── models/                     # ResidualSRNet architecture, ensemble, loss
│   ├── evaluation/                 # Metrics, consistency, fusion, trust receipts
│   ├── dashboard/                  # Streamlit inspection application
│   └── utils/                      # Configuration parsing & device resolution
└── tests/                          # 57 comprehensive automated unit and integration tests
```
