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

