# Phase 11 Report: Experimental Benchmarking of Sub-4m Scale-3x Super-Resolution

**Document Status**: Official Repository Benchmark & Evaluation Report  
**Repository**: `GeoFUSE SentinelGuard`  
**Experiment ID**: `exp007_scale3x`  
**Checkpoint Path**: [`checkpoints/sr_experiment_scale3x/best_model.pth`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/sr_experiment_scale3x/best_model.pth)  
**Host Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (`cuda:0`, 8188 MiB VRAM)  
**Date**: September 17, 2026  
**Git Commit**: Tracked in `experiments/experiments_log.jsonl`  

---

## 1. Executive Summary & Stop-Condition Recommendation

> [!CAUTION]
> ### STOP CONDITION FORMAL RECOMMENDATION: KEEP THE 2x MODEL FOR PRODUCTION & DEMO
> **The 3x super-resolution model (`exp007_scale3x`) performs substantially worse than the 2x model across every single physical, spectral, and structural fidelity metric.**
> 
> Although the 3x model successfully projects features onto a **nominal 3.33m grid** (<4m GSD), the $8:1$ pixel synthesis challenge on 10m native Sentinel-2 imagery degrades reconstruction quality:
> - **Reconstruction PSNR drops by $2.59\text{ dB}$** (from $38.75\text{ dB}$ to $36.16\text{ dB}$).
> - **Gain over Bicubic collapses** from $+0.12\text{ dB}$ to barely **$+0.03\text{ dB}$**.
> - **SSIM drops from $0.9280$ to $0.8817$** (failing the 0.90 structural threshold).
> - **Spectrally inconsistent pixels double** from $6.14\%$ to **$12.23\%$**.
> - **Canny Edge IoU drops by 40%** (from $0.3839$ to $0.2278$), and gradient correlation falls from $0.8138$ to **$0.6435$**.
> 
> **Decision**: In strict accordance with the Phase 11 stop condition (*"If results are substantially worse or unstable, stop and recommend keeping the 2x model for the demo rather than adopting a weaker higher-scale variant for a marketing claim"*), **we recommend retaining the 2x model (`ResidualSRNet-6b`, exp005 / ensemble_v2) as the operational standard.**
> 
> The 3x variant is archived under `checkpoints/sr_experiment_scale3x/` as an experimental research artifact, but should **NOT** replace the superior 2x model in the interactive dashboard or evidence pipeline.

---

## 2. Experimental Setup & Hardware Telemetry

Both models were trained and benchmarked under identical scientific conditions on the **RTX 4060 Laptop GPU**:

| Configuration Parameter | 2x Winning Model (`exp005_scaled_depth`) | 3x Experimental Variant (`exp007_scale3x`) |
| :--- | :---: | :---: |
| **Nominal Ground Sampling Distance (GSD)** | $5.00\text{ m}$ | **$3.333...\text{ m}$ (<4m nominal grid)** |
| **Synthesis Ratio (Unobserved : Observed)** | $3 : 1$ ($4\times$ pixel area expansion) | **$8 : 1$ ($9\times$ pixel area expansion)** |
| **Architecture** | `ResidualSRNet` (6 blocks, 48 features) | `ResidualSRNet` (6 blocks, 48 features) |
| **Upsampling Operator** | `PixelShuffle(2)` | `PixelShuffle(3)` |
| **Trainable Parameters** | $356,836$ | $460,756$ (+29.1%) |
| **Training Patch Size (LR $\rightarrow$ HR)** | $64 \times 64 \rightarrow 128 \times 128$ | $32 \times 32 \rightarrow 96 \times 96$ |
| **Geographic Split** | Phase 5 Clean Split v2 (Zero Leakage) | Phase 5 Clean Split v2 (Zero Leakage) |
| **Training Patches / Validation Patches** | 105 train / 25 val | 132 train / 36 val |
| **Loss Function** | $L_1 + 0.20 \cdot \nabla_{\text{Sobel}}$ | $L_1 + 0.20 \cdot \nabla_{\text{Sobel}}$ |
| **Optimizer / LR Schedule** | AdamW (5e-4) / Cosine Annealing (15 ep) | AdamW (5e-4) / Cosine Annealing (15 ep) |
| **Mixed Precision** | `torch.amp.autocast('cuda', fp16)` | `torch.amp.autocast('cuda', fp16)` |
| **GPU Peak VRAM** | $178.79\text{ MiB}$ | **$79.30\text{ MiB}$** |
| **Training Duration (15 epochs)** | $4.96\text{ s}$ | **$3.45\text{ s}$** |
| **Single-Patch Inference Latency** | $1.38\text{ ms}$ (724.6 patches/s) | **$1.91\text{ ms}$ (524.0 patches/s)** |

---

## 3. Comprehensive Comparative Benchmark Table (2x vs. 3x)

Evaluated across all held-out geographic validation patches in the Southeast Quadrant with zero spatial leakage:

| Evaluation Dimension | Metric | 2x Best Model (`exp005`) | 3x Experimental Model (`exp007`) | Delta ($3\times - 2\times$) | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Reconstruction Fidelity** | **Model PSNR (dB)** | **$38.75 \pm 0.21$** | **$36.16 \pm 0.42$** | **−2.59 dB** | Substantially Worse |
| | Bicubic Benchmark PSNR | $38.64 \pm 0.20$ | $36.13 \pm 0.41$ | −2.51 dB | Baseline lower |
| | **PSNR Gain over Bicubic** | **$+0.12 \pm 0.04$** | **$+0.03 \pm 0.05$** | **−0.09 dB** | Marginal advantage |
| | Model MAE (reflectance) | $0.00892 \pm 0.00021$ | $0.01211 \pm 0.00062$ | +0.00319 | +35.8% error |
| **Structural Alignment** | **Model SSIM** | **$0.9280 \pm 0.0034$** | **$0.8817 \pm 0.0101$** | **−0.0463** | Severely Degraded (<0.90) |
| | Bicubic Benchmark SSIM | $0.9252 \pm 0.0036$ | $0.8809 \pm 0.0101$ | −0.0443 | Baseline lower |
| | **SSIM Gain over Bicubic** | **$+0.0028 \pm 0.0006$** | **$+0.0008 \pm 0.0007$** | **−0.0020** | Collapsed |
| | **Sobel Gradient Corr ($r$)** | **$0.8138 \pm 0.0248$** | **$0.6435 \pm 0.0482$** | **−0.1703** | Boundaries blurred |
| | **Canny Edge IoU** | **$0.3839 \pm 0.0212$** | **$0.2278 \pm 0.0315$** | **−0.1561** | −40.7% edge overlap |
| | **Canny Edge F1** | **$0.5545 \pm 0.0225$** | **$0.3703 \pm 0.0396$** | **−0.1842** | −33.2% edge F1 |
| **Spectral Radiometry** | **Mean $\Delta$-NDVI** | **$0.02046 \pm 0.00058$** | **$0.02550 \pm 0.00138$** | **+0.00504** | +24.6% distortion |
| | **Inconsistent Pixels ($\Delta > 0.05$)** | **$6.14 \pm 0.68\%$** | **$12.23 \pm 1.84\%$** | **+6.09%** | **Double the error rate** |
| **Robustness & Compute** | **Perturbation Stability Var** | $0.000505$ | $0.000400$ | −0.000105 | Slightly lower variance |
| | **Inference Latency** | $1.38\text{ ms}$ | $1.91\text{ ms}$ | +0.53 ms | Still very fast |
| | **Peak Training VRAM** | $178.79\text{ MiB}$ | $79.30\text{ MiB}$ | −99.49 MiB | Extremely lightweight |

---

## 4. Deep Scientific Analysis: Why Does 3x Perform Substantially Worse?

### 4.1 The $8 : 1$ Ill-Posedness Penalty
In single-image super-resolution, ill-posedness scales quadratically ($s^2$):
- At $2\times$, the model reconstructs $4$ pixels from $1$. The network has $25\%$ of physical ground evidence directly available.
- At $3\times$, the model reconstructs $9$ pixels from $1$. The network has only $11.1\%$ of physical ground evidence available; **$88.9\%$ of all output pixel values must be synthesized.**
- With only 132 training patches extracted from a single Sentinel-2 geographic scene, the network does not have sufficient multi-scene contextual diversity to learn an 8:1 mapping without smoothing or blurring high frequencies.

### 4.2 Receptive Field & LR Context Deficit
- At $3\times$, the low-resolution input patch is only $32 \times 32$ pixels ($320\text{m} \times 320\text{m}$ ground footprint).
- In a $32 \times 32$ tile, urban structures and field boundaries occupy only 1–3 pixels.
- The 6-block residual CNN's receptive field ($13 \times 13$ pixels) spans over $40\%$ of the entire input patch dimension, causing boundary features to bleed together and producing over-smoothed representations.

### 4.3 Severe Structural & Edge Collapse
The drop in structural alignment is stark:
- Sobel gradient correlation drops from $r = 0.8138$ to $r = 0.6435$.
- Edge IoU drops from $0.3839$ to $0.2278$.
- At $3\times$, the network struggles to distinguish true ground boundaries from PSF blur and sensor noise, resulting in fuzzy, low-contrast edges that barely outperform bicubic interpolation (+0.03 dB gain vs. +0.12 dB for 2x).

---

## 5. Mandatory Scientific Disclosure: Nominal Grid vs. Validated Resolution

> [!IMPORTANT]
> **Mandatory Disclosure**:
> 1. The 3x model outputs an array with pixel pitch corresponding to **3.33 meters** on the ground ($10\text{m} / 3$).
> 2. This is purely a **nominal raster coordinate grid**.
> 3. It does **not** provide validated physical sub-4m optical resolving power.
> 4. Under degrade-and-recover validation, the model's actual ability to recover 10m detail from simulated 30m input is substantially inferior to the 2x model's ability to recover 10m detail from 20m input.
> 5. Adopting this model under the marketing banner of "<4m resolution" would be misleading and scientifically unsound.

---

## 6. Clear Operational Recommendation for the Dashboard Demo

| Option | Scale Factor | Nominal Grid | Real Reconstruction PSNR | SSIM | Inconsistent Pixels | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Option A (2x Standard)** | **$2\times$** | **$5.00\text{ m}$** | **$38.75\text{ dB}$ (+0.12 dB gain)** | **$0.9280$** | **$6.14\%$** | **RECOMMENDED FOR DEMO** |
| **Option B (3x Research Variant)** | $3\times$ | $3.33\text{ m}$ | $36.16\text{ dB}$ (+0.03 dB gain) | $0.8817$ | $12.23\%$ | **ARCHIVE AS RESEARCH ARTIFACT ONLY** |

### Final Conclusion
- **Retain the 2x model** (`ResidualSRNet-6b`, 48 features, $L_1 + 0.20 \cdot \nabla$) as the production standard for the interactive Streamlit dashboard, downstream building footprint analysis, and trust receipts.
- Preserve the 3x checkpoint in [`checkpoints/sr_experiment_scale3x/`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/sr_experiment_scale3x/) and experiment log `exp007_scale3x` in [`experiments/experiments_log.jsonl`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/experiments/experiments_log.jsonl) as proof of rigorous empirical negative-result reporting.
