# Phase 12 Report: Full Validation & Ablation Study — Final Model Selection

**Document Status**: Official Repository Consolidation & Final Validation Artifact  
**Repository**: `GeoFUSE SentinelGuard`  
**Host Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (`cuda:0`, 8188 MiB VRAM)  
**Evaluation Date**: September 17, 2026  
**Final Checkpoint**: [`checkpoints/ensemble_v2/`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/ensemble_v2/) & [`checkpoints/exp005_scaled_depth/best_model.pth`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/exp005_scaled_depth/best_model.pth)  

---

## 1. Executive Summary & Final Model Recommendation

Over Phases 3 through 11, seven distinct architectural, loss, data, and scaling variants were trained and benchmarked under strict GPU verification (Phase 1) and leak-free geographic partitioning (Phase 5).

### Final Recommended Architecture: `ResidualSRNet-6b` Ensemble (`ensemble_v2`)
- **Backbone Architecture**: 6 residual blocks, 48 feature channels, sub-pixel convolution (`PixelShuffle(2)`), global bicubic residual skip connection.
- **Parameter Count**: $356,836$ parameters per member ($1,070,508$ parameters total across 3 members), well within the 1.5M parameter budget.
- **Loss Formulation**: `CompoundSRLoss` with $L_1 + 0.20 \cdot \nabla_{\text{Sobel}}$ (edge-tuned gradient weight).
- **Training Strategy**: Sequential GPU training with random seeds `42`, `101`, and `2024` on the leak-free Phase 5 split (105 training patches, 25 validation patches).
- **Final Validation Performance (Full 25-Patch Held-Out Set)**:
  - **PSNR**: **$38.83 \pm 0.21\text{ dB}$** (**$+0.19 \pm 0.02\text{ dB}$ gain over Bicubic**, doubling the baseline gain).
  - **SSIM**: **$0.9290 \pm 0.0036$** (**$+0.0038\text{ gain}$**).
  - **Spectral Radiometry**: Mean $\Delta$-NDVI dropped to **$0.02029$**; spectrally inconsistent pixels reduced to **$5.98\%$**.
  - **Structural Boundaries**: Sobel gradient correlation $r = \mathbf{0.8133}$, Canny Edge IoU = **$0.3836$**, F1 = **$0.5543$**.
  - **Downstream Building Footprint Agreement**: **$0.9092\text{ IoU}$** with Bicubic; **$0.5198\text{ IoU}$** relative accuracy against ground truth.
  - **Inference Latency**: **$12.64\text{ ms}$** for complete 3-member ensemble prediction + uncertainty computation on GPU (**79.1 patches/sec**).

*(For resource-constrained edge deployments where ensemble memory or latency cannot be accommodated, the single-model checkpoint `exp005_scaled_depth` provides the fastest alternative at **$1.38\text{ ms}$ latency** and **$38.75\text{ dB PSNR}$**).*

---

## 2. Complete Side-by-Side Ablation Matrix (Phases 3–11)

The table below compiles every experimental trial recorded in [`experiments/experiments_log.jsonl`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/experiments/experiments_log.jsonl) under identical degrade-and-recover validation standards:

| Exp ID | Architecture | Scale | Params | Split | Loss Formula | Val PSNR (dB) | PSNR Gain | Val SSIM | SSIM Gain | Mean $\Delta$-NDVI | Inconsistent Pixels | Grad Corr ($r$) | Edge IoU | Stability Var | Peak VRAM | Latency / patch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_exp001`** | 4b / 48c | $2\times$ | 274k | v1 *(Leakage)* | $L_1 + 0.10 \nabla$ | 38.67 ± 0.22 | +0.08 dB | 0.9272 | +0.0023 | 0.02091 | 6.63% | 0.8173 | 0.3838 | — | 158.2 MiB | 1.52 ms |
| **`exp002_baseline_v2`** | 4b / 48c | $2\times$ | 274k | v2 *(Clean)* | $L_1 + 0.10 \nabla$ | 38.65 ± 0.20 | +0.01 dB | 0.9265 | +0.0012 | 0.02086 | 6.54% | 0.8098 | 0.3787 | — | 153.7 MiB | 1.57 ms |
| **`exp003_augmented`** | 4b / 48c | $2\times$ | 274k | v2 *(Clean)* | $L_1 + 0.10 \nabla$ + D4 | 38.64 ± 0.20 | +0.00 dB | 0.9263 | +0.0011 | 0.02089 | 6.57% | 0.8085 | 0.3781 | — | 153.7 MiB | 1.53 ms |
| **`exp004_edge_tuned`** | 4b / 48c | $2\times$ | 274k | v2 *(Clean)* | $L_1 + \mathbf{0.20} \nabla$ | 38.74 ± 0.21 | +0.10 dB | 0.9282 | +0.0030 | 0.02060 | 6.28% | 0.8117 | 0.3820 | — | 153.7 MiB | 1.77 ms |
| **`exp005_scaled_depth`** | **6b / 48c** | $2\times$ | 357k | v2 *(Clean)* | $L_1 + \mathbf{0.20} \nabla$ | **38.75 ± 0.21** | **+0.12 dB** | **0.9280** | **+0.0028** | **0.02046** | **6.14%** | **0.8138** | **0.3839** | 0.000505 | 178.8 MiB | **1.38 ms** |
| **`exp006_ensemble_v2`** | **3x (6b/48c)** | $2\times$ | **1.07M** | v2 *(Clean)* | $L_1 + \mathbf{0.20} \nabla$ (Ensemble) | **38.83 ± 0.21** | **+0.19 dB** | **0.9290** | **+0.0038** | **0.02029** | **5.98%** | **0.8133** | **0.3836** | 0.000490 | 178.8 MiB | 12.64 ms *(Ens)* |
| **`exp007_scale3x`** | 6b / 48c | **$3\times$** | 461k | v2 *(Clean)* | $L_1 + 0.20 \nabla$ (3x Head) | 36.16 ± 0.42 | +0.03 dB | 0.8817 | +0.0008 | 0.02550 | 12.23% | 0.6435 | 0.2278 | 0.000400 | 79.3 MiB | 1.91 ms |

---

## 3. Scientific Attribution & Ablation Analysis

### 1. Removing Spatial Leakage Revealed True Performance (`exp001` vs. `exp002`)
- In `baseline_exp001`, evaluating on Split v1 (center-point split with 24 boundary overlap pairs) reported an apparent PSNR gain of $+0.08\text{ dB}$.
- When tested on the leak-free Phase 5 split (`exp002_baseline_v2`), the baseline 4-block model achieved only **$+0.01\text{ dB}$ gain** over Bicubic.
- **Attribution**: Spatial boundary autocorrelation between adjacent training and validation patches had silently inflated the perceived baseline gain by $8\times$. Split v2 established an honest benchmark.

### 2. Geometric Augmentation Did Not Accelerate Convergence (`exp002` vs. `exp003`)
- Adding random D4 dihedral flips and 90° rotations (`exp003_augmented`) yielded $38.64\text{ dB PSNR}$ and $0.9263\text{ SSIM}$ (statistically indistinguishable from unaugmented $38.65\text{ dB}$).
- **Attribution**: Within a 15-epoch training schedule on 105 patches, data augmentation increased sample variance without sufficient epochs to reach lower training loss. It was discarded for subsequent single-model tuning.

### 3. Edge Gradient Loss Weight Tuning Drove Major Gains (`exp002` vs. `exp004`)
- Increasing Sobel gradient loss weight from $0.10$ to $0.20$ produced the single largest jump in the ablation:
  - PSNR jumped from $38.65\text{ dB}$ to **$38.74\text{ dB}$** (+0.10 dB gain over Bicubic).
  - SSIM rose from $0.9265$ to **$0.9282$**.
  - Spectral $\Delta$-NDVI dropped from $0.02086$ to **$0.02060$**.
  - Edge IoU improved from $0.3787$ to **$0.3820$**.
- **Attribution**: Penalizing high-frequency gradient mismatch forced the network to actively resolve sub-pixel contrast transitions rather than averaging them into smooth $L_1$ medians.

### 4. Depth Scaling Provided the Best Single-Model Efficiency (`exp004` vs. `exp005`)
- Scaling from 4 to 6 residual blocks (adding 83k parameters) lifted PSNR to **$38.75\text{ dB}$** (+0.12 dB gain), dropped $\Delta$-NDVI to **$0.02046$**, and achieved the highest single-model edge alignment ($r = 0.8138$, IoU $0.3839$).
- **Attribution**: Increased representational capacity allowed the network to learn more complex spatial-spectral receptive fields while remaining ultra-fast ($1.38\text{ ms}$ latency).

### 5. Multi-Seed Ensembling Delivered the Global Optimum (`exp005` vs. `exp006`)
- Averaging predictions across 3 diverse members retrained with seeds `42`, `101`, and `2024` achieved the overall best fidelity:
  - PSNR reached **$38.83\text{ dB}$** (**$+0.19\text{ dB}$ gain over Bicubic**).
  - SSIM reached **$0.9290$**.
  - Inconsistent pixels dropped below $6\%$ (**$5.98\%$**).
- **Attribution**: Variance reduction through ensemble averaging cancels idiosyncratic high-frequency prediction noise across distinct local minima in the loss basin.

### 6. Sub-4m Scale-3x Experiment Confirmed Severe Degradation (`exp005` vs. `exp007`)
- Shifting from $2\times$ ($5\text{m}$ grid) to $3\times$ ($3.33\text{m}$ grid) resulted in a sharp quality drop:
  - PSNR dropped by **$2.59\text{ dB}$** ($36.16\text{ dB}$).
  - SSIM dropped to **$0.8817$** (failing the 0.90 standard).
  - Spectrally inconsistent pixels doubled to **$12.23\%$**.
  - Edge IoU collapsed to **$0.2278$**.
- **Attribution**: The $8:1$ pixel synthesis ratio ($88.9\%$ unobserved pixels) is too severely ill-posed for 10m Sentinel-2 data without external high-resolution training targets.

---

## 4. Final Full Validation Pass Distributions (25 Held-Out Patches)

Run on the winning `ensemble_v2` model through the full end-to-end evidence pipeline on `cuda:0` (saved to [`logs/final_model_validation_distribution.csv`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/logs/final_model_validation_distribution.csv)):

| Pipeline Metric | Mean ± Std | Min | 25% | Median | 75% | Max | Unit / Meaning |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model PSNR** | **$38.83 \pm 0.21$** | 38.36 | 38.71 | 38.84 | 38.99 | 39.19 | dB (Reconstruction fidelity) |
| **Bicubic Baseline PSNR** | $38.64 \pm 0.21$ | 38.20 | 38.51 | 38.65 | 38.81 | 38.96 | dB |
| **PSNR Gain over Bicubic** | **$+0.19 \pm 0.02$** | +0.15 | +0.17 | +0.19 | +0.20 | +0.23 | dB (Statistically positive on 100% of patches) |
| **Model SSIM** | **$0.9290 \pm 0.0036$** | 0.9224 | 0.9268 | 0.9292 | 0.9318 | 0.9339 | [-1, 1] (Structural similarity) |
| **SSIM Gain over Bicubic** | **$+0.0038 \pm 0.0004$** | +0.0032 | +0.0036 | +0.0038 | +0.0041 | +0.0044 | score delta |
| **Mean Delta-NDVI** | **$0.02029 \pm 0.00060$** | 0.01921 | 0.01984 | 0.02035 | 0.02071 | 0.02145 | Absolute vegetative index error |
| **Inconsistent Pixels ($\Delta > 0.05$)** | **$5.98 \pm 0.65$** | 4.98 | 5.57 | 6.01 | 6.45 | 7.20 | % of tile pixels |
| **Sobel Gradient Corr ($r$)** | **$0.8133 \pm 0.0254$** | 0.7650 | 0.7932 | 0.8164 | 0.8315 | 0.8736 | Pearson correlation |
| **Canny Edge IoU** | **$0.3836 \pm 0.0179$** | 0.3510 | 0.3705 | 0.3852 | 0.3980 | 0.4246 | Intersection over Union |
| **Canny Edge F1 Score** | **$0.5543 \pm 0.0187$** | 0.5196 | 0.5403 | 0.5562 | 0.5694 | 0.5961 | Harmonic mean precision/recall |
| **Composite Evidence Score** | **$84.60 \pm 0.94$** | 82.12 | 83.95 | 84.68 | 85.34 | 85.82 | % (Empirical heuristic trust indicator) |
| **Mean Composite Risk** | **$0.1540 \pm 0.0094$** | 0.1418 | 0.1466 | 0.1532 | 0.1605 | 0.1788 | [0, 1] (Normalized composite risk) |
| **Downstream Footprint Agreement** | **$0.9092 \pm 0.0138$** | 0.8767 | 0.9008 | 0.9098 | 0.9205 | 0.9298 | IoU (Bicubic vs SR building footprints) |
| **Downstream Accuracy (vs Ref HR)** | **$0.5198 \pm 0.0267$** | 0.4718 | 0.4994 | 0.5218 | 0.5401 | 0.5618 | IoU (SR footprints vs reference HR extraction) |
| **Inference Latency** | **$12.64 \pm 31.53$** | 4.64 | 5.12 | 5.34 | 5.82 | 163.78 | ms per patch (3 members + disagreement) |

---

## 5. End-to-End Pipeline Integrity Verification

During the final validation pass, all 9 stages of the GeoFUSE evidence pipeline executed without errors or warnings:
1. **Model Weight Loading**: Checkpoints [`checkpoints/ensemble_v2/member_{1,2,3}.pt`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/ensemble_v2/) loaded cleanly.
2. **Inference & Epistemic Uncertainty**: Produced non-identical predictions (pairwise MSE $\approx 10^{-5}$) and scalar disagreement map.
3. **Perturbation Testing**: Computed perturbation sensitivity across noise and jitter trials.
4. **Spectral & Structural Auditing**: Generated $\Delta$-NDVI and Sobel gradient difference maps.
5. **Multi-Criteria Fusion**: Synthesized normalized spatial risk and trust heatmaps.
6. **Downstream Footprint Analysis**: Extracted morphological building footprints and verified positive agreement.
7. **Trust Receipt Generation**: Generated cryptographically signed receipt `TR-Sentinel-2A-43PGQ-T00-1789585554` containing full provenance metadata.

---

## 6. Deployment Recommendation Summary

| Deployment Context | Recommended Model | Checkpoint Location | Key Rationale |
| :--- | :--- | :--- | :--- |
| **Mission Control Dashboard (Primary Demo)** | **`ensemble_v2` (3-member ensemble)** | [`checkpoints/ensemble_v2/`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/ensemble_v2/) | Best PSNR ($38.83\text{ dB}$), highest SSIM ($0.9290$), lowest spectral error, and provides required epistemic disagreement uncertainty layer. |
| **Edge / Low-Latency Deployment (Alternative)** | **`exp005_scaled_depth` (Single model)** | [`checkpoints/exp005_scaled_depth/best_model.pth`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/exp005_scaled_depth/best_model.pth) | Ultra-fast ($1.38\text{ ms}$ latency, $725\text{ p/s}$), $357\text{k}$ parameters, $38.75\text{ dB PSNR}$ (+0.12 dB gain). |
| **Sub-4m Scale-3x Variant** | *Archived Research Artifact Only* | [`checkpoints/sr_experiment_scale3x/best_model.pth`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/sr_experiment_scale3x/best_model.pth) | Do not deploy in demo: $36.16\text{ dB PSNR}$, $0.8817\text{ SSIM}$, $12.23\%\text{ spectral error}$. |
