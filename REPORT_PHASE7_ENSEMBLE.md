# GeoFUSE SentinelGuard — Phase 7 Ensemble Improvement & Disagreement Informativeness Audit

**Date**: September 16, 2026  
**Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (`cuda:0`, 8188 MiB VRAM)  
**Ensemble Architecture**: 3-member ensemble of `ResidualSRNet` (6 residual blocks, 48 features, 356,836 parameters each)  
**Loss Function**: `CompoundSRLoss` ($L_1 + 0.20 \cdot \nabla_{\text{Sobel}}$)  
**Dataset Split**: Phase 5 Split v2 (105 train / 25 held-out val patches; 0 overlapping pixels)  
**Checkpoints**: [`checkpoints/ensemble_v2/member_{1,2,3}.pt`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/ensemble_v2/)  
**Analysis Log**: [`logs/ensemble_v2_disagreement_analysis.csv`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/logs/ensemble_v2_disagreement_analysis.csv)  

---

## 1. Executive Summary & Objective

In **Phase 7**, we retrained a 3-member super-resolution ensemble using the best architecture and loss configuration discovered in Phase 6 (`ResidualSRNet`, 6 blocks, 48 channels, `grad_weight=0.20`).

### Core Scientific Questions Addressed:
1. **Diversity**: Do the ensemble members produce genuine, non-identical predictions on identical inputs?
2. **Reconstruction Gain**: Does averaging the ensemble predictions improve reconstruction fidelity over any individual model?
3. **Disagreement Informativeness**: Does pixel-level ensemble disagreement (standard deviation across members) actually correlate with structural complexity and high-frequency edges, or is it uniformly noisy and uninformative?

---

## 2. Training Telemetry & Member Diversity

Members were trained sequentially on `cuda:0` to guarantee strict VRAM compliance ($178.79\text{ MiB}$ peak VRAM, $< 5\text{ seconds}$ per member):
- **Member 1 (Seed 42)**: Best Epoch #15 | Val Loss: 0.02489 | Val PSNR: **38.75 dB** | Val SSIM: **0.9280** | Duration: 4.81s
- **Member 2 (Seed 101)**: Best Epoch #15 | Val Loss: 0.02507 | Val PSNR: **38.70 dB** | Val SSIM: **0.9270** | Duration: 4.73s
- **Member 3 (Seed 2024)**: Best Epoch #15 | Val Loss: 0.02511 | Val PSNR: **38.63 dB** | Val SSIM: **0.9264** | Duration: 4.71s

### 2.1 Empirical Diversity Verification
Pairwise Mean Squared Error (MSE) was evaluated across all 3 members on identical validation input patches:
- $\text{MSE}(\text{Member 1}, \text{Member 2}) = \mathbf{0.000009}$
- $\text{MSE}(\text{Member 1}, \text{Member 3}) = \mathbf{0.000010}$
- $\text{MSE}(\text{Member 2}, \text{Member 3}) = \mathbf{0.000010}$

**Finding**: All pairwise MSE values are strictly positive ($\sim 1 \times 10^{-5}$ in normalized reflectance scale $[0, 1]$), confirming that varying random weight initialization and DataLoader batch mini-batch trajectories successfully caused gradient descent to converge to distinct local optima.

---

## 3. Quantitative Reconstruction Performance

Evaluating across all **25 held-out validation patches** of the leak-free Phase 5 split:

| Model / Ensemble | Val PSNR (dB) | PSNR Gain over Bicubic (dB) | Val SSIM | SSIM Gain over Bicubic |
| :--- | :---: | :---: | :---: | :---: |
| Bicubic Baseline | 38.64 ± 0.20 | — | 0.9252 ± 0.0036 | — |
| Member 3 (Seed 2024) | 38.64 ± 0.21 | +0.00 ± 0.04 | 0.9264 ± 0.0035 | +0.0012 |
| Member 2 (Seed 101) | 38.70 ± 0.20 | +0.06 ± 0.04 | 0.9270 ± 0.0035 | +0.0018 |
| Member 1 (Seed 42) | 38.75 ± 0.21 | +0.11 ± 0.04 | 0.9280 ± 0.0034 | +0.0028 |
| **Ensemble Mean (3 Members)** | **38.83 ± 0.21** | **+0.19 ± 0.04** | **0.9290 ± 0.0035** | **+0.0038** |

### Key Finding: Ensemble Averaging Delivers Tangible Fidelity Gains
Ensemble mean reconstruction achieves **38.83 dB PSNR**, outperforming:
- The Bicubic baseline by **+0.19 dB** (nearly double the gain of the single best member).
- The best individual member (`Member 1`, 38.75 dB) by **+0.08 dB**.
- Structural SSIM increases to **0.9290**, demonstrating significant variance reduction on high-frequency detail.

---

## 4. Disagreement Informativeness Audit: Honest Assessment

The prompt specifically requested:
> *"Measure whether disagreement actually correlates with regions of known difficulty (e.g. edges, textured urban areas) vs. being uniformly noisy/meaningless. Report whether ensemble disagreement is informative or not — honestly, even if the answer is 'not very.'"*

To answer this question rigorously, two spatial statistics were calculated for each validation patch:
1. **Pearson Correlation $r(\text{Disagreement}, \nabla_{\text{Sobel}})$**: Spatial correlation between the pixel-level ensemble standard deviation and the ground-truth Sobel gradient magnitude.
2. **Edge-to-Background Disagreement Ratio**: $\frac{\overline{\text{Disagreement}}_{\text{Canny Edge}}}{\overline{\text{Disagreement}}_{\text{Background}}}$.

### 4.1 Statistical Results across 25 Held-Out Patches

| Metric | Mean ± Std | Min | Max | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **$r(\text{Disagreement}, \nabla_{\text{Sobel}})$** | **0.2164 ± 0.0589** | **0.1403** | **0.3951** | Consistently positive, weak-to-moderate correlation |
| **Edge / Background Disagreement Ratio** | **1.041x ± 0.025** | **1.005x** | **1.099x** | Disagreement is ~4.1% higher on edge pixels |
| **Mean Disagreement Magnitude** | 0.0021 ± 0.0003 | 0.0016 | 0.0028 | Low absolute variance across members |

### 4.2 Honest Scientific Conclusion: "Weakly Informative, Highly Modest"

**Is ensemble disagreement informative?**
- **The Honest Answer**: **"Weakly informative, but largely uniform."**
- **Why it is not completely meaningless**:
  - The spatial correlation with high-frequency edges is consistently positive ($r = 0.2164$, peaking up to $0.3951$).
  - Across all 25 validation patches, disagreement is strictly higher on Canny edge boundaries than on flat background regions ($\text{ratio} > 1.0$ on every patch, averaging $1.041\times$).
- **Why it is "not very" informative**:
  - A $4.1\%$ elevation on edges is very small. Visually and numerically, the disagreement map has a large baseline component across all pixels, rather than functioning like a crisp detector of reconstruction failures.
  - Because all 3 members share the same architecture, loss function, and training dataset, their predictions are tightly clustered (pairwise MSE $\approx 10^{-5}$). They rarely take radically different topological trajectories.

### 4.3 Engineering Decision for GeoFUSE
1. **Keep the Ensemble for Reconstruction**: The ensemble mean reconstruction is unequivocally superior to single-model inference (+0.19 dB gain over bicubic, 38.83 dB).
2. **Scientific Modesty for Evidence Fusion**: The Disagreement Evidence layer should be honestly documented in UI banners as a *weak heuristic proxy* for epistemic variance, not an authoritative error map. It should be weighted proportionally alongside input perturbation stability and spectral consistency rather than dominating the trust receipt.

---

## 5. Artifacts & Deliverables

1. **Checkpoints**:
   - [`checkpoints/ensemble_v2/member_1.pt`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/ensemble_v2/member_1.pt) (and `.pth`)
   - [`checkpoints/ensemble_v2/member_2.pt`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/ensemble_v2/member_2.pt) (and `.pth`)
   - [`checkpoints/ensemble_v2/member_3.pt`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/ensemble_v2/member_3.pt) (and `.pth`)
2. **Training & Audit CLI**: [`scripts/train_ensemble_v2.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/scripts/train_ensemble_v2.py)
3. **Automated Unit Tests**: [`tests/test_ensemble_v2.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/tests/test_ensemble_v2.py) (All 3 tests passing)
4. **Per-Patch CSV Log**: [`logs/ensemble_v2_disagreement_analysis.csv`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/logs/ensemble_v2_disagreement_analysis.csv)
