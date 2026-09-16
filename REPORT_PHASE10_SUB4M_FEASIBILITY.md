# Phase 10 Report: Scientific Feasibility Investigation for Sub-4m Super-Resolution

**Document Status**: Official Repository Research & Feasibility Report  
**Repository**: `GeoFUSE SentinelGuard`  
**Target Research Question**: What scale factor beyond 2x could be scientifically pursued to achieve a nominal grid finer than 4m, what does "<4m" scientifically mean, and is an experimental trial (Phase 11) justified?  
**Host Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM)  
**Date**: September 17, 2026  

---

## 1. Executive Summary & Core Recommendation

> [!IMPORTANT]
> ### Core Recommendation: CONDITIONAL GO for Nominal 3.33m Grid Benchmark
> - **Arithmetic Feasibility**: To achieve a nominal grid finer than $4.0\text{ m}$ from $10.0\text{ m}$ native Sentinel-2 imagery, an integer scale factor of **$s = 3\times$** is required ($10.0\text{ m} / 3 = \mathbf{3.333...\text{ m}}$ nominal Ground Sampling Distance).
> - **Architectural Feasibility**: Supported natively by `ResidualSRNet` via sub-pixel convolution (`nn.PixelShuffle(3)`). Parameter count rises modestly from **356,836** to **460,756** (+29.1%), well within the 1.5M budget.
> - **Compute / VRAM Feasibility**: Training at $3\times$ requires only **76.3 MiB to 166.8 MiB peak VRAM** on the RTX 4060 (< 2.1% of available VRAM).
> - **Scientific Caveat**: **A nominal grid finer than 4m ($3.33\text{ m}$) is NOT equivalent to validated true sub-4m spatial information.**
>   Because Sentinel-2 L2A ground truth is strictly 10m, $3\times$ degrade-and-recover training can only be quantitatively evaluated on the $30\text{ m} \rightarrow 10\text{ m}$ recovery task. Applying the resulting weights to native 10m imagery produces a $3.33\text{ m}$ nominal raster grid, but its effective optical resolving power is heuristic and cannot be verified without independent sub-4m aerial or commercial satellite truth.

---

## 2. Spatial Arithmetic: Defining Nominal Grid vs. Validated Resolution

### 2.1 Exact Scale Factor Arithmetic

Native optical bands of Sentinel-2 L2A (B02 Blue, B03 Green, B04 Red, B08 NIR) are delivered at a physical Ground Sampling Distance (GSD) of **$10.0\text{ m} \times 10.0\text{ m}$**:

$$\text{Nominal Output GSD} = \frac{\text{Native GSD}}{\text{Scale Factor } s} = \frac{10.0\text{ m}}{s}$$

| Scale Factor ($s$) | Upsampling Operator | Nominal Output GSD | Grid Spacing Relative to 4m | Feasibility in PyTorch |
| :---: | :---: | :---: | :---: | :---: |
| **$1.0\times$** | Identity | $10.00\text{ m}$ | Coarser than 4m | Native |
| **$2.0\times$** *(Current)* | `PixelShuffle(2)` | $5.00\text{ m}$ | Coarser than 4m | Native (current baseline) |
| **$2.5\times$** | Fractional / Resample | $4.00\text{ m}$ | Exactly 4m | Non-integer (requires fractional interpolation) |
| **$3.0\times$** *(Candidate)* | `PixelShuffle(3)` | **$3.333...\text{ m}$** | **Finer than 4m (<4m target)** | **Native integer operator** |
| **$4.0\times$** | `PixelShuffle(4)` | $2.50\text{ m}$ | Sub-3m ($16\times$ pixel synthesis) | Native (high ill-posedness) |

To reach a nominal resolution strictly finer than $4.0\text{ m}$, the scale factor must satisfy:
$$s > \frac{10.0\text{ m}}{4.0\text{ m}} = 2.5\times$$

Because sub-pixel convolution (`PixelShuffle`) maps periodic channels to a 2D grid via $C \cdot s^2$, integer scale factors are required to preserve efficient tensor operations. The smallest integer scale factor achieving $< 4\text{m}$ nominal grid spacing is **$s = 3$** ($\mathbf{3.333...\text{ m}}$).

---

### 2.2 Crucial Scientific Distinction: Nominal Grid vs. Validated Information

> [!CAUTION]
> **A nominal $3.33\text{ m}$ raster grid does NOT represent validated sub-4m spatial resolving power.**

In satellite remote sensing, there is a fundamental physical difference between:
1. **Nominal Grid Resolution (Pixel Pitch / GSD)**:
   The geometric spacing of pixels in the raster coordinate system. Any image can be resampled with bicubic or nearest-neighbor interpolation to a $1\text{ mm}$ grid, but this creates zero physical resolving power.
2. **Effective Spatial Resolution (Resolving Power / MTF)**:
   The physical ability of an imaging system to independently resolve two distinct, closely spaced ground targets (governed by the satellite sensor's optical aperture, Point Spread Function, atmospheric scattering, and detector modulation transfer function).

In single-image super-resolution (SISR) trained on Sentinel-2 data alone:
- **No ground truth exists below 10m.** The satellite sensor never recorded optical photon counts at $<10\text{m}$.
- When a neural network reconstructs native 10m imagery at $3\times$ ($3.33\text{m}$ grid), it synthesizes high-frequency patterns based purely on statistical priors learned during training.
- While this sharpens visual edges and provides finer geometric rasterization for downstream building footprint contours, **it cannot be certified as true ground-verified sub-4m physical detail.**
- Any claim resulting from Phase 11 must explicitly describe the output as a **"nominal 3.33m reconstruction grid,"** never as "validated 3.33m optical imagery."

---

## 3. Architecture & Upsampling Head Adaptability

The current model architecture ([`src/models/model.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/models/model.py)) is a lightweight residual network (`ResidualSRNet`):

```
Input Tensor (B, 4, H, W)
      │
      ▼
Head Conv: Conv2d(4, 48, kernel=3, padding=1)
      │
      ▼
6x Residual Blocks: 2x Conv2d(48, 48, kernel=3) + LeakyReLU
      │
      ▼
Trunk Conv: Conv2d(48, 48, kernel=3) + Residual Skip
      │
      ▼
Upsample Head: Conv2d(48, 48 * s², kernel=3) ──► PixelShuffle(s) ──► LeakyReLU
      │                                                                  │
      ▼                                                                  ▼
Base Bicubic Interpolation: F.interpolate(scale=s)                  Tail Conv: Conv2d(48, 4)
      │                                                                  │
      └────────────────────────── (+) ───────────────────────────────────┘
                                   │
                                   ▼
                       Output Tensor (B, 4, s·H, s·W)
```

### Parameter Breakdown for $s = 2$ vs. $s = 3$

| Component | Layer Configuration | Parameters ($s = 2$) | Parameters ($s = 3$) | Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Shallow Head** | `Conv2d(4, 48, k=3, p=1)` | $1,776$ | $1,776$ | $0$ |
| **Residual Body** | 6 blocks $\times$ 2 convs `(48, 48, k=3)` | $249,408$ | $249,408$ | $0$ |
| **Trunk Conv** | `Conv2d(48, 48, k=3, p=1)` | $20,784$ | $20,784$ | $0$ |
| **Upsample Conv** | `Conv2d(48, 48 * s², k=3, p=1)` | **$83,136$** ($48 \times 192$) | **$187,056$** ($48 \times 432$) | **$+103,920$** |
| **PixelShuffle** | `PixelShuffle(s)` | $0$ | $0$ | $0$ |
| **Tail Conv** | `Conv2d(48, 4, k=3, p=1)` | $1,732$ | $1,732$ | $0$ |
| **Total Model Parameters** | — | **$356,836$** | **$460,756$** | **$+103,920$ (+29.1%)** |

- **Budget Compliance**: At **460,756 parameters**, the $3\times$ architecture occupies less than **31%** of the 1.5M parameter ceiling.
- **Implementation Status**: Natively supported by PyTorch `nn.PixelShuffle(3)`. Validated by empirical execution on CUDA:
  ```python
  x = torch.randn(2, 4, 32, 32, device='cuda')
  out = model_3x(x)  # Shape: torch.Size([2, 4, 96, 96])
  ```
  No architectural redesign, attention layers, or heavy components are required.

---

## 4. Hardware, VRAM, and Compute Feasibility

Using the NVIDIA GeForce RTX 4060 Laptop GPU (`cuda:0`, 8,188 MiB VRAM), peak memory and latency were directly benchmarked for forward + backward passes (AdamW optimizer, mixed-precision `torch.amp.autocast('cuda', dtype=torch.float16)`, batch size 16):

### Empirical VRAM Benchmark

| Configuration | Input LR Patch | Output HR Patch | Batch Size | Peak VRAM Allocated | % of RTX 4060 VRAM | Feasibility Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$2\times$ Baseline (exp005)** | $64 \times 64$ | $128 \times 128$ | 16 | **$178.79\text{ MiB}$** | $2.18\%$ | Fully verified |
| **$3\times$ Option A** | $32 \times 32$ | $96 \times 96$ | 16 | **$76.27\text{ MiB}$** | $0.93\%$ | **Negligible VRAM** |
| **$3\times$ Option B** | $48 \times 48$ | $144 \times 144$ | 16 | **$166.83\text{ MiB}$** | $2.04\%$ | **Negligible VRAM** |

- **Memory Assessment**: Both $3\times$ configurations consume less than $170\text{ MiB}$ of VRAM. There is zero risk of Out-Of-Memory (OOM) errors on the RTX 4060.
- **Inference Latency**: Benchmarked single-patch latency at $3\times$ ($32 \times 32 \rightarrow 96 \times 96$) is **$1.85\text{ ms} \pm 0.32\text{ ms}$** (>500 patches/sec), maintaining real-time telemetry performance.

---

## 5. Dataset Volume, Degradation Realism & Overfitting Risk

### 5.1 Dataset Volume Analysis
The repository currently stores one $512 \times 512$ 4-band Sentinel-2 L2A raster in `data/raw/`. Under Phase 5's leak-free geographic split (`split_mode="v2"`):
- Southeast Quadrant `(rows 256..512, cols 256..512)` is held out for validation.
- Remaining 3 quadrants are used for training.

When tiling with candidate $3\times$ patch sizes:

| HR Patch Size | LR Patch Size | Stride | Training Patches | Validation Patches | Total Patches |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$96 \times 96$** | $32 \times 32$ | 32 | **132** | **36** | 168 |
| **$120 \times 120$** | $40 \times 40$ | 32 | 115 | 36 | 151 |
| **$144 \times 144$** | $48 \times 48$ | 32 | 88 | 25 | 113 |

### 5.2 The $3\times$ Ill-Posedness Challenge
- In $2\times$ super-resolution, each LR pixel is expanded into $2 \times 2 = 4$ HR pixels ($1$ observed, $3$ synthesized: **$3:1$ synthesis ratio**).
- In $3\times$ super-resolution, each LR pixel is expanded into $3 \times 3 = 9$ HR pixels ($1$ observed, $8$ synthesized: **$8:1$ synthesis ratio**).
- Synthesizing $8$ unobserved pixels for every input pixel is significantly more ill-posed.
- Training a 460k-parameter model on only **132 training patches** extracted from a single scene creates a real risk of **texture memorization and edge overfitting**.
- To counteract this, training at $3\times$ must incorporate:
  1. Data augmentation (random D4 dihedral rotations and flips, which was tested in Phase 6 as `exp003`).
  2. Modest patch overlap (stride 24–32).
  3. Strict monitoring of validation loss to halt before overfitting.

---

## 6. The Degradation Paradigm for $3\times$: What is Actually Evaluated?

If Phase 11 trains a $3\times$ super-resolution model, what does the degrade-and-recover validation actually measure?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRAINING & VALIDATION PARADIGM                       │
├─────────────────────────────────────────────────────────────────────────┤
│ Ground Truth Target (HR):  10m Native Sentinel-2 Tile (96x96 pixels)    │
│                                    │                                    │
│                              3x Downsampling                            │
│                              PSF Optical Blur                           │
│                              Sensor Radiometric Noise                   │
│                                    │                                    │
│                                    ▼                                    │
│ Synthetic Input (LR):      30m Pseudo-LR Tile (32x32 pixels)            │
│                            (Equivalent to Landsat-8 GSD)                │
│                                    │                                    │
│                              3x ResidualSRNet                           │
│                                    │                                    │
│                                    ▼                                    │
│ Reconstructed Output (SR): 10m Pseudo-HR Tile (96x96 pixels)            │
│                            Evaluated against Ground Truth 10m           │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   INFERENCE / DEPLOYMENT PARADIGM                       │
├─────────────────────────────────────────────────────────────────────────┤
│ Real-World Input:          10m Native Sentinel-2 Tile                   │
│                                    │                                    │
│                              3x ResidualSRNet                           │
│                                    │                                    │
│                                    ▼                                    │
│ Reconstructed Output:      3.33m Nominal GSD Raster Grid                │
│                            (No sub-4m ground truth exists to validate!) │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **Validation Domain**: The model is trained and validated on **$30\text{ m} \rightarrow 10\text{ m}$ reconstruction**. Quantitative metrics (PSNR, SSIM, $\Delta$-NDVI) evaluate its ability to recover 10m Sentinel-2 from simulated 30m inputs.
2. **Deployment Domain**: When applied to native 10m Sentinel-2 inputs, the network projects features onto a **nominal $3.33\text{ m}$ grid**.
3. **Validation Gap**: Because no physical $3.33\text{ m}$ ground truth exists in the Sentinel-2 archive, the quality of the nominal $3.33\text{ m}$ output cannot be evaluated with PSNR or SSIM against true reality. It can only be evaluated via:
   - Consistency metrics (NDVI preservation against native 10m input).
   - Relative downstream building footprint agreement.
   - Empirical evidence indicators (ensemble disagreement, stability).

---

## 7. Go / No-Go Decision Framework for Phase 11

### Decision: CONDITIONAL GO

An experimental trial in Phase 11 is **scientifically justified and computationally feasible**, provided the following explicit constraints are maintained:

| Decision Criterion | Status | Justification |
| :--- | :---: | :--- |
| **Arithmetic Integrity** | **PASS** | $s = 3\times$ strictly achieves nominal GSD $= 3.33\text{ m} < 4.0\text{ m}$. |
| **Architecture Compatibility** | **PASS** | `ResidualSRNet` supports `PixelShuffle(3)` out-of-the-box (460,756 params). |
| **Hardware & VRAM** | **PASS** | Peak VRAM $< 170\text{ MiB}$ on RTX 4060; inference $< 2\text{ ms}$ per patch. |
| **Evaluation Honesty** | **CONDITIONAL** | Must report quantitative degrade-and-recover metrics on $30\text{ m} \rightarrow 10\text{ m}$ test set. |
| **Scientific Nomenclature** | **CONDITIONAL** | Must refer to results as **"nominal 3.33m grid"**, never as "validated physical sub-4m resolving power." |
| **Stop Condition Readiness** | **PASS** | If $3\times$ training exhibits severe spatial artifacts, loss divergence, or negative PSNR gain over $3\times$ Bicubic, report failure immediately. |

---

## 8. Proposed Specification for Phase 11 Experiment (`exp007_scale3x`)

If proceeding to Phase 11:
- **Scale Factor**: $s = 3$ ($\mathbf{3.333...\text{ m}}$ nominal GSD).
- **Architecture**: `ResidualSRNet(num_channels=4, num_features=48, num_blocks=6, scale_factor=3)`.
- **Patch Size**: HR $96 \times 96$, LR $32 \times 32$, stride 32 (yielding 132 train patches, 36 validation patches on clean Split v2).
- **Loss**: `CompoundSRLoss` with $L_1 + 0.20 \cdot \nabla_{\text{Sobel}}$.
- **Benchmark**: Direct comparison against **$3\times$ Bicubic upsampling** on the 36 held-out validation patches.
- **Primary Hypothesis to Test**: Does a 6-block residual network achieve a statistically significant PSNR/SSIM gain over $3\times$ Bicubic under $30\text{m} \rightarrow 10\text{m}$ degrade-and-recover validation?
