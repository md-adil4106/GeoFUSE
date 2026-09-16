# GeoFUSE SentinelGuard — Phase 2 Existing Training Pipeline Audit

**Date of Audit**: September 16, 2026  
**Auditor**: Antigravity Assistant (Google DeepMind)  
**Status**: Read-Only Pipeline Audit (Ground truth code inspection)  

---

## 1. Executive Summary & Pipeline Architecture

This document establishes the exact ground-truth specification of the existing training pipeline in GeoFUSE SentinelGuard as implemented in code. All values and behaviors described herein were extracted directly from the codebase without assumptions or placeholder estimates.

---

## 2. Super-Resolution Model Architecture

- **Source Code**: [`src/models/model.py:20-148`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/models/model.py#L20-L148)
- **Model Class**: `ResidualSRNet(nn.Module)`
- **Factory Function**: `build_model(config)` in [`src/models/model.py:125-147`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/models/model.py#L125-L147)

### 2.1 Topology & Layer Configuration
1. **Shallow Feature Extraction Head**:
   - `nn.Conv2d(in_channels=4, out_channels=48, kernel_size=3, stride=1, padding=1, bias=True)`
2. **Deep Residual Backbone**:
   - **4 Residual Blocks** (configured via `config.yaml:46: num_residual_blocks: 4`).
   - Each `ResidualBlock` contains:
     - `nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True)`
     - `nn.LeakyReLU(negative_slope=0.2, inplace=True)`
     - `nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True)`
     - Local residual addition with scaling factor `res_scale = 0.1`:
       $$x_{\text{out}} = x_{\text{in}} + 0.1 \times \text{body}(x_{\text{in}})$$
3. **Mid-Feature Trunk Conv**:
   - `nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True)`
   - Skip addition connecting feature trunk with shallow head features:
     $$f_{\text{trunk}} = \text{trunk}(f_{\text{res}}) + f_{\text{init}}$$
4. **Sub-Pixel Convolution Upsampling Head**:
   - `nn.Conv2d(48, 48 * (2^2) = 192, kernel_size=3, padding=1, bias=True)`
   - `nn.PixelShuffle(upscale_factor=2)` (reorganizes shape $(B, 192, H, W) \rightarrow (B, 48, 2H, 2W)$)
   - `nn.LeakyReLU(negative_slope=0.2, inplace=True)`
5. **Reconstruction Layer**:
   - `nn.Conv2d(48, out_channels=4, kernel_size=3, padding=1, bias=True)`
6. **Global Residual Skip (Bicubic Base Bypass)**:
   - Interpolates low-resolution input tensor directly to $2\times$ using bicubic interpolation:
     $$\text{base} = \text{F.interpolate}(x, \text{scale\_factor}=2.0, \text{mode}=\text{"bicubic"}, \text{align\_corners}=\text{False})$$
   - Final output combines interpolated baseline and learned high-frequency residual detail:
     $$\text{out} = \text{base} + \text{residual}_{\text{HR}}$$

### 2.2 Parameter Count
- **Trainable Parameters**: **`273,700`** parameters.
  - Computed via: `sum(p.numel() for p in model.parameters() if p.requires_grad)`.
- **Ensemble Total (3 Models)**: $3 \times 273,700 = \mathbf{821,100}$ parameters (~0.82M).
- **Compliance**: Well within the project budget ceiling of 1.5M parameters.
- **Architectural Constraints**: Zero GAN discriminators, zero self-attention modules, zero transformer blocks.

---

## 3. Loss Functions & Formulation

- **Source Code**: [`src/models/loss.py:15-112`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/models/loss.py#L15-L112)
- **Class**: `CompoundSRLoss(nn.Module)`

### 3.1 Mathematical Formulation
The loss function combines a pixel-wise $\ell_1$ radiometric loss with a multi-channel spatial gradient alignment loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{L1}} + \lambda_{\text{grad}} \cdot \mathcal{L}_{\text{grad}}$$

- **Hyperparameters & Weights**:
  - $\lambda_{\text{L1}} = 1.0$
  - $\lambda_{\text{grad}} = 0.1$ (`grad_weight = 0.1` in `src/models/loss.py:81`)

### 3.2 Component Loss Definitions
1. **Radiometric $\ell_1$ Loss**:
   $$\mathcal{L}_{\text{L1}}(I_{\text{SR}}, I_{\text{HR}}) = \frac{1}{B \cdot C \cdot H \cdot W} \sum_{b, c, y, x} \left| I_{\text{SR}}(b, c, y, x) - I_{\text{HR}}(b, c, y, x) \right|$$
2. **Sobel Gradient Loss (`SobelGradientLoss`)**:
   - Penalizes blurred edge transitions across high-frequency spectral boundaries:
     $$\mathcal{L}_{\text{grad}}(I_{\text{SR}}, I_{\text{HR}}) = \frac{1}{B \cdot C \cdot H \cdot W} \left( \|\nabla_x I_{\text{SR}} - \nabla_x I_{\text{HR}}\|_1 + \|\nabla_y I_{\text{SR}} - \nabla_y I_{\text{HR}}\|_1 \right)$$
   - Implemented as depthwise separable 2D convolutions (`groups = channels = 4`) with replication padding using fixed $3\times 3$ Sobel kernels:
     $$K_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

---

## 4. Optimization & Training Schedule

- **Source Code**: [`src/models/train.py:130-285`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/models/train.py#L130-L285)
- **Configuration**: [`config.yaml:49-64`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/config.yaml#L49-L64)

| Parameter | Exact Value in Code / Config | Code Location |
| :--- | :--- | :--- |
| **Optimizer** | `torch.optim.AdamW` | `src/models/train.py:183` |
| **Base Learning Rate ($\eta_0$)** | `0.0005` ($5 \times 10^{-4}$) | `config.yaml:52`, `src/models/train.py:151` |
| **Weight Decay** | `0.0001` ($1 \times 10^{-4}$) | `config.yaml:53`, `src/models/train.py:183` |
| **AdamW Betas** | `(0.9, 0.999)` (PyTorch default) | `src/models/train.py:183` |
| **AdamW Epsilon** | `1e-8` (PyTorch default) | `src/models/train.py:183` |
| **Learning Rate Scheduler** | `CosineAnnealingLR` | `src/models/train.py:184` |
| **Scheduler $T_{\max}$** | `15` (matches total epochs) | `src/models/train.py:184` |
| **Scheduler $\eta_{\min}$** | `1e-6` | `src/models/train.py:184` |
| **Scheduler Step** | Per epoch (after validation loop) | `src/models/train.py:236` |
| **Epoch Count** | `15` epochs | `config.yaml:51`, `src/models/train.py:149` |
| **Batch Size** | `16` | `config.yaml:50`, `src/models/train.py:150` |
| **OOM Recovery Strategy** | Automatic batch size halving down to $\min=4$ | `src/models/train.py:218-226` |
| **Mixed Precision** | `torch.amp.autocast(device_type="cuda")` with `torch.amp.GradScaler("cuda")` | `src/models/train.py:58, 65, 185` |

---

## 5. Dataset, Degradation & Augmentation Pipeline

- **Source Code**: [`src/data/dataset.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/data/dataset.py), [`src/data/degrade.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/data/degrade.py)
- **Dataset Class**: `SentinelSRDataset(Dataset)`

### 5.1 Patch Dimensions & Tiling
- **Raw Scene Dimensions**: $512 \times 512 \times 4$ pixels (`EPSG:32643`, 10m GSD).
- **High-Resolution (HR) Patch Size**: $128 \times 128 \times 4$ pixels (`patch_size_hr = 128`).
- **Low-Resolution (LR) Patch Size**: $64 \times 64 \times 4$ pixels (derived from $2\times$ downsampling).
- **Tiling Stride**: `stride = 32` pixels (`src/models/train.py:153`).
- **Total Candidate Grid Patches**: 169 patches ($13 \times 13$ spatial grid).
- **Partition Counts**:
  - **Train Patches**: **120**
  - **Validation Patches**: **49**

### 5.2 Synthetic Degradation Pipeline (`synthesize_pseudo_lr`)
Generates paired LR tiles on-the-fly via three physical degradation steps:
1. **Sensor Optical PSF Blur**:
   - `cv2.GaussianBlur(hr, (3, 3), sigmaX=0.5, sigmaY=0.5)` (`kernel_size=3, sigma=0.5`).
2. **Spatial Sub-Sampling (Coarser Ground Resolution)**:
   - `cv2.resize(blurred, (64, 64), interpolation=cv2.INTER_CUBIC)`.
3. **Radiometric Sensor Noise**:
   - Zero-mean Gaussian noise with standard deviation `noise_std = 0.01` added to normalized reflectance, clipped to $[0.0, \infty)$.
   - Train split: stochastic noise (`seed=None`).
   - Val split: deterministic noise seeded per patch index (`seed = base_seed + idx`).

### 5.3 Augmentation Operations
- **Finding**: **No data augmentations are performed in the training pipeline.**
  - Zero geometric transformations: No random horizontal flips, no random vertical flips, no 90-degree rotations.
  - Zero photometric transformations: No random color jitter, no contrast scaling, no cutout/mixup.
  - The dataset strictly feeds unaugmented cropped patches to the model.

---

## 6. Train/Validation Split Logic & Spatial Leakage Audit

### 6.1 Split Mechanism
- Partitioning is defined by an intended geographic quadrant hold-out:
  - Validation Quadrant: Southeast Quadrant `val_quadrant = (256, 512, 256, 512)` on the $512 \times 512$ scene.
- Patch Assignment Logic ([`src/data/dataset.py:68-77`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/data/dataset.py#L68-L77)):
  ```python
  patch_center_y = y + patch_size_hr // 2
  patch_center_x = x + patch_size_hr // 2

  in_val_zone = (
      val_r_min <= patch_center_y < val_r_max
      and val_c_min <= patch_center_x < val_c_max
  )
  if (split == "val" and in_val_zone) or (split == "train" and not in_val_zone):
      self.patches.append(...)
  ```

### 6.2 Critical Finding: Presence of Spatial Data Leakage
> [!WARNING]
> **Spatial Data Leakage Detected**:
> 1. **Center-Point Splitting Defect**: Assignment to train vs. validation is decided solely by the patch *center coordinate* (`patch_center_y`, `patch_center_x`), **not** by the patch bounding box.
> 2. **Physical Boundary Overhang**:
>    - Patches have size $128 \times 128$ pixels.
>    - A training patch whose center is at $y = 224$ ($< 256$) begins at $y = 160$ and extends to $y = 288$.
>    - Its bottom 32 rows ($y \in [256, 288]$) physically lie **inside** the validation quadrant $[256:512, 256:512]$.
> 3. **Quantified Overlap**:
>    - **15 out of 120 training patches (12.5%)** physically intersect and overlap with the geographic validation zone.
>    - Furthermore, with a small extraction stride of $32$ pixels on $128 \times 128$ patches, neighboring patches overlap with each other by **$96$ pixels ($75\%$ spatial overlap)**.
>    - As a result, the boundary training patches and validation patches observe identical surface textures and topographic features from the same geographic region.
> 4. **Remediation Target**: This spatial autocorrelation leakage must be eliminated in Phase 5 by enforcing an explicit guard buffer ($\ge 128\text{ px}$) between train and validation partitions or adopting completely disjoint non-overlapping scenes.

---

## 7. Ensemble & Random Seed Handling

- **Source Code**: [`src/models/train.py:142-146, 295-344`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/models/train.py#L142-L146)
- **Configuration**: [`config.yaml:59-64`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/config.yaml#L59-L64)
- **Ensemble Architecture**: 3-member ensemble trained sequentially to maintain lightweight memory footprint on host GPU.
- **Member Seed Assignment**:
  - Member 0: `seed = 42`
  - Member 1: `seed = 101`
  - Member 2: `seed = 2024`
- **RNG Seeding per Member**:
  ```python
  torch.manual_seed(seed)
  np.random.seed(seed)
  if device.type == "cuda":
      torch.cuda.manual_seed_all(seed)
  ```
- **DataLoader Worker Determinism**: DataLoaders use default single-process fetching (`num_workers=0` in DataLoader constructor within `src/models/train.py:174-175`).

---

## 8. Checkpoint Format & Save Triggers

- **Source Code**: [`src/models/train.py:249-266`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/models/train.py#L249-L266)
- **Destination Path**: `outputs/checkpoints/ensemble_member_{member_idx}.pth`
- **Save Trigger**: Saved strictly when total validation loss improves:
  $$\text{val\_total} < \text{best\_val\_loss}$$
  *(Initialized at $\text{best\_val\_loss} = \infty$).*
- **Saved Payload Structure**:
  ```python
  {
      "member_idx": member_idx,
      "seed": seed,
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "val_loss": val_m["val_total"],
      "val_psnr": val_m["val_psnr"],
      "val_ssim": val_m["val_ssim"],
      "config": config,
  }
  ```
- **Metrics Log Output**: Appended epoch-by-epoch to `outputs/checkpoints/training_log_member_{member_idx}.csv` with columns:
  `epoch, train_l1, train_grad, train_total, val_l1, val_grad, val_total, val_psnr, val_ssim, lr, epoch_time_s`.
