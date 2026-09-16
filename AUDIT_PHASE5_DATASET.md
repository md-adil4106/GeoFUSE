# GeoFUSE SentinelGuard — Phase 5 Dataset & Degradation Audit

**Date**: September 16, 2026  
**Status**: Leakage Eliminated & Automated Test Verified  
**Artifacts**: [`data/split_config_v2.json`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/data/split_config_v2.json), [`tests/test_no_patch_leakage.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/tests/test_no_patch_leakage.py)  

---

## 1. Executive Summary

During the Phase 2 training pipeline audit, a critical flaw in train/validation partitioning was discovered: **center-point based quadrant splitting** permitted 128x128 pixel patches whose geometric centers fell outside the validation quadrant to extend up to 64 pixels inside the validation quadrant, causing physical pixel overlap and spatial autocorrelation leakage.

In **Phase 5**, we:
1. Formulated and implemented **Split v2**, a strictly disjoint, boundary-constrained spatial partitioning algorithm that mathematically and empirically guarantees **zero spatial pixel overlap** between training and validation sets.
2. Formulated [`data/split_config_v2.json`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/data/split_config_v2.json), defining exact scene coordinates, bounding boxes, patch yields, and guard buffer options.
3. Created an automated regression test suite ([`tests/test_no_patch_leakage.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/tests/test_no_patch_leakage.py)) proving zero patch overlap.
4. Preserved exact backward compatibility (`split_mode="v1"`) within [`src/data/dataset.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/data/dataset.py) so all Phase 3 and Phase 4 baseline numbers remain reproducible bit-for-bit.
5. Documented the physical optics and degradation rationale behind the sensor PSF blur ($\sigma=0.5$).

---

## 2. Spatial Data Leakage: Defect Diagnosis (Split v1)

### 2.1 The Defect Mechanism
The original Sentinel-2 scene stack in `data/raw/` has dimensions $512 \times 512 \times 4$. Patches are extracted with $H=128, W=128$ and a sliding stride of $32$ pixels ($13 \times 13 = 169$ candidate origins).

In the legacy split (`v1`):
- Southeast Quadrant was intended for validation: `val_quadrant = (256, 512, 256, 512)`.
- However, assignment was decided solely on patch center coordinates:
  $$\text{in\_val\_zone} = (256 \le y + 64 < 512) \land (256 \le x + 64 < 512)$$
- This condition evaluated true for any patch with top-left origin $y \ge 192$ and $x \ge 192$.
- Consequently:
  - Validation patches spanned rows $[192, 512]$ and columns $[192, 512]$ (7x7 = 49 patches).
  - Training patches with center $y < 256$ (e.g. $y = 160$, center = 224) covered rows $[160, 288]$ and intersected the validation zone across rows $[192, 288]$.

### 2.2 Quantified Leakage Impact
- **Training Patches Overlapping Validation**: **51 out of 120 (42.5%)**
- **Total Intersecting (Train, Val) Pairs**: **480 intersecting pairs**
- **Maximum Pixel Overlap per Pair**: $96 \times 128 = 12,288$ pixels ($75\%$ spatial overlap)

---

## 3. The Remediated Split Architecture (Split v2)

### 3.1 Mathematical Formulation of Disjoint Partitioning
Split v2 replaces center-point testing with **strict 2D bounding box containment**:

1. **Validation Set ($\mathcal{V}$)**:
   A candidate patch $[y, y + S] \times [x, x + S]$ belongs to validation if and only if it is **strictly contained** inside the validation bounding box $[R_{\min}, R_{\max}] \times [C_{\min}, C_{\max}]$:
   $$\mathcal{V} = \{ (y, x) \mid y \ge R_{\min} \land y + S \le R_{\max} \land x \ge C_{\min} \land x + S \le C_{\max} \}$$

2. **Training Set ($\mathcal{T}$)**:
   A candidate patch belongs to training if and only if it is **strictly disjoint** from the validation bounding box (with optional guard buffer $B \ge 0$):
   $$\mathcal{T} = \{ (y, x) \mid (y + S \le R_{\min} - B) \lor (x + S \le C_{\min} - B) \lor (y \ge R_{\max} + B) \lor (x \ge C_{\max} + B) \}$$

Because any training patch satisfies $(y + S \le 256) \lor (x + S \le 256)$, and every validation patch satisfies $(y \ge 256) \land (x \ge 256)$, their 2D intersection is:
$$\forall t \in \mathcal{T}, \forall v \in \mathcal{V} \implies \operatorname{Area}(t \cap v) = 0$$

### 3.2 Partition Boundaries and Patch Yields

| Split Configuration | Guard Buffer | Train Patches | Val Patches | Total Patches | Overlapping Pairs | Min Boundary Separation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Split v1 (Legacy)** | N/A | 120 | 49 | 169 | **480 (42.5% leakage)** | 0 px (overlapping) |
| **Split v2 (Contiguous)** | **0 px** | **105** | **25** | **130** | **0 (Zero Leakage)** | **0 px (contact at seam)** |
| **Split v2 (Buffered)** | **32 px** | **88** | **25** | **113** | **0 (Zero Leakage)** | **32 px (320m guard band)** |

Both Split v2 variants provide sufficient data volume for training while completely eliminating spatial contamination.

---

## 4. Degradation Realism & Physical Sensor Optics

### 4.1 Optical Point Spread Function (PSF) Modeling
The synthetic degradation pipeline in [`src/data/degrade.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/data/degrade.py) applies a Gaussian blur ($\text{kernel} = 3\times 3, \sigma = 0.5$) before $2\times$ downsampling.

**Physical Remote Sensing Rationale**:
- For Sentinel-2 MSI 10m bands (B02, B03, B04, B08), the optical telescope diffraction and detector aperture sampling correspond to a spatial Full Width at Half Maximum (FWHM) of approximately $1.1$ to $1.2$ pixels.
- Under Gaussian optics:
  $$\text{FWHM} = 2\sqrt{2\ln 2} \cdot \sigma \approx 2.355 \, \sigma \implies \sigma \approx \frac{1.2}{2.355} \approx 0.51 \text{ pixels}$$
- Setting $\sigma = 0.5$ on a $3\times 3$ kernel directly mirrors this optical MTF (Modulation Transfer Function) roll-off.
- Testing larger blurs ($\sigma \ge 1.0$) simulates severe atmospheric defocusing or uncalibrated optics rather than nominal satellite imaging.
- Therefore, the $\sigma=0.5$ PSF parameter represents the optimal physical balance for 2x super-resolution modeling on Sentinel-2 data.

### 4.2 Scientific Integrity Declaration
> [!IMPORTANT]
> **Synthetic Degrade-and-Recover Setup**:
> All experiments in GeoFUSE remain synthetic degrade-and-recover pipelines. High-resolution reference tiles are pre-degradation Sentinel-2 L2A tiles, not independently acquired airborne or sub-meter satellite imagery. This distinction is preserved in all user-facing documentation and UI readouts.

---

## 5. Automated Verification & Test Results

All 5 tests in [`tests/test_no_patch_leakage.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/tests/test_no_patch_leakage.py) passed:
- `test_v2_split_zero_patch_overlap`: PASSED (0 overlapping pairs).
- `test_v2_split_with_guard_buffer`: PASSED ($\ge 32\text{ px}$ boundary separation).
- `test_v1_split_backward_compatibility`: PASSED (reproduces 120/49 patches).
- `test_split_config_v2_json_validity`: PASSED (valid JSON schema).
- `test_real_sentinel2_stack_leak_free`: PASSED (0 overlapping pairs on real data).

Total test suite: **98 passed tests across 19 modules**.
