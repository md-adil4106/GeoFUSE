# Phase 8 Audit: Composite Evidence Score Formula & Pipeline Analysis

**Document Status**: Official Repository Audit Artifact  
**Repository**: `GeoFUSE SentinelGuard`  
**Target Code**: [`src/evaluation/fusion.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/evaluation/fusion.py), [`src/models/ensemble.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/models/ensemble.py), [`src/evaluation/stability.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/evaluation/stability.py), [`src/evaluation/spectral_check.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/evaluation/spectral_check.py), [`src/evaluation/edge_check.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/evaluation/edge_check.py), [`src/dashboard/app.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/dashboard/app.py), [`config.yaml`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/config.yaml)  
**Date**: September 2026  

---

## 1. Executive Summary & Foundational Scientific Mandate

> [!CAUTION]
> ### Foundational Scientific Mandate
> The Composite Evidence Score is an **empirical heuristic multi-criteria reliability proxy**, **NOT a calibrated Bayesian posterior probability, conformal prediction guarantee, or certainty certificate of correctness.**
> 
> A score of $86.75 / 100$ does **not** denote an $86.75\%$ probability that the super-resolved pixels are accurate. It indicates an ad-hoc linear combination of four separately min-max normalized heuristic signals representing parameter disagreement, perturbation sensitivity, spectral index divergence, and edge gradient magnitude discrepancy.

---

## 2. Complete Mathematical Formulation & Aggregation Logic

The score is generated through a 6-stage pipeline executing in [`fuse_trust_risk_maps(...)`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/evaluation/fusion.py#L51-L213):

```
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│  Ensemble Disagreement Map     │   │  Input Stability Map            │
│  (std across 3 models)          │   │  (variance across noise trials) │
└───────────────┬─────────────────┘   └────────────────┬────────────────┘
                │ Min-Max Tile Norm                    │ Min-Max Tile Norm
                ▼                                      ▼
     [norm_disag ∈ [0, 1]]                  [norm_stab ∈ [0, 1]]
                │                                      │
                ├─────────────── (w₁ = 0.25) ──────────┤
                │                                      │
┌───────────────┴─────────────────┐   ┌────────────────┴────────────────┐
│  Spectral Delta-NDVI Map        │   │  Structural Gradient Diff Map   │
│  (|NDVI_sr - NDVI_ref|)         │   │  (|∇_sr - ∇_ref|)               │
└───────────────┬─────────────────┘   └────────────────┬────────────────┘
                │ Min-Max Tile Norm                    │ Min-Max Tile Norm
                ▼                                      ▼
     [norm_spec ∈ [0, 1]]                   [norm_struct ∈ [0, 1]]
                │                                      │
                └─────────────── (w₃ = 0.25) ──────────┘
                                      │
                                      ▼
                     Weighted Composite Risk Map (0 to 1)
                     Risk(u, v) = ∑ wᵢ · norm_signalᵢ(u, v)
                                      │
                                      ▼
                     Complementary Trust Map (0 to 1)
                     Trust(u, v) = 1.0 - Risk(u, v)
                                      │
                                      ▼
                     Spatial Average Trust Score (%)
                     Score = round(100.0 · mean(Trust), 2)
```

### Mathematical Equations

1. **Per-Pixel Risk Map**:
   $$\text{Risk}(u, v) = \text{clip}\left( \sum_{i \in \mathcal{S}} w_i \cdot \widetilde{S}_i(u, v), \, 0.0, \, 1.0 \right)$$
   where $\mathcal{S} = \{\text{disagreement}, \, \text{stability}, \, \text{spectral}, \, \text{structural}\}$, and $\sum w_i = 1.0$.

2. **Per-Pixel Trust Map**:
   $$\text{Trust}(u, v) = 1.0 - \text{Risk}(u, v)$$

3. **Composite Evidence Score (Scalar Percentage)**:
   $$\text{Evidence\_Score} = 100.0 \times \left( \frac{1}{H \cdot W} \sum_{u=1}^H \sum_{v=1}^W \text{Trust}(u, v) \right)$$

---

## 3. Component Signals: Extraction, Definitions, and Formulas

### Signal 1: Ensemble Disagreement Map ($\sigma_{\text{ensemble}}$)
- **Source Code**: [`src/models/ensemble.py:150-152`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/models/ensemble.py#L150-L152)
- **Raw Definition**: Given $M$ ensemble members predicting output tensors $Y_m \in \mathbb{R}^{H \times W \times C}$ from low-resolution input $X$:
  $$\sigma_c(u, v) = \sqrt{\frac{1}{M} \sum_{m=1}^M \left(Y_{m, c}(u, v) - \bar{Y}_c(u, v)\right)^2}$$
  $$\text{Disagreement}(u, v) = \frac{1}{C} \sum_{c=1}^C \sigma_c(u, v)$$
- **Physical Meaning**: Epistemic uncertainty proxy reflecting divergence across local minima reached by different random seeds and patch mini-batch trajectories.
- **Typical Raw Values**: Mean $\approx 0.0015$, Max $\approx 0.0110$ (reflectance units $\in [0, 1]$).

### Signal 2: Input-Perturbation Stability Map ($\text{Var}_{\text{pert}}$)
- **Source Code**: [`src/evaluation/stability.py:117-121`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/evaluation/stability.py#L117-L121)
- **Raw Definition**: Given unperturbed input $X$ and $K$ perturbed inputs $X^{(k)}$ generated by additive Gaussian noise $\mathcal{N}(0, \sigma_{\text{noise}}^2)$ and multiplicative brightness jitter $(1 + \mathcal{N}(0, \sigma_{\text{jitter}}^2))$:
  $$\text{Var}_c(u, v) = \frac{1}{K+1} \sum_{k=0}^K \left( \bar{Y}_c^{(k)}(u, v) - \langle \bar{Y}_c \rangle(u, v) \right)^2$$
  $$\text{Stability}(u, v) = \frac{1}{C} \sum_{c=1}^C \text{Var}_c(u, v)$$
- **Physical Meaning**: Empirical sensitivity to input radiometric noise.
- **Typical Raw Values**: Mean $\approx 0.0005$, Max $\approx 0.0024$.

### Signal 3: Spectral NDVI Inconsistency Map ($\Delta \text{NDVI}$)
- **Source Code**: [`src/evaluation/spectral_check.py:96`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/evaluation/spectral_check.py#L96)
- **Raw Definition**:
  $$\text{NDVI}(u, v) = \frac{\text{NIR}(u, v) - \text{Red}(u, v)}{\text{NIR}(u, v) + \text{Red}(u, v) + \epsilon}$$
  $$\Delta \text{NDVI}(u, v) = \left| \text{NDVI}_{\text{sr}}(u, v) - \text{NDVI}_{\text{ref}}(u, v) \right|$$
  where $\text{NIR} = \text{Band 8}$ (channel 3) and $\text{Red} = \text{Band 4}$ (channel 2).
- **Physical Meaning**: Radiometric distortion in vegetative biophysical index.
- **Typical Raw Values**: Mean $\approx 0.0214$, Max $\approx 0.1650$.

### Signal 4: Structural Gradient Difference Map ($|\Delta \nabla|$)
- **Source Code**: [`src/evaluation/edge_check.py:55-66`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/evaluation/edge_check.py#L55-L66), [`src/dashboard/app.py:202`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/dashboard/app.py#L202)
- **Raw Definition**: Grayscale conversion $I = 0.299 R + 0.587 G + 0.114 B$. Sobel gradient magnitude:
  $$\nabla I(u, v) = \sqrt{(I * S_x)^2 + (I * S_y)^2}$$
  $$\text{Structural\_Diff}(u, v) = \left| \nabla I_{\text{sr}}(u, v) - \nabla I_{\text{ref}}(u, v) \right|$$
- **Physical Meaning**: High-frequency boundary distortion and edge displacement.
- **Typical Raw Values**: Mean $\approx 0.0327$, Max $\approx 0.3611$.

---

## 4. Normalization Mechanics: The Per-Tile Min-Max Trap

Every evidence stream is normalized by [`normalize_evidence_map(...)`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/evaluation/fusion.py#L24-L48):

```python
def normalize_evidence_map(map_2d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    clean_map = np.nan_to_num(map_2d, nan=0.0, posinf=1.0, neginf=0.0)
    v_min = float(np.min(clean_map))
    v_max = float(np.max(clean_map))
    rng = v_max - v_min
    if rng > eps:
        norm = (clean_map - v_min) / rng
    else:
        norm = np.zeros_like(clean_map, dtype=np.float32)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)
```

$$\widetilde{S}_i(u, v) = \frac{S_i(u, v) - \min_{(u, v)} S_i(u, v)}{\max_{(u, v)} S_i(u, v) - \min_{(u, v)} S_i(u, v) + \epsilon}$$

### Critical Flaws of Per-Tile Min-Max Scaling

1. **Destruction of Absolute Error Scale**:
   - Because normalization is computed independently per-tile, the absolute magnitude of error is discarded.
   - If Patch A is pristine with raw disagreement $\in [0.0001, 0.0005]$, its worst pixel is scaled to $1.0$ (maximum risk).
   - If Patch B is heavily distorted with raw disagreement $\in [0.05, 0.25]$, its best pixel is scaled to $0.0$ (zero risk).
   - Both patches end up with nearly identical normalized spatial means ($\sim 0.15$), yielding virtually identical Composite Evidence Scores (~85.0%).
2. **Artificial Dynamic Range Compression**:
   - Across all 25 held-out validation patches of Split v2, the standard deviation of `trust_score_pct` is **only 0.60%** (Mean: $84.99\%$, Min: $83.78\%$, Max: $86.47\%$). The entire variation across 25 scenes spans just 2.69 percentage points!
3. **Outlier Skew**:
   - A single extreme edge artifact or anomalous pixel elevates $v_{\max}$, suppressing the normalized values of the rest of the tile.

---

## 5. Weights & Attribution: The Equal-Weight Assumption

From [`config.yaml:86-90`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/config.yaml#L86-L90):
```yaml
weights:
  disagreement: 0.25   # Uncertainty proxy from ensemble variance
  stability: 0.25      # Input-perturbation sensitivity
  spectral: 0.25       # Radiometric/NDVI consistency
  structural: 0.25     # High-frequency edge/gradient consistency
```

- **Are the weights justified?** **No. They are completely arbitrary equal weights ($25\%$ each).**
- There is no regression against ground truth error, no Bayesian parameter estimation, no calibration on validation loss, and no stakeholder utility optimization.
- The weights were selected purely as an unweighted linear average.

---

## 6. Numerical Dominance Analysis

Across the 25 held-out validation patches of Split v2, the average normalized values and their risk contributions are:

| Evidence Signal | Raw Mean | Raw Max | Normalized Mean ($\widetilde{S}_i$) | Std of $\widetilde{S}_i$ | Weight ($w_i$) | Risk Contribution ($w_i \cdot \widetilde{S}_i$) | Share of Total Risk |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Perturbation Stability** | $0.00051$ | $0.00237$ | **$0.1890$** | $0.0008$ | $0.25$ | **$0.0472$** | **31.5%** (Dominant) |
| **Ensemble Disagreement** | $0.00155$ | $0.01104$ | **$0.1585$** | $0.0151$ | $0.25$ | **$0.0396$** | **26.4%** |
| **Spectral Inconsistency ($\Delta$-NDVI)** | $0.02143$ | $0.15143$ | **$0.1480$** | $0.0136$ | $0.25$ | **$0.0370$** | **24.7%** |
| **Structural Difference ($|\Delta \nabla|$)** | $0.03278$ | $0.36914$ | **$0.1047$** | $0.0165$ | $0.25$ | **$0.0262$** | **17.4%** (Suppressed) |
| **Composite Total** | — | — | — | — | **1.00** | **$0.1501$** | **100.0%** |

### Key Findings on Dominance
1. **Stability dominates the risk penalty (31.5%)**:
   Because output variance to Gaussian noise has a continuous bell-shaped distribution, min-max scaling yields a normalized spatial mean of $\approx 0.189$. As a result, synthetic perturbation noise consistently penalizes the trust score more than any other signal.
2. **Structural error is numerically suppressed (17.4%)**:
   Edges and gradient boundaries are spatially sparse. Most pixels in a tile are smooth background with $\nabla \approx 0$. Even when edge errors are sharp, their area-weighted spatial mean is low ($\approx 0.105$). Thus, structural errors have the smallest voice in the final scalar score.

---

## 7. Redundancy & Correlation Matrix

The mean spatial Pearson correlation matrix across all 25 validation patches:

| Correlation ($r$) | Disagreement | Stability | Spectral ($\Delta$-NDVI) | Structural ($|\Delta \nabla|$) |
| :--- | :---: | :---: | :---: | :---: |
| **Disagreement** | $1.0000$ | $+0.0501$ | $+0.0251$ | $+0.0918$ |
| **Stability** | $+0.0501$ | $1.0000$ | $+0.0099$ | $-0.0018$ |
| **Spectral ($\Delta$-NDVI)** | $+0.0251$ | $+0.0099$ | $1.0000$ | $+0.0409$ |
| **Structural ($|\Delta \nabla|$)** | $+0.0918$ | $-0.0018$ | $+0.0409$ | $1.0000$ |

### Scientific Interpretation
- All inter-signal correlations are $|r| < 0.10$.
- The 4 components are **statistically orthogonal (non-redundant)**.
- However, they are orthogonal for the *wrong reason*: `Stability` has $r \approx 0.00$ with true spectral and structural errors because it measures the network's reaction to synthetic Gaussian white noise, which does not reflect the scene's real radiometric or geometric difficulty.

---

## 8. Threshold Audit & "Magic Numbers"

The pipeline contains multiple hardcoded thresholds across [`config.yaml`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/config.yaml) and Python functions:

### 1. `high_risk_threshold: 0.65`
- **Location**: [`src/evaluation/fusion.py:57`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/evaluation/fusion.py#L57), [`config.yaml:92`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/config.yaml#L92)
- **Role**: Pixels with $\text{Risk}(u, v) \ge 0.65$ are classified as high risk (`pct_high_risk_pixels`).
- **Empirical Reality**: Across all 4 demo tiles and all 25 validation patches, `pct_high_risk_pixels` is **$0.00\%$**.
- **Assessment**: Inoperative / dead threshold. Because the average risk is $\sim 0.15$ and signals are orthogonal, virtually no pixel ever simultaneously spikes across all 4 streams to reach $0.65$.

### 2. `min_trust_score_threshold: 86.5`
- **Location**: [`config.yaml:115`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/config.yaml#L115)
- **Role**: If $\text{Score} < 86.5\%$, the UI and Trust Receipt trigger an **"Operational Advisory"**.
- **Historical Derivation**: Hand-tuned against the 4 offline demo tiles:
  - Tile 0: $86.75\%$ (Pass)
  - Tile 8: $86.70\%$ (Pass)
  - Tile 16: $86.41\%$ (Advisory)
  - Tile 24: $86.02\%$ (Advisory)
- **Empirical Reality under Retrained Ensemble (Split v2)**:
  - Across the 25 held-out validation patches, the mean score is **$84.99\%$** and the max score is **$86.47\%$**.
  - Under the $86.5\%$ threshold, **$100\%$ of validation patches are flagged as Operational Advisories**!
- **Assessment**: Undocumented magic number with zero statistical justification.

### 3. `trust_partition_threshold: 0.85`
- **Location**: [`config.yaml:105`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/config.yaml#L105)
- **Role**: Partitions tiles into high-trust ($\text{Trust} \ge 0.85$, i.e. $\text{Risk} \le 0.15$) and low-trust regions for downstream building footprint comparison.
- **Empirical Reality**: Since mean trust is $\approx 0.850$, this arbitrary threshold simply bifurcates every tile down the middle ($\sim 50\%-70\%$ high trust).

### 4. `ndvi_threshold: 0.05`
- **Location**: [`config.yaml:78`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/config.yaml#L78)
- **Role**: Flags pixels with $\Delta \text{NDVI} > 0.05$ as spectrally inconsistent.
- **Assessment**: Standard remote-sensing heuristic for vegetation index deviation, reasonable as a rule of thumb.

---

## 9. Silent Ground Truth Substitution in Live Upload Mode

A critical discrepancy exists between offline evaluation and live user upload mode:

### Offline Evaluation Mode
- Ground truth tile ($10\text{m}$ HR) is known:
  $$\Delta \text{NDVI} = |\text{NDVI}_{\text{sr}} - \text{NDVI}_{\text{gt}}|$$
  $$\text{Structural\_Diff} = |\nabla_{\text{sr}} - \nabla_{\text{gt}}|$$

### Live Upload Mode ([`src/dashboard/app.py:346, 355`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/dashboard/app.py#L346-L355))
- Independent ground truth is **not available**. The system silently substitutes the upsampled low-resolution image (`bicubic_tile`) as the reference:
  ```python
  spectral_metrics = compute_spectral_consistency(
      gt_tile=bicubic_tile,  # <-- BICUBIC SUBSTITUTED FOR GROUND TRUTH
      sr_tile=sr_tile,
      red_idx=red_idx,
      nir_idx=nir_idx,
  )
  structural_diff = np.abs(grad_sr - grad_bic)  # <-- MEASURES DEPARTURE FROM BICUBIC
  ```
- **Consequence**:
  - In upload mode, the formula does **not** evaluate error against true high-resolution reality.
  - It evaluates **departure from bicubic upsampling**.
  - If the super-resolution model generates sharp, realistic high-frequency edges where bicubic was blurry, $\text{Structural\_Diff}$ increases, and the **model is penalized with higher risk and a lower Trust Score** for actually doing its job.

---

## 10. Calibration Against Real Outcomes

- **Calibrated probabilities**: **Zero.** There is no Platt scaling, isotonic regression, beta calibration, or temperature scaling.
- **Conformal prediction guarantees**: **Zero.** No error coverage intervals or bounded risk sets.
- **Empirical outcome validation**: **Zero.** No verification that high-trust regions actually yield higher building footprint accuracy or land-cover classification accuracy in field-verified datasets.

---

## 11. Recommendations for Phase 9 (Evidence Calibration & Reform)

1. **Replace Min-Max Per-Tile Normalization**:
   Use global, dataset-level normalization percentiles (e.g. 1st and 99th percentiles computed over the training corpus) so that clean tiles actually receive higher scores than noisy/corrupted tiles.
2. **De-weight or Redesign Perturbation Stability**:
   Either reduce $w_{\text{stability}}$ from $0.25$ to a lower fraction (e.g. $0.10$), or test structured adversarial perturbations rather than uninformative Gaussian sensor noise.
3. **Resolve Upload Mode Reference Trap**:
   In reference-free mode, replace $|\nabla_{\text{sr}} - \nabla_{\text{bicubic}}|$ with a true reference-free sharpness/blurriness metric (e.g. Tenengrad / Laplacian energy gradient score) or local variance metric, rather than penalizing divergence from bicubic.
4. **Recalibrate Operational Advisory Threshold**:
   Replace the hardcoded magic number $86.5$ with a data-driven percentile threshold derived from the empirical distribution on held-out validation data (e.g. 5th percentile of validation patches).
