# GeoFUSE SentinelGuard — Phase 0 Repository & Environment Audit

**Date of Audit**: September 16, 2026  
**Auditor**: Antigravity Assistant (Google DeepMind)  
**Execution Type**: Read-Only Ground Truth Audit (No modifications to codebase or pipelines)  

---

## 1. Executive Summary & Environment Ground Truth

### 1.1 Python & Compute Runtime
- **Operating System**: Windows 11 (AMD64 architecture)
- **Python Version**: `3.12.10 (tags/v3.12.10:0cc8128, Apr 8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`
- **PyTorch Version**: `2.14.0+cpu`
- **`torch.cuda.is_available()`**: `False`
- **`torch.version.cuda`**: `None`
- **Active PyTorch Device**: CPU-only fallback

### 1.2 Hardware & Physical GPU Status
- **Host GPU Detected via `nvidia-smi`**:
  - **GPU Model**: `NVIDIA GeForce RTX 4060 Laptop GPU`
  - **VRAM Total**: `8188 MiB` (~8 GB GDDR6)
  - **Driver Version**: `610.47`
  - **CUDA Driver API Version**: `13.3`
- **Environment Discrepancy & Finding for Phase 1**:
  - A physical NVIDIA RTX 4060 GPU is present on the machine, but the current active Python 3.12 virtual environment has the CPU build of PyTorch (`2.14.0+cpu`) installed.
  - To enable GPU training and live GPU inference in Phase 1, PyTorch should be updated to a CUDA-enabled wheel (e.g. CUDA 12.1 / 12.4).

### 1.3 Dependency Files
- **`requirements.txt`**: Present in repository root. Specifies:
  ```text
  numpy>=1.26.0
  torch>=2.0.0
  torchvision>=0.15.0
  rasterio>=1.3.0
  opencv-python-headless>=4.8.0
  scikit-image>=0.22.0
  scikit-learn>=1.3.0
  matplotlib>=3.8.0
  pyyaml>=6.0.0
  tqdm>=4.66.0
  streamlit>=1.30.0
  ```
- **`environment.yml`**: Not present in repository.
- **`pyproject.toml`**: Not present in repository.
- **`pytest.ini`**: Present (sets test discovery options).

---

## 2. Repository Structure

```text
GeoFUSE/
├── .gitignore
├── AUDIT_PHASE0.md                  # This audit document
├── DATASET_SETUP.md                 # Raw Sentinel-2 acquisition protocol
├── PROJECT_SUMMARY.md               # Summary of methodology and benchmark results
├── README.md                        # Master project documentation
├── REPRODUCIBILITY.md               # Audit trail and step-by-step reproduction guide
├── config.yaml                      # Master operational configuration
├── pytest.ini                       # PyTest configuration
├── requirements.txt                 # Python dependencies
├── checkpoints/
│   └── .gitkeep                     # Default checkpoints directory
├── data/
│   ├── aoi.geojson                  # Geographic boundary (Hyderabad, Telangana)
│   ├── raw/                         # 4-band Sentinel-2 L2A GeoTIFFs (B02, B03, B04, B08, 512x512 px)
│   ├── interim/                     # Temporary processing artifacts
│   └── processed/                   # Prepared training patches and baseline triplets
├── examples/
│   └── sample_upload/
│       └── sample_s2_4band_128px.tif # Bundled presenter demo GeoTIFF (128x128 px)
├── outputs/
│   ├── checkpoints/                 # Trained model checkpoints & training logs
│   │   ├── best_model.pth           # Single model best checkpoint (3.32 MB)
│   │   ├── latest_model.pth         # Single model final epoch checkpoint (1.10 MB)
│   │   ├── ensemble_member_0.pth    # Ensemble member 0 checkpoint, seed 42 (1.11 MB)
│   │   ├── ensemble_member_1.pth    # Ensemble member 1 checkpoint, seed 101 (1.11 MB)
│   │   ├── ensemble_member_2.pth    # Ensemble member 2 checkpoint, seed 2024 (1.11 MB)
│   │   ├── training_log.csv         # Epoch-by-epoch loss & PSNR/SSIM log (single model)
│   │   ├── training_log_member_0.csv# Epoch log for ensemble member 0
│   │   ├── training_log_member_1.csv# Epoch log for ensemble member 1
│   │   └── training_log_member_2.csv# Epoch log for ensemble member 2
│   ├── demo_cache/                  # Precomputed offline assets for Demo Mode
│   │   ├── manifest.json            # Manifest of indexed demo tiles & trust scores
│   │   ├── demo_tile_0.pkl          # Tile #0 full pipeline bundle
│   │   ├── demo_tile_0_receipt.json # Tile #0 auditable Trust Receipt
│   │   ├── demo_tile_8.pkl          # Tile #8 bundle
│   │   ├── demo_tile_8_receipt.json # Tile #8 receipt
│   │   ├── demo_tile_16.pkl         # Tile #16 bundle
│   │   ├── demo_tile_16_receipt.json# Tile #16 receipt
│   │   ├── demo_tile_24.pkl         # Tile #24 bundle
│   │   └── demo_tile_24_receipt.json# Tile #24 receipt
│   ├── presentation/                # Exported figures, diagrams, and summaries
│   ├── previews/                    # RGB previews of raw and processed imagery
│   └── receipts/                    # Trust Receipt JSON files
├── scripts/
│   ├── download_sample_data.py      # Real Sentinel-2 synthetic fallback downloader
│   ├── evaluate_downstream_task.py  # Downstream building footprint extractor & IoU evaluator
│   ├── export_presentation_assets.py# Generates presentation figures and comparison grids
│   ├── generate_baseline_triplets.py# Degrades HR to synthesize LR/Bicubic triplets
│   ├── generate_trust_receipts.py   # CLI tool to generate auditable Trust Receipts
│   ├── generate_trust_risk_maps.py  # Standalone multi-criteria evidence fusion script
│   ├── inspect_sentinel2.py         # Inspects metadata, CRS, GSD, and statistics
│   ├── precompute_demo_cache.py     # Precomputes offline demo bundles
│   ├── reproduce_all.py             # End-to-end master reproduction pipeline
│   ├── run_ensemble_inference.py    # 3-member PyTorch ensemble inference
│   ├── smoke_test.py                # Environment and pipeline sanity checks
│   ├── test_consistency_checks.py  # Spectral and structural consistency tests
│   ├── test_stability.py            # Input perturbation stability analysis
│   └── train.py                     # CLI entry point for model training
├── src/
│   ├── dashboard/
│   │   └── app.py                   # Streamlit mission-control telemetry dashboard
│   ├── data/
│   │   ├── dataset.py               # PyTorch Dataset for Sentinel-2 patches
│   │   ├── degrade.py               # 2x sensor blur, downsampling, and noise injection
│   │   ├── inspect_data.py          # Data ingestion and geospatial validation
│   │   ├── tiling.py                # Non-overlapping grid tiling (64x64)
│   │   └── upload.py                # User GeoTIFF validation, parsing, and stacking
│   ├── evaluation/
│   │   ├── downstream_eval.py       # Morphological building footprint extraction & IoU
│   │   ├── edge_check.py            # Sobel/Canny boundary gradient correlation
│   │   ├── fusion.py                # Multi-evidence heuristic trust/risk fusion
│   │   ├── spectral_check.py        # Absolute Delta-NDVI error computation
│   │   ├── stability.py             # Perturbation testing (noise + brightness jitter)
│   │   └── trust_receipt.py         # JSON and HTML Trust Receipt generator
│   ├── models/
│   │   ├── ensemble.py              # PyTorch 3-model forward pass & disagreement std
│   │   ├── loss.py                  # Compound loss (L1 + Sobel gradient loss)
│   │   ├── model.py                 # ResidualSRNet deep neural network architecture
│   │   └── train.py                 # Training loop, optimizer, AMP, holdout split
│   └── utils/
│       └── config.py                # Configuration loader and device resolver
└── tests/                           # 19 automated test suites (89 passed tests)
```

---

## 3. Super-Resolution Architecture Ground Truth

Inspected directly from [`src/models/model.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/models/model.py) and [`config.yaml`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/config.yaml):

### 3.1 Network Topology: `ResidualSRNet`
- **Class**: `ResidualSRNet(nn.Module)` in [`src/models/model.py:36-118`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/models/model.py#L36-L118)
- **Scale Factor**: **`2x`** (nominally $64\times 64 \rightarrow 128\times 128$ pixels; input 20m/10m GSD reconstructed to $2\times$ finer spatial sampling).
  - Confirmed in `config.yaml:42` (`scale_factor: 2`)
  - Confirmed in `src/models/model.py:54` (`scale_factor: int = 2`)
  - Confirmed in `src/dashboard/app.py:1769` (`Reconstructed GSD: 5.0m (2x SR)`)
- **Input Channels**: **4 spectral bands** (`B02` Blue, `B03` Green, `B04` Red, `B08` NIR).
- **Output Channels**: **4 spectral bands** (`B02`, `B03`, `B04`, `B08`).
- **Feature Depth (`num_features`)**: **48 channels**.
- **Residual Backbone**:
  - Configured residual blocks (`num_residual_blocks`): **4 blocks** (from `config.yaml:46`).
  - Each `ResidualBlock` contains two $3\times 3$ Conv2D layers with `LeakyReLU(negative_slope=0.2)` and local skip addition scaled by `res_scale = 0.1`.
- **Upsampling Head**:
  - Sub-pixel convolution via `nn.Conv2d(48, 48 * (2^2) = 192, 3, padding=1)` followed by `nn.PixelShuffle(upscale_factor=2)` and `LeakyReLU(0.2)`.
- **Global Residual Learning**:
  - A global base skip connection interpolates the input tensor directly to $2\times$ via bicubic interpolation (`F.interpolate(x, scale_factor=2.0, mode="bicubic")`).
  - The network learns only the high-frequency residual detail ($\text{Output} = \text{Bicubic Base} + \text{Residual Detail}$).
- **Parameter Count**:
  - Single model with `config.yaml` parameters (4 residual blocks, 48 features): **273,700 parameters** (~0.27M parameters).
  - 3-member ensemble total parameters: $3 \times 273,700 =$ **821,100 parameters** (~0.82M parameters).
  - Well below the strict 1.5M parameter budget constraint.
  - Zero GAN discriminators, zero attention maps, zero transformer layers.

---

## 4. Metrics & Score Verification: Ground Truth vs. Planning Assumptions

### 4.1 Verification of Example Numbers from Prior Planning

| Metric / Value Cited in Prompt | Ground Truth in Code / Logs | Status | Location / Source Reference |
| :--- | :--- | :--- | :--- |
| **Reconstruction Scale Factor: 2x** | **2x** confirmed | **VERIFIED** | `config.yaml:42`, `src/models/model.py:54`, `src/dashboard/app.py:1769` |
| **Evidence Score: 86.75 / 100** | **86.75%** confirmed | **VERIFIED** | `outputs/demo_cache/manifest.json:9`, `outputs/demo_cache/demo_tile_0_receipt.json:57` |
| **PSNR: 38.23 dB** | **38.23 dB** confirmed at Epoch 4 | **VERIFIED** | `outputs/checkpoints/training_log.csv:5` (val_psnr at epoch 4) |
| **PSNR: 38.09 dB** | **Not found** anywhere in codebase | **UNGROUNDED** | Zero occurrences in logs/code. Likely an informal estimate from early planning. |

### 4.2 Where PSNR & SSIM Are Actually Computed
1. **Fidelity Benchmark Function**:
   - Location: [`src/data/degrade.py:166-200`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/data/degrade.py#L166-L200), `evaluate_reconstruction_fidelity(gt, rec, data_range=1.0)`.
   - Uses `skimage.metrics.peak_signal_noise_ratio` and `skimage.metrics.structural_similarity(channel_axis=0)`.
2. **Training Validation Loop**:
   - Location: [`src/models/train.py:110-125`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/models/train.py#L110-L125), `validate_epoch(...)`.
   - Computes per-patch PSNR and SSIM across the geographic hold-out split (southeast quadrant, rows 256..512, cols 256..512) and logs epoch averages to CSV.
3. **Live Dashboard Evaluation**:
   - Location: [`src/dashboard/app.py:1685-1695`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/dashboard/app.py#L1685-L1695).
   - Computes real PSNR/SSIM when independent ground-truth reference is available (`has_reference=True` in Demo Mode).
   - For real user uploads where ground truth does not exist, honestly marks PSNR/SSIM as `N/A`.

### 4.3 Actual Most-Recent Real Logged Benchmark Values (Epoch 15 Final Convergence)
From `outputs/checkpoints/training_log*.csv`, `PROJECT_SUMMARY.md:39-44`, and `REPRODUCIBILITY.md:153-158`:

| Model / Configuration | Parameters | Val L1 Loss | Val Gradient Loss | Val PSNR (dB) | Val SSIM | Convergence Epoch |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bicubic Baseline (2x)** | 0 (Interpolation) | 0.01018 | 0.09021 | **38.19 dB** | **0.9208** | Baseline |
| **ResidualSRNet (Single, Seed 42)** | 273,700 (~0.27M) | **0.00912** | **0.08143** | **38.68 dB** | **0.9270** | Epoch 15 |
| **Ensemble Member 0 (Seed 42)** | 273,700 (~0.27M) | 0.00912 | 0.08158 | **38.67 dB** | **0.9272** | Epoch 15 |
| **Ensemble Member 1 (Seed 101)** | 273,700 (~0.27M) | 0.00915 | 0.08181 | **38.66 dB** | **0.9265** | Epoch 15 |
| **Ensemble Member 2 (Seed 2024)**| 273,700 (~0.27M) | 0.00917 | 0.08200 | **38.62 dB** | **0.9261** | Epoch 15 |
| **Ensemble Mean** | 821,100 (3×0.27M) | **0.00915** | **0.08180** | **38.65 dB** | **0.9266** | Ensemble |

*Summary*: GeoFUSE SR achieves **+0.49 dB PSNR** and **+0.0062 SSIM** over the standard $2\times$ bicubic baseline on real Sentinel-2 validation data.

---

## 5. Evidence-Score Computation Code & Pipeline Verification

### 5.1 Computation Location & Logic
- **File & Function**: [`src/evaluation/fusion.py:51-200`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/evaluation/fusion.py#L51-L200), `fuse_trust_risk_maps(...)`.
- **4 Evidence Streams**:
  1. **Disagreement Map**: Model uncertainty proxy derived from the pixel-wise standard deviation $\sigma(x, y)$ across the 3 ensemble predictions (`src/models/ensemble.py`).
  2. **Stability Map**: Input perturbation sensitivity measured by the variance of outputs under Gaussian sensor noise ($\sigma=0.01, 0.02, 0.05$) and multiplicative brightness jitter ($\sigma=0.02$) across 3 trials (`src/evaluation/stability.py`).
  3. **Spectral Error Map**: Absolute deviation in Normalized Difference Vegetation Index ($|\Delta\text{NDVI}| = |\text{NDVI}_{\text{SR}} - \text{NDVI}_{\text{LR}}|$) against input baseline (`src/evaluation/spectral_check.py`).
  4. **Structural Error Map**: Deviation in high-frequency Sobel gradient magnitude between reconstructed output and baseline (`src/evaluation/edge_check.py`).

### 5.2 Heuristic Fusion Formulation
- **Signal Normalization**: Each raw map $M_i$ is min-max normalized to $[0.0, 1.0]$ via `normalize_evidence_map(arr)`:
  $$\hat{M}_i = \frac{M_i - \min(M_i)}{\max(M_i) - \min(M_i) + 10^{-8}}$$
  *(Prevents any single evidence stream from dominating due to raw scale magnitude).*
- **Weights**: Configured as equal weights $w_i = 0.25$ (`config.yaml:86-90`).
- **Composite Risk Map**:
  $$R(x, y) = \text{clip}\left(\sum_{i=1}^4 w_i \cdot \hat{M}_i(x, y), 0.0, 1.0\right)$$
- **Composite Trust Map**:
  $$T(x, y) = 1.0 - R(x, y)$$
- **Composite Evidence / Trust Score**:
  $$\text{trust\_score\_pct} = \text{round}\left(\frac{1}{H \times W}\sum_{x, y} T(x, y) \times 100.0, 2\right)$$
- **Scientific Qualification**: The codebase explicitly documents in code and UI banners that this score is a heuristic multi-criteria reliability proxy, **not** a calibrated Bayesian posterior probability or conformal prediction guarantee.

### 5.3 Verified Demo Tile Scores in Offline Cache (`outputs/demo_cache/manifest.json`)
- **Tile #0** (`demo_tile_0.pkl`): **86.75%** (`NOMINAL_HIGH_TRUST`) — Central Settlement Cluster
- **Tile #8** (`demo_tile_8.pkl`): **86.70%** (`WARNING_LOW_TRUST`) — Mixed Agricultural & Roads
- **Tile #16** (`demo_tile_16.pkl`): **86.41%** (`WARNING_LOW_TRUST`) — Rural River Corridor
- **Tile #24** (`demo_tile_24.pkl`): **86.02%** (`WARNING_LOW_TRUST`) — Complex Terrain Transition

---

## 6. Audit Summary & Readiness for Subsequent Phases

1. **Stop Conditions Check**:
   - The repository structure matches expected architecture completely.
   - Evidence-score computation, model architecture, training routines, and testing suites are fully present and operational.
   - All 89 tests in the test suite pass cleanly (`pytest tests/ -v`).
   - No stopping condition triggered.

2. **Phase 1 Takeaway**:
   - PyTorch is currently CPU-only (`2.14.0+cpu`).
   - The physical hardware possesses an `NVIDIA GeForce RTX 4060 Laptop GPU` (Driver 610.47, CUDA 13.3, 8 GB VRAM).
   - Upgrading PyTorch to a CUDA build will enable GPU-accelerated training and inference for subsequent phases.
