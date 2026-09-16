# GeoFUSE SentinelGuard — Phase 4 Baseline Evaluation & Experiment Tracking Audit (exp001)

**Date of Evaluation**: September 16, 2026  
**Experiment ID**: `baseline_exp001`  
**Checkpoint Evaluated**: [`checkpoints/baseline_exp001/best_model.pth`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/checkpoints/baseline_exp001/best_model.pth)  
**Host Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM)  
**Evaluation Environment**: Python 3.12.10 | PyTorch 2.6.0+cu124 (CUDA 12.4)  
**Evaluation Paradigm**: Explicitly labeled as **synthetic degrade-and-recover validation**  

---

## 1. Executive Summary & Evaluation Paradigm

The Phase 3 baseline model (`ResidualSRNet`, 273,700 parameters) was thoroughly evaluated across all **49 held-out geographic validation patches** located in the Southeast Quadrant `(rows: 256–512, cols: 256–512)` of `data/raw/`.

### 1.1 Synthetic Degrade-and-Recover Validation Paradigm
In alignment with rigorous scientific integrity, this evaluation is explicitly designated as **synthetic degrade-and-recover validation**:
- Pseudo-HR patches ($128\times 128$, 4-band: B02, B03, B04, B08) were degraded via:
  1. Optical PSF blur (Gaussian kernel size $3\times 3$, $\sigma = 0.5$) simulating sensor point-spread function.
  2. Spatial bicubic downsampling ($2\times$) generating $64\times 64$ pseudo-LR inputs.
  3. Sensor radiometric Gaussian noise ($\sigma = 0.01$, deterministic per-patch seeding).
- The model reconstructs the $128\times 128$ tile from the $64\times 64$ LR input.
- Reconstructed outputs are evaluated directly against the original pre-degradation reference tile.

---

## 2. Quantitative Distribution Statistics (49 Validation Patches)

Rather than presenting cherry-picked or isolated scalar metrics, full distribution statistics ($\text{Mean} \pm \text{Std}$, $\text{Min}$, $\text{Max}$) across all 49 validation tiles are recorded below. All individual per-patch evaluations are preserved in [`logs/baseline_exp001_eval_distribution.csv`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/logs/baseline_exp001_eval_distribution.csv).

### 2.1 Summary Metrics Distribution Table

| Dimension | Metric | Mean ± Std | Min | Max | Unit / Range |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fidelity** | **Model PSNR** | **38.67 ± 0.22** | **38.14** | **39.05** | dB (higher is better) |
| | Bicubic Baseline PSNR | 38.59 ± 0.22 | 38.08 | 38.94 | dB |
| | **PSNR Gain over Bicubic** | **+0.08 ± 0.03** | **+0.01** | **+0.12** | dB |
| | Model MAE | 0.00902 ± 0.00022 | 0.00870 | 0.00950 | normalized reflectance [0, 1] |
| | Bicubic Baseline MAE | 0.00913 ± 0.00022 | 0.00880 | 0.00960 | normalized reflectance [0, 1] |
| **Structural** | **Model SSIM** | **0.9272 ± 0.0031** | **0.9202** | **0.9327** | [-1, 1] (higher is better) |
| | Bicubic Baseline SSIM | 0.9249 ± 0.0033 | 0.9181 | 0.9305 | [-1, 1] |
| | **SSIM Gain over Bicubic** | **+0.0023 ± 0.0005** | **+0.0015** | **+0.0030** | score delta |
| | **Gradient Correlation ($r$)** | **0.8173 ± 0.0243** | **0.7635** | **0.8721** | Pearson correlation [-1, 1] |
| | **Canny Edge IoU** | **0.3838 ± 0.0262** | **0.3098** | **0.4263** | [0, 1] |
| | **Canny Edge F1** | **0.5542 ± 0.0278** | **0.4730** | **0.5977** | [0, 1] |
| **Spectral** | **Mean $\Delta$-NDVI** | **0.02091 ± 0.00053** | **0.01992** | **0.02204** | absolute NDVI error |
| | Inconsistent Pixels ($\Delta > 0.05$) | 6.63 ± 0.57 % | 5.59 % | 8.05 % | % of tile pixels |

---

## 3. Hardware Telemetry & GPU Inference Benchmark

Inference was benchmarked directly on the target host hardware using CUDA stream synchronization:
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (Driver: 610.47, CUDA 12.4)
- **Batch Size for Latency Benchmark**: 1 patch ($1\times 4\times 64\times 64 \rightarrow 1\times 4\times 128\times 128$)
- **Mean Inference Latency**: **1.52 ms ± 0.64 ms** per patch
- **Min / Max Latency**: 0.93 ms / 3.56 ms
- **Inference Throughput**: **657.9 patches/sec**
- **Peak Training VRAM**: 158.18 MiB

---

## 4. Experiment Tracking Schema (`experiments/experiments_log.jsonl`)

All future training and evaluation runs append to [`experiments/experiments_log.jsonl`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/experiments/experiments_log.jsonl). The schema enforces complete reproducibility, hardware provenance, configuration integrity, and statistical distributions.

### 4.1 Schema Fields Specification
```json
{
  "experiment_id": "string",
  "timestamp": "ISO-8601 UTC string",
  "git_commit": "40-character SHA-1 git commit hash",
  "random_seed": "integer",
  "evaluation_type": "string (e.g. synthetic degrade-and-recover validation)",
  "dataset": {
    "name": "string",
    "channels": ["list of strings"],
    "patch_size_hr": "integer",
    "stride": "integer",
    "val_quadrant": ["r_min", "r_max", "c_min", "c_max"],
    "num_val_patches": "integer"
  },
  "degradation_config": {
    "optical_psf_blur": {"kernel_size": "int", "sigma": "float"},
    "downsample_factor": "int",
    "downsample_method": "string",
    "sensor_noise_std": "float"
  },
  "architecture_config": {
    "model_name": "string",
    "num_channels": "int",
    "num_features": "int",
    "num_residual_blocks": "int",
    "upsample_method": "string",
    "scale_factor": "int"
  },
  "param_count": "integer",
  "training_config": {
    "batch_size": "int",
    "epochs": "int",
    "learning_rate": "float",
    "weight_decay": "float",
    "optimizer": "string",
    "scheduler": "string",
    "loss": "string",
    "mixed_precision": "boolean"
  },
  "training_results": {
    "best_epoch": "int",
    "train_loss_final": "float",
    "val_loss_best": "float",
    "total_duration_s": "float",
    "peak_vram_mib": "float"
  },
  "hardware_telemetry": {
    "device": "string",
    "gpu_name": "string",
    "cuda_version": "string",
    "torch_version": "string"
  },
  "inference_benchmark": {
    "device": "string",
    "mean_latency_ms_per_patch": "float",
    "std_latency_ms": "float",
    "min_latency_ms": "float",
    "max_latency_ms": "float",
    "throughput_patches_per_sec": "float"
  },
  "metrics_summary": {
    "<metric_name>": {
      "mean": "float",
      "std": "float",
      "min": "float",
      "max": "float"
    }
  }
}
```

---

## 5. Artifacts and Verification

1. **Evaluation Script**: [`scripts/evaluate_baseline.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/scripts/evaluate_baseline.py)
   - Executable CLI supporting GPU enforcement, per-patch metric extraction, CSV distribution output, and JSONL logging.
2. **Experiment Tracking Utility**: [`src/utils/experiment_tracker.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/src/utils/experiment_tracker.py)
   - Handles schema validation, git SHA extraction, round-trip JSONL loading/saving, and statistical distribution computations.
3. **Unit Tests**: [`tests/test_experiment_tracking.py`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/tests/test_experiment_tracking.py)
   - Validates schema validation failure modes, distribution computation, and round-trip append/read operations (all tests passing).
4. **Per-Patch Distribution CSV**: [`logs/baseline_exp001_eval_distribution.csv`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/logs/baseline_exp001_eval_distribution.csv)
   - 49 rows capturing exact per-patch coordinates, fidelity metrics, Bicubic comparisons, $\Delta$-NDVI, gradient correlation, edge IoU, and inference latency.
5. **Append-Only Experiment Log**: [`experiments/experiments_log.jsonl`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/experiments/experiments_log.jsonl)
   - Initialized and recorded entry for `baseline_exp001`.
