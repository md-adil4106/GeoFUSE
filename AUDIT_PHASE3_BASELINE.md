# GeoFUSE SentinelGuard — Phase 3 Reproducible Baseline Training Run Audit (exp001)

**Date of Run**: September 16, 2026  
**Experiment ID**: `baseline_exp001`  
**Host Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM)  
**Execution Environment**: Python 3.12.10 | PyTorch 2.6.0+cu124 (CUDA 12.4)  
**Status**: Completed Successfully on `cuda:0` (Zero CPU fallback)  

---

## 1. Executive Summary & Hardware Verification

A trustworthy, reproducible baseline training run (`baseline_exp001`) was executed using the exact Phase 2 documented architecture and hyperparameters. All forward passes, loss calculations, backpropagation, and optimizer updates ran exclusively on `cuda:0`.

### 1.1 Device Telemetry at Training Launch
```text
============================================================================
   GeoFUSE SentinelGuard — Baseline Training Run (baseline_exp001)
============================================================================
PyTorch Version       : 2.6.0+cu124
CUDA Version          : 12.4
GPU Name              : NVIDIA GeForce RTX 4060 Laptop GPU
Total GPU VRAM        : 8187.5 MiB
Selected Device       : cuda:0
Model Device          : cuda:0
First Batch Device    : cuda:0
Model Parameters      : 273,700
Training Patches      : 120 (Batch Size: 16)
Validation Patches    : 49
Epochs                : 15
Mixed Precision       : Enabled (torch.amp.autocast('cuda'))
----------------------------------------------------------------------------
```

---

## 2. Real Measured Training Metrics (Epoch-by-Epoch)

Logged in [`logs/baseline_exp001_metrics.csv`](file:///c:/Users/Mohommed%20Adil/Desktop/GeoFUSE/logs/baseline_exp001_metrics.csv):

| Epoch | Train L1 | Train Grad | Train Total Loss | Val L1 | Val Grad | Val Total Loss | Val PSNR (dB) | Val SSIM | Learning Rate | Peak VRAM | Time (s) | Best Model Flag |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.02383 | 0.09062 | 0.03283 | 0.01396 | 0.08661 | 0.02265 | 34.95 dB | 0.8766 | 0.000495 | 158.18 MiB | 0.69s | `[BEST]` |
| **2** | 0.01224 | 0.08823 | 0.02106 | 0.01112 | 0.08557 | 0.01966 | 36.98 dB | 0.9041 | 0.000478 | 158.18 MiB | 0.40s | `[BEST]` |
| **3** | 0.01061 | 0.08716 | 0.01933 | 0.01011 | 0.08473 | 0.01858 | 37.79 dB | 0.9162 | 0.000452 | 158.18 MiB | 0.39s | `[BEST]` |
| **4** | 0.01004 | 0.08642 | 0.01868 | 0.00966 | 0.08403 | 0.01807 | 38.17 dB | 0.9207 | 0.000417 | 158.18 MiB | 0.39s | `[BEST]` |
| **5** | 0.00976 | 0.08587 | 0.01835 | 0.00946 | 0.08350 | 0.01781 | 38.36 dB | 0.9230 | 0.000375 | 158.18 MiB | 0.41s | `[BEST]` |
| **6** | 0.00963 | 0.08535 | 0.01817 | 0.00936 | 0.08309 | 0.01767 | 38.46 dB | 0.9242 | 0.000328 | 158.18 MiB | 0.39s | `[BEST]` |
| **7** | 0.00951 | 0.08506 | 0.01802 | 0.00929 | 0.08278 | 0.01757 | 38.52 dB | 0.9249 | 0.000277 | 158.18 MiB | 0.40s | `[BEST]` |
| **8** | 0.00947 | 0.08470 | 0.01794 | 0.00923 | 0.08246 | 0.01748 | 38.57 dB | 0.9256 | 0.000224 | 158.18 MiB | 0.39s | `[BEST]` |
| **9** | 0.00944 | 0.08436 | 0.01787 | 0.00920 | 0.08219 | 0.01742 | 38.60 dB | 0.9261 | 0.000173 | 158.18 MiB | 0.39s | `[BEST]` |
| **10** | 0.00940 | 0.08416 | 0.01781 | 0.00917 | 0.08197 | 0.01737 | 38.63 dB | 0.9265 | 0.000126 | 158.18 MiB | 0.40s | `[BEST]` |
| **11** | 0.00937 | 0.08396 | 0.01777 | 0.00915 | 0.08181 | 0.01733 | 38.64 dB | 0.9268 | 0.000084 | 158.18 MiB | 0.41s | `[BEST]` |
| **12** | 0.00935 | 0.08386 | 0.01773 | 0.00913 | 0.08170 | 0.01730 | 38.66 dB | 0.9270 | 0.000049 | 158.18 MiB | 0.40s | `[BEST]` |
| **13** | 0.00932 | 0.08382 | 0.01770 | 0.00912 | 0.08164 | 0.01729 | 38.66 dB | 0.9271 | 0.000023 | 158.18 MiB | 0.40s | `[BEST]` |
| **14** | 0.00933 | 0.08370 | 0.01770 | 0.00912 | 0.08161 | 0.01728 | 38.67 dB | 0.9272 | 0.000006 | 158.18 MiB | 0.40s | `[BEST]` |
| **15** | 0.00933 | 0.08369 | 0.01770 | 0.00912 | 0.08160 | 0.01728 | 38.67 dB | 0.9272 | 0.000001 | 158.18 MiB | 0.39s | `[BEST]` |

---

## 3. Computational Performance & VRAM Efficiency

- **Total Training Duration**: **`6.40 seconds`** (15 full epochs).
- **Average Epoch Duration**: **`0.41 seconds`** per epoch.
- **Speedup over CPU**: Previous CPU-only execution averaged ~3.95 seconds/epoch (~60s total). RTX 4060 GPU with mixed precision achieved a **`~9.5x` speedup**.
- **Peak Allocated VRAM**: **`158.18 MiB`** (utilizing less than 2% of the available 8,188 MiB VRAM).
- **Out of Memory (OOM) Events**: **0**. The batch size of 16 ran without memory pressure.

---

## 4. Final Converged Benchmark Metrics

- **Bicubic Baseline (2x)**: **38.19 dB** PSNR | **0.9208** SSIM
- **Baseline Exp001 Model (Best Epoch 15)**: **38.67 dB** PSNR | **0.9272** SSIM
- **Advantage Over Bicubic**:
  - **$\Delta$ PSNR**: **`+0.48 dB`**
  - **$\Delta$ SSIM**: **`+0.0064`**
  - **$\Delta$ Total Loss**: Decreased from $0.02265$ (Epoch 1) to **$0.01728$** (Epoch 15).

---

## 5. Generated Artifacts & Preservation

All experiment assets have been generated and preserved:
1. `checkpoints/baseline_exp001/best_model.pth` (3.32 MB) — Checkpoint from best validation loss epoch (Epoch 15).
2. `checkpoints/baseline_exp001/latest_model.pth` (3.32 MB) — Checkpoint from final epoch.
3. `checkpoints/baseline_exp001/config.yaml` — Complete hyperparameter and hardware record.
4. `checkpoints/baseline_exp001/metrics.csv` — Exact epoch metric logs.
5. `logs/baseline_exp001_metrics.csv` — Centralized experiment metrics log.
6. `scripts/train_baseline.py` — Dedicated one-command reproduction runner.

---

## 6. How to Reproduce

Run via either CLI entrypoint:
```powershell
python scripts/train_baseline.py
# Or:
python scripts/train.py --baseline
```
