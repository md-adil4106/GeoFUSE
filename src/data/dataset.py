"""PyTorch Dataset for Sentinel-2 Super-Resolution with Geographic Partitioning.

Enforces strict geographic hold-out splitting to prevent spatial autocorrelation
and data leakage between training and validation sets:
- Split v1: Legacy center-point splitting (retained for Phase 3/4 baseline reproduction).
- Split v2: Strict bounding-box spatial containment with zero spatial leakage and
            optional guard buffer separation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.degrade import synthesize_pseudo_lr


class SentinelSRDataset(Dataset):
    """Dataset yielding paired (pseudo-LR, pseudo-HR) Sentinel-2 multi-spectral patches.

    Args:
        full_image: Multi-band array of shape (H, W, C) in normalized reflectance [0, 1].
        patch_size_hr: Dimension of the high-resolution patch (e.g., 128 for 64x64 LR at 2x).
        stride: Extraction step size between adjacent patches.
        split: Mode, either 'train' or 'val'.
        val_quadrant: Bounding box (row_min, row_max, col_min, col_max) defining the
                      independent geographic validation hold-out zone.
        downsample_factor: Spatial scale factor (default: 2).
        blur_kernel_size: Gaussian optical PSF blur size.
        noise_std: Additive sensor radiometric noise standard deviation.
        seed: Base random seed for reproducible noise degradation.
        split_mode: Splitting algorithm:
                    - 'v1': Legacy center-point assignment (has spatial boundary overlap;
                            preserved for baseline reproduction).
                    - 'v2': Strict bounding-box spatial containment with zero leakage.
        buffer_pixels: Guard buffer in pixels around validation zone for split_mode='v2'
                       (e.g., 0 for contiguous boundary contact, 32 for 320m guard buffer).
    """

    def __init__(
        self,
        full_image: np.ndarray,
        patch_size_hr: int = 128,
        stride: int = 64,
        split: str = "train",
        val_quadrant: Tuple[int, int, int, int] = (256, 512, 256, 512),
        downsample_factor: int = 2,
        blur_kernel_size: int = 3,
        noise_std: float = 0.01,
        seed: int = 42,
        split_mode: str = "v2",
        buffer_pixels: int = 0,
    ) -> None:
        super().__init__()
        assert split in ("train", "val"), f"Invalid split: {split}. Must be 'train' or 'val'."
        assert split_mode in ("v1", "v2"), f"Invalid split_mode: {split_mode}. Must be 'v1' or 'v2'."
        self.split = split
        self.split_mode = split_mode
        self.buffer_pixels = buffer_pixels
        self.patch_size_hr = patch_size_hr
        self.downsample_factor = downsample_factor
        self.blur_kernel_size = blur_kernel_size
        self.noise_std = noise_std
        self.seed = seed
        self.val_quadrant = val_quadrant

        val_r_min, val_r_max, val_c_min, val_c_max = val_quadrant
        h, w, c = full_image.shape

        # Extract all candidate patch coordinates
        y_coords = list(range(0, h - patch_size_hr + 1, stride))
        x_coords = list(range(0, w - patch_size_hr + 1, stride))
        if y_coords and y_coords[-1] != h - patch_size_hr:
            y_coords.append(h - patch_size_hr)
        if x_coords and x_coords[-1] != w - patch_size_hr:
            x_coords.append(w - patch_size_hr)

        self.patches: List[Dict[str, Any]] = []

        for y in y_coords:
            for x in x_coords:
                if self.split_mode == "v1":
                    # Legacy center-point splitting (retained for Phase 3/4 baseline reproduction)
                    patch_center_y = y + patch_size_hr // 2
                    patch_center_x = x + patch_size_hr // 2

                    in_val_zone = (
                        val_r_min <= patch_center_y < val_r_max
                        and val_c_min <= patch_center_x < val_c_max
                    )
                    selected = (split == "val" and in_val_zone) or (split == "train" and not in_val_zone)

                else:  # split_mode == "v2"
                    # Strict bounding-box spatial containment with zero spatial leakage
                    # Validation patch: must be strictly contained inside the validation quadrant
                    is_val_contained = (
                        y >= val_r_min
                        and y + patch_size_hr <= val_r_max
                        and x >= val_c_min
                        and x + patch_size_hr <= val_c_max
                    )
                    # Training patch: must be strictly disjoint from the validation quadrant (with optional buffer)
                    is_train_disjoint = (
                        (y + patch_size_hr <= val_r_min - buffer_pixels)
                        or (x + patch_size_hr <= val_c_min - buffer_pixels)
                        or (y >= val_r_max + buffer_pixels)
                        or (x >= val_c_max + buffer_pixels)
                    )
                    selected = (split == "val" and is_val_contained) or (split == "train" and is_train_disjoint)

                if selected:
                    hr_patch = full_image[y : y + patch_size_hr, x : x + patch_size_hr, :].copy()
                    self.patches.append({
                        "hr": hr_patch,
                        "y": y,
                        "x": x,
                        "size": patch_size_hr,
                        "bbox": (y, y + patch_size_hr, x, x + patch_size_hr),
                    })

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.patches[idx]
        hr_patch = record["hr"]

        # Synthesize degraded pseudo-LR tile
        lr_patch = synthesize_pseudo_lr(
            hr_tile=hr_patch,
            downsample_factor=self.downsample_factor,
            blur_kernel_size=self.blur_kernel_size,
            noise_std=self.noise_std,
            seed=self.seed + idx if self.split == "val" else None,
        )

        # Convert to PyTorch format (Channels, Height, Width)
        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1).contiguous().float()
        lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).contiguous().float()

        meta = {
            "y": record["y"],
            "x": record["x"],
            "split": self.split,
            "split_mode": self.split_mode,
            "bbox": record.get("bbox", (record["y"], record["y"] + self.patch_size_hr, record["x"], record["x"] + self.patch_size_hr)),
        }
        return lr_tensor, hr_tensor, meta

    @staticmethod
    def count_overlapping_pairs(
        train_dataset: "SentinelSRDataset",
        val_dataset: "SentinelSRDataset",
    ) -> int:
        """Count how many (train, val) patch pairs share any spatial pixel area.

        Args:
            train_dataset: Training split dataset.
            val_dataset: Validation split dataset.

        Returns:
            int: Total number of intersecting pairs. Zero signifies a leak-free split.
        """
        overlap_count = 0
        for t in train_dataset.patches:
            ty_min, ty_max = t["y"], t["y"] + t["size"]
            tx_min, tx_max = t["x"], t["x"] + t["size"]
            for v in val_dataset.patches:
                vy_min, vy_max = v["y"], v["y"] + v["size"]
                vx_min, vx_max = v["x"], v["x"] + v["size"]

                # 2D Bounding box intersection
                if max(ty_min, vy_min) < min(ty_max, vy_max) and max(tx_min, vx_min) < min(tx_max, vx_max):
                    overlap_count += 1

        return overlap_count
