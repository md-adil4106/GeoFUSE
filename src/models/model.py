"""Lightweight Residual CNN Super-Resolution Model for GeoFUSE SentinelGuard.

Implements a streamlined, memory-efficient deep learning super-resolution network
tailored for medium-resolution satellite imagery (Sentinel-2 L2A 4-band):
- 4 to 6 residual blocks with local skip connections
- 32 to 48 feature channels
- Sub-pixel convolution (PixelShuffle) 2x upsampling head
- Global residual learning via base upsampling bypass
- Parameter budget: ~0.3M - 0.7M parameters (well under the 1.5M ceiling)
- Zero GAN discriminators, attention modules, or heavy transformer blocks.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard residual block with two 3x3 convolutions and LeakyReLU activation."""

    def __init__(self, channels: int, res_scale: float = 1.0) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class ResidualSRNet(nn.Module):
    """Lightweight Residual Super-Resolution Network for Multi-Band Satellite Data.

    Args:
        in_channels: Number of input spectral bands (e.g. 4 for B02, B03, B04, B08).
        out_channels: Number of output spectral bands (matches in_channels).
        num_features: Intermediate feature representation depth (32 to 48).
        num_blocks: Number of chained residual blocks (4 to 6).
        scale_factor: Spatial upsampling factor (default: 2).
        res_scale: Residual scaling factor for numerical stability.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        num_features: int = 48,
        num_blocks: int = 6,
        scale_factor: int = 2,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        # 1. Shallow Feature Extraction Head
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True)

        # 2. Deep Residual Backbone
        self.body = nn.Sequential(
            *[ResidualBlock(channels=num_features, res_scale=res_scale) for _ in range(num_blocks)]
        )

        # 3. Mid-Feature Trunk Conv (Global Feature Integration)
        self.trunk = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True)

        # 4. Pixel-Shuffle Sub-Pixel Convolution Upsampling Head
        self.upsample = nn.Sequential(
            nn.Conv2d(
                num_features,
                num_features * (scale_factor ** 2),
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # 5. Final Reconstruction Layer
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (Batch, in_channels, H, W).

        Returns:
            torch.Tensor: Super-resolved tensor of shape (Batch, out_channels, H * scale, W * scale).
        """
        # Global base skip connection (bicubic interpolation of input)
        base = F.interpolate(
            x,
            scale_factor=float(self.scale_factor),
            mode="bicubic",
            align_corners=False,
        )

        # Feature extraction & residual learning
        f_init = self.head(x)
        f_res = self.body(f_init)
        f_trunk = self.trunk(f_res) + f_init

        # PixelShuffle upsampling
        f_up = self.upsample(f_trunk)
        residual_hr = self.tail(f_up)

        # Reconstructed HR = Interpolated Base + Learned High-Frequency Detail
        out = base + residual_hr
        return out


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: Optional[Dict[str, Any]] = None) -> ResidualSRNet:
    """Factory function to instantiate ResidualSRNet from project config.yaml.

    Args:
        config: Optional configuration dictionary. If None, default settings are used.

    Returns:
        ResidualSRNet: Instantiated model.
    """
    model_cfg = config.get("model", {}) if config else {}
    in_channels = int(model_cfg.get("num_channels", 4))
    num_features = int(model_cfg.get("num_features", 48))
    scale_factor = int(model_cfg.get("scale_factor", 2))
    num_blocks = int(model_cfg.get("num_residual_blocks", 6))

    model = ResidualSRNet(
        in_channels=in_channels,
        out_channels=in_channels,
        num_features=num_features,
        num_blocks=num_blocks,
        scale_factor=scale_factor,
    )
    return model
