"""Loss functions for GeoFUSE SentinelGuard Super-Resolution Training.

Implements:
1. SobelGradientLoss: Multi-channel spatial gradient alignment loss.
2. CompoundSRLoss: L1 loss + Sobel gradient loss for edge fidelity without GAN artifacts.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelGradientLoss(nn.Module):
    """Multi-channel Sobel gradient loss to penalize blurred edge transitions.

    Applies depthwise separable Sobel filtering across all spectral channels
    and computes the L1 norm between gradient magnitudes of prediction and target.

    Args:
        channels: Number of spectral channels (e.g., 4 for B02, B03, B04, B08).
    """

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.channels = channels

        # Standard 3x3 Sobel kernels
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        # Replicate for depthwise convolution across all input channels
        weight_x = sobel_x.repeat(channels, 1, 1, 1)
        weight_y = sobel_y.repeat(channels, 1, 1, 1)

        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient difference loss.

        Args:
            pred: Super-resolved prediction tensor (B, C, H, W).
            target: Ground truth HR tensor (B, C, H, W).

        Returns:
            torch.Tensor: Scalar gradient loss.
        """
        # Ensure weights match tensor device and dtype
        wx = self.weight_x.to(dtype=pred.dtype, device=pred.device)
        wy = self.weight_y.to(dtype=pred.dtype, device=pred.device)

        # Depthwise 2D convolutions (groups = channels) with reflection padding
        pred_pad = F.pad(pred, (1, 1, 1, 1), mode="replicate")
        target_pad = F.pad(target, (1, 1, 1, 1), mode="replicate")

        pred_dx = F.conv2d(pred_pad, wx, groups=self.channels)
        pred_dy = F.conv2d(pred_pad, wy, groups=self.channels)

        target_dx = F.conv2d(target_pad, wx, groups=self.channels)
        target_dy = F.conv2d(target_pad, wy, groups=self.channels)

        grad_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
        return grad_loss


class CompoundSRLoss(nn.Module):
    """Compound Super-Resolution Loss: L1 Loss + Weighted Sobel Gradient Loss.

    Args:
        channels: Number of input spectral bands.
        grad_weight: Weight for Sobel edge gradient term (default: 0.1).
    """

    def __init__(self, channels: int = 4, grad_weight: float = 0.1) -> None:
        super().__init__()
        self.grad_weight = grad_weight
        self.l1_loss = nn.L1Loss()
        self.gradient_loss = SobelGradientLoss(channels=channels)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute compound loss.

        Args:
            pred: Predicted HR tensor (B, C, H, W).
            target: Ground truth HR tensor (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - Total backpropagatable loss tensor.
                - Dictionary containing individual loss components as floats.
        """
        l1 = self.l1_loss(pred, target)
        grad = self.gradient_loss(pred, target)

        total = l1 + self.grad_weight * grad

        loss_dict = {
            "l1": float(l1.detach().item()),
            "grad": float(grad.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, loss_dict
