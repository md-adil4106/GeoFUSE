"""Ensemble Super-Resolution and Disagreement (Uncertainty Proxy) Module.

Implements sequential ensemble prediction and pixel-level disagreement calculation
across multiple lightweight SR model checkpoints:
1. Ensemble Mean Reconstruction: Nominal super-resolved prediction.
2. Disagreement Map: Per-pixel standard deviation across ensemble members,
   acting as an empirical uncertainty proxy indicating where reconstructions are unstable.

Scientific Honesty:
Ensemble disagreement is an empirical UNCERTAINTY PROXY (model epistemic disagreement),
NOT calibrated Bayesian posterior uncertainty.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.models.model import ResidualSRNet, build_model


def load_ensemble_members(
    checkpoint_paths: List[Union[str, Path]],
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> List[nn.Module]:
    """Load an ensemble of trained SR models from checkpoint files.

    Args:
        checkpoint_paths: List of paths to .pth checkpoint files.
        config: Optional project configuration dictionary.
        device: PyTorch device ('cuda' or 'cpu').

    Returns:
        List[nn.Module]: List of loaded models set to evaluation mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models: List[nn.Module] = []
    for p in checkpoint_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Ensemble checkpoint not found at: {path.resolve()}")

        model = build_model(config).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)

        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    return models


def predict_ensemble(
    models: List[nn.Module],
    lr_input: Union[torch.Tensor, np.ndarray],
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Compute ensemble mean reconstruction and spatial disagreement map.

    Args:
        models: List of evaluation-ready PyTorch SR models.
        lr_input: Input low-resolution data as either:
                  - PyTorch Tensor of shape (Batch, Channels, H, W) or (Channels, H, W)
                  - NumPy array of shape (H, W, Channels)
        device: PyTorch device for inference.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
            - mean_recon: Ensemble mean array of shape (H_hr, W_hr, Channels), dtype float32.
            - disagreement_map: Per-pixel standard deviation across members of shape (H_hr, W_hr), dtype float32.
            - individual_preds: List of individual member predictions (each H_hr, W_hr, Channels).
    """
    if not models:
        raise ValueError("Cannot perform ensemble prediction with an empty models list.")

    if device is None:
        device = next(models[0].parameters()).device

    # Convert NumPy (H, W, C) to Tensor (1, C, H, W)
    if isinstance(lr_input, np.ndarray):
        if lr_input.ndim == 3:
            # (H, W, C) -> (1, C, H, W)
            tensor_input = (
                torch.from_numpy(lr_input).permute(2, 0, 1).unsqueeze(0).float().to(device)
            )
        elif lr_input.ndim == 4:
            # (B, H, W, C) -> (B, C, H, W)
            tensor_input = (
                torch.from_numpy(lr_input).permute(0, 3, 1, 2).float().to(device)
            )
        else:
            raise ValueError(f"Unsupported numpy input shape: {lr_input.shape}")
    else:
        tensor_input = lr_input.to(device)
        if tensor_input.dim() == 3:
            tensor_input = tensor_input.unsqueeze(0)

    member_predictions: List[np.ndarray] = []

    with torch.no_grad():
        for model in models:
            model.eval()
            pred_tensor = model(tensor_input)
            # Remove batch dim and convert to (H, W, C) numpy array
            pred_np = pred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
            member_predictions.append(pred_np)

    # Stack along ensemble dimension: shape (M, H, W, C)
    stacked = np.stack(member_predictions, axis=0)

    # (a) Ensemble Mean Reconstruction: (H, W, C)
    mean_recon = np.mean(stacked, axis=0)

    # (b) Per-pixel Standard Deviation across members: (H, W, C)
    # Aggregated across channels (spectral mean) to form a scalar 2D uncertainty proxy map (H, W)
    std_per_band = np.std(stacked, axis=0)
    disagreement_map = np.mean(std_per_band, axis=-1)

    return mean_recon, disagreement_map, member_predictions
