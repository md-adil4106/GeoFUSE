"""Sentinel-2 Multi-Band Ingestion, Normalization, and Patch Tiling Module."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio

from src.data.inspect_data import find_band_files, check_spatial_consistency


def load_sentinel2_stack(
    data_dir: Path,
    target_bands: Optional[List[str]] = None,
    to_reflectance: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load Sentinel-2 bands from directory and stack into (H, W, C) float array.

    Args:
        data_dir: Path to directory containing band GeoTIFFs.
        target_bands: List of bands, defaults to ['B02', 'B03', 'B04', 'B08'].
        to_reflectance: If True and data is integer DN (> 10), converts to [0, 1] reflectance.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Multi-band array of shape (H, W, num_bands), dtype float32.
            - Metadata dictionary containing CRS, transform, band names, bounds.
    """
    if target_bands is None:
        target_bands = ["B02", "B03", "B04", "B08"]

    band_files = find_band_files(data_dir, target_bands)

    band_datasets = {}
    band_arrays = []
    for band in target_bands:
        ds = rasterio.open(band_files[band])
        band_datasets[band] = ds
        arr = ds.read(1).astype(np.float32)
        band_arrays.append(arr)

    # Validate spatial alignment across all bands
    spatial_meta = check_spatial_consistency(band_datasets)

    # Stack along channels: (H, W, C)
    stack = np.stack(band_arrays, axis=-1)

    # Normalize to [0.0, 1.0] surface reflectance if in raw DN (> 10.0)
    if to_reflectance and np.max(stack) > 10.0:
        stack = stack / 10000.0

    metadata = {
        **spatial_meta,
        "bands": target_bands,
        "dtype": str(stack.dtype),
        "is_reflectance": to_reflectance,
    }

    for ds in band_datasets.values():
        ds.close()

    return stack, metadata


def extract_tiles(
    image: np.ndarray,
    patch_size: int = 64,
    stride: int = 32,
) -> List[Dict[str, Any]]:
    """Extract regular overlapping spatial tiles from a multi-band image.

    Args:
        image: Multi-band array of shape (H, W, C).
        patch_size: Tile spatial dimension (height and width).
        stride: Step size between adjacent tiles.

    Returns:
        List[Dict[str, Any]]: List of extracted tile records with pixel coordinates.
    """
    h, w, c = image.shape
    tiles = []

    y_steps = list(range(0, h - patch_size + 1, stride))
    x_steps = list(range(0, w - patch_size + 1, stride))

    # Ensure right and bottom boundaries are covered
    if y_steps[-1] != h - patch_size:
        y_steps.append(h - patch_size)
    if x_steps[-1] != w - patch_size:
        x_steps.append(w - patch_size)

    tile_id = 0
    for y in y_steps:
        for x in x_steps:
            patch = image[y : y + patch_size, x : x + patch_size, :]
            tiles.append({
                "tile_id": tile_id,
                "y": y,
                "x": x,
                "patch_size": patch_size,
                "data": patch,
            })
            tile_id += 1

    return tiles
