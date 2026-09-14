"""Unit tests for Sentinel-2 data inspection module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.data.inspect_data import (
    check_spatial_consistency,
    compute_band_stats,
    find_band_files,
    generate_rgb_preview,
    inspect_sentinel2_data,
    MissingBandError,
    SpatialAlignmentError,
)


def create_dummy_band_geotiff(
    path: Path,
    data: np.ndarray,
    crs: str = "EPSG:32643",
    res: float = 10.0,
    nodata: float = 0.0,
) -> None:
    """Helper to write a georeferenced GeoTIFF for testing."""
    height, width = data.shape
    transform = from_origin(500000.0, 1000000.0, res, res)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def test_compute_band_stats():
    # Simulated 100x100 band with typical Sentinel-2 DN values (0 to 4000)
    data = np.full((100, 100), 2000, dtype=np.uint16)
    data[:10, :10] = 0  # 100 nodata pixels out of 10000 (1%)

    stats = compute_band_stats("B02", data, nodata_val=0)
    assert stats["band"] == "B02"
    assert stats["total_pixels"] == 10000
    assert stats["valid_pixels"] == 9900
    assert stats["nodata_percentage"] == 1.0
    assert stats["raw_mean"] == 2000.0
    assert stats["reflectance_mean"] == 0.2000  # 2000 / 10000
    assert stats["is_scaled_dn"] is True
    assert stats["physically_plausible"] is True


def test_generate_rgb_preview(tmp_path: Path):
    r = np.full((50, 50), 1500, dtype=np.uint16)
    g = np.full((50, 50), 1200, dtype=np.uint16)
    b = np.full((50, 50), 1000, dtype=np.uint16)

    preview_path = tmp_path / "preview.png"
    out = generate_rgb_preview(r, g, b, preview_path)
    assert out.exists()
    assert out.stat().st_size > 0


def test_inspect_sentinel2_data_synthetic(tmp_path: Path):
    # Create 4 matching band files
    bands = ["B02", "B03", "B04", "B08"]
    for b in bands:
        data = np.full((64, 64), 1800, dtype=np.uint16)
        file_path = tmp_path / f"T43REQ_20260901_{b}_10m.tif"
        create_dummy_band_geotiff(file_path, data)

    preview_path = tmp_path / "rgb_preview.png"
    results = inspect_sentinel2_data(
        data_dir_or_file=tmp_path,
        output_preview_path=preview_path,
        target_bands=bands,
    )

    assert results["spatial_metadata"]["shape"] == (64, 64)
    assert results["all_bands_plausible"] is True
    assert preview_path.exists()
    for b in bands:
        assert b in results["band_statistics"]


def test_spatial_mismatch_detection(tmp_path: Path):
    # Create band with mismatching shape
    b2_data = np.full((64, 64), 1800, dtype=np.uint16)
    b3_data = np.full((32, 32), 1800, dtype=np.uint16)

    create_dummy_band_geotiff(tmp_path / "test_B02.tif", b2_data)
    create_dummy_band_geotiff(tmp_path / "test_B03.tif", b3_data)

    with pytest.raises(SpatialAlignmentError):
        inspect_sentinel2_data(
            data_dir_or_file=tmp_path,
            target_bands=["B02", "B03"],
        )
