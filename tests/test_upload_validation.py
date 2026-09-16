"""Unit tests for Phase B: GeoTIFF Upload Workflow and Validation Module."""

from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from src.data.upload import validate_uploaded_raster, load_uploaded_stack
from src.utils.config import get_project_root


def create_in_memory_geotiff(
    shape=(128, 128),
    bands=4,
    dtype="uint16",
    crs="EPSG:32643",
    res=(10.0, 10.0),
    data_range=(500, 3000),
    filename="test_raster.tif",
) -> bytes:
    """Helper to generate in-memory synthetic GeoTIFF bytes."""
    h, w = shape
    data = np.random.randint(data_range[0], data_range[1], (bands, h, w), dtype=np.uint16)
    transform = from_origin(500000, 3000000, res[0], res[1])

    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            height=h,
            width=w,
            count=bands,
            dtype=dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data)
        return bytes(mem.getbuffer())


def test_validate_real_sentinel2_bands_pass():
    """Verify that real Sentinel-2 4-band files from data/raw pass all 6 validation checks."""
    root = get_project_root()
    raw_dir = root / "data/raw"
    band_files = sorted(list(raw_dir.glob("S2A_*.tif")))
    assert len(band_files) == 4, f"Expected 4 raw files in {raw_dir}, found {len(band_files)}"

    result = validate_uploaded_raster(band_files)
    assert result["is_valid"] is True, f"Validation unexpectedly failed: {result.get('error_message')}"
    assert result["mode"] == "multi_file"
    assert len(result["checks"]) == 6
    assert all(c["passed"] for c in result["checks"])

    meta = result["metadata"]
    assert meta["height"] == 512
    assert meta["width"] == 512
    assert meta["crs"] == "EPSG:32643"
    assert meta["band_count"] == 4
    assert meta["needs_reflectance_scaling"] is True

    # Test loading
    stack, load_meta = load_uploaded_stack(band_files, result)
    assert stack.shape == (512, 512, 4)
    assert stack.dtype == np.float32
    assert 0.0 <= stack.min() and stack.max() <= 1.5


def test_validate_single_4band_geotiff_pass():
    """Verify that a single 4-band GeoTIFF passes all 6 validation checks."""
    buf = create_in_memory_geotiff(shape=(128, 128), bands=4)
    # Wrap in BytesIO with name
    from io import BytesIO
    bio = BytesIO(buf)
    bio.name = "single_scene_4band.tif"

    result = validate_uploaded_raster([bio])
    assert result["is_valid"] is True
    assert result["mode"] == "single_file"
    assert len(result["checks"]) == 6
    assert all(c["passed"] for c in result["checks"])

    meta = result["metadata"]
    assert meta["height"] == 128
    assert meta["width"] == 128
    assert meta["crs"] == "EPSG:32643"
    assert meta["band_count"] == 4


def test_validate_corrupt_file_fail():
    """Verify that corrupt or invalid raster bytes fail Check 1 gracefully without crashing."""
    corrupt_bytes = b"NOT_A_VALID_TIFF_HEADER_RANDOM_CORRUPT_BYTES"
    from io import BytesIO
    bio = BytesIO(corrupt_bytes)
    bio.name = "corrupted_raster.tif"

    result = validate_uploaded_raster([bio])
    assert result["is_valid"] is False
    assert result["checks"][0]["passed"] is False
    assert "Unreadable" in result["error_message"] or "Failed" in result["error_message"]


def test_validate_missing_bands_fail():
    """Verify that uploading only 3 bands fails Check 2 with clear guidance."""
    root = get_project_root()
    raw_dir = root / "data/raw"
    band_files = sorted(list(raw_dir.glob("S2A_*.tif")))[:3]  # Only B02, B03, B04

    result = validate_uploaded_raster(band_files)
    assert result["is_valid"] is False
    # Check 2 (Required Spectral Bands) should fail
    bands_check = next(c for c in result["checks"] if "Spectral Bands" in c["name"])
    assert bands_check["passed"] is False
    assert "B08" in bands_check["message"]


def test_validate_insufficient_dimensions_fail():
    """Verify that an image smaller than 64x64 fails Check 3 (spatial dimensions)."""
    buf = create_in_memory_geotiff(shape=(32, 32), bands=4)
    from io import BytesIO
    bio = BytesIO(buf)
    bio.name = "tiny_patch_32x32.tif"

    result = validate_uploaded_raster([bio], min_patch_size=64)
    assert result["is_valid"] is False
    dim_check = next(c for c in result["checks"] if "Spatial Dimensions" in c["name"])
    assert dim_check["passed"] is False
    assert "too small" in dim_check["message"]


def test_validate_missing_crs_fail():
    """Verify that a raster lacking CRS metadata fails Check 5."""
    buf = create_in_memory_geotiff(shape=(128, 128), bands=4, crs=None)
    from io import BytesIO
    bio = BytesIO(buf)
    bio.name = "no_crs_scene.tif"

    result = validate_uploaded_raster([bio])
    assert result["is_valid"] is False
    crs_check = next(c for c in result["checks"] if "CRS" in c["name"])
    assert crs_check["passed"] is False
    assert "missing" in crs_check["message"].lower()


def test_load_uploaded_stack_blocks_invalid():
    """Verify that load_uploaded_stack raises ValueError if given invalid validation result."""
    with pytest.raises(ValueError, match="Cannot load invalid raster"):
        load_uploaded_stack([], {"is_valid": False, "error_message": "Invalid CRS"})
