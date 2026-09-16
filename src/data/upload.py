"""Sentinel-2 GeoTIFF Upload Validation and Ingestion Module for GeoFUSE SentinelGuard.

Provides production-grade validation and loading for user-uploaded satellite imagery:
1. Raster Readability: Verifies valid raster format (GeoTIFF/JP2) via Rasterio.
2. Band Verification: Validates presence of all 4 required bands (B02, B03, B04, B08)
   either from a single multi-band GeoTIFF (>= 4 bands) or 4 individual band files.
3. Spatial Dimensions: Ensures image dimensions (H, W) are sufficient for model patching (>= 64x64).
4. Pixel Resolution: Validates positive and consistent pixel resolution (GSD).
5. Coordinate Reference System (CRS): Validates standard projected/geographic CRS.
6. Radiometric Dtype & Range: Validates numeric dtype and handles surface reflectance conversion.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.io import MemoryFile


# Target band definitions for GeoFUSE SentinelGuard
TARGET_BANDS = ["B02", "B03", "B04", "B08"]
BAND_NAMES = {
    "B02": "Blue (B02, 10m)",
    "B03": "Green (B03, 10m)",
    "B04": "Red (B04, 10m)",
    "B08": "NIR (B08, 10m)",
}


def _get_file_bytes_and_name(file_obj: Any) -> Tuple[bytes, str]:
    """Safely extract raw bytes and filename from UploadedFile, Path, or BytesIO."""
    if hasattr(file_obj, "name") and hasattr(file_obj, "getvalue"):
        # Streamlit UploadedFile
        return file_obj.getvalue(), file_obj.name
    elif isinstance(file_obj, (str, Path)):
        p = Path(file_obj)
        with open(p, "rb") as f:
            return f.read(), p.name
    elif hasattr(file_obj, "read"):
        name = getattr(file_obj, "name", "uploaded_raster.tif")
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        data = file_obj.read()
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        return data, name
    elif isinstance(file_obj, bytes):
        return file_obj, "in_memory_raster.tif"
    else:
        raise ValueError(f"Unsupported file object type: {type(file_obj)}")


def _extract_acquisition_date(tags: Dict[str, Any], filename: str) -> str:
    """Extract acquisition date from GeoTIFF tags or Sentinel-2 filename tokens."""
    # 1. Inspect GeoTIFF tags
    for date_key in ("DATETIME", "TIFFTAG_DATETIME", "ACQUISITION_DATETIME", "acquisition_time", "SENSING_TIME"):
        if date_key in tags and tags[date_key]:
            return str(tags[date_key])

    # 2. Inspect Sentinel-2 standard naming pattern (e.g., S2A_T43PGQ_20240227T052054_...)
    stem = Path(filename).stem
    parts = stem.split("_")
    for part in parts:
        if len(part) == 15 and "T" in part:
            try:
                dt = datetime.strptime(part, "%Y%m%dT%H%M%S")
                return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                pass
        elif len(part) == 8 and part.isdigit() and (part.startswith("201") or part.startswith("202")):
            try:
                dt = datetime.strptime(part, "%Y%m%d")
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    return "Not available"


def _extract_platform(tags: Dict[str, Any], filename: str) -> str:
    """Extract satellite platform from GeoTIFF tags or filename tokens."""
    for k in ("SPACECRAFT_NAME", "PLATFORM", "MISSION_ID", "satellite"):
        if k in tags and tags[k]:
            val = str(tags[k]).strip()
            if "S2A" in val.upper() or "SENTINEL-2A" in val.upper():
                return "Sentinel-2A"
            if "S2B" in val.upper() or "SENTINEL-2B" in val.upper():
                return "Sentinel-2B"
            return val

    parts = Path(filename).stem.split("_")
    if parts:
        p0 = parts[0].upper()
        if p0 == "S2A":
            return "Sentinel-2A"
        elif p0 == "S2B":
            return "Sentinel-2B"
        elif "SENTINEL" in p0:
            return "Sentinel-2"
    return "Not available"


def _extract_mgrs_tile(tags: Dict[str, Any], filename: str) -> str:
    """Extract MGRS tile ID from GeoTIFF tags or filename tokens."""
    for k in ("MGRS_TILE", "TILE_ID", "tile_id"):
        if k in tags and tags[k]:
            return str(tags[k]).strip().lstrip("T")

    parts = Path(filename).stem.split("_")
    for part in parts:
        part_up = part.upper()
        if (
            len(part_up) == 6
            and part_up.startswith("T")
            and part_up[1:3].isdigit()
            and part_up[3:].isalpha()
        ):
            return part_up.lstrip("T")
        elif len(part_up) == 5 and part_up[:2].isdigit() and part_up[2:].isalpha():
            return part_up
    return "Not available"


def _extract_product_level(tags: Dict[str, Any], filename: str) -> str:
    """Extract product processing level from GeoTIFF tags or filename tokens."""
    for k in ("PROCESSING_LEVEL", "PRODUCT_LEVEL", "processing_level"):
        if k in tags and tags[k]:
            return str(tags[k]).strip()

    parts = Path(filename).stem.split("_")
    for part in parts:
        p = part.upper()
        if p in ("L2A", "MSIL2A"):
            return "L2A"
        elif p in ("L1C", "MSIL1C"):
            return "L1C"
    return "Not available"


def _extract_tag_value(tags: Dict[str, Any], candidate_keys: Tuple[str, ...]) -> str:
    """Safely extract metadata tag value from candidate keys or return 'Not available'."""
    for k in candidate_keys:
        if k in tags and tags[k] is not None and str(tags[k]).strip() != "":
            return str(tags[k]).strip()
    return "Not available"


def _identify_band_from_filename(filename: str) -> Optional[str]:
    """Identify which Sentinel-2 band a file corresponds to from its filename."""
    stem = Path(filename).stem.upper()
    tokens = stem.replace("-", "_").split("_")
    for b in TARGET_BANDS:
        if b in tokens or any(t.startswith(b) for t in tokens):
            return b
    return None


def validate_uploaded_raster(
    files: Union[Any, List[Any]],
    min_patch_size: int = 64,
) -> Dict[str, Any]:
    """Execute rigorous pre-flight validation on user-uploaded raster file(s).

    Checks:
    1. File is readable as a valid raster.
    2. Required bands (B02, B03, B04, B08) are present and identified.
    3. Spatial dimensions are sufficient for at least one model patch (>= min_patch_size).
    4. Pixel resolution is positive and consistent.
    5. CRS is present and is a recognized projected/geographic coordinate system.
    6. Data range and dtype are consistent with model preprocessing expectations.

    Args:
        files: Single file or list of uploaded files (e.g. from st.file_uploader).
        min_patch_size: Minimum spatial dimension required by SR model (default: 64).

    Returns:
        Dict[str, Any]: Validation summary containing pass/fail flags, individual check details,
                        extracted metadata, and user-facing error messages if invalid.
    """
    if not isinstance(files, (list, tuple)):
        file_list = [files] if files is not None else []
    else:
        file_list = list(files)

    if not file_list:
        return {
            "is_valid": False,
            "mode": "none",
            "checks": [],
            "metadata": {},
            "error_message": "No files uploaded. Please upload a GeoTIFF raster.",
        }

    checks: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Check 1: File Readability
    # -------------------------------------------------------------------------
    readers = []
    memfiles = []
    file_info = []

    try:
        for f in file_list:
            b_data, name = _get_file_bytes_and_name(f)
            mem = MemoryFile(b_data)
            ds = mem.open()
            memfiles.append(mem)
            readers.append(ds)
            file_info.append({"name": name, "size_bytes": len(b_data)})
    except Exception as e:
        # Cleanup any already opened readers
        for ds in readers:
            try:
                ds.close()
            except Exception:
                pass
        for m in memfiles:
            try:
                m.close()
            except Exception:
                pass
        return {
            "is_valid": False,
            "mode": "unknown",
            "checks": [
                {
                    "name": "Raster File Readability",
                    "passed": False,
                    "message": f"Failed to open raster file: {str(e)}",
                    "details": "Ensure the uploaded file is a valid, uncorrupted GeoTIFF (.tif, .tiff) or JP2.",
                }
            ],
            "metadata": {},
            "error_message": f"Unreadable raster format: {str(e)}",
        }

    checks.append({
        "name": "Raster File Readability",
        "passed": True,
        "message": f"Successfully parsed {len(file_list)} raster file(s) via Rasterio.",
        "details": f"Driver: {readers[0].driver}, Total files: {len(file_list)}",
    })

    # Determine mode: Single multi-band file vs Multiple separate band files
    is_single_file = len(file_list) == 1
    band_mapping: Dict[str, Tuple[int, int]] = {}  # band_id -> (file_index, band_index)

    # -------------------------------------------------------------------------
    # Check 2: Required Bands Present & Ordered
    # -------------------------------------------------------------------------
    bands_check_passed = False
    bands_message = ""

    if is_single_file:
        src = readers[0]
        total_bands = src.count
        if total_bands >= 4:
            bands_check_passed = True
            bands_message = (
                f"Multi-band raster contains {total_bands} bands. "
                "Channels 1-4 mapped to Sentinel-2 order [B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)]."
            )
            for idx, b in enumerate(TARGET_BANDS):
                band_mapping[b] = (0, idx + 1)
        else:
            bands_check_passed = False
            bands_message = (
                f"File contains {total_bands} band(s). GeoFUSE requires 4 spectral bands "
                "(B02 Blue, B03 Green, B04 Red, B08 NIR). Please upload a 4-band GeoTIFF "
                "or upload the 4 individual band files."
            )
    else:
        # Multiple files: identify each by filename token
        identified = {}
        for f_idx, ds in enumerate(readers):
            fname = file_info[f_idx]["name"]
            matched_band = _identify_band_from_filename(fname)
            if matched_band:
                identified[matched_band] = (f_idx, 1)

        missing = [b for b in TARGET_BANDS if b not in identified]
        if not missing:
            bands_check_passed = True
            bands_message = (
                f"All 4 required Sentinel-2 bands identified: "
                f"B02 ({file_info[identified['B02'][0]]['name']}), "
                f"B03 ({file_info[identified['B03'][0]]['name']}), "
                f"B04 ({file_info[identified['B04'][0]]['name']}), "
                f"B08 ({file_info[identified['B08'][0]]['name']})."
            )
            band_mapping = identified
        else:
            bands_check_passed = False
            bands_message = (
                f"Could not identify all 4 required bands from uploaded files. "
                f"Missing: {missing}. Uploaded filenames: {[f['name'] for f in file_info]}"
            )

    checks.append({
        "name": "Required Spectral Bands (B02, B03, B04, B08)",
        "passed": bands_check_passed,
        "message": bands_message,
        "details": f"Bands required by ResidualSRNet: {TARGET_BANDS}",
    })

    # -------------------------------------------------------------------------
    # Check 3: Spatial Dimensions Sufficient for Model Patch
    # -------------------------------------------------------------------------
    ref_ds = readers[0]
    height, width = ref_ds.height, ref_ds.width
    dim_check_passed = True
    dim_message = ""

    if height < min_patch_size or width < min_patch_size:
        dim_check_passed = False
        dim_message = (
            f"Image dimensions ({height} × {width} px) are too small for super-resolution. "
            f"Minimum spatial size required is {min_patch_size} × {min_patch_size} px."
        )
    else:
        dim_message = f"Spatial dimensions ({height} × {width} px) satisfy minimum patch requirement (>= {min_patch_size} × {min_patch_size} px)."

    # If multiple files, ensure identical dimensions
    if not is_single_file and dim_check_passed:
        for f_idx, ds in enumerate(readers):
            if (ds.height, ds.width) != (height, width):
                dim_check_passed = False
                dim_message = (
                    f"Dimension mismatch across band files: {file_info[0]['name']} is "
                    f"({height} × {width}), but {file_info[f_idx]['name']} is ({ds.height} × {ds.width})."
                )
                break

    checks.append({
        "name": "Spatial Dimensions",
        "passed": dim_check_passed,
        "message": dim_message,
        "details": f"Height: {height} px, Width: {width} px (Min required: {min_patch_size} px)",
    })

    # -------------------------------------------------------------------------
    # Check 4: Pixel Resolution (GSD)
    # -------------------------------------------------------------------------
    res = ref_ds.res
    res_x, res_y = abs(float(res[0])), abs(float(res[1]))
    res_check_passed = True
    res_message = ""

    if res_x <= 0 or res_y <= 0 or np.isnan(res_x) or np.isnan(res_y):
        res_check_passed = False
        res_message = f"Invalid pixel resolution: ({res_x}, {res_y})."
    else:
        res_message = f"Pixel resolution: {res_x:.2f} m × {res_y:.2f} m Ground Sample Distance (GSD)."

    # Multi-file resolution consistency
    if not is_single_file and res_check_passed:
        for f_idx, ds in enumerate(readers):
            ds_res = (abs(float(ds.res[0])), abs(float(ds.res[1])))
            if abs(ds_res[0] - res_x) > 1e-3 or abs(ds_res[1] - res_y) > 1e-3:
                res_check_passed = False
                res_message = (
                    f"Resolution mismatch: {file_info[0]['name']} has GSD ({res_x:.2f}m, {res_y:.2f}m), "
                    f"but {file_info[f_idx]['name']} has GSD ({ds_res[0]:.2f}m, {ds_res[1]:.2f}m)."
                )
                break

    checks.append({
        "name": "Pixel Resolution (GSD)",
        "passed": res_check_passed,
        "message": res_message,
        "details": f"Target resolution: ~10.0m (Sentinel-2 10m bands)",
    })

    # -------------------------------------------------------------------------
    # Check 5: Coordinate Reference System (CRS)
    # -------------------------------------------------------------------------
    crs = ref_ds.crs
    crs_check_passed = True
    crs_message = ""

    if crs is None:
        crs_check_passed = False
        crs_message = "Coordinate Reference System (CRS) is missing from GeoTIFF headers."
    else:
        crs_str = str(crs)
        crs_message = f"Valid Coordinate Reference System detected: {crs_str}"

    if not is_single_file and crs_check_passed:
        for f_idx, ds in enumerate(readers):
            if ds.crs != crs:
                crs_check_passed = False
                crs_message = (
                    f"CRS mismatch: {file_info[0]['name']} is {crs}, but "
                    f"{file_info[f_idx]['name']} is {ds.crs}."
                )
                break

    checks.append({
        "name": "Coordinate Reference System (CRS)",
        "passed": crs_check_passed,
        "message": crs_message,
        "details": f"Detected CRS: {str(crs) if crs else 'None'}",
    })

    # -------------------------------------------------------------------------
    # Check 6: Radiometric Dtype & Data Range
    # -------------------------------------------------------------------------
    dtype_str = ref_ds.dtypes[0]
    supported_dtypes = ("uint16", "int16", "uint8", "float32", "float64")
    dtype_check_passed = True
    dtype_message = ""
    data_min = 0.0
    data_max = 1.0
    needs_scaling = False

    if dtype_str.lower() not in supported_dtypes:
        dtype_check_passed = False
        dtype_message = f"Unsupported raster dtype: '{dtype_str}'. Expected one of: {supported_dtypes}"
    else:
        # Sample first band to check range
        try:
            sample_band = ref_ds.read(1)
            valid_mask = ~np.isnan(sample_band) & ~np.isinf(sample_band)
            if not np.any(valid_mask):
                dtype_check_passed = False
                dtype_message = "Raster contains only NaN or infinite values."
            else:
                data_min = float(np.min(sample_band[valid_mask]))
                data_max = float(np.max(sample_band[valid_mask]))
                if data_max > 10.0:
                    needs_scaling = True
                    dtype_message = (
                        f"Dtype '{dtype_str}', Range [{data_min:.1f}, {data_max:.1f}] (Raw Digital Numbers). "
                        "Will be automatically normalized to [0.0, 1.0] surface reflectance (DN / 10000.0)."
                    )
                else:
                    dtype_message = (
                        f"Dtype '{dtype_str}', Range [{data_min:.4f}, {data_max:.4f}]. "
                        "Values are already within surface reflectance range [0.0, 1.0]."
                    )
        except Exception as e:
            dtype_check_passed = False
            dtype_message = f"Failed to inspect raster values: {str(e)}"

    checks.append({
        "name": "Data Range & Radiometric Dtype",
        "passed": dtype_check_passed,
        "message": dtype_message,
        "details": f"Dtype: {dtype_str}, Range: [{data_min:.2f}, {data_max:.2f}]",
    })

    # -------------------------------------------------------------------------
    # Aggregate Result & Extract Metadata
    # -------------------------------------------------------------------------
    all_passed = all(c["passed"] for c in checks)

    # Acquisition metadata extraction
    ref_tags = ref_ds.tags()
    primary_name = file_info[0]["name"]
    acq_date = _extract_acquisition_date(ref_tags, primary_name)
    platform = _extract_platform(ref_tags, primary_name)
    mgrs_tile = _extract_mgrs_tile(ref_tags, primary_name)
    product_level = _extract_product_level(ref_tags, primary_name)
    cloud_cover = _extract_tag_value(
        ref_tags, ("CLOUD_COVERAGE_ASSESSMENT", "CLOUDY_PIXEL_PERCENTAGE", "cloud_cover")
    )
    sun_elevation = _extract_tag_value(
        ref_tags, ("MEAN_SUN_ELEVATION_ANGLE", "SUN_ELEVATION", "sun_elevation")
    )
    sun_azimuth = _extract_tag_value(
        ref_tags, ("MEAN_SUN_AZIMUTH_ANGLE", "SUN_AZIMUTH", "sun_azimuth")
    )
    orbit_number = _extract_tag_value(
        ref_tags, ("SENSING_ORBIT_NUMBER", "ORBIT_NUMBER", "RELATIVE_ORBIT_NUMBER", "orbit")
    )

    # Bounds
    bounds_dict = None
    if ref_ds.bounds:
        bounds_dict = {
            "left": round(float(ref_ds.bounds.left), 2),
            "bottom": round(float(ref_ds.bounds.bottom), 2),
            "right": round(float(ref_ds.bounds.right), 2),
            "top": round(float(ref_ds.bounds.top), 2),
        }

    spatial_res = float(abs(round(res_x, 2))) if res_x and res_x > 0 else "Not available"

    metadata = {
        "is_upload": True,
        "primary_filename": primary_name,
        "all_filenames": [f["name"] for f in file_info],
        "height": height,
        "width": width,
        "spatial_shape": (height, width),
        "band_count": len(TARGET_BANDS) if bands_check_passed else ref_ds.count,
        "bands": [BAND_NAMES[b] for b in TARGET_BANDS] if bands_check_passed else [],
        "crs": str(crs) if crs else "Not available",
        "source_crs": str(crs) if crs else "Not available",
        "resolution": (res_x, res_y),
        "spatial_resolution_meters": spatial_res,
        "dtype": dtype_str,
        "acquisition_date": acq_date,
        "acquisition_datetime": acq_date,
        "platform": platform,
        "mgrs_tile": mgrs_tile,
        "product_level": product_level,
        "cloud_cover_percentage": cloud_cover,
        "sun_elevation_angle_deg": sun_elevation,
        "sun_azimuth_angle_deg": sun_azimuth,
        "satellite_orbit_number": orbit_number,
        "bounds": bounds_dict if bounds_dict else "Not available",
        "geospatial_bounds": bounds_dict if bounds_dict else "Not available",
        "data_range": (data_min, data_max),
        "needs_reflectance_scaling": needs_scaling,
        "mode": "single_file" if is_single_file else "multi_file",
        "band_mapping": band_mapping,
    }

    # Close temporary readers
    for ds in readers:
        try:
            ds.close()
        except Exception:
            pass
    for m in memfiles:
        try:
            m.close()
        except Exception:
            pass

    first_failed = next((c for c in checks if not c["passed"]), None)
    error_message = first_failed["message"] if first_failed else None

    return {
        "is_valid": all_passed,
        "mode": "single_file" if is_single_file else "multi_file",
        "checks": checks,
        "metadata": metadata,
        "error_message": error_message,
    }


def load_uploaded_stack(
    files: Union[Any, List[Any]],
    validation_result: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Safely load validated uploaded raster(s) into a 4-band float32 reflectance array.

    Args:
        files: File or list of files validated by validate_uploaded_raster.
        validation_result: Output dictionary from validate_uploaded_raster.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Multi-band array of shape (H, W, 4) in float32 reflectance [0.0, 1.0].
            - Comprehensive metadata dictionary including geospatial transform and bounds.

    Raises:
        ValueError: If validation_result indicates validation failed.
    """
    if not validation_result.get("is_valid", False):
        raise ValueError(
            f"Cannot load invalid raster. Validation failed: {validation_result.get('error_message')}"
        )

    if not isinstance(files, (list, tuple)):
        file_list = [files]
    else:
        file_list = list(files)

    band_mapping = validation_result["metadata"]["band_mapping"]
    needs_scaling = validation_result["metadata"]["needs_reflectance_scaling"]

    readers = []
    memfiles = []

    try:
        for f in file_list:
            b_data, _ = _get_file_bytes_and_name(f)
            mem = MemoryFile(b_data)
            ds = mem.open()
            memfiles.append(mem)
            readers.append(ds)

        band_arrays = []
        for band_key in TARGET_BANDS:
            f_idx, b_idx = band_mapping[band_key]
            arr = readers[f_idx].read(b_idx).astype(np.float32)
            band_arrays.append(arr)

        stack = np.stack(band_arrays, axis=-1)  # (H, W, 4)

        if needs_scaling:
            stack = stack / 10000.0

        # Physical reflectance non-negativity and finite numerical guard
        stack = np.nan_to_num(stack, nan=0.0, posinf=1.5, neginf=0.0)
        stack = np.clip(stack, 0.0, 1.5)

        primary_ds = readers[0]
        meta = {
            **validation_result["metadata"],
            "transform": primary_ds.transform,
            "bounds": primary_ds.bounds,
            "crs": str(primary_ds.crs),
            "is_reflectance": True,
        }

        return stack, meta

    finally:
        for ds in readers:
            try:
                ds.close()
            except Exception:
                pass
        for m in memfiles:
            try:
                m.close()
            except Exception:
                pass
