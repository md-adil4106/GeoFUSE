"""Command-line script to inspect Sentinel-2 L2A data for GeoFUSE SentinelGuard.

Loads bands specified in config.yaml, checks spatial alignment & CRS,
prints radiometric reflectance statistics, and saves an RGB preview.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_project_root
from src.data.inspect_data import (
    inspect_sentinel2_data,
    SpatialAlignmentError,
    MissingBandError,
)


def main() -> int:
    print("=" * 72)
    print("   GeoFUSE SentinelGuard — Sentinel-2 L2A Data Inspection")
    print("=" * 72)

    root = get_project_root()
    try:
        config = load_config()
    except Exception as e:
        print(f"\n[ERROR] Failed to load config.yaml: {e}")
        return 1

    # Resolve data path from config.yaml
    raw_path_str = config.get("paths", {}).get("raw_data_dir", "data/raw")
    raw_path = Path(raw_path_str)
    if not raw_path.is_absolute():
        raw_path = root / raw_path

    target_bands = config.get("preprocessing", {}).get("bands", ["B02", "B03", "B04", "B08"])
    preview_output = root / config.get("paths", {}).get("outputs_dir", "outputs") / "previews" / "rgb_preview.png"

    print(f"\nConfigured raw data path : {raw_path.resolve()}")
    print(f"Target bands             : {target_bands}")
    print(f"Target preview output    : {preview_output.resolve()}")

    # Check if directory or file exists and has content
    if not raw_path.exists():
        print(f"\n[STOP CONDITION TRIGGERED] Data path does not exist: {raw_path.resolve()}")
        print("Please update 'paths.raw_data_dir' in config.yaml with the path to your Sentinel-2 data.")
        return 2

    # Check if empty directory
    if raw_path.is_dir():
        files = list(raw_path.glob("*.tif")) + list(raw_path.glob("*.tiff")) + list(raw_path.glob("*.jp2")) + list(raw_path.glob("**/*.tif"))
        if not files:
            print(f"\n[STOP CONDITION TRIGGERED] No GeoTIFF/JP2 satellite band files found in: {raw_path.resolve()}")
            print("Please place your Sentinel-2 L2A band files (e.g. B02, B03, B04, B08) into 'data/raw/'")
            print("or update 'paths.raw_data_dir' in config.yaml with the folder path containing your scene.")
            return 2

    # Run inspection
    try:
        results = inspect_sentinel2_data(
            data_dir_or_file=raw_path,
            output_preview_path=preview_output,
            target_bands=target_bands,
        )
    except MissingBandError as e:
        print(f"\n[STOP CONDITION TRIGGERED: Missing Band(s)]\n{e}")
        return 2
    except SpatialAlignmentError as e:
        print(f"\n[STOP CONDITION TRIGGERED: Spatial Alignment / CRS Mismatch]\n{e}")
        return 2
    except Exception as e:
        print(f"\n[ERROR] Inspection failed unexpectedly: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Report spatial metadata
    spatial = results["spatial_metadata"]
    print("\n" + "-" * 72)
    print(" SPATIAL ALIGNMENT & METADATA CHECK: PASSED")
    print("-" * 72)
    print(f"  CRS         : {spatial['crs']}")
    print(f"  Shape (H, W): {spatial['shape']}")
    print(f"  Resolution  : {spatial['resolution']}")
    print(f"  Bounds      : {spatial['bounds']}")

    # Report band statistics
    print("\n" + "-" * 72)
    print(" RADIOMETRIC REFLECTANCE STATISTICS & HEALTH CHECK")
    print("-" * 72)
    header = f"{'Band':<6} | {'Raw Min':<9} | {'Raw Max':<9} | {'Raw Mean':<9} | {'Refl Mean':<10} | {'NoData %':<9} | {'Plausible?'}"
    print(header)
    print("-" * len(header))

    stats = results["band_statistics"]
    for band in target_bands:
        b_info = stats[band]
        refl_mean_str = f"{b_info['reflectance_mean']:.4f}" if b_info['reflectance_mean'] is not None else "N/A"
        plausible_str = "YES [OK]" if b_info["physically_plausible"] else "WARN [CHECK]"
        print(
            f"{band:<6} | "
            f"{str(b_info['raw_min']):<9} | "
            f"{str(b_info['raw_max']):<9} | "
            f"{str(b_info['raw_mean']):<9} | "
            f"{refl_mean_str:<10} | "
            f"{b_info['nodata_percentage']:<8}% | "
            f"{plausible_str}"
        )

    # Preview report
    print("\n" + "-" * 72)
    print(f" RGB PREVIEW GENERATED: {results['preview_path']}")
    print("-" * 72)

    if results["all_bands_plausible"]:
        print("\n[SUCCESS] Input data is verified, spatially aligned, and ready for pipeline integration!")
        return 0
    else:
        print("\n[WARNING] Some bands have values outside typical surface reflectance ranges. Please verify sensor calibration.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
