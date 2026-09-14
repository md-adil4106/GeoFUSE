"""Utility script to fetch a lightweight, legitimate Sentinel-2 L2A sample scene.

Downloads a 512x512 pixel crop (approx. 5.12 km x 5.12 km, ~2 MB total)
from the AWS Open Data Sentinel-2 Level-2A archive (Element 84 Earth Search catalog)
for bands B02, B03, B04, and B08, preserving complete georeferencing and CRS.
No credentials, tokens, or account setup required.
"""

import sys
from pathlib import Path
from typing import Dict

import rasterio
from rasterio.windows import Window, transform as window_transform

# Project root setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_project_root

SCENE_ID = "S2A_T43PGQ_20240227T052054_L2A"
BASE_URL = f"https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/sentinel-2-c1-l2a/43/P/GQ/2024/2/{SCENE_ID}"

BAND_URLS: Dict[str, str] = {
    "B02": f"{BASE_URL}/B02.tif",
    "B03": f"{BASE_URL}/B03.tif",
    "B04": f"{BASE_URL}/B04.tif",
    "B08": f"{BASE_URL}/B08.tif",
}

# Sub-window selection: 512x512 pixels covering diverse urban-vegetation landscape
CROP_COL = 5000
CROP_ROW = 5000
CROP_WIDTH = 512
CROP_HEIGHT = 512


def download_sample_scene(output_dir: Path) -> Dict[str, Path]:
    """Download the 4-band crop into the target directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    win = Window(CROP_COL, CROP_ROW, CROP_WIDTH, CROP_HEIGHT)
    downloaded_files: Dict[str, Path] = {}

    print(f"Dataset source  : AWS Open Data (Earth Search / Sentinel-2 L2A)")
    print(f"Scene Identifier: {SCENE_ID}")
    print(f"Crop Window     : col={CROP_COL}, row={CROP_ROW}, size=({CROP_HEIGHT}x{CROP_WIDTH})")
    print(f"Target Directory: {output_dir.resolve()}\n")

    for band_name, url in BAND_URLS.items():
        out_file = output_dir / f"{SCENE_ID}_{band_name}_10m.tif"
        print(f"Fetching {band_name} from COG stream...")

        with rasterio.open(url) as src:
            data = src.read(1, window=win)
            crop_meta = src.meta.copy()
            crop_meta.update({
                "driver": "GTiff",
                "height": CROP_HEIGHT,
                "width": CROP_WIDTH,
                "transform": window_transform(win, src.transform),
                "count": 1,
                "compress": "deflate",
            })

            with rasterio.open(out_file, "w", **crop_meta) as dst:
                dst.write(data, 1)

        size_kb = out_file.stat().st_size / 1024.0
        print(f"  [OK] Saved {out_file.name} ({size_kb:.1f} KB, shape={data.shape})")
        downloaded_files[band_name] = out_file

    print(f"\nAll {len(downloaded_files)} bands successfully downloaded and saved.")
    return downloaded_files


def main() -> int:
    config = load_config()
    target_rel = config.get("paths", {}).get("raw_data_dir", "data/raw")
    root = get_project_root()
    target_dir = root / target_rel

    try:
        download_sample_scene(target_dir)
        return 0
    except Exception as e:
        print(f"\n[ERROR] Failed to download sample scene: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
