"""Data preprocessing, satellite band handling, degradation, and patch tiling module."""

from .inspect_data import (
    inspect_sentinel2_data,
    find_band_files,
    check_spatial_consistency,
    compute_band_stats,
    generate_rgb_preview,
    SpatialAlignmentError,
    MissingBandError,
)

from .degrade import (
    bicubic_downsample,
    bicubic_upsample,
    apply_sensor_blur,
    add_sensor_noise,
    synthesize_pseudo_lr,
    evaluate_reconstruction_fidelity,
)

from .tiling import (
    load_sentinel2_stack,
    extract_tiles,
)

__all__ = [
    "inspect_sentinel2_data",
    "find_band_files",
    "check_spatial_consistency",
    "compute_band_stats",
    "generate_rgb_preview",
    "SpatialAlignmentError",
    "MissingBandError",
    "bicubic_downsample",
    "bicubic_upsample",
    "apply_sensor_blur",
    "add_sensor_noise",
    "synthesize_pseudo_lr",
    "evaluate_reconstruction_fidelity",
    "load_sentinel2_stack",
    "extract_tiles",
]
