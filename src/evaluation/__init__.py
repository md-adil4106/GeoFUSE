"""Evaluation, stability testing, spectral consistency, and structural checks."""

from .stability import (
    apply_controlled_perturbation,
    compute_stability_map,
    compare_disagreement_and_stability,
)
from .spectral_check import (
    compute_ndvi,
    compute_spectral_consistency,
    render_spectral_ndvi_overlay,
    SpectralBandError,
)
from .edge_check import (
    extract_canny_edges,
    compute_gradient_magnitude,
    compute_edge_consistency,
    render_edge_consistency_overlay,
)
from .fusion import (
    normalize_evidence_map,
    fuse_trust_risk_maps,
    render_trust_risk_overlay,
)
from .downstream_eval import (
    extract_building_footprints,
    compare_downstream_footprints,
    render_downstream_overlay,
    FootprintExtractionError,
)

__all__ = [
    "apply_controlled_perturbation",
    "compute_stability_map",
    "compare_disagreement_and_stability",
    "compute_ndvi",
    "compute_spectral_consistency",
    "render_spectral_ndvi_overlay",
    "SpectralBandError",
    "extract_canny_edges",
    "compute_gradient_magnitude",
    "compute_edge_consistency",
    "render_edge_consistency_overlay",
    "normalize_evidence_map",
    "fuse_trust_risk_maps",
    "render_trust_risk_overlay",
    "extract_building_footprints",
    "compare_downstream_footprints",
    "render_downstream_overlay",
    "FootprintExtractionError",
]
