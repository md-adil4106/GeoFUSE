"""Unit tests for Phase 11: Auditable Trust Receipt Generation."""

import json
from pathlib import Path

import pytest

from src.evaluation.trust_receipt import (
    extract_geotiff_metadata,
    generate_trust_receipt,
    render_trust_receipt_html,
    save_trust_receipt,
)
from src.utils.config import get_project_root, load_config


def test_extract_geotiff_metadata_real():
    root = get_project_root()
    raw_dir = root / "data/raw"

    meta = extract_geotiff_metadata(raw_dir)

    assert meta["platform"] == "Sentinel-2A"
    assert meta["mgrs_tile"] == "43PGQ"
    assert meta["source_crs"] == "EPSG:32643"
    assert meta["spatial_resolution_meters"] == 10.0
    assert "geospatial_bounds" in meta
    # Unextractable fields must be marked strictly as "unavailable" per scientific honesty mandate
    assert meta["cloud_cover_percentage"] == "unavailable"
    assert meta["sun_elevation_angle_deg"] == "unavailable"
    assert meta["sun_azimuth_angle_deg"] == "unavailable"
    assert meta["satellite_orbit_number"] == "unavailable"


def test_extract_geotiff_metadata_empty_dir(tmp_path):
    meta = extract_geotiff_metadata(tmp_path)
    assert meta["source_file"] == "unavailable"
    assert meta["source_crs"] == "unavailable"
    assert meta["cloud_cover_percentage"] == "unavailable"


def test_generate_trust_receipt_nominal():
    root = get_project_root()
    raw_dir = root / "data/raw"
    config = load_config()

    mock_pipeline_data = {
        "fusion_result": {
            "trust_score_pct": 88.50,
            "mean_risk_score": 0.1150,
            "pct_high_risk_pixels": 0.0,
            "weights_used": {"disagreement": 0.25, "stability": 0.25, "spectral": 0.25, "structural": 0.25},
            "component_stats": {
                "disagreement": {"raw_mean": 0.0012, "raw_max": 0.008},
                "stability": {"raw_mean": 0.0001, "raw_max": 0.0005},
                "spectral": {"raw_mean": 0.015, "raw_max": 0.09},
                "structural": {"raw_mean": 0.035, "raw_max": 0.20},
            },
        },
        "spectral_metrics": {
            "mean_delta_ndvi": 0.0185,
            "max_delta_ndvi": 0.095,
            "pct_inconsistent_pixels": 4.2,
            "is_spectrally_consistent": True,
        },
        "edge_metrics": {
            "gradient_correlation": 0.8250,
            "edge_iou": 0.4420,
            "edge_f1": 0.6130,
        },
        "downstream_comp": {
            "scientific_honesty_label": "Bicubic-vs-SR Footprint Agreement",
            "overall_bic_sr": {"iou": 0.9350, "dice": 0.9664},
            "high_trust_bic_sr": {"iou": 0.9420},
            "low_trust_bic_sr": {"iou": 0.9210},
            "high_trust_area_pct": 72.5,
            "reference_comparison": {"sr_vs_ref_iou": 0.5820},
        },
    }

    receipt = generate_trust_receipt(
        tile_idx=0,
        raw_dir=raw_dir,
        config=config,
        pipeline_data=mock_pipeline_data,
        min_trust_threshold=86.5,
    )

    assert receipt["trust_evaluation"]["status"] == "NOMINAL_HIGH_TRUST"
    assert receipt["trust_evaluation"]["is_trusted"] is True
    assert len(receipt["trust_evaluation"]["warnings_and_advisories"]) == 0
    assert receipt["evidence_metrics"]["fused_trust_score_pct"] == 88.50
    assert "ResidualSRNet" in receipt["model_provenance"]["architecture"]


def test_generate_trust_receipt_low_trust_warning():
    """Verify that a low trust score triggers the plain-language warning."""
    root = get_project_root()
    raw_dir = root / "data/raw"
    config = load_config()

    mock_pipeline_data = {
        "fusion_result": {
            "trust_score_pct": 84.20,  # Below threshold 86.5%
            "mean_risk_score": 0.1580,
            "pct_high_risk_pixels": 2.5,
            "weights_used": {"disagreement": 0.25, "stability": 0.25, "spectral": 0.25, "structural": 0.25},
            "component_stats": {
                "disagreement": {"raw_mean": 0.0035, "raw_max": 0.025},
                "stability": {"raw_mean": 0.0008, "raw_max": 0.004},
                "spectral": {"raw_mean": 0.038, "raw_max": 0.22},
                "structural": {"raw_mean": 0.082, "raw_max": 0.45},
            },
        },
        "spectral_metrics": {
            "mean_delta_ndvi": 0.038,
            "max_delta_ndvi": 0.22,
            "pct_inconsistent_pixels": 12.8,  # Triggers spectral advisory (> 7.5%)
            "is_spectrally_consistent": False,
        },
        "edge_metrics": {
            "gradient_correlation": 0.6850,
            "edge_iou": 0.3120,
            "edge_f1": 0.4750,
        },
        "downstream_comp": {
            "scientific_honesty_label": "Bicubic-vs-SR Footprint Agreement",
            "overall_bic_sr": {"iou": 0.8650, "dice": 0.9276},
            "high_trust_bic_sr": {"iou": 0.8920},
            "low_trust_bic_sr": {"iou": 0.8410},
            "high_trust_area_pct": 54.0,
            "reference_comparison": {"sr_vs_ref_iou": 0.4120},
        },
    }

    receipt = generate_trust_receipt(
        tile_idx=3,
        raw_dir=raw_dir,
        config=config,
        pipeline_data=mock_pipeline_data,
        min_trust_threshold=86.5,
    )

    assert receipt["trust_evaluation"]["status"] == "WARNING_LOW_TRUST"
    assert receipt["trust_evaluation"]["is_trusted"] is False
    warnings = receipt["trust_evaluation"]["warnings_and_advisories"]
    assert len(warnings) >= 1
    # Check for plain-language text
    assert any("LOW TRUST WARNING" in w for w in warnings)
    assert any("falls 2.3% below" in w for w in warnings)


def test_save_and_load_trust_receipt(tmp_path):
    root = get_project_root()
    raw_dir = root / "data/raw"
    config = load_config()

    mock_pipeline_data = {
        "fusion_result": {
            "trust_score_pct": 87.0,
            "mean_risk_score": 0.13,
            "pct_high_risk_pixels": 0.0,
            "weights_used": {"disagreement": 0.25, "stability": 0.25, "spectral": 0.25, "structural": 0.25},
            "component_stats": {
                "disagreement": {"raw_mean": 0.001, "raw_max": 0.01},
                "stability": {"raw_mean": 0.0001, "raw_max": 0.001},
                "spectral": {"raw_mean": 0.02, "raw_max": 0.1},
                "structural": {"raw_mean": 0.04, "raw_max": 0.2},
            },
        },
        "spectral_metrics": {"mean_delta_ndvi": 0.02, "max_delta_ndvi": 0.1, "pct_inconsistent_pixels": 5.0, "is_spectrally_consistent": True},
        "edge_metrics": {"gradient_correlation": 0.80, "edge_iou": 0.40, "edge_f1": 0.57},
        "downstream_comp": {
            "scientific_honesty_label": "Bicubic-vs-SR Footprint Agreement",
            "overall_bic_sr": {"iou": 0.92, "dice": 0.95},
            "high_trust_bic_sr": {"iou": 0.92},
            "low_trust_bic_sr": {"iou": 0.92},
            "high_trust_area_pct": 65.0,
            "reference_comparison": None,
        },
    }

    receipt = generate_trust_receipt(1, raw_dir, config, mock_pipeline_data)
    out_file = tmp_path / "test_receipt.json"

    saved_path = save_trust_receipt(receipt, out_file)
    assert saved_path.exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["receipt_id"] == receipt["receipt_id"]
    assert loaded["evidence_metrics"]["fused_trust_score_pct"] == 87.0


def test_render_trust_receipt_html():
    root = get_project_root()
    raw_dir = root / "data/raw"
    config = load_config()

    mock_pipeline_data = {
        "fusion_result": {
            "trust_score_pct": 86.0,
            "mean_risk_score": 0.14,
            "pct_high_risk_pixels": 0.0,
            "weights_used": {"disagreement": 0.25, "stability": 0.25, "spectral": 0.25, "structural": 0.25},
            "component_stats": {
                "disagreement": {"raw_mean": 0.001, "raw_max": 0.01},
                "stability": {"raw_mean": 0.0001, "raw_max": 0.001},
                "spectral": {"raw_mean": 0.02, "raw_max": 0.1},
                "structural": {"raw_mean": 0.04, "raw_max": 0.2},
            },
        },
        "spectral_metrics": {"mean_delta_ndvi": 0.02, "max_delta_ndvi": 0.1, "pct_inconsistent_pixels": 5.0, "is_spectrally_consistent": True},
        "edge_metrics": {"gradient_correlation": 0.80, "edge_iou": 0.40, "edge_f1": 0.57},
        "downstream_comp": {
            "scientific_honesty_label": "Bicubic-vs-SR Footprint Agreement",
            "overall_bic_sr": {"iou": 0.92, "dice": 0.95},
            "high_trust_bic_sr": {"iou": 0.92},
            "low_trust_bic_sr": {"iou": 0.92},
            "high_trust_area_pct": 65.0,
            "reference_comparison": None,
        },
    }

    receipt = generate_trust_receipt(0, raw_dir, config, mock_pipeline_data, min_trust_threshold=86.5)
    html = render_trust_receipt_html(receipt)

    assert isinstance(html, str)
    assert "GeoFUSE Trust Receipt" in html
    assert "WARNING -- LOW TRUST" in html
    assert "Sentinel-2A" in html
    assert "ResidualSRNet" in html
