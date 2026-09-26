import geopandas as gpd
from shapely.geometry import LineString

from src.postprocessing.double_row_validation import (
    estimate_double_row_spacing,
    validate_double_rows,
)


def _alternating_rows(pair_count=20):
    y = 0.0
    rows = []
    for _ in range(pair_count):
        rows.append(LineString([(0.0, y), (20.0, y)]))
        y += 0.74
        rows.append(LineString([(0.0, y), (20.0, y)]))
        y += 1.24
    return rows


def _config():
    return {
        "postprocessing": {
            "double_row_validation": {
                "enabled": True,
                "mode": "annotate",
                "tolerance_m": 0.04,
                "max_angle_difference_deg": 5.0,
                "min_longitudinal_overlap_ratio": 0.8,
                "auto_spacing": {
                    "enabled": True,
                    "min_samples": 20,
                    "max_sample_lines": 200,
                    "max_search_distance_m": 2.0,
                    "minimum_half_width_m": 0.03,
                    "relative_half_width": 0.06,
                    "mad_multiplier": 2.0,
                    "cluster_separation_margin_m": 0.02,
                },
            }
        }
    }


def test_spacing_is_calibrated_from_each_image_geometry():
    spacing = estimate_double_row_spacing(_alternating_rows(), _config())

    assert spacing is not None
    assert abs(spacing["intra_pair_center_m"] - 0.74) < 0.05
    assert abs(spacing["inter_pair_center_m"] - 1.24) < 0.08


def test_validation_records_automatic_spacing_provenance():
    frame = gpd.GeoDataFrame(geometry=_alternating_rows(), crs="EPSG:3857")

    result = validate_double_rows(frame, _config())

    assert set(result["spacing_calibration_source"]) == {"automatic"}
    assert result["pair_status"].eq("valid_pair").sum() >= 30
