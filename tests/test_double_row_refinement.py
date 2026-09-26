import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon

from src.postprocessing.double_row_refinement import (
    evaluate_line_quality,
    smooth_line_center_constrained,
    straighten_and_extend_terminals,
    refine_double_rows,
    _snap_to_probability_ridge,
)


def base_config():
    return {
        "double_row_refinement": {
            "enabled": True,
            "quality": {
                "sample_spacing_m": 0.20,
                "turn_window_m": 0.40,
                "kink_threshold_deg": 10.0,
                "good_max_turn_deg": 8.0,
                "good_p95_turn_deg": 5.0,
                "max_good_kink_fraction": 0.01,
                "min_good_length_m": 4.0,
            },
            "smoothing": {
                "enabled": True,
                "resample_spacing_m": 0.20,
                "window_m": 1.20,
                "max_deviation_m": 0.10,
                "max_hausdorff_deviation_m": 0.12,
                "smooth_good_when_kinked": True,
            },
            "center_mass_snap": {"enabled": False},
            "track_assignment": {
                "max_same_track_gap_m": 6.0,
                "max_same_track_angle_deg": 8.0,
                "max_connector_angle_deg": 12.0,
                "max_lateral_offset_m": 0.18,
            },
            "reference_guided_repair": {
                "max_completion_gap_m": 6.0,
                "min_completion_gap_m": 0.20,
                "guide_spacing_m": 0.20,
                "min_reference_offset_m": 0.30,
                "max_reference_distance_m": 4.0,
                "min_probability_median": 0.0,
                "min_probability_fraction": 0.0,
                "join_tolerance_m": 0.25,
                "repair_kink_threshold_deg": 12.0,
            },
            "terminal_extension": {
                "enabled": True,
                "trigger_distance_to_boundary_m": 2.0,
                "replace_terminal_m": 2.0,
                "direction_fit_window_m": 2.0,
                "max_extension_m": 2.5,
                "require_probability_or_sister_support": False,
            },
        }
    }


def test_constrained_smoothing_reduces_kink_without_large_shift():
    cfg = base_config()
    line = LineString([(0, 0), (2, 0), (3, 0.15), (4, 0), (8, 0)])
    before = evaluate_line_quality(line, cfg)
    smoothed, meta = smooth_line_center_constrained(line, cfg)
    after = evaluate_line_quality(smoothed, cfg)
    assert meta["smoothed"] is True
    assert line.hausdorff_distance(smoothed) <= 0.12 + 1e-6
    assert after.p95_turn_deg <= before.p95_turn_deg + 1e-6


def test_refinement_keeps_established_smoothing_schema_order():
    cfg = base_config()
    cfg["double_row_refinement"]["terminal_extension"]["enabled"] = False
    line = LineString([(0, 0), (2, 0), (3, 0.15), (4, 0), (8, 0)])
    gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:31982")
    roi = Polygon([(-1, -2), (9, -2), (9, 2), (-1, 2)])

    result, _ = refine_double_rows(gdf, roi, cfg, collect_debug=False)

    expected = (
        "p95_turn_after_deg",
        "smoothing_hausdorff_m",
        "smoothing_reason",
        "smoothed",
    )
    positions = [result.columns.get_loc(name) for name in expected]
    assert positions == sorted(positions)


def test_terminal_last_two_metres_are_replaced_by_straight_extension():
    cfg = base_config()
    roi = Polygon([(0, -2), (10, -2), (10, 2), (0, 2)])
    line = LineString([(1, 0), (6, 0), (7, 0.2), (8.3, 0.5)])
    result, meta = straighten_and_extend_terminals(line, roi, cfg)
    assert meta["terminal_straightened"] is True
    assert abs(result.coords[-1][0] - 10.0) < 1e-6
    tail = np.asarray(result.coords[-2:], dtype=float)
    assert np.linalg.norm(tail[-1] - tail[-2]) > 0


def test_broken_row_uses_only_missing_interval_from_good_reference():
    cfg = base_config()
    roi = Polygon([(-1, -3), (11, -3), (11, 3), (-1, 3)])
    # Broken target at y=0, good reference at y=0.9.
    lines = [
        LineString([(0, 0), (4, 0)]),
        LineString([(6, 0), (10, 0)]),
        LineString([(0, 0.9), (10, 0.9)]),
    ]
    gdf = gpd.GeoDataFrame(
        {
            "pair_id": [1, 1, 1],
            "pair_status": ["valid_pair", "valid_pair", "valid_pair"],
            "pair_confidence": [0.9, 0.9, 0.9],
        },
        geometry=lines,
        crs="EPSG:31982",
    )
    result, debug = refine_double_rows(gdf, roi, cfg)
    repaired = result[result["reference_repair"].eq(True)]
    assert len(repaired) == 1
    repaired_line = repaired.geometry.iloc[0]
    assert repaired_line.length >= 9.9
    # The guide must remain on the target row, not replace it with y=0.9.
    ys = np.asarray(repaired_line.coords)[:, 1]
    assert np.max(np.abs(ys)) < 0.05
    assert float(repaired["reference_completed_m"].iloc[0]) < 3.0
    assert {
        "trechos_substituidos_pela_irma",
        "extensoes_terminais",
    }.issubset(set(debug["debug_layer"]))
    result_without_debug, no_debug = refine_double_rows(
        gdf, roi, cfg, collect_debug=False
    )
    assert no_debug.empty
    assert [geometry.wkb for geometry in result_without_debug.geometry] == [
        geometry.wkb for geometry in result.geometry
    ]


def test_two_broken_pair_members_do_not_use_unconfirmed_neighbor_reference():
    cfg = base_config()
    roi = Polygon([(-1, -4), (11, -4), (11, 4), (-1, 4)])
    lines = [
        LineString([(0, 0), (4, 0)]),
        LineString([(6, 0), (10, 0)]),
        LineString([(0, 0.9), (3.5, 0.9)]),
        LineString([(6.5, 0.9), (10, 0.9)]),
        LineString([(0, 2.2), (10, 2.2)]),
    ]
    gdf = gpd.GeoDataFrame(
        {
            "pair_id": [1, 1, 1, 1, 2],
            "pair_status": ["unmatched", "unmatched", "unmatched", "unmatched", "valid_pair"],
            "pair_confidence": [0.0, 0.0, 0.0, 0.0, 0.95],
        },
        geometry=lines,
        crs="EPSG:31982",
    )
    result, _ = refine_double_rows(gdf, roi, cfg)
    reference_repair = result.get(
        "reference_repair", pd.Series(False, index=result.index)
    ).fillna(False)
    assert not reference_repair.any()
    # Other conservative refinements may extend an endpoint to the ROI boundary,
    # but the four broken members must not be completed from pair 2.
    assert all(length < 7.0 for length in result.geometry.length.iloc[:4])



def test_center_mass_snap_moves_interior_toward_probability_ridge():
    from affine import Affine
    cfg = base_config()
    cfg["double_row_refinement"]["center_mass_snap"] = {
        "enabled": True,
        "search_radius_m": 0.12,
        "step_m": 0.02,
        "min_probability": 0.10,
        "offset_smoothing_window_m": 0.60,
        "max_total_shift_m": 0.12,
    }
    transform = Affine.translation(-2.0, 2.0) * Affine.scale(0.02, -0.02)
    raster = np.zeros((200, 200), dtype=np.float32)
    for row in range(200):
        _, y = transform * (0, row + 0.5)
        raster[row, :] = np.exp(-0.5 * (y / 0.035) ** 2)
    original = np.column_stack([np.linspace(-1.5, 1.5, 21), np.full(21, 0.08)])
    snapped = _snap_to_probability_ridge(original, original.copy(), cfg, raster, transform)
    assert np.mean(np.abs(snapped[2:-2, 1])) < 0.08
    assert np.max(np.linalg.norm(snapped - original, axis=1)) <= 0.12 + 1e-6


def test_good_line_is_preserved_with_audit_metadata():
    cfg = base_config()
    roi = Polygon([(-5, -3), (15, -3), (15, 3), (-5, 3)])
    original = LineString([(0, 0), (10, 0)])
    gdf = gpd.GeoDataFrame(geometry=[original], crs="EPSG:31982")

    result, debug = refine_double_rows(gdf, roi, cfg)

    assert result.geometry.iloc[0].equals_exact(original, tolerance=1e-9)
    assert result["quality_before"].iloc[0] == "GOOD"
    assert result["smoothed"].iloc[0] is False or not bool(result["smoothed"].iloc[0])
    assert result["smoothing_reason"].iloc[0] == "good_preserved"
    assert result["refinement_source"].iloc[0] == "original"
    assert debug.empty
