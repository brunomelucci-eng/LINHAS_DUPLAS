"""Tests for continuous vector smoothing and track bridging."""

import math
import numpy as np
import pytest
from shapely.geometry import LineString
import geopandas as gpd

from src.postprocessing.continuous_fairing import (
    smooth_linestring_safely,
    bridge_and_merge_track_fragments,
    continuous_fairing_gdf,
)


def test_smooth_linestring_safely_removes_serration():
    # Construct a serrated/staircase line (zigzagging at 45 degrees along the X axis)
    x = np.linspace(0, 50, 100)
    # 0.1m amplitude high frequency zigzag
    y = 0.08 * np.sin(x * 5)
    serrated_line = LineString(np.column_stack([x, y]))

    smoothed, method = smooth_linestring_safely(serrated_line, maximum_deviation_m=0.12)

    assert smoothed.is_valid
    assert smoothed.is_simple
    assert not smoothed.is_empty
    assert smoothed.hausdorff_distance(serrated_line) <= 0.12
    assert method in ("bspline", "bspline_simplified", "chaikin", "chaikin_simplified")

    # Verify that turn angles on the smoothed line are significantly lower than the serrated line
    smoothed_coords = np.asarray(smoothed.coords)
    diffs = np.diff(smoothed_coords, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angle_turns = np.abs(np.diff(angles))
    angle_turns = np.minimum(angle_turns, 2 * np.pi - angle_turns)
    max_turn_deg = np.degrees(np.max(angle_turns)) if len(angle_turns) > 0 else 0.0

    # The original serrated line had sharp turns; smoothed should be gentle
    assert max_turn_deg < 45.0


def test_bridge_and_merge_track_fragments():
    # Create two collinear fragments along a row with a 1.5m gap
    seg1 = LineString([(0, 0), (10, 0)])
    seg2 = LineString([(11.5, 0), (25, 0)])

    merged = bridge_and_merge_track_fragments([seg1, seg2], max_gap_m=3.0, max_lateral_offset_m=0.20)
    assert len(merged) == 1
    assert merged[0].length >= 24.9
    assert merged[0].is_valid


def test_bridge_does_not_connect_across_parallel_rows():
    # Two parallel lines 0.90m apart (like double rows)
    seg1 = LineString([(0, 0), (10, 0)])
    seg2 = LineString([(0, 0.90), (10, 0.90)])

    # Should not connect them into one line because lateral offset > 0.25m
    merged = bridge_and_merge_track_fragments([seg1, seg2], max_gap_m=3.0, max_lateral_offset_m=0.25)
    assert len(merged) == 2


def test_continuous_fairing_gdf():
    seg1 = LineString([(0, 0), (5, 0.05), (10, 0)])
    seg2 = LineString([(12, 0), (18, 0.05), (25, 0)])
    gdf = gpd.GeoDataFrame(
        {
            "track_id": [1, 1],
            "pair_id": [10, 10],
        },
        geometry=[seg1, seg2],
        crs="EPSG:32722",
    )

    out_gdf = continuous_fairing_gdf(gdf, maximum_deviation_m=0.10)
    assert len(out_gdf) == 1
    assert out_gdf.iloc[0].geometry.length > 20.0
    assert bool(out_gdf.iloc[0]["smoothing_applied"]) is True
    assert out_gdf.iloc[0]["serration_status"] == "passed"
