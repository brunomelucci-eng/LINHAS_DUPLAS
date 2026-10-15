"""Unit tests for neighbor-guided gap bridging and consensus boundary extension."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Polygon

from rowgraphnet.domain import RowRecord
from rowgraphnet.geometry.enhanced_refinement import (
    EnhancedRefinementConfig,
    bridge_long_gaps_with_neighbors,
    export_enhanced_geopackage,
    extend_to_boundary_with_consensus,
)


def _make_row(row_id: str, line: LineString, spacing_m: float = 1.5) -> RowRecord:
    return RowRecord(
        row_id=row_id,
        geometry=line,
        confidence=0.95,
        observed_ratio=1.0,
        repaired_ratio=0.0,
        spacing_local_m=spacing_m,
        orientation_deg=0.0,
        topology_status="valid",
        repair_reason=json.dumps({"test": True}),
        model_version="test-v1",
    )


def test_bridge_3m20_gap_with_neighbor_support() -> None:
    """Test that a 3.20m gap is bridged when a parallel continuous neighbor is present."""
    spacing = 1.5
    # Row 1 (neighbor): continuous from y=0 to y=30 at x=1.5
    row_neighbor = _make_row("row_neighbor", LineString([(1.5, 0.0), (1.5, 30.0)]), spacing_m=spacing)

    # Row 2 (broken into two parts separated by a 3.20m gap):
    # Part A: y=0 to y=12 at x=0
    # Part B: y=15.2 to y=30 at x=0 (gap of 3.20m from y=12 to y=15.2)
    part_a = _make_row("part_a", LineString([(0.0, 0.0), (0.0, 12.0)]), spacing_m=spacing)
    part_b = _make_row("part_b", LineString([(0.0, 15.2), (0.0, 30.0)]), spacing_m=spacing)

    config = EnhancedRefinementConfig(max_gap_distance_m=5.0)
    bridged_rows, count = bridge_long_gaps_with_neighbors(
        [part_a, part_b, row_neighbor],
        spacing_m=spacing,
        config=config,
        model_version="test-v1",
    )

    assert count == 1, "Expected exactly 1 gap to be bridged"
    assert len(bridged_rows) == 2, "Expected 2 total rows after bridging (1 merged + 1 neighbor)"
    merged = next(r for r in bridged_rows if r.row_id != "row_neighbor")
    assert merged.geometry.length >= 29.5, "Merged line should span from ~0 to ~30"


def test_bridge_gap_rejected_without_neighbor_support() -> None:
    """Test that a long gap (e.g. 4m) is rejected if no parallel neighbor exists to validate it."""
    spacing = 1.5
    # Isolated broken row with no neighbor
    part_a = _make_row("part_a", LineString([(0.0, 0.0), (0.0, 10.0)]), spacing_m=spacing)
    part_b = _make_row("part_b", LineString([(0.0, 14.0), (0.0, 25.0)]), spacing_m=spacing)

    config = EnhancedRefinementConfig(
        max_gap_distance_m=5.0,
        require_neighbor_support_above_m=2.0,
    )
    bridged_rows, count = bridge_long_gaps_with_neighbors(
        [part_a, part_b],
        spacing_m=spacing,
        config=config,
        model_version="test-v1",
    )

    assert count == 0, "Should not bridge 4m gap without neighbor support"
    assert len(bridged_rows) == 2


def test_extend_to_boundary_with_consensus_ignores_tip_waviness() -> None:
    """Test extending rows to a boundary polygon, where a row with a wiggly tip uses neighbor consensus."""
    spacing = 1.5
    # Field boundary: rectangle [ -5, 30 ] x [ -5, 30 ]
    boundary = Polygon([(-5.0, -5.0), (30.0, -5.0), (30.0, 30.0), (-5.0, 30.0)])

    # Neighbors 1, 2, 3: perfectly straight lines heading north (y=0 to y=25) at x=0, 1.5, 3.0
    r0 = _make_row("r0", LineString([(0.0, 0.0), (0.0, 25.0)]), spacing_m=spacing)
    r1 = _make_row("r1", LineString([(1.5, 0.0), (1.5, 25.0)]), spacing_m=spacing)
    r2 = _make_row("r2", LineString([(3.0, 0.0), (3.0, 25.0)]), spacing_m=spacing)

    # Row 3 (at x=4.5): straight up to y=24.5, but has a wavy tip turning towards east at (5.0, 25.0)
    # The wiggly tip points towards east (~90 deg), but neighbors point north (0 deg)
    r3 = _make_row(
        "r3",
        LineString([(4.5, 0.0), (4.5, 24.5), (5.0, 25.0)]),
        spacing_m=spacing,
    )

    config = EnhancedRefinementConfig(
        max_boundary_extension_m=8.0,
        boundary_consensus_neighbors=3,
    )

    extended, count = extend_to_boundary_with_consensus(
        [r0, r1, r2, r3],
        field_boundary=boundary,
        spacing_m=spacing,
        config=config,
        model_version="test-v1",
    )

    assert count >= 1, "Expected extensions to boundary"
    # Find extended r3
    ext_r3 = [r for r in extended if "r3" in r.repair_reason or r.row_id == "r3"]
    assert len(ext_r3) == 1
    # Check that extension touched the boundary exterior at y=30
    last_pt = ext_r3[0].geometry.coords[-1]
    assert abs(last_pt[1] - 30.0) < 0.1, f"Expected touch at northern boundary y=30, got {last_pt}"


def test_export_enhanced_geopackage(tmp_path: Path) -> None:
    """Test exporting enhanced rows to GeoPackage and reading them back."""
    line = LineString([(500000.0, 7000000.0), (500000.0, 7000050.0)])
    row = _make_row("row_1", line, spacing_m=1.5)
    out_gpkg = tmp_path / "linhas_completas_v2.gpkg"

    path = export_enhanced_geopackage(out_gpkg, [row], crs="EPSG:32722")
    assert path.exists()

    gdf = gpd.read_file(path)
    assert len(gdf) == 1
    assert gdf.loc[0, "row_id"] == "row_1"
    assert gdf.crs.to_epsg() == 32722
