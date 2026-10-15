"""ROI-level quality filter in dataset preparation for double lines.

A ROI is discarded when it holds distinct lines closer than 0.40 m (indicative of
digitization errors, spurious crossings, or duplicates) or a line geometry deleted
by the repair step; the other ROIs of the scene are kept.
Lines with agronomic double-row spacing (in-pair 0.60m - 0.90m, between-pair >= 1.00m)
are strictly preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from scripts.prepare_manifests import (
    MIN_LINE_SPACING_M,
    repair_line_geometry_detailed,
    sanitize_annotations,
)
from shapely.geometry import LineString, MultiLineString, box

CRS_UTM = "EPSG:32722"
X0, Y0 = 500_000.0, 7_000_000.0


def _rois() -> gpd.GeoDataFrame:
    # ROI 0: [0, 20] x [0, 20]; ROI 1: [30, 50] x [0, 20] (metres, local offsets).
    return gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[
            box(X0, Y0, X0 + 20, Y0 + 20),
            box(X0 + 30, Y0, X0 + 50, Y0 + 20),
        ],
        crs=CRS_UTM,
    )


def _vline(x: float, y0: float = 1.0, y1: float = 19.0) -> LineString:
    return LineString([(X0 + x, Y0 + y0), (X0 + x, Y0 + y1)])


def _write_lines(path: Path, geoms: list) -> Path:
    gpd.GeoDataFrame(geometry=geoms, crs=CRS_UTM).to_file(path, driver="GPKG")
    return path


def _run(tmp_path: Path, geoms: list, rois: gpd.GeoDataFrame | None = None):
    raw = _write_lines(tmp_path / "linhas_99999.gpkg", geoms)
    clean = tmp_path / "clean" / "linhas_99999.geojson"
    return sanitize_annotations("99999", raw, clean, roi_gdf=rois if rois is not None else _rois()), clean


def test_threshold_is_forty_cm():
    assert pytest.approx(0.40) == MIN_LINE_SPACING_M


def test_close_lines_discard_only_their_roi(tmp_path):
    geoms = [
        _vline(5.0), _vline(5.3),            # ROI 0: 0.30 m apart (< 0.40 m) -> discard
        _vline(33.0), _vline(36.0),          # ROI 1: 3.00 m apart -> keep
    ]
    (gdf, stats), clean = _run(tmp_path, geoms)
    assert stats["discarded_roi_ids"] == [0]
    assert stats["discarded_roi_ids_close_lines"] == [0]
    assert stats["discarded_roi_ids_repair_loss"] == []
    assert stats["close_lines_min_distance_m"]["0"] == pytest.approx(0.30, abs=1e-6)
    assert stats["dropped_in_discarded_roi"] == 2
    assert gdf is not None and len(gdf) == 2
    assert clean.exists()
    json.dumps(stats)  # audit report must stay serialisable


def test_double_row_narrow_and_wide_spacing_are_preserved(tmp_path):
    # Agronomic geometry of double lines:
    # ROI 0: in-pair narrow spacing = 0.60 m (5.0 and 5.6)
    # ROI 1: between-pair wide spacing = 1.00 m (33.0 and 34.0)
    geoms = [
        _vline(5.0), _vline(5.6),            # narrow in-pair spacing: 0.60 m -> kept!
        _vline(33.0), _vline(34.0),          # wide inter-pair spacing: 1.00 m -> kept!
    ]
    (gdf, stats), clean = _run(tmp_path, geoms)
    assert stats["discarded_roi_ids"] == []
    assert len(gdf) == 4
    assert clean.exists()


def test_spacing_exactly_at_threshold_is_kept(tmp_path):
    geoms = [_vline(5.0), _vline(5.4), _vline(33.0), _vline(36.0)]  # 0.40 m apart
    (gdf, stats), _ = _run(tmp_path, geoms)
    assert stats["discarded_roi_ids"] == []
    assert len(gdf) == 4


def test_closeness_is_measured_inside_each_roi(tmp_path):
    # Lines are 0.2 m apart only outside every ROI (y in [25, 30]); inside ROI 0 they are 2 m apart.
    a = LineString([(X0 + 5, Y0 + 1), (X0 + 5, Y0 + 30)])
    b = LineString([(X0 + 7, Y0 + 1), (X0 + 7, Y0 + 22), (X0 + 5.2, Y0 + 25), (X0 + 5.2, Y0 + 30)])
    (gdf, stats), _ = _run(tmp_path, [a, b, _vline(33.0), _vline(36.0)])
    assert stats["discarded_roi_ids"] == []
    assert len(gdf) == 4


def test_parts_of_the_same_feature_are_not_compared(tmp_path):
    same_row = MultiLineString([[(X0 + 5, Y0 + 1), (X0 + 5, Y0 + 9)], [(X0 + 5, Y0 + 9), (X0 + 5, Y0 + 19)]])
    (gdf, stats), _ = _run(tmp_path, [same_row, _vline(8.0), _vline(33.0), _vline(36.0)])
    assert stats["discarded_roi_ids"] == []
    assert len(gdf) == 4


def test_line_crossing_into_kept_roi_is_preserved(tmp_path):
    crossing = LineString([(X0 + 10, Y0 + 10), (X0 + 40, Y0 + 10)])  # ROI 0 -> ROI 1
    geoms = [_vline(5.0), _vline(5.2), crossing, _vline(45.0, 12.0, 19.0)]
    (gdf, stats), _ = _run(tmp_path, geoms)
    assert stats["discarded_roi_ids"] == [0]
    assert stats["dropped_in_discarded_roi"] == 2
    assert len(gdf) == 2
    assert any(g.equals(crossing) for g in gdf.geometry)


def test_geometry_deleted_by_repair_discards_its_roi(tmp_path):
    degenerate = LineString([(X0 + 40, Y0 + 5), (X0 + 40, Y0 + 5)])  # inside ROI 1, deleted by repair
    geoms = [_vline(5.0), _vline(8.0), _vline(33.0), _vline(36.0), degenerate]
    (gdf, stats), _ = _run(tmp_path, geoms)
    assert stats["lost_in_repair"] == 1
    assert stats["discarded_roi_ids_repair_loss"] == [1]
    assert stats["discarded_roi_ids"] == [1]
    assert len(gdf) == 2
    xs = sorted(round(g.coords[0][0] - X0, 6) for g in gdf.geometry)
    assert xs == [5.0, 8.0]


def test_all_rois_discarded_rejects_scene(tmp_path):
    geoms = [_vline(5.0), _vline(5.1), _vline(33.0), _vline(33.2)]
    (gdf, stats), clean = _run(tmp_path, geoms)
    assert gdf is None
    assert sorted(stats["discarded_roi_ids"]) == [0, 1]
    assert not clean.exists()


def test_roi_labels_follow_original_index(tmp_path):
    rois = _rois()
    rois.index = [7, 11]  # e.g. after earlier invalid ROIs were filtered out
    (gdf, stats), _ = _run(tmp_path, [_vline(5.0), _vline(8.0), _vline(33.0), _vline(33.3)], rois)
    assert stats["discarded_roi_ids"] == [11]
    assert len(gdf) == 2


def test_close_lines_detected_in_geographic_crs(tmp_path):
    rois = _rois().to_crs("EPSG:4326")
    geoms = gpd.GeoSeries([_vline(5.0), _vline(5.3), _vline(33.0), _vline(36.0)], crs=CRS_UTM).to_crs("EPSG:4326")
    raw = tmp_path / "linhas_99999.gpkg"
    gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326").to_file(raw, driver="GPKG")
    gdf, stats = sanitize_annotations("99999", raw, tmp_path / "c.geojson", roi_gdf=rois)
    assert stats["discarded_roi_ids"] == [0]
    assert stats["close_lines_min_distance_m"]["0"] == pytest.approx(0.30, abs=0.01)
    assert len(gdf) == 2


def test_repair_detailed_flags_losses():
    ok, rep, lost = repair_line_geometry_detailed(LineString([(0, 0), (1, 1), (2, 2)]))
    assert ok is not None and not rep and not lost

    ok, rep, lost = repair_line_geometry_detailed(LineString([(0, 0), (0, 0), (1, 1), (2, 2)]))
    assert ok is not None and rep and not lost  # duplicate vertex clean-up is not a loss
