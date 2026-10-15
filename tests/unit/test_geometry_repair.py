"""Unit tests for vector geometry verification and auto-repair."""

from scripts.prepare_manifests import (
    repair_line_geometry,
    repair_roi_geometry,
    safe_read_vector,
)
from shapely.geometry import LineString, Polygon


def test_repair_line_geometry_valid():
    line = LineString([(0, 0), (1, 1), (2, 2)])
    repaired, was_repaired = repair_line_geometry(line)
    assert repaired is not None
    assert repaired.is_valid
    assert not was_repaired


def test_repair_line_geometry_duplicate_vertices():
    # Consecutive duplicate vertices
    line = LineString([(0, 0), (0, 0), (1, 1), (1, 1), (2, 2)])
    repaired, was_repaired = repair_line_geometry(line)
    assert repaired is not None
    assert repaired.is_valid
    assert was_repaired
    assert len(repaired.coords) == 3


def test_repair_line_geometry_degenerate_zero_length():
    # Exactly identical points (CAD click error)
    line = LineString([(5.0, 5.0), (5.0, 5.0)])
    repaired, was_repaired = repair_line_geometry(line)
    assert repaired is None
    assert was_repaired


def test_repair_roi_geometry_valid():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    repaired, was_repaired = repair_roi_geometry(poly)
    assert repaired is not None
    assert repaired.is_valid
    assert not was_repaired


def test_repair_roi_geometry_self_intersecting_bowtie():
    # Classic self-intersecting bowtie polygon
    bowtie = Polygon([(0, 0), (10, 10), (0, 10), (10, 0), (0, 0)])
    assert not bowtie.is_valid
    repaired, was_repaired = repair_roi_geometry(bowtie)
    assert repaired is not None
    assert repaired.is_valid
    assert was_repaired
    assert repaired.area > 0


def test_safe_read_vector_corrupted_or_missing(tmp_path):
    missing_file = tmp_path / "nao_existe.geojson"
    result = safe_read_vector(missing_file, "99999", "anotacao")
    assert result is None
