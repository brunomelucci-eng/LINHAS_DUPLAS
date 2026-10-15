from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pyproj import CRS
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from rowgraphnet.domain import ExtractedSegment, RasterPrediction, SpacingEstimate
from rowgraphnet.geometry.boundary_extension import (
    BoundaryExtensionConfig,
    check_boundary_extension_vetoes,
    compute_robust_terminal_tangent,
    create_boundary_repair_path,
    find_boundary_intersection,
    generate_boundary_extension_candidates,
)


def _make_segment(
    segment_id: str,
    coords: list[tuple[float, float]],
    endpoint_confidence: tuple[float, float] = (0.0, 0.0),
) -> ExtractedSegment:
    geom = LineString(coords)
    return ExtractedSegment(
        segment_id=segment_id,
        geometry=geom,
        confidence=0.95,
        observed_length_m=float(geom.length),
        mean_uncertainty=0.05,
        endpoint_confidence=endpoint_confidence,
        attrs={"spacing_reliable": True, "spacing_local_m": 2.0},
    )


def _dummy_prediction() -> RasterPrediction:
    shape = (100, 100)
    return RasterPrediction(
        center=np.full(shape, 0.8, dtype=np.float32),
        band=np.full(shape, 0.5, dtype=np.float32),
        orientation_sin2=np.zeros(shape, dtype=np.float32),
        orientation_cos2=np.ones(shape, dtype=np.float32),
        endpoints=np.zeros(shape, dtype=np.float32),
        distance=np.zeros(shape, dtype=np.float32),
        uncertainty=np.full(shape, 0.1, dtype=np.float32),
        transform=Affine.translation(0.0, 100.0) * Affine.scale(1.0, -1.0),
        crs=CRS.from_epsg(31983),
    )


def test_robust_terminal_tangent_straight() -> None:
    config = BoundaryExtensionConfig()
    seg = _make_segment("s1", [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])
    # Side 1: endpoint at (3.0, 0.0), pointing outward (+x)
    tangent_end = compute_robust_terminal_tangent(seg, side=1, spacing_m=2.0, config=config)
    assert tangent_end.stable
    assert tangent_end.tangent_vector[0] == pytest.approx(1.0, abs=1e-3)
    assert tangent_end.tangent_vector[1] == pytest.approx(0.0, abs=1e-3)
    assert tangent_end.dispersion_deg == pytest.approx(0.0, abs=1e-3)

    # Side 0: endpoint at (0.0, 0.0), pointing outward (-x)
    tangent_start = compute_robust_terminal_tangent(seg, side=0, spacing_m=2.0, config=config)
    assert tangent_start.stable
    assert tangent_start.tangent_vector[0] == pytest.approx(-1.0, abs=1e-3)
    assert tangent_start.tangent_vector[1] == pytest.approx(0.0, abs=1e-3)


def test_robust_terminal_tangent_unstable_curvature() -> None:
    config = BoundaryExtensionConfig(max_tangent_angular_dispersion_deg=15.0)
    # Zig-zag oscillating points
    coords = [(0.0, 0.0), (1.0, 1.0), (2.0, -1.0), (3.0, 1.0), (4.0, 0.0)]
    seg = _make_segment("curved", coords)
    tangent = compute_robust_terminal_tangent(seg, side=1, spacing_m=2.0, config=config)
    # The zig-zag creates high transverse eigenvalue ratio and angular dispersion
    assert not tangent.stable or tangent.eigenvalue_ratio > 0.1


def test_boundary_intersection_convex_polygon() -> None:
    field = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    seg = _make_segment("s1", [(1.0, 5.0), (4.0, 5.0), (8.0, 5.0)])
    config = BoundaryExtensionConfig()
    tangent = compute_robust_terminal_tangent(seg, side=1, spacing_m=2.0, config=config)
    inter_xy, dist, reason = find_boundary_intersection(
        (8.0, 5.0), tangent, field, max_distance_m=5.0
    )
    assert reason is None
    assert dist == pytest.approx(2.0, abs=1e-3)
    assert inter_xy == pytest.approx((10.0, 5.0), abs=1e-3)


def test_boundary_intersection_ignores_interior_holes() -> None:
    exterior = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
    hole = [(12.0, 4.0), (16.0, 4.0), (16.0, 8.0), (12.0, 8.0)]
    field_with_hole = Polygon(exterior, holes=[hole])

    # Segment at y=6, pointing east from x=10 towards the hole at x=12 and exterior at x=20
    seg = _make_segment("s1", [(2.0, 6.0), (5.0, 6.0), (10.0, 6.0)])
    config = BoundaryExtensionConfig()
    tangent = compute_robust_terminal_tangent(seg, side=1, spacing_m=2.0, config=config)

    # find_boundary_intersection must intersect ONLY the exterior ring (x=20), ignoring the hole boundary
    inter_xy, dist, reason = find_boundary_intersection(
        (10.0, 6.0), tangent, field_with_hole, max_distance_m=15.0
    )
    assert reason is None
    assert inter_xy == pytest.approx((20.0, 6.0), abs=1e-3)
    assert dist == pytest.approx(10.0, abs=1e-3)


def test_boundary_extension_veto_true_end() -> None:
    field = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    # Endpoint confidence is 0.9 (confident physical end)
    seg = _make_segment("s1", [(1.0, 5.0), (4.0, 5.0), (8.0, 5.0)], endpoint_confidence=(0.0, 0.9))
    config = BoundaryExtensionConfig(true_endpoint_probability_threshold=0.5)
    pred = _dummy_prediction()
    spacing = SpacingEstimate(median_m=2.0, mad_m=0.1, entropy=0.2, confidence=0.9)

    candidates = generate_boundary_extension_candidates([seg], pred, spacing, field, config)
    # Side 1 must be vetoed due to true_end
    side_1_candidates = [c for c in candidates if c.side_a == 1]
    assert len(side_1_candidates) == 0


def test_boundary_extension_veto_neighbor_corridor() -> None:
    field = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    first = _make_segment("first", [(1.0, 5.0), (4.0, 5.0), (8.0, 5.0)])
    # Neighbor row at y=5.2 (too close, within 0.35 * 2.0 = 0.70m)
    neighbor = _make_segment("neighbor", [(8.5, 5.2), (9.5, 5.2)])
    config = BoundaryExtensionConfig(safe_corridor_fraction=0.35)

    candidate_line = LineString([(8.0, 5.0), (10.0, 5.0)])
    passed, reason = check_boundary_extension_vetoes(
        candidate_line,
        "first",
        [first, neighbor],
        spacing_m=2.0,
        field_boundary=field,
        config=config,
    )
    assert not passed
    assert "safe_corridor" in str(reason)


def test_boundary_candidate_generation_and_repair_path() -> None:
    field = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    seg = _make_segment("s1", [(2.0, 5.0), (5.0, 5.0), (8.5, 5.0)])
    config = BoundaryExtensionConfig(enabled=True)
    pred = _dummy_prediction()
    spacing = SpacingEstimate(median_m=2.0, mad_m=0.1, entropy=0.2, confidence=0.9)

    candidates = generate_boundary_extension_candidates([seg], pred, spacing, field, config)
    assert len(candidates) == 2
    cands_by_side = {c.side_a: c for c in candidates}
    assert 0 in cands_by_side and 1 in cands_by_side
    cand_0 = cands_by_side[0]
    cand_1 = cands_by_side[1]
    assert cand_0.boundary_intersection_xy == pytest.approx((0.0, 5.0), abs=1e-3)
    assert cand_1.boundary_intersection_xy == pytest.approx((10.0, 5.0), abs=1e-3)
    assert cand_1.probability > 0.0
    assert cand_1.evidence["repair_type"] == "boundary_extension"

    # Create repair path
    path = create_boundary_repair_path(cand_1, pred)
    assert path.repair_type == "boundary_extension"
    assert path.candidate_id == cand_1.candidate_id
    assert path.geometry.coords[-1] == pytest.approx((10.0, 5.0), abs=1e-3)
    assert 0.0 <= path.mean_center_support <= 1.0


def test_multipolygon_boundary_extension() -> None:
    poly1 = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    poly2 = Polygon([(20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)])
    multipoly = MultiPolygon([poly1, poly2])

    seg = _make_segment("s1", [(2.0, 5.0), (5.0, 5.0), (8.5, 5.0)])
    config = BoundaryExtensionConfig(enabled=True)
    pred = _dummy_prediction()
    spacing = SpacingEstimate(median_m=2.0, mad_m=0.1, entropy=0.2, confidence=0.9)

    candidates = generate_boundary_extension_candidates([seg], pred, spacing, multipoly, config)
    assert len(candidates) == 2
    cands_by_side = {c.side_a: c for c in candidates}
    assert cands_by_side[0].boundary_intersection_xy == pytest.approx((0.0, 5.0), abs=1e-3)
    assert cands_by_side[1].boundary_intersection_xy == pytest.approx((10.0, 5.0), abs=1e-3)


def test_true_endpoint_suppresses_boundary_extension() -> None:
    field = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    # Endpoint side 1 has confidence 0.90 (true agricultural headland/end)
    seg = _make_segment(
        "s1",
        [(2.0, 5.0), (5.0, 5.0), (8.5, 5.0)],
        endpoint_confidence=(0.0, 0.90),
    )
    config = BoundaryExtensionConfig(enabled=True, true_endpoint_probability_threshold=0.50)
    pred = _dummy_prediction()
    spacing = SpacingEstimate(median_m=2.0, mad_m=0.1, entropy=0.2, confidence=0.9)

    candidates = generate_boundary_extension_candidates([seg], pred, spacing, field, config)
    # Side 1 is suppressed because it's a true end! Only side 0 is generated
    assert len(candidates) == 1
    assert candidates[0].side_a == 0


def test_unstable_tangent_rejects_boundary_extension() -> None:
    # A jagged/oscillating extremity with high dispersion
    coords = [(5.0, 5.0), (6.0, 5.5), (7.0, 4.5), (8.0, 5.5), (8.5, 4.5)]
    seg = _make_segment("jagged", coords)
    config = BoundaryExtensionConfig(max_tangent_angular_dispersion_deg=10.0)
    tangent = compute_robust_terminal_tangent(seg, side=1, spacing_m=2.0, config=config)
    assert not tangent.stable


def test_boundary_defaults_do_not_hide_a_universal_metric_cap() -> None:
    config = BoundaryExtensionConfig()
    assert config.max_absolute_distance_m is None


def test_minimum_observed_length_factor_is_configurable() -> None:
    field = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    segment = _make_segment("short", [(7.0, 5.0), (8.0, 5.0), (9.0, 5.0)])
    spacing = SpacingEstimate(median_m=2.0, mad_m=0.1, entropy=0.2, confidence=0.9)
    candidates = generate_boundary_extension_candidates(
        [segment],
        _dummy_prediction(),
        spacing,
        field,
        BoundaryExtensionConfig(min_observed_length_spacing_factor=2.0),
    )
    assert not candidates


def test_boundary_extension_rejects_invalid_raster_corridor() -> None:
    field = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    segment = _make_segment("s1", [(2.0, 5.0), (5.0, 5.0), (8.5, 5.0)])
    prediction = _dummy_prediction()
    prediction.valid_mask = np.ones(prediction.center.shape, dtype=bool)
    prediction.valid_mask[94, 8:10] = False
    spacing = SpacingEstimate(median_m=2.0, mad_m=0.1, entropy=0.2, confidence=0.9)

    candidates = generate_boundary_extension_candidates(
        [segment],
        prediction,
        spacing,
        field,
        BoundaryExtensionConfig(),
    )
    assert not [candidate for candidate in candidates if candidate.side_a == 1]


def test_boundary_extension_rejects_a_hole_in_the_path() -> None:
    field = Polygon(
        [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
        holes=[[(12.0, 4.0), (16.0, 4.0), (16.0, 8.0), (12.0, 8.0)]],
    )
    segment = _make_segment("s1", [(2.0, 6.0), (5.0, 6.0), (10.0, 6.0)])
    spacing = SpacingEstimate(median_m=4.0, mad_m=0.1, entropy=0.2, confidence=0.9)
    candidates = generate_boundary_extension_candidates(
        [segment],
        _dummy_prediction(),
        spacing,
        field,
        BoundaryExtensionConfig(max_boundary_distance_factor=3.0),
    )
    assert not [candidate for candidate in candidates if candidate.side_a == 1]


def test_boundary_extension_rejects_non_polygon_boundary_contract() -> None:
    segment = _make_segment("s1", [(2.0, 5.0), (5.0, 5.0), (8.5, 5.0)])
    spacing = SpacingEstimate(median_m=2.0, mad_m=0.1, entropy=0.2, confidence=0.9)
    with pytest.raises(ValueError, match="Polygon"):
        generate_boundary_extension_candidates(
            [segment],
            _dummy_prediction(),
            spacing,
            Point(10.0, 5.0),
            BoundaryExtensionConfig(),
        )
