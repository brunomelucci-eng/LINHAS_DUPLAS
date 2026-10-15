from __future__ import annotations

from shapely.geometry import LineString

from rowgraphnet.domain import CandidateConnection, ExtractedSegment, RepairPath
from rowgraphnet.geometry.global_optimizer import OptimizationResult
from rowgraphnet.geometry.vector_refinement import refine_rows


def _segment(identifier: str, coords: list[tuple[float, float]]) -> ExtractedSegment:
    geometry = LineString(coords)
    return ExtractedSegment(
        segment_id=identifier,
        geometry=geometry,
        confidence=0.9,
        observed_length_m=geometry.length,
        mean_uncertainty=0.1,
        attrs={"spacing_local_m": 2.0},
    )


def test_disconnected_accepted_component_is_not_silently_reduced_to_longest_part() -> None:
    first = _segment("a", [(0.0, 0.0), (1.0, 0.0)])
    second = _segment("b", [(4.0, 0.0), (5.0, 0.0)])
    candidate = CandidateConnection(
        candidate_id="broken",
        segment_a="a",
        side_a=1,
        segment_b="b",
        side_b=0,
        geometry=LineString([(1.0, 0.0), (4.0, 0.0)]),
        probability=0.9,
        cost=0.1,
        evidence={},
    )
    disconnected_path = RepairPath(
        candidate_id="broken",
        geometry=LineString([(10.0, 10.0), (11.0, 10.0)]),
        total_cost=0.1,
        mean_center_support=0.8,
        mean_uncertainty=0.2,
        reason={},
    )
    result = refine_rows(
        [first, second],
        OptimizationResult([candidate], [], [], 1.0, True),
        [disconnected_path],
        model_version="test",
    )
    assert not result.rows
    assert not result.repairs
    assert {row.geometry.wkt for row in result.uncertain_rows} == {
        first.geometry.wkt,
        second.geometry.wkt,
    }


def test_collective_validation_abstains_from_crossing_final_rows() -> None:
    horizontal = _segment("horizontal", [(-1.0, 0.0), (1.0, 0.0)])
    vertical = _segment("vertical", [(0.0, -1.0), (0.0, 1.0)])
    result = refine_rows(
        [horizontal, vertical],
        OptimizationResult([], [], [], 0.0, True),
        [],
        model_version="test",
    )
    assert not result.rows
    assert len(result.uncertain_rows) == 2
    assert all("crosses_" in row.repair_reason for row in result.uncertain_rows)
