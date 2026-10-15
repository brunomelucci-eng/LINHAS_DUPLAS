from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from pyproj import CRS
from shapely.geometry import LineString

from rowgraphnet.domain import (
    CandidateConnection,
    ExtractedSegment,
    RasterPrediction,
    SpacingEstimate,
)
from rowgraphnet.geometry.candidate_graph import generate_candidates
from rowgraphnet.geometry.global_optimizer import optimize_connections
from rowgraphnet.geometry.orientation_field import (
    discover_orientation_regimes,
    fuse_orientation_fields,
    pixel_theta_to_world,
    structure_tensor_orientation,
)
from rowgraphnet.geometry.path_repair import PathCostWeights
from rowgraphnet.geometry.segment_extractor import adaptive_hysteresis
from rowgraphnet.geometry.spacing_estimator import estimate_spacing
from rowgraphnet.synthetic import generate_synthetic_scene


def _candidate(
    identifier: str,
    segment_a: str,
    side_a: int,
    segment_b: str,
    side_b: int,
    coords: list[tuple[float, float]],
    probability: float,
) -> CandidateConnection:
    return CandidateConnection(
        candidate_id=identifier,
        segment_a=segment_a,
        side_a=side_a,
        segment_b=segment_b,
        side_b=side_b,
        geometry=LineString(coords),
        probability=probability,
        cost=-np.log(probability),
        evidence={},
    )


def test_axial_orientation_treats_zero_and_180_as_identical() -> None:
    scene = generate_synthetic_scene("straight", with_gaps=False)
    tensor = structure_tensor_orientation(scene.prediction.center, sigma_px=1.0)
    field = fuse_orientation_fields(
        scene.prediction.orientation_sin2,
        scene.prediction.orientation_cos2,
        scene.prediction.uncertainty,
        tensor,
    )
    rows = scene.prediction.center > 0.5
    axial_alignment = np.cos(2.0 * field.theta_rad[rows])
    assert np.median(axial_alignment) > 0.99


def test_pixel_orientation_uses_complete_rotated_affine_basis() -> None:
    transform = Affine(0.0, -2.0, 10.0, 3.0, 0.0, 20.0)
    column_direction = float(pixel_theta_to_world(0.0, transform))
    row_direction = float(pixel_theta_to_world(np.pi / 2.0, transform))
    assert column_direction == pytest.approx(np.pi / 2.0)
    assert row_direction == pytest.approx(0.0)


def test_hysteresis_keeps_positive_saturated_and_quantized_ridges() -> None:
    saturated, _ = adaptive_hysteresis(np.ones((4, 4), dtype=float))
    assert saturated.all()

    quantized = np.zeros((9, 9), dtype=float)
    quantized[4, 2:7] = 1.0
    ridge, _ = adaptive_hysteresis(quantized)
    assert np.array_equal(ridge, quantized.astype(bool))


def test_orientation_fusion_treats_uncertainty_as_an_unbounded_scale() -> None:
    shape = (3, 4)
    tensor = structure_tensor_orientation(np.zeros(shape), sigma_px=1.0)
    field = fuse_orientation_fields(
        np.zeros(shape),
        np.ones(shape),
        np.full(shape, 2.0),
        tensor,
    )
    assert np.all(field.confidence > 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        fuse_orientation_fields(
            np.zeros(shape),
            np.ones(shape),
            np.full(shape, -0.1),
            tensor,
        )


def test_path_cost_weights_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        PathCostWeights(background=float("nan")).normalized()
    with pytest.raises(ValueError, match="finite"):
        PathCostWeights(uncertainty=float("inf")).normalized()


def test_raster_prediction_rejects_invalid_values_inside_valid_mask() -> None:
    shape = (3, 4)
    arrays = {
        "center": np.full(shape, 0.5),
        "band": np.full(shape, 0.5),
        "orientation_sin2": np.zeros(shape),
        "orientation_cos2": np.ones(shape),
        "endpoints": np.zeros(shape),
        "distance": np.zeros(shape),
        "uncertainty": np.ones(shape),
    }
    arrays["center"][1, 1] = np.nan
    with pytest.raises(ValueError, match="center"):
        RasterPrediction(
            **arrays,
            transform=Affine.scale(0.1, -0.1),
            crs=CRS.from_epsg(31983),
        )


def test_direction_regime_count_is_selected_from_evidence() -> None:
    straight = generate_synthetic_scene("straight", with_gaps=False)
    straight_tensor = structure_tensor_orientation(straight.prediction.center, sigma_px=2.0)
    straight_field = fuse_orientation_fields(
        straight.prediction.orientation_sin2,
        straight.prediction.orientation_cos2,
        straight.prediction.uncertainty,
        straight_tensor,
    )
    one_regime, _ = discover_orientation_regimes(straight_field, straight.prediction.center)
    assert len(one_regime) == 1

    mixed = generate_synthetic_scene("two_regimes", with_gaps=False)
    mixed_tensor = structure_tensor_orientation(mixed.prediction.center, sigma_px=2.0)
    mixed_field = fuse_orientation_fields(
        mixed.prediction.orientation_sin2,
        mixed.prediction.orientation_cos2,
        mixed.prediction.uncertainty,
        mixed_tensor,
    )
    regimes, _ = discover_orientation_regimes(mixed_field, mixed.prediction.center)
    assert len(regimes) == 2
    separation = abs(regimes[0].orientation_deg - regimes[1].orientation_deg)
    separation = min(separation, 180.0 - separation)
    assert separation > 10.0


def test_spacing_is_inferred_and_noise_causes_abstention() -> None:
    scene = generate_synthetic_scene("straight", spacing_px=24.0)
    spacing = estimate_spacing(scene.prediction.center, 0.0, pixel_size_m=0.1)
    assert spacing.reliable
    assert spacing.global_estimate is not None
    assert spacing.global_estimate.median_m == pytest.approx(2.4, abs=0.11)

    noise = np.random.default_rng(9).random((128, 128))
    uncertain = estimate_spacing(noise, 0.0, pixel_size_m=0.1)
    assert not uncertain.reliable


def test_global_optimizer_prevents_crossing_and_is_order_invariant() -> None:
    horizontal = _candidate(
        "horizontal",
        "west",
        1,
        "east",
        0,
        [(-1.0, 0.0), (1.0, 0.0)],
        0.91,
    )
    vertical = _candidate(
        "vertical",
        "south",
        1,
        "north",
        0,
        [(0.0, -1.0), (0.0, 1.0)],
        0.82,
    )
    first = optimize_connections([vertical, horizontal])
    second = optimize_connections([horizontal, vertical])
    assert [item.candidate_id for item in first.accepted] == ["horizontal"]
    assert [item.candidate_id for item in second.accepted] == ["horizontal"]
    assert first.objective == pytest.approx(second.objective)


def test_global_optimizer_enforces_endpoint_degree_and_abstention() -> None:
    preferred = _candidate("a", "one", 1, "two", 0, [(0, 0), (1, 0)], 0.9)
    competing = _candidate("b", "one", 1, "three", 0, [(0, 0), (1, 1)], 0.8)
    result = optimize_connections([competing, preferred])
    assert [item.candidate_id for item in result.accepted] == ["a"]

    competing.ambiguous = True
    preferred.ambiguous = True
    abstained = optimize_connections([preferred, competing])
    assert not abstained.accepted
    assert {item.candidate_id for item in abstained.ambiguous} == {"a", "b"}


def test_global_optimizer_rejects_boundary_extensions_sharing_target() -> None:
    first = CandidateConnection(
        candidate_id="ext_a",
        segment_a="row_a",
        side_a=1,
        segment_b="",
        side_b=-1,
        geometry=LineString([(0.0, 0.0), (2.0, 1.0)]),
        probability=0.9,
        cost=-np.log(0.9),
        evidence={},
        is_boundary_extension=True,
        boundary_intersection_xy=(2.0, 1.0),
    )
    second = CandidateConnection(
        candidate_id="ext_b",
        segment_a="row_b",
        side_a=1,
        segment_b="",
        side_b=-1,
        geometry=LineString([(0.0, 2.0), (2.0, 1.0)]),
        probability=0.8,
        cost=-np.log(0.8),
        evidence={},
        is_boundary_extension=True,
        boundary_intersection_xy=(2.0, 1.0),
    )
    result = optimize_connections([second, first])
    assert [candidate.candidate_id for candidate in result.accepted] == ["ext_a"]


def test_global_optimizer_rejects_duplicate_ids_and_non_finite_probabilities() -> None:
    first = _candidate("duplicate", "a", 1, "b", 0, [(0.0, 0.0), (1.0, 0.0)], 0.9)
    second = _candidate("duplicate", "c", 1, "d", 0, [(0.0, 1.0), (1.0, 1.0)], 0.8)
    with pytest.raises(ValueError, match="unique"):
        optimize_connections([first, second])

    first.probability = float("nan")
    with pytest.raises(ValueError, match="probabilities"):
        optimize_connections([first])


def test_spacing_contract_can_represent_local_uncertainty() -> None:
    estimate = SpacingEstimate(
        median_m=1.7,
        mad_m=0.2,
        entropy=0.4,
        confidence=0.8,
        samples_m=(1.5, 1.7, 1.9),
        position_xy=(10.0, 20.0),
    )
    assert estimate.mad_m > 0
    assert len(estimate.samples_m) == 3


def test_true_endpoint_evidence_vetoes_a_visual_continuation() -> None:
    first = ExtractedSegment(
        segment_id="first",
        geometry=LineString([(0.0, 0.0), (1.0, 0.0)]),
        confidence=0.99,
        observed_length_m=1.0,
        mean_uncertainty=0.01,
        endpoint_confidence=(1.0, 1.0),
    )
    second = ExtractedSegment(
        segment_id="second",
        geometry=LineString([(1.5, 0.0), (2.5, 0.0)]),
        confidence=0.99,
        observed_length_m=1.0,
        mean_uncertainty=0.01,
        endpoint_confidence=(1.0, 1.0),
    )
    spacing = SpacingEstimate(2.0, 0.0, 0.0, 1.0, (2.0,))
    candidates = generate_candidates([first, second], spacing)
    assert len(candidates) == 1
    assert candidates[0].probability == 0.0
    assert not optimize_connections(candidates).accepted
