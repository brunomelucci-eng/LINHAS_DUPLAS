from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString

from rowgraphnet.validation import (
    binary_cldice,
    compute_axial_orientation_error,
    compute_calibration_metrics,
    compute_geometry_metrics,
    compute_raster_metrics,
    compute_topology_metrics,
)


def test_binary_raster_metrics_have_exact_confusion_scores() -> None:
    probability = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.4]])
    target = np.array([[1, 0, 0], [0, 1, 1]])

    result = compute_raster_metrics(probability, target, threshold=0.5)

    assert (result.true_positive, result.false_positive) == (2, 1)
    assert (result.false_negative, result.true_negative) == (1, 2)
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)
    assert result.iou == pytest.approx(0.5)
    assert result.dice == pytest.approx(2 / 3)


def test_raster_metrics_ignore_invalid_pixels_and_define_empty_as_perfect() -> None:
    probability = np.array([[0.9, 0.1, np.nan], [0.2, 0.7, np.nan]])
    target = np.array([[1, 0, 1], [0, 1, 0]])
    valid = np.array([[1, 1, 0], [1, 1, 0]], dtype=bool)

    result = compute_raster_metrics(probability, target, valid_mask=valid)
    empty = compute_raster_metrics(np.zeros((3, 3)), np.zeros((3, 3)))

    assert result.valid_pixels == 4
    assert result.f1 == result.iou == result.dice == result.cldice == 1.0
    assert empty.f1 == empty.iou == empty.dice == empty.cldice == 1.0


def test_cldice_penalizes_a_break_in_an_otherwise_correct_centerline() -> None:
    target = np.zeros((7, 15), dtype=bool)
    target[3, 2:13] = True
    broken = target.copy()
    broken[3, 7] = False

    score = binary_cldice(broken, target)

    assert 0.0 < score < 1.0
    assert binary_cldice(target, target) == 1.0


def test_axial_error_treats_zero_and_180_degrees_as_equivalent() -> None:
    predicted_theta = np.deg2rad(np.array([[0.0, 180.0, 10.0]]))
    target_theta = np.zeros((1, 3), dtype=float)

    result = compute_axial_orientation_error(
        np.sin(2.0 * predicted_theta),
        np.cos(2.0 * predicted_theta),
        np.sin(2.0 * target_theta),
        np.cos(2.0 * target_theta),
    )

    assert result.valid_pixels == 3
    assert result.mean_error_deg == pytest.approx(10 / 3)
    assert result.median_error_deg == pytest.approx(0.0, abs=1e-6)
    assert result.p95_error_deg == pytest.approx(10.0)


def test_calibration_metrics_compute_ece_and_brier() -> None:
    probability = np.array([[0.1, 0.9]])
    target = np.array([[0.0, 1.0]])

    result = compute_calibration_metrics(probability, target, n_bins=2)

    assert result.valid_pixels == 2
    assert result.ece == pytest.approx(0.1)
    assert result.brier == pytest.approx(0.01)
    assert sum(item.count for item in result.bins) == 2


def test_geometry_metrics_match_rows_and_measure_symmetric_distance() -> None:
    predicted = [LineString([(0.0, 1.0), (10.0, 1.0)])]
    reference = [LineString([(0.0, 0.0), (10.0, 0.0)])]

    result = compute_geometry_metrics(
        predicted,
        reference,
        max_match_distance_m=2.0,
        sample_spacing_m=1.0,
    )
    rejected = compute_geometry_metrics(predicted, reference, max_match_distance_m=0.5)

    assert result.matched_rows == 1
    assert result.row_precision == result.row_recall == result.row_f1 == 1.0
    assert result.mean_distance_m == pytest.approx(1.0)
    assert result.p95_distance_m == pytest.approx(1.0)
    assert result.mean_absolute_length_error_m == 0.0
    assert rejected.matched_rows == 0
    assert rejected.mean_distance_m is None


def test_topology_metrics_detect_crossing_duplicates_and_high_endpoint_degree() -> None:
    horizontal = LineString([(0.0, 0.0), (2.0, 0.0)])
    vertical = LineString([(1.0, -1.0), (1.0, 1.0)])
    duplicate_reversed = LineString([(2.0, 0.0), (0.0, 0.0)])
    third_at_origin = LineString([(0.0, 0.0), (-1.0, 1.0)])

    result = compute_topology_metrics([horizontal, vertical, duplicate_reversed, third_at_origin])

    assert result.valid_feature_count == 4
    assert result.crossing_pairs >= 1
    assert result.duplicate_pairs == 1
    assert result.endpoint_nodes_degree_gt2 >= 1
    assert result.connected_components == 1


def test_metric_inputs_reject_non_finite_valid_probability() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        compute_raster_metrics(np.array([[np.nan]]), np.array([[0.0]]))
