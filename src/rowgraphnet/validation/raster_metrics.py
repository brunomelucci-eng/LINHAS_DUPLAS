"""Raster, orientation, and calibration metrics for RowGraphNet outputs.

All functions honor an explicit validity mask.  Non-finite values outside that
mask are ignored; non-finite values inside it are rejected so a broken raster
cannot silently improve a score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from skimage.morphology import skeletonize


@dataclass(slots=True, frozen=True)
class RasterMetrics:
    precision: float
    recall: float
    f1: float
    iou: float
    dice: float
    cldice: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    valid_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class AxialOrientationMetrics:
    mean_error_deg: float | None
    median_error_deg: float | None
    p95_error_deg: float | None
    valid_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    positive_fraction: float | None


@dataclass(slots=True, frozen=True)
class CalibrationMetrics:
    ece: float | None
    brier: float | None
    valid_pixels: int
    n_bins: int
    bins: tuple[CalibrationBin, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_common(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(prediction)
    expected = np.asarray(target)
    if predicted.ndim != 2 or expected.ndim != 2:
        raise ValueError("prediction and target must be two-dimensional")
    if predicted.shape != expected.shape:
        raise ValueError("prediction and target must have the same shape")
    if valid_mask is None:
        valid = np.ones(predicted.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != predicted.shape:
            raise ValueError("valid_mask must have the same shape as prediction")
    if np.any(valid & ~np.isfinite(predicted)):
        raise ValueError("prediction contains non-finite values in valid pixels")
    if np.any(valid & ~np.isfinite(expected)):
        raise ValueError("target contains non-finite values in valid pixels")
    return predicted, expected, valid


def _unit_score(numerator: int, denominator: int) -> float:
    """Return one for a vacuous, perfectly empty comparison."""

    return float(numerator / denominator) if denominator else 1.0


def binary_cldice(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Compute discrete clDice for binary tubular masks.

    Skeleton precision measures whether the predicted skeleton lies inside the
    target.  Skeleton sensitivity performs the converse check.  Empty/empty is
    defined as a perfect result, while one empty mask and one non-empty mask
    produce zero.
    """

    predicted, expected, valid = _validate_common(prediction, target, valid_mask)
    predicted_binary = predicted.astype(bool) & valid
    target_binary = expected.astype(bool) & valid
    if not predicted_binary.any() and not target_binary.any():
        return 1.0
    if not predicted_binary.any() or not target_binary.any():
        return 0.0

    predicted_skeleton = skeletonize(predicted_binary)
    target_skeleton = skeletonize(target_binary)
    topology_precision = _unit_score(
        int(np.count_nonzero(predicted_skeleton & target_binary)),
        int(np.count_nonzero(predicted_skeleton)),
    )
    topology_sensitivity = _unit_score(
        int(np.count_nonzero(target_skeleton & predicted_binary)),
        int(np.count_nonzero(target_skeleton)),
    )
    denominator = topology_precision + topology_sensitivity
    if denominator == 0:
        return 0.0
    return float(2.0 * topology_precision * topology_sensitivity / denominator)


def compute_raster_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float = 0.5,
    valid_mask: np.ndarray | None = None,
) -> RasterMetrics:
    """Threshold a probability raster and compute binary segmentation scores."""

    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")
    predicted, expected, valid = _validate_common(probability, target, valid_mask)
    if np.any(valid & ((predicted < 0.0) | (predicted > 1.0))):
        raise ValueError("probability values in valid pixels must be within [0, 1]")
    if np.any(valid & ((expected < 0.0) | (expected > 1.0))):
        raise ValueError("target values in valid pixels must be within [0, 1]")

    predicted_binary = (predicted >= threshold) & valid
    target_binary = (expected >= 0.5) & valid
    true_positive = int(np.count_nonzero(predicted_binary & target_binary))
    false_positive = int(np.count_nonzero(predicted_binary & ~target_binary & valid))
    false_negative = int(np.count_nonzero(~predicted_binary & target_binary & valid))
    true_negative = int(np.count_nonzero(~predicted_binary & ~target_binary & valid))

    precision = _unit_score(true_positive, true_positive + false_positive)
    recall = _unit_score(true_positive, true_positive + false_negative)
    f1 = _unit_score(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    iou = _unit_score(true_positive, true_positive + false_positive + false_negative)
    dice = _unit_score(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    cldice = binary_cldice(predicted_binary, target_binary, valid)
    return RasterMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou,
        dice=dice,
        cldice=cldice,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        valid_pixels=int(np.count_nonzero(valid)),
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = quantile * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, cutoff, side="left")), len(values) - 1)
    return float(sorted_values[index])


def compute_axial_orientation_error(
    predicted_sin2: np.ndarray,
    predicted_cos2: np.ndarray,
    target_sin2: np.ndarray,
    target_cos2: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> AxialOrientationMetrics:
    """Measure axial angular error, where theta and theta + 180 are equal."""

    arrays = [
        np.asarray(predicted_sin2, dtype=np.float64),
        np.asarray(predicted_cos2, dtype=np.float64),
        np.asarray(target_sin2, dtype=np.float64),
        np.asarray(target_cos2, dtype=np.float64),
    ]
    shape = arrays[0].shape
    if len(shape) != 2 or any(item.shape != shape for item in arrays[1:]):
        raise ValueError("all axial orientation arrays must share the same two-dimensional shape")
    valid = np.ones(shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != shape:
        raise ValueError("valid_mask must match the orientation arrays")
    if weights is None:
        weight_array = np.ones(shape, dtype=np.float64)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != shape:
            raise ValueError("weights must match the orientation arrays")
        if np.any(valid & (~np.isfinite(weight_array) | (weight_array < 0.0))):
            raise ValueError("weights must be finite and non-negative in valid pixels")

    finite = np.logical_and.reduce([np.isfinite(item) for item in arrays])
    if np.any(valid & ~finite):
        raise ValueError("orientation contains non-finite values in valid pixels")
    predicted_norm = np.hypot(arrays[0], arrays[1])
    target_norm = np.hypot(arrays[2], arrays[3])
    usable = valid & finite & (predicted_norm > 0.0) & (target_norm > 0.0) & (weight_array > 0.0)
    count = int(np.count_nonzero(usable))
    if count == 0:
        return AxialOrientationMetrics(None, None, None, 0)

    dot = (arrays[0] * arrays[2] + arrays[1] * arrays[3]) / (predicted_norm * target_norm)
    error_deg = np.degrees(0.5 * np.arccos(np.clip(dot[usable], -1.0, 1.0)))
    selected_weights = weight_array[usable]
    mean = float(np.average(error_deg, weights=selected_weights))
    median = _weighted_quantile(error_deg, selected_weights, 0.5)
    p95 = _weighted_quantile(error_deg, selected_weights, 0.95)
    return AxialOrientationMetrics(mean, median, p95, count)


def compute_calibration_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    n_bins: int = 15,
) -> CalibrationMetrics:
    """Compute equal-width expected calibration error and Brier score."""

    if n_bins < 1:
        raise ValueError("n_bins must be at least one")
    predicted, expected, valid = _validate_common(probability, target, valid_mask)
    if np.any(valid & ((predicted < 0.0) | (predicted > 1.0))):
        raise ValueError("probability values in valid pixels must be within [0, 1]")
    if np.any(valid & ((expected < 0.0) | (expected > 1.0))):
        raise ValueError("target values in valid pixels must be within [0, 1]")
    probabilities = predicted[valid].astype(np.float64)
    labels = expected[valid].astype(np.float64)
    count = len(probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    if count == 0:
        bins = tuple(
            CalibrationBin(float(edges[index]), float(edges[index + 1]), 0, None, None)
            for index in range(n_bins)
        )
        return CalibrationMetrics(None, None, 0, n_bins, bins)

    assignments = np.minimum((probabilities * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    bins_list: list[CalibrationBin] = []
    for index in range(n_bins):
        selected = assignments == index
        bin_count = int(np.count_nonzero(selected))
        if bin_count:
            mean_probability = float(np.mean(probabilities[selected]))
            positive_fraction = float(np.mean(labels[selected]))
            ece += bin_count / count * abs(mean_probability - positive_fraction)
        else:
            mean_probability = None
            positive_fraction = None
        bins_list.append(
            CalibrationBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=bin_count,
                mean_probability=mean_probability,
                positive_fraction=positive_fraction,
            )
        )
    brier = float(np.mean((probabilities - labels) ** 2))
    return CalibrationMetrics(float(ece), brier, count, n_bins, tuple(bins_list))


# Short aliases are convenient in experiment notebooks while the explicit
# ``compute_*`` names remain the public, self-documenting API.
raster_metrics = compute_raster_metrics
axial_orientation_error = compute_axial_orientation_error
calibration_metrics = compute_calibration_metrics
