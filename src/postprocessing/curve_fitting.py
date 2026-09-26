"""Constrained curve fitting for crop-row branches."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from scipy.interpolate import UnivariateSpline
from shapely.geometry import LineString

logger = logging.getLogger(__name__)


def _disabled_fit_metadata(points: np.ndarray, config: dict) -> dict:
    """Reproduce disabled-fit audit metadata without constructing geometries."""
    points = np.asarray(points, dtype=np.float64)
    metadata = {"fit_quality_class": "INVALID", "fit_type": "original"}
    if len(points) < 2 or not np.all(np.isfinite(points)):
        return metadata

    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-12:
        return metadata

    # GEOS' default discrete Hausdorff against the two-vertex chord is the
    # maximum perpendicular distance of the input vertices to that chord.
    relative = points - points[0]
    deviations = np.abs(chord[0] * relative[:, 1] - chord[1] * relative[:, 0]) / chord_length
    tolerance = float(config.get("curve_fitting", {}).get("smoothing_tolerance_m", 0.05))
    if len(points) < 4 or float(np.max(deviations)) <= tolerance:
        return {"fit_quality_class": "GOOD", "fit_type": "preserved"}
    return {"fit_quality_class": "SUSPECT", "fit_type": "original"}


def fit_branches_if_enabled(
    branches_world: List[np.ndarray],
    config: dict,
    *,
    return_metadata: bool = False,
):
    """Bypass every geometric fitting operation when fitting is disabled."""
    if not config.get("curve_fitting", {}).get("enabled", True):
        fitted = branches_world
        records = [_disabled_fit_metadata(branch, config) for branch in branches_world]
        return (fitted, records) if return_metadata else fitted
    return smooth_and_fit_branches(
        branches_world,
        config,
        return_metadata=return_metadata,
    )


def _fallback(points: np.ndarray, simplify_tolerance: float, enabled: bool) -> np.ndarray:
    if not enabled:
        return points.copy()
    simplified = LineString(points).simplify(simplify_tolerance, preserve_topology=True)
    if simplified.is_empty or simplified.geom_type != "LineString":
        return points.copy()
    result = np.asarray(simplified.coords, dtype=np.float64)
    result[0] = points[0]
    result[-1] = points[-1]
    return result


def fit_curve_to_points(
    points: np.ndarray,
    config: dict,
    *,
    return_metadata: bool = False,
):
    """Fit a spline only to SUSPECT lines and enforce geometric safeguards."""
    points = np.asarray(points, dtype=np.float64)
    curve_cfg = config.get("curve_fitting", {})
    simplify_tolerance = float(config.get("vector", {}).get("simplify_tolerance_m", 0.05))
    smoothing_tolerance = float(curve_cfg.get("smoothing_tolerance_m", 0.05))
    max_deviation = float(curve_cfg.get("max_deviation_m", 0.10))
    fallback_enabled = bool(curve_cfg.get("fallback_to_simplified", True))
    reject_self_intersection = bool(curve_cfg.get("reject_self_intersection", True))

    metadata = {"fit_quality_class": "INVALID", "fit_type": "original"}
    if len(points) < 2:
        result = points.copy()
        return (result, metadata) if return_metadata else result

    original = LineString(points)
    if original.is_empty or not original.is_valid or not original.is_simple or original.length < 1e-9:
        result = _fallback(points, simplify_tolerance, fallback_enabled)
        metadata["fit_type"] = "simplified_fallback" if not np.array_equal(result, points) else "original"
        return (result, metadata) if return_metadata else result

    chord = LineString([points[0], points[-1]])
    straightness_deviation = original.hausdorff_distance(chord)
    if straightness_deviation <= smoothing_tolerance or len(points) < 4:
        metadata = {"fit_quality_class": "GOOD", "fit_type": "preserved"}
        result = points.copy()
        return (result, metadata) if return_metadata else result

    metadata["fit_quality_class"] = "SUSPECT"
    if not curve_cfg.get("enabled", True):
        result = points.copy()
        return (result, metadata) if return_metadata else result

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-9))
    unique_points = points[keep]
    if len(unique_points) < 4:
        result = _fallback(points, simplify_tolerance, fallback_enabled)
        metadata["fit_type"] = "simplified_fallback"
        return (result, metadata) if return_metadata else result

    t = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(unique_points, axis=0), axis=1))))
    spacing = float(config.get("vector", {}).get("vertex_spacing_m", 1.5))
    sample_count = max(3, int(np.ceil(t[-1] / max(spacing, 1e-3))) + 1)
    t_new = np.linspace(0.0, t[-1], sample_count)

    try:
        degree = min(3, len(unique_points) - 1)
        smoothing = len(unique_points) * smoothing_tolerance**2
        spline_x = UnivariateSpline(t, unique_points[:, 0], k=degree, s=smoothing)
        spline_y = UnivariateSpline(t, unique_points[:, 1], k=degree, s=smoothing)
        candidate = np.column_stack((spline_x(t_new), spline_y(t_new)))
        candidate[0] = points[0]
        candidate[-1] = points[-1]
        candidate_line = LineString(candidate)
        deviation = original.hausdorff_distance(candidate_line)
        unsafe = (
            candidate_line.is_empty
            or not candidate_line.is_valid
            or (reject_self_intersection and not candidate_line.is_simple)
            or deviation > max_deviation
        )
        if unsafe:
            raise ValueError(f"unsafe fitted curve (Hausdorff deviation={deviation:.4f}m)")
        metadata["fit_type"] = "spline"
        metadata["fit_hausdorff_deviation_m"] = float(deviation)
        result = candidate
    except Exception as exc:
        logger.debug("Constrained spline rejected: %s", exc)
        result = _fallback(points, simplify_tolerance, fallback_enabled)
        metadata["fit_type"] = "simplified_fallback" if fallback_enabled else "original"

    return (result, metadata) if return_metadata else result


def smooth_and_fit_branches(
    branches_world: List[np.ndarray],
    config: dict,
    *,
    return_metadata: bool = False,
):
    fitted, records = [], []
    for branch in branches_world:
        result, metadata = fit_curve_to_points(branch, config, return_metadata=True)
        fitted.append(result)
        records.append(metadata)
    return (fitted, records) if return_metadata else fitted
