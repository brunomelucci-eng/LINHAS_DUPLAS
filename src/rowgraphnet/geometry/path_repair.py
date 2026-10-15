"""Orientation-aware A* repair over continuous model evidence."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
from pyproj import CRS
from scipy.ndimage import label
from shapely.geometry import LineString

from rowgraphnet.domain import CandidateConnection, RasterPrediction, RepairPath, SpacingEstimate
from rowgraphnet.geometry.orientation_field import (
    FusedOrientationField,
    orientation_field_to_world,
)


@dataclass(slots=True, frozen=True)
class PathCostWeights:
    """Dimensionless costs selected on validation data or learned by the GNN."""

    background: float = 1.0
    orientation: float = 1.0
    curvature: float = 1.0
    uncertainty: float = 1.0

    def normalized(self) -> PathCostWeights:
        values = np.asarray(
            [self.background, self.orientation, self.curvature, self.uncertainty],
            dtype=float,
        )
        if not np.isfinite(values).all() or np.any(values < 0) or not np.any(values > 0):
            raise ValueError(
                "path cost weights must be finite, non-negative, and include a positive value"
            )
        values /= values.sum()
        return PathCostWeights(*map(float, values))


_MOVES: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _is_metric(crs: CRS) -> bool:
    return (
        crs.is_projected
        and bool(crs.axis_info)
        and all(axis.unit_name.lower() in {"metre", "meter"} for axis in crs.axis_info)
    )


def _pixel_from_world(prediction: RasterPrediction, xy: tuple[float, float]) -> tuple[int, int]:
    col, row = (~prediction.transform) * xy
    return int(np.clip(round(row - 0.5), 0, prediction.center.shape[0] - 1)), int(
        np.clip(round(col - 0.5), 0, prediction.center.shape[1] - 1)
    )


def _world_from_pixel(prediction: RasterPrediction, pixel: tuple[int, int]) -> tuple[float, float]:
    row, col = pixel
    x, y = prediction.transform * (col + 0.5, row + 0.5)
    return float(x), float(y)


def _metric_pixel_size(prediction: RasterPrediction) -> float:
    x_step = math.hypot(prediction.transform.a, prediction.transform.d)
    y_step = math.hypot(prediction.transform.b, prediction.transform.e)
    return float(math.sqrt(x_step * y_step))


def _source_component_mask(
    observed: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> np.ndarray:
    labels, _ = label(observed, structure=np.ones((3, 3), dtype=np.uint8))
    allowed_labels = {int(labels[start]), int(labels[goal])} - {0}
    if not allowed_labels:
        return np.zeros_like(observed, dtype=bool)
    return np.isin(labels, list(allowed_labels))


def _corridor_mask(
    prediction: RasterPrediction,
    connector: LineString,
    half_width_m: float,
) -> np.ndarray:
    height, width = prediction.center.shape
    rows, cols = np.indices((height, width), dtype=float)
    xs = (
        prediction.transform.a * (cols + 0.5)
        + prediction.transform.b * (rows + 0.5)
        + prediction.transform.c
    )
    ys = (
        prediction.transform.d * (cols + 0.5)
        + prediction.transform.e * (rows + 0.5)
        + prediction.transform.f
    )
    start = np.asarray(connector.coords[0], dtype=float)
    end = np.asarray(connector.coords[-1], dtype=float)
    delta = end - start
    length_squared = float(np.dot(delta, delta))
    if length_squared <= np.finfo(float).eps:
        return np.zeros((height, width), dtype=bool)
    projection = ((xs - start[0]) * delta[0] + (ys - start[1]) * delta[1]) / length_squared
    projection = np.clip(projection, 0.0, 1.0)
    closest_x = start[0] + projection * delta[0]
    closest_y = start[1] + projection * delta[1]
    distance = np.hypot(xs - closest_x, ys - closest_y)
    return distance <= half_width_m


def _reconstruct_path(
    came_from: dict[tuple[int, int, int], tuple[int, int, int]],
    state: tuple[int, int, int],
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = [(state[0], state[1])]
    while state in came_from:
        state = came_from[state]
        result.append((state[0], state[1]))
    result.reverse()
    compact: list[tuple[int, int]] = []
    for pixel in result:
        if not compact or compact[-1] != pixel:
            compact.append(pixel)
    return compact


def _geometry_stays_traversable(
    geometry: LineString,
    prediction: RasterPrediction,
    traversable: np.ndarray,
    pixel_size_m: float,
) -> bool:
    sample_count = max(math.ceil(geometry.length / max(pixel_size_m / 2.0, np.finfo(float).eps)), 2)
    inverse = ~prediction.transform
    for distance in np.linspace(0.0, geometry.length, sample_count):
        point = geometry.interpolate(float(distance))
        col, row = inverse * (point.x, point.y)
        row_index, col_index = round(row - 0.5), round(col - 0.5)
        if not (
            0 <= row_index < traversable.shape[0]
            and 0 <= col_index < traversable.shape[1]
            and traversable[row_index, col_index]
        ):
            return False
    return True


def repair_candidate_path(
    candidate: CandidateConnection,
    prediction: RasterPrediction,
    spacing: SpacingEstimate,
    weights: PathCostWeights | None = None,
    orientation_field: FusedOrientationField | None = None,
) -> RepairPath | None:
    """Find the lowest-cost smooth path, or abstain if none is feasible."""

    crs = CRS.from_user_input(prediction.crs)
    if not _is_metric(crs):
        raise ValueError("path repair requires a projected CRS with metre axes")
    if spacing.median_m <= 0:
        raise ValueError("spacing median must be positive")
    cost_weights = (weights or PathCostWeights()).normalized()
    start_xy = tuple(map(float, candidate.geometry.coords[0]))
    goal_xy = tuple(map(float, candidate.geometry.coords[-1]))
    start = _pixel_from_world(prediction, start_xy)
    goal = _pixel_from_world(prediction, goal_xy)
    if start == goal:
        return None

    valid = np.isfinite(prediction.center) & np.isfinite(prediction.uncertainty)
    if prediction.valid_mask is not None:
        valid &= prediction.valid_mask.astype(bool)
    center = np.clip(np.nan_to_num(prediction.center, nan=0.0), 0.0, 1.0)
    raw_uncertainty = np.asarray(prediction.uncertainty, dtype=float)
    uncertainty = np.divide(
        raw_uncertainty,
        1.0 + raw_uncertainty,
        out=np.ones_like(raw_uncertainty),
        where=np.isfinite(raw_uncertainty) & (raw_uncertainty >= 0.0),
    )
    valid_values = center[valid]
    if not len(valid_values):
        return None
    observed_threshold = float(np.quantile(valid_values, 0.90))
    observed = valid & (center >= observed_threshold)
    source_mask = _source_component_mask(observed, start, goal)
    crossing_mask = observed & ~source_mask

    pixel_size = _metric_pixel_size(prediction)
    distribution_width = max(
        1.4826 * spacing.mad_m,
        spacing.median_m * spacing.entropy,
        pixel_size,
    )
    # The midline between adjacent rows is the natural maximum corridor.  A
    # broad spacing posterior narrows it, encouraging abstention under uncertainty.
    corridor_half_width = max((spacing.median_m - distribution_width) / 2.0, pixel_size)
    corridor = _corridor_mask(prediction, candidate.geometry, corridor_half_width)
    traversable = valid & corridor & ~crossing_mask
    traversable[start] = True
    traversable[goal] = True

    background_cost = 1.0 - center
    base_cost = cost_weights.background * background_cost + cost_weights.uncertainty * uncertainty
    if orientation_field is None:
        learned = FusedOrientationField(
            sin2=np.asarray(prediction.orientation_sin2, dtype=float),
            cos2=np.asarray(prediction.orientation_cos2, dtype=float),
            confidence=np.ones_like(prediction.center, dtype=float),
        )
        world_orientation = orientation_field_to_world(learned, prediction.transform)
    else:
        if not (
            orientation_field.sin2.shape
            == orientation_field.cos2.shape
            == orientation_field.confidence.shape
            == prediction.center.shape
        ):
            raise ValueError("orientation_field must match the prediction grid")
        world_orientation = orientation_field
    axial_s = np.asarray(world_orientation.sin2, dtype=float)
    axial_c = np.asarray(world_orientation.cos2, dtype=float)
    axial_confidence = np.clip(np.asarray(world_orientation.confidence, dtype=float), 0.0, 1.0)

    move_world: tuple[tuple[float, float, float, float], ...] = tuple(
        (
            prediction.transform.a * d_col + prediction.transform.b * d_row,
            prediction.transform.d * d_col + prediction.transform.e * d_row,
            math.hypot(
                prediction.transform.a * d_col + prediction.transform.b * d_row,
                prediction.transform.d * d_col + prediction.transform.e * d_row,
            ),
            math.atan2(
                prediction.transform.d * d_col + prediction.transform.e * d_row,
                prediction.transform.a * d_col + prediction.transform.b * d_row,
            ),
        )
        for d_row, d_col in _MOVES
    )

    start_state = (start[0], start[1], -1)
    queue: list[tuple[float, float, int, int, int]] = [(0.0, 0.0, *start_state)]
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    best_cost: dict[tuple[int, int, int], float] = {start_state: 0.0}
    final_state: tuple[int, int, int] | None = None

    minimum_cell_cost = float(np.min(base_cost[traversable])) if np.any(traversable) else 0.0
    while queue:
        _, current_cost, row, col, previous_direction = heapq.heappop(queue)
        state = (row, col, previous_direction)
        if current_cost > best_cost.get(state, math.inf):
            continue
        if (row, col) == goal:
            final_state = state
            break
        for direction_index, (d_row, d_col) in enumerate(_MOVES):
            next_row, next_col = row + d_row, col + d_col
            if not (0 <= next_row < center.shape[0] and 0 <= next_col < center.shape[1]):
                continue
            if not traversable[next_row, next_col]:
                continue
            _, _, step_length, move_theta = move_world[direction_index]
            orientation_alignment = axial_c[next_row, next_col] * math.cos(
                2.0 * move_theta
            ) + axial_s[next_row, next_col] * math.sin(2.0 * move_theta)
            orientation_penalty = (1.0 - float(np.clip(orientation_alignment, -1.0, 1.0))) / 2.0
            orientation_penalty *= axial_confidence[next_row, next_col]
            if previous_direction >= 0:
                old_theta = move_world[previous_direction][3]
                curvature_penalty = (1.0 - math.cos(move_theta - old_theta)) / 2.0
            else:
                curvature_penalty = 0.0
            step_cost = step_length * (
                base_cost[next_row, next_col]
                + cost_weights.orientation * orientation_penalty
                + cost_weights.curvature * curvature_penalty
            )
            new_cost = current_cost + step_cost
            next_state = (next_row, next_col, direction_index)
            if new_cost >= best_cost.get(next_state, math.inf):
                continue
            best_cost[next_state] = new_cost
            came_from[next_state] = state
            next_xy = _world_from_pixel(prediction, (next_row, next_col))
            heuristic = math.hypot(goal_xy[0] - next_xy[0], goal_xy[1] - next_xy[1])
            heuristic *= minimum_cell_cost
            heapq.heappush(
                queue,
                (new_cost + heuristic, new_cost, next_row, next_col, direction_index),
            )

    if final_state is None:
        return None
    pixels = _reconstruct_path(came_from, final_state)
    coords = [_world_from_pixel(prediction, pixel) for pixel in pixels]
    coords[0] = start_xy
    coords[-1] = goal_xy
    geometry = LineString(coords).simplify(pixel_size / 2.0, preserve_topology=True)
    if (
        geometry.is_empty
        or not geometry.is_valid
        or not geometry.is_simple
        or not _geometry_stays_traversable(geometry, prediction, traversable, pixel_size)
    ):
        return None
    values = np.array([center[pixel] for pixel in pixels], dtype=float)
    uncertainties = np.array([uncertainty[pixel] for pixel in pixels], dtype=float)
    total_cost = float(best_cost[final_state])
    path_length_m = max(float(geometry.length), np.finfo(float).eps)
    normalized_cost = total_cost / path_length_m
    path_quality = float(
        np.mean(
            (
                float(values.mean()),
                1.0 - float(uncertainties.mean()),
                math.exp(-normalized_cost),
            )
        )
    )
    reason = dict(candidate.evidence)
    reason.update(
        {
            "algorithm": "orientation_aware_a_star",
            "path_pixels": len(pixels),
            "corridor_half_width_m": corridor_half_width,
            "normalized_cost_per_m": normalized_cost,
            "path_quality": path_quality,
            "crossing_policy": "hard_veto",
        }
    )
    return RepairPath(
        candidate_id=candidate.candidate_id,
        geometry=geometry,
        total_cost=total_cost,
        mean_center_support=float(values.mean()),
        mean_uncertainty=float(uncertainties.mean()),
        reason=reason,
    )
