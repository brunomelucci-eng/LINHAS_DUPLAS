"""Basic vector-geometry, row matching, and topology validation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Point

from rowgraphnet.domain import RepairPath, RowRecord


@dataclass(slots=True, frozen=True)
class GeometryMetrics:
    predicted_rows: int
    reference_rows: int
    matched_rows: int
    unmatched_predicted: int
    unmatched_reference: int
    row_precision: float
    row_recall: float
    row_f1: float
    mean_distance_m: float | None
    p95_distance_m: float | None
    mean_absolute_length_error_m: float | None
    mean_relative_length_error: float | None
    total_length_difference_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class TopologyMetrics:
    feature_count: int
    valid_feature_count: int
    empty_count: int
    invalid_count: int
    non_simple_count: int
    crossing_pairs: int
    overlapping_pairs: int
    duplicate_pairs: int
    connected_components: int
    endpoint_nodes_degree_gt2: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_line_geometry(geometry: LineString, *, name: str = "geometry") -> None:
    """Reject geometries that cannot be safely measured or exported."""

    if not isinstance(geometry, LineString):
        raise TypeError(f"{name} must be a LineString")
    if geometry.is_empty:
        raise ValueError(f"{name} must not be empty")
    if not geometry.is_valid:
        raise ValueError(f"{name} is invalid")
    if not geometry.is_simple:
        raise ValueError(f"{name} must not self-intersect")
    coordinates = np.asarray(geometry.coords, dtype=np.float64)
    if coordinates.ndim != 2 or len(coordinates) < 2:
        raise ValueError(f"{name} must contain at least two coordinates")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} contains non-finite coordinates")
    if not np.isfinite(geometry.length) or geometry.length <= 0.0:
        raise ValueError(f"{name} must have positive finite length")


def _finite_unit_interval(value: float, name: str) -> None:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")


def validate_row_records(records: Iterable[RowRecord]) -> tuple[RowRecord, ...]:
    """Validate mandatory row attributes and return a reusable tuple."""

    materialized = tuple(records)
    seen_ids: set[str] = set()
    for index, record in enumerate(materialized):
        if not isinstance(record, RowRecord):
            raise TypeError(f"rows[{index}] must be a RowRecord")
        if not record.row_id or not record.row_id.strip():
            raise ValueError(f"rows[{index}].row_id must not be empty")
        if record.row_id in seen_ids:
            raise ValueError(f"duplicate row_id: {record.row_id}")
        seen_ids.add(record.row_id)
        validate_line_geometry(record.geometry, name=f"rows[{index}].geometry")
        _finite_unit_interval(record.confidence, f"rows[{index}].confidence")
        _finite_unit_interval(record.observed_ratio, f"rows[{index}].observed_ratio")
        _finite_unit_interval(record.repaired_ratio, f"rows[{index}].repaired_ratio")
        if record.observed_ratio + record.repaired_ratio > 1.0 + 1e-9:
            raise ValueError("observed_ratio + repaired_ratio must not exceed one")
        if record.spacing_local_m is not None and (
            not np.isfinite(record.spacing_local_m) or record.spacing_local_m <= 0.0
        ):
            raise ValueError(f"rows[{index}].spacing_local_m must be positive or None")
        if not np.isfinite(record.orientation_deg) or not 0.0 <= record.orientation_deg < 180.0:
            raise ValueError(f"rows[{index}].orientation_deg must be within [0, 180)")
        if record.topology_status not in {"valid", "ambiguous", "rejected"}:
            raise ValueError(f"rows[{index}].topology_status is unsupported")
        if not isinstance(record.repair_reason, str):
            raise TypeError(f"rows[{index}].repair_reason must be a string")
        if not record.model_version or not record.model_version.strip():
            raise ValueError(f"rows[{index}].model_version must not be empty")
    return materialized


def validate_repair_paths(paths: Iterable[RepairPath]) -> tuple[RepairPath, ...]:
    """Validate reconstructed path IDs, geometries, scores, and evidence."""

    materialized = tuple(paths)
    seen_ids: set[str] = set()
    for index, path in enumerate(materialized):
        if not isinstance(path, RepairPath):
            raise TypeError(f"repairs[{index}] must be a RepairPath")
        if not path.candidate_id or not path.candidate_id.strip():
            raise ValueError(f"repairs[{index}].candidate_id must not be empty")
        if path.candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id: {path.candidate_id}")
        seen_ids.add(path.candidate_id)
        validate_line_geometry(path.geometry, name=f"repairs[{index}].geometry")
        if not np.isfinite(path.total_cost) or path.total_cost < 0.0:
            raise ValueError(f"repairs[{index}].total_cost must be finite and non-negative")
        _finite_unit_interval(
            path.mean_center_support,
            f"repairs[{index}].mean_center_support",
        )
        _finite_unit_interval(path.mean_uncertainty, f"repairs[{index}].mean_uncertainty")
        if not isinstance(path.reason, dict):
            raise TypeError(f"repairs[{index}].reason must be a dictionary")
        if path.repair_type not in {"internal_gap_repair", "boundary_extension"}:
            raise ValueError(f"repairs[{index}].repair_type is unsupported")
        declared_type = path.reason.get("repair_type")
        if declared_type is not None and declared_type != path.repair_type:
            raise ValueError(f"repairs[{index}] has conflicting repair_type provenance")
    return materialized


def _extract_lines(items: Sequence[RowRecord | LineString]) -> list[LineString]:
    lines: list[LineString] = []
    for index, item in enumerate(items):
        geometry = item.geometry if isinstance(item, RowRecord) else item
        validate_line_geometry(geometry, name=f"features[{index}]")
        lines.append(geometry)
    return lines


def _sample_line(
    line: LineString,
    sample_spacing_m: float | None,
    max_samples_per_line: int,
) -> list[Point]:
    if sample_spacing_m is None:
        sample_count = min(max_samples_per_line, max(2, len(line.coords) * 16))
    else:
        sample_count = min(
            max_samples_per_line,
            max(2, int(np.ceil(line.length / sample_spacing_m)) + 1),
        )
    distances = np.linspace(0.0, line.length, sample_count)
    return [line.interpolate(float(distance)) for distance in distances]


def _symmetric_sample_distances(
    first: LineString,
    second: LineString,
    sample_spacing_m: float | None,
    max_samples_per_line: int,
) -> np.ndarray:
    first_points = _sample_line(first, sample_spacing_m, max_samples_per_line)
    second_points = _sample_line(second, sample_spacing_m, max_samples_per_line)
    first_distances = [point.distance(second) for point in first_points]
    second_distances = [point.distance(first) for point in second_points]
    return np.asarray(first_distances + second_distances, dtype=np.float64)


def _empty_aware_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 1.0


def compute_geometry_metrics(
    predicted: Sequence[RowRecord | LineString],
    reference: Sequence[RowRecord | LineString],
    *,
    max_match_distance_m: float | None = None,
    sample_spacing_m: float | None = None,
    max_samples_per_line: int = 512,
) -> GeometryMetrics:
    """Match rows one-to-one and report symmetric line-distance statistics.

    Matching minimizes Hausdorff distance.  ``max_match_distance_m`` is
    intentionally supplied by the validation protocol (normally from GSD and
    local spacing); no agronomic distance is hidden in this implementation.
    """

    if max_match_distance_m is not None and (
        not np.isfinite(max_match_distance_m) or max_match_distance_m < 0.0
    ):
        raise ValueError("max_match_distance_m must be finite and non-negative")
    if sample_spacing_m is not None and (
        not np.isfinite(sample_spacing_m) or sample_spacing_m <= 0.0
    ):
        raise ValueError("sample_spacing_m must be positive and finite")
    if max_samples_per_line < 2:
        raise ValueError("max_samples_per_line must be at least two")

    predicted_lines = _extract_lines(predicted)
    reference_lines = _extract_lines(reference)
    predicted_count = len(predicted_lines)
    reference_count = len(reference_lines)
    pairs: list[tuple[int, int]] = []
    if predicted_lines and reference_lines:
        costs = np.asarray(
            [
                [
                    predicted_line.hausdorff_distance(reference_line)
                    for reference_line in reference_lines
                ]
                for predicted_line in predicted_lines
            ],
            dtype=np.float64,
        )
        predicted_indices, reference_indices = linear_sum_assignment(costs)
        for predicted_index, reference_index in zip(
            predicted_indices.tolist(), reference_indices.tolist(), strict=True
        ):
            if (
                max_match_distance_m is None
                or costs[predicted_index, reference_index] <= max_match_distance_m
            ):
                pairs.append((predicted_index, reference_index))

    matched_count = len(pairs)
    row_precision = _empty_aware_ratio(matched_count, predicted_count)
    row_recall = _empty_aware_ratio(matched_count, reference_count)
    denominator = row_precision + row_recall
    row_f1 = 2.0 * row_precision * row_recall / denominator if denominator else 0.0

    all_distances: list[np.ndarray] = []
    absolute_length_errors: list[float] = []
    relative_length_errors: list[float] = []
    for predicted_index, reference_index in pairs:
        predicted_line = predicted_lines[predicted_index]
        reference_line = reference_lines[reference_index]
        all_distances.append(
            _symmetric_sample_distances(
                predicted_line,
                reference_line,
                sample_spacing_m,
                max_samples_per_line,
            )
        )
        length_error = abs(predicted_line.length - reference_line.length)
        absolute_length_errors.append(float(length_error))
        relative_length_errors.append(float(length_error / reference_line.length))

    if all_distances:
        distances = np.concatenate(all_distances)
        mean_distance = float(np.mean(distances))
        p95_distance = float(np.percentile(distances, 95))
        mean_length_error = float(np.mean(absolute_length_errors))
        mean_relative_length_error = float(np.mean(relative_length_errors))
    else:
        mean_distance = None
        p95_distance = None
        mean_length_error = None
        mean_relative_length_error = None

    total_predicted_length = sum(line.length for line in predicted_lines)
    total_reference_length = sum(line.length for line in reference_lines)
    return GeometryMetrics(
        predicted_rows=predicted_count,
        reference_rows=reference_count,
        matched_rows=matched_count,
        unmatched_predicted=predicted_count - matched_count,
        unmatched_reference=reference_count - matched_count,
        row_precision=row_precision,
        row_recall=row_recall,
        row_f1=float(row_f1),
        mean_distance_m=mean_distance,
        p95_distance_m=p95_distance,
        mean_absolute_length_error_m=mean_length_error,
        mean_relative_length_error=mean_relative_length_error,
        total_length_difference_m=float(total_predicted_length - total_reference_length),
    )


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _endpoint_degree_violations(lines: list[LineString], tolerance_m: float) -> int:
    endpoints = [
        Point(coordinate) for line in lines for coordinate in (line.coords[0], line.coords[-1])
    ]
    if not endpoints:
        return 0
    groups = _DisjointSet(len(endpoints))
    for first in range(len(endpoints)):
        for second in range(first + 1, len(endpoints)):
            if endpoints[first].distance(endpoints[second]) <= tolerance_m:
                groups.union(first, second)
    degrees: dict[int, int] = {}
    for index in range(len(endpoints)):
        root = groups.find(index)
        degrees[root] = degrees.get(root, 0) + 1
    return sum(degree > 2 for degree in degrees.values())


def compute_topology_metrics(
    features: Sequence[RowRecord | LineString],
    *,
    endpoint_snap_tolerance_m: float = 0.0,
) -> TopologyMetrics:
    """Count invalidities and pairwise violations in a collection of rows."""

    if not np.isfinite(endpoint_snap_tolerance_m) or endpoint_snap_tolerance_m < 0.0:
        raise ValueError("endpoint_snap_tolerance_m must be finite and non-negative")
    raw_geometries = [item.geometry if isinstance(item, RowRecord) else item for item in features]
    valid_lines: list[LineString] = []
    empty_count = 0
    invalid_count = 0
    non_simple_count = 0
    for geometry in raw_geometries:
        if not isinstance(geometry, LineString):
            invalid_count += 1
            continue
        if geometry.is_empty:
            empty_count += 1
            continue
        coordinates = np.asarray(geometry.coords, dtype=np.float64)
        finite = (
            coordinates.ndim == 2 and len(coordinates) >= 2 and np.all(np.isfinite(coordinates))
        )
        positive_length = np.isfinite(geometry.length) and geometry.length > 0.0
        if not geometry.is_valid or not finite or not positive_length:
            invalid_count += 1
            continue
        valid_lines.append(geometry)
        if not geometry.is_simple:
            non_simple_count += 1

    sets = _DisjointSet(len(valid_lines))
    crossing_pairs = 0
    overlapping_pairs = 0
    duplicate_pairs = 0
    for first in range(len(valid_lines)):
        for second in range(first + 1, len(valid_lines)):
            first_line = valid_lines[first]
            second_line = valid_lines[second]
            if first_line.equals(second_line):
                duplicate_pairs += 1
            elif first_line.crosses(second_line):
                crossing_pairs += 1
            elif first_line.overlaps(second_line):
                overlapping_pairs += 1
            if first_line.intersects(second_line):
                sets.union(first, second)
    connected_components = len({sets.find(index) for index in range(len(valid_lines))})
    return TopologyMetrics(
        feature_count=len(raw_geometries),
        valid_feature_count=len(valid_lines),
        empty_count=empty_count,
        invalid_count=invalid_count,
        non_simple_count=non_simple_count,
        crossing_pairs=crossing_pairs,
        overlapping_pairs=overlapping_pairs,
        duplicate_pairs=duplicate_pairs,
        connected_components=connected_components,
        endpoint_nodes_degree_gt2=_endpoint_degree_violations(
            valid_lines,
            endpoint_snap_tolerance_m,
        ),
    )


geometry_metrics = compute_geometry_metrics
topology_metrics = compute_topology_metrics
