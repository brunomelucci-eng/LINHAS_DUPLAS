"""Straight tangent extension of eligible row endpoints to physical field boundaries.

This module implements Phase 2 of the RowGraphNet evolution plan:
- Robust terminal tangent estimation using orthogonal distance regression (PCA).
- Strict endpoint eligibility (unconnected, observed support, no true_end).
- Raycasting and intersection exclusively against the exterior boundary ring.
- Hard topological vetoes (crossings, neighbor corridor intrusion, leaving boundary).
- Unilateral candidate generation and auditable repair path provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from rowgraphnet.domain import (
    CandidateConnection,
    ExtractedSegment,
    RasterPrediction,
    RepairPath,
    SpacingEstimate,
)
from rowgraphnet.geometry.orientation_field import FusedOrientationField


@dataclass(slots=True, frozen=True)
class BoundaryExtensionConfig:
    """Configurable parameters for unilateral field boundary extensions."""

    enabled: bool = True
    max_boundary_distance_factor: float = 2.5
    max_absolute_distance_m: float | None = None
    min_terminal_vertices: int = 3
    min_observed_length_spacing_factor: float = 0.5
    terminal_length_spacing_factor: float = 1.5
    max_tangent_angular_dispersion_deg: float = 25.0
    max_tangent_eigenvalue_ratio: float = 0.35
    min_orientation_alignment: float = 0.40
    safe_corridor_fraction: float = 0.35
    true_endpoint_probability_threshold: float = 0.50
    minimum_probability: float = 0.35
    raster_evidence_weight: float = 0.0
    minimum_valid_fraction: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "max_boundary_distance_factor": self.max_boundary_distance_factor,
            "min_observed_length_spacing_factor": self.min_observed_length_spacing_factor,
            "terminal_length_spacing_factor": self.terminal_length_spacing_factor,
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError("boundary distance and terminal length factors must be positive")
        if self.max_absolute_distance_m is not None and (
            not math.isfinite(self.max_absolute_distance_m) or self.max_absolute_distance_m <= 0.0
        ):
            raise ValueError("max_absolute_distance_m must be positive or None")
        if (
            isinstance(self.min_terminal_vertices, bool)
            or not isinstance(self.min_terminal_vertices, int)
            or self.min_terminal_vertices < 2
        ):
            raise ValueError("min_terminal_vertices must be an integer of at least two")
        if not math.isfinite(self.max_tangent_angular_dispersion_deg) or not (
            0.0 <= self.max_tangent_angular_dispersion_deg <= 180.0
        ):
            raise ValueError("max tangent angular dispersion must lie in [0, 180]")
        unit_interval = {
            "max_tangent_eigenvalue_ratio": self.max_tangent_eigenvalue_ratio,
            "min_orientation_alignment": self.min_orientation_alignment,
            "safe_corridor_fraction": self.safe_corridor_fraction,
            "true_endpoint_probability_threshold": self.true_endpoint_probability_threshold,
            "minimum_probability": self.minimum_probability,
            "raster_evidence_weight": self.raster_evidence_weight,
            "minimum_valid_fraction": self.minimum_valid_fraction,
        }
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in unit_interval.values()
        ):
            raise ValueError("boundary extension probabilities and fractions must lie in [0, 1]")


@dataclass(slots=True)
class TerminalTangent:
    """Robust orientation estimate at a row segment extremity."""

    tangent_vector: tuple[float, float]  # unit vector (dx, dy) pointing outward from interior
    azimuth_deg: float  # direction in degrees [0, 360) clockwise from north
    dispersion_deg: float
    eigenvalue_ratio: float  # lambda_2 / lambda_1 (transverse scatter / longitudinal variance)
    stable: bool
    sampled_coords: list[tuple[float, float]] = field(default_factory=list)


def compute_robust_terminal_tangent(
    segment: ExtractedSegment,
    side: int,
    spacing_m: float,
    config: BoundaryExtensionConfig,
) -> TerminalTangent:
    """Estimate the outward terminal tangent using total least squares / PCA.

    Parameters
    ----------
    segment : ExtractedSegment
        Observed segment geometry and attributes.
    side : int
        0 for start point, 1 for end point.
    spacing_m : float
        Local row spacing in meters.
    config : BoundaryExtensionConfig
        Algorithm thresholds.
    """

    if side not in (0, 1):
        raise ValueError("endpoint side must be zero or one")
    coords = list(segment.geometry.coords)
    if len(coords) < 2:
        return TerminalTangent((0.0, 0.0), 0.0, 180.0, 1.0, False)

    # Order coordinates from interior towards the endpoint
    oriented_coords = list(reversed(coords)) if side == 0 else coords

    # Select terminal window adaptively based on spacing
    if not math.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive and finite")
    target_length = config.terminal_length_spacing_factor * spacing_m
    accumulated = 0.0
    selected: list[tuple[float, float]] = [oriented_coords[-1]]
    for index in range(len(oriented_coords) - 2, -1, -1):
        p1 = oriented_coords[index + 1]
        p0 = oriented_coords[index]
        edge_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        accumulated += edge_len
        selected.append(p0)
        if len(selected) >= config.min_terminal_vertices and accumulated >= target_length:
            break

    # selected is in order: [endpoint, p_prev1, p_prev2, ...]
    # Put in chronological order from interior towards endpoint
    selected.reverse()
    pts = np.asarray(selected, dtype=np.float64)

    if len(pts) < config.min_terminal_vertices:
        return TerminalTangent((0.0, 0.0), 0.0, 180.0, 1.0, False)

    # Compute mean and covariance
    center = np.mean(pts, axis=0)
    centered = pts - center
    cov = np.cov(centered, rowvar=False)

    if cov.shape != (2, 2) or np.any(np.isnan(cov)):
        # Fallback to direct chord
        dx = float(pts[-1, 0] - pts[0, 0])
        dy = float(pts[-1, 1] - pts[0, 1])
        norm = math.hypot(dx, dy)
        if norm <= 1e-12:
            return TerminalTangent((0.0, 0.0), 0.0, 180.0, 1.0, False)
        vec = (dx / norm, dy / norm)
        azimuth = float(math.degrees(math.atan2(vec[0], vec[1])) % 360.0)
        return TerminalTangent(vec, azimuth, 0.0, 0.0, True, selected)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns eigenvalues in ascending order: idx 1 is largest
    principal_vec = eigenvectors[:, 1]
    lambda_1 = float(max(eigenvalues[1], 1e-12))
    lambda_2 = float(max(eigenvalues[0], 0.0))
    eigenvalue_ratio = float(lambda_2 / lambda_1)

    # Direct chord from interior to tip
    chord_dx = float(pts[-1, 0] - pts[0, 0])
    chord_dy = float(pts[-1, 1] - pts[0, 1])
    chord_norm = math.hypot(chord_dx, chord_dy)
    if chord_norm > 1e-12:
        chord_vec = (chord_dx / chord_norm, chord_dy / chord_norm)
    else:
        chord_vec = (float(principal_vec[0]), float(principal_vec[1]))

    # Orient principal eigenvector towards the endpoint (same general direction as chord)
    dot = principal_vec[0] * chord_vec[0] + principal_vec[1] * chord_vec[1]
    if dot < 0:
        principal_vec = -principal_vec

    norm = math.hypot(principal_vec[0], principal_vec[1])
    if norm <= 1e-12:
        return TerminalTangent((0.0, 0.0), 0.0, 180.0, 1.0, False)
    u_vec = (float(principal_vec[0] / norm), float(principal_vec[1] / norm))

    # Measure angular dispersion between fitted tangent and individual segments
    angles_diff = []
    for i in range(len(pts) - 1):
        seg_dx = float(pts[i + 1, 0] - pts[i, 0])
        seg_dy = float(pts[i + 1, 1] - pts[i, 1])
        seg_norm = math.hypot(seg_dx, seg_dy)
        if seg_norm > 1e-12:
            seg_dot = np.clip(
                (seg_dx * u_vec[0] + seg_dy * u_vec[1]) / seg_norm,
                -1.0,
                1.0,
            )
            angles_diff.append(math.degrees(math.acos(seg_dot)))

    dispersion_deg = float(np.mean(angles_diff)) if angles_diff else 0.0
    azimuth_deg = float(math.degrees(math.atan2(u_vec[0], u_vec[1])) % 360.0)

    # Stability criteria: dispersion below threshold and eigenvalue ratio indicates clear 1D orientation
    stable = (
        dispersion_deg <= config.max_tangent_angular_dispersion_deg
        and eigenvalue_ratio <= config.max_tangent_eigenvalue_ratio
    )

    return TerminalTangent(
        tangent_vector=u_vec,
        azimuth_deg=azimuth_deg,
        dispersion_deg=dispersion_deg,
        eigenvalue_ratio=eigenvalue_ratio,
        stable=stable,
        sampled_coords=selected,
    )


def _field_polygons(field_boundary: BaseGeometry) -> list[Polygon]:
    """Return valid polygon components without silently accepting other types."""

    if isinstance(field_boundary, Polygon):
        polygons = [field_boundary]
    elif isinstance(field_boundary, MultiPolygon):
        polygons = list(field_boundary.geoms)
    else:
        return []
    return [polygon for polygon in polygons if not polygon.is_empty and polygon.is_valid]


def _polygon_for_endpoint(
    field_boundary: BaseGeometry,
    endpoint: tuple[float, float],
) -> Polygon | None:
    point = Point(endpoint)
    containing = [polygon for polygon in _field_polygons(field_boundary) if polygon.covers(point)]
    if len(containing) != 1:
        return None
    return containing[0]


def _coordinate_tolerance(*coordinates: tuple[float, float]) -> float:
    magnitude = max((abs(value) for pair in coordinates for value in pair), default=1.0)
    return max(float(np.spacing(max(magnitude, 1.0))) * 64.0, np.finfo(float).eps)


def find_boundary_intersection(
    endpoint: tuple[float, float],
    tangent: TerminalTangent,
    field_boundary: BaseGeometry,
    max_distance_m: float,
) -> tuple[tuple[float, float] | None, float | None, str | None]:
    """Cast outward ray from endpoint along tangent and find exterior ring intersection.

    Returns (intersection_xy, distance_m, rejection_reason).
    """

    polygon = _polygon_for_endpoint(field_boundary, endpoint)
    if polygon is None:
        return None, None, "endpoint_not_in_exactly_one_field_polygon"

    ux, uy = tangent.tangent_vector
    # Ray reaches beyond max_distance_m to detect intersections within threshold
    ray_reach = max_distance_m * 1.5
    ray = LineString([endpoint, (endpoint[0] + ray_reach * ux, endpoint[1] + ray_reach * uy)])

    best_intersection: tuple[float, float] | None = None
    min_distance = float("inf")

    inter = ray.intersection(polygon.exterior)
    if inter.is_empty:
        return None, None, "no_frontal_intersection_with_exterior"

    candidate_points: list[Point] = []
    if isinstance(inter, Point):
        candidate_points = [inter]
    elif hasattr(inter, "geoms"):
        candidate_points = [geom for geom in inter.geoms if isinstance(geom, Point)]
    else:
        return None, None, "ray_overlaps_field_boundary"

    tolerance = _coordinate_tolerance(endpoint, tuple(ray.coords[-1]))
    projected: list[tuple[float, tuple[float, float]]] = []
    for point in candidate_points:
        projection = (point.x - endpoint[0]) * ux + (point.y - endpoint[1]) * uy
        if projection > tolerance:
            projected.append((projection, (float(point.x), float(point.y))))
    if projected:
        projected.sort(key=lambda item: (item[0], item[1]))
        min_distance, best_intersection = projected[0]
        equally_near = [item for item in projected if abs(item[0] - min_distance) <= tolerance]
        unique_points = {
            (round(item[1][0] / tolerance), round(item[1][1] / tolerance)) for item in equally_near
        }
        if len(unique_points) > 1:
            return None, None, "ambiguous_first_exterior_intersection"

    if best_intersection is None:
        return None, None, "no_frontal_intersection_with_exterior"

    if min_distance > max_distance_m + tolerance:
        return None, None, f"intersection_exceeds_max_distance_{min_distance:.2f}m"

    return best_intersection, min_distance, None


def _metric_pixel_size(prediction: RasterPrediction) -> float:
    x_step = math.hypot(prediction.transform.a, prediction.transform.d)
    y_step = math.hypot(prediction.transform.b, prediction.transform.e)
    if min(x_step, y_step) <= 0.0:
        raise ValueError("prediction affine has a collapsed pixel basis")
    return float(math.sqrt(x_step * y_step))


def _sample_prediction_evidence(
    line: LineString,
    prediction: RasterPrediction,
) -> tuple[float, float, float, int]:
    """Sample only the interior of a proposed extension on the raster grid."""

    pixel_size = _metric_pixel_size(prediction)
    sample_count = max(math.ceil(line.length / pixel_size) + 1, 2)
    inverse = ~prediction.transform
    valid_count = 0
    supports: list[float] = []
    uncertainties: list[float] = []
    for fraction in np.linspace(0.0, 1.0, sample_count, endpoint=False):
        point = line.interpolate(float(fraction), normalized=True)
        col, row = inverse * (point.x, point.y)
        row_index = round(row - 0.5)
        col_index = round(col - 0.5)
        if not (
            0 <= row_index < prediction.center.shape[0]
            and 0 <= col_index < prediction.center.shape[1]
        ):
            continue
        index = (row_index, col_index)
        if prediction.valid_mask is not None and not bool(prediction.valid_mask[index]):
            continue
        center_value = float(prediction.center[index])
        uncertainty_value = float(prediction.uncertainty[index])
        if not math.isfinite(center_value) or not math.isfinite(uncertainty_value):
            continue
        valid_count += 1
        supports.append(float(np.clip(center_value, 0.0, 1.0)))
        uncertainties.append(
            uncertainty_value / (1.0 + uncertainty_value) if uncertainty_value >= 0.0 else 1.0
        )
    valid_fraction = valid_count / sample_count
    mean_support = float(np.mean(supports)) if supports else 0.0
    mean_uncertainty = float(np.mean(uncertainties)) if uncertainties else 1.0
    return valid_fraction, mean_support, mean_uncertainty, sample_count


def check_boundary_extension_vetoes(
    candidate_line: LineString,
    segment_id: str,
    segments: list[ExtractedSegment],
    spacing_m: float,
    field_boundary: BaseGeometry,
    config: BoundaryExtensionConfig,
    pixel_size_m: float | None = None,
    spatial_tree: Any | None = None,
    segment_map: dict[str, ExtractedSegment] | None = None,
) -> tuple[bool, str | None]:
    """Verify all hard veto rules for a boundary extension.

    Returns (passed, failure_reason).
    """

    if not candidate_line.is_valid or not candidate_line.is_simple:
        return False, "geometry_invalid_or_not_simple"

    if not math.isfinite(spacing_m) or spacing_m <= 0.0:
        return False, "invalid_local_spacing"
    tolerance = (
        pixel_size_m * np.sqrt(np.finfo(float).eps)
        if pixel_size_m is not None and pixel_size_m > 0.0
        else _coordinate_tolerance(*tuple(candidate_line.coords))
    )
    # 1. Must remain covered by the same field polygon (tolerance is numerical,
    # not a physical buffer that could bridge a hole or leave a concave field).
    source_polygon = _polygon_for_endpoint(field_boundary, tuple(candidate_line.coords[0]))
    if source_polygon is None:
        return False, "extension_source_not_in_exactly_one_field_polygon"
    buffered_boundary = source_polygon.buffer(tolerance)
    if not buffered_boundary.covers(candidate_line):
        return False, "extension_leaves_field_boundary"

    # 2. Cannot cross or overlap any observed segment
    safe_corridor = spacing_m * config.safe_corridor_fraction

    # Check for self-crossing against the rest of the own segment
    own_segment = (
        segment_map.get(segment_id)
        if segment_map is not None
        else next((s for s in segments if s.segment_id == segment_id), None)
    )
    if own_segment is not None:
        coords = list(own_segment.geometry.coords)
        if len(coords) >= 3:
            # If candidate touches coords[0], check against coords[1:]
            # If candidate touches coords[-1], check against coords[:-1]
            p_start = candidate_line.coords[0]
            d0 = math.hypot(p_start[0] - coords[0][0], p_start[1] - coords[0][1])
            rest = LineString(coords[1:]) if d0 <= tolerance else LineString(coords[:-1])
            if candidate_line.intersects(rest):
                return False, "extension_intersects_own_segment"

    if spatial_tree is not None:
        hit_indices = spatial_tree.query(candidate_line.buffer(safe_corridor))
        other_candidates = [segments[int(idx)] for idx in hit_indices]
    else:
        other_candidates = segments

    for other in other_candidates:
        if other.segment_id == segment_id:
            continue

        if candidate_line.crosses(other.geometry):
            return False, f"extension_crosses_segment_{other.segment_id}"

        if candidate_line.overlaps(other.geometry):
            return False, f"extension_overlaps_segment_{other.segment_id}"

        # 3. Cannot intrude into neighbor row corridor
        dist = candidate_line.distance(other.geometry)
        if dist < safe_corridor:
            return False, f"extension_violates_safe_corridor_of_{other.segment_id}_{dist:.2f}m"

    return True, None


def generate_boundary_extension_candidates(
    segments: list[ExtractedSegment],
    prediction: RasterPrediction,
    spacing: SpacingEstimate,
    field_boundary: BaseGeometry | None,
    config: BoundaryExtensionConfig,
    world_orientation: FusedOrientationField | None = None,
) -> list[CandidateConnection]:
    """Generate unilateral boundary extension candidates for eligible row endpoints."""

    if not config.enabled or field_boundary is None:
        return []
    if (
        field_boundary.is_empty
        or not field_boundary.is_valid
        or not isinstance(field_boundary, (Polygon, MultiPolygon))
    ):
        raise ValueError("field_boundary must be a valid, non-empty Polygon or MultiPolygon")
    if not math.isfinite(spacing.median_m) or spacing.median_m <= 0.0:
        raise ValueError("boundary extension requires a positive local spacing estimate")
    if world_orientation is not None and (
        world_orientation.sin2.shape != prediction.center.shape
        or world_orientation.cos2.shape != prediction.center.shape
        or world_orientation.confidence.shape != prediction.center.shape
    ):
        raise ValueError("world_orientation must match the prediction grid")

    local_spacing = spacing.median_m
    max_reach = config.max_boundary_distance_factor * local_spacing
    if config.max_absolute_distance_m is not None:
        max_reach = min(max_reach, config.max_absolute_distance_m)
    pixel_size = _metric_pixel_size(prediction)

    candidates: list[CandidateConnection] = []
    spatial_tree = STRtree([s.geometry for s in segments]) if segments else None
    segment_map = {s.segment_id: s for s in segments}

    for seg in segments:
        # Minimum observed length required for stable endpoint direction
        if (
            seg.observed_length_m
            < config.min_observed_length_spacing_factor * local_spacing
        ):
            continue

        endpoints = seg.endpoints  # Point(side 0), Point(side 1)
        for side in (0, 1):
            pt = endpoints[side]
            pt_xy = (float(pt.x), float(pt.y))

            # Check true endpoint veto
            end_conf = seg.endpoint_confidence[side]
            if not math.isfinite(end_conf):
                continue
            if end_conf >= config.true_endpoint_probability_threshold:
                continue

            # Check distance to boundary exterior before expensive computations
            polygon = _polygon_for_endpoint(field_boundary, pt_xy)
            if polygon is None:
                continue
            min_dist_to_boundary = polygon.exterior.distance(pt)
            if min_dist_to_boundary > max_reach:
                continue

            # Compute robust outward tangent
            tangent = compute_robust_terminal_tangent(seg, side, local_spacing, config)
            if not tangent.stable:
                continue

            # Find exterior intersection
            intersection_xy, dist_m, _fail_reason = find_boundary_intersection(
                pt_xy, tangent, field_boundary, max_reach
            )
            if intersection_xy is None or dist_m is None:
                continue

            candidate_line = LineString([pt_xy, intersection_xy])

            # Evaluate hard vetoes
            passed, _veto_reason = check_boundary_extension_vetoes(
                candidate_line,
                seg.segment_id,
                segments,
                local_spacing,
                field_boundary,
                config,
                pixel_size_m=pixel_size,
                spatial_tree=spatial_tree,
                segment_map=segment_map,
            )
            if not passed:
                continue

            valid_fraction, visual_support, mean_uncertainty, sample_count = (
                _sample_prediction_evidence(candidate_line, prediction)
            )
            if valid_fraction < config.minimum_valid_fraction:
                continue

            # Compute orientation alignment with local raster field in world CRS
            alignment = 1.0
            raw_alignment = 1.0
            orientation_confidence = 0.0
            if world_orientation is not None:
                # Sample local orientation near endpoint
                # Convert world XY to pixel column/row
                inv_transform = ~prediction.transform
                col, row = inv_transform * pt_xy
                r, c = round(row - 0.5), round(col - 0.5)
                if not (
                    0 <= r < world_orientation.sin2.shape[0]
                    and 0 <= c < world_orientation.sin2.shape[1]
                ):
                    continue
                theta_world = 0.5 * math.atan2(
                    float(world_orientation.sin2[r, c]),
                    float(world_orientation.cos2[r, c]),
                )
                orientation_confidence = float(world_orientation.confidence[r, c])
                if not math.isfinite(theta_world) or not math.isfinite(orientation_confidence):
                    continue
                orientation_confidence = float(np.clip(orientation_confidence, 0.0, 1.0))
                tan_theta = math.atan2(tangent.tangent_vector[1], tangent.tangent_vector[0])
                raw_alignment = float((1.0 + math.cos(2.0 * (tan_theta - theta_world))) / 2.0)
                alignment = orientation_confidence * raw_alignment + 1.0 - orientation_confidence
                if not math.isfinite(alignment) or alignment < config.min_orientation_alignment:
                    continue

            # Score probabilities
            # 1. Tangent stability factor
            scatter_scale = max(config.max_tangent_eigenvalue_ratio, np.finfo(float).eps)
            p_tangent = math.exp(-tangent.eigenvalue_ratio / scatter_scale)
            # 2. Distance factor: shorter extensions are more reliable over reach window
            p_dist = math.exp(-dist_m / max_reach)
            # 3. Endpoint factor
            p_end = 1.0 - float(end_conf)

            raster_quality = float(
                np.clip((visual_support + 1.0 - mean_uncertainty) / 2.0, 0.0, 1.0)
            )
            probability = float(
                np.clip(
                    p_tangent
                    * p_dist
                    * math.sqrt(alignment)
                    * p_end
                    * raster_quality**config.raster_evidence_weight,
                    1e-6,
                    0.999,
                )
            )

            if probability < config.minimum_probability:
                continue

            cost = float(-math.log(probability))

            cand_id = f"ext_{seg.segment_id}_{side}"
            evidence: dict[str, float | str | bool] = {
                "repair_type": "boundary_extension",
                "distance_m": round(dist_m, 3),
                "tangent_azimuth_deg": round(tangent.azimuth_deg, 2),
                "dispersion_deg": round(tangent.dispersion_deg, 2),
                "eigenvalue_ratio": round(tangent.eigenvalue_ratio, 4),
                "orientation_alignment": round(alignment, 3),
                "raw_orientation_alignment": round(raw_alignment, 3),
                "orientation_confidence": round(orientation_confidence, 3),
                "valid_raster_fraction": round(valid_fraction, 4),
                "visual_support": round(visual_support, 4),
                "mean_uncertainty": round(mean_uncertainty, 4),
                "raster_sample_count": sample_count,
                "confidence": round(probability, 4),
                "endpoint_confidence": round(end_conf, 3),
                "intersection_x": round(intersection_xy[0], 3),
                "intersection_y": round(intersection_xy[1], 3),
            }

            candidates.append(
                CandidateConnection(
                    candidate_id=cand_id,
                    segment_a=seg.segment_id,
                    side_a=side,
                    segment_b="",
                    side_b=-1,
                    geometry=candidate_line,
                    probability=probability,
                    cost=cost,
                    evidence=evidence,
                    ambiguous=False,
                    is_boundary_extension=True,
                    boundary_intersection_xy=intersection_xy,
                )
            )

    return candidates


def create_boundary_repair_path(
    candidate: CandidateConnection,
    prediction: RasterPrediction,
) -> RepairPath:
    """Construct a RepairPath representation for an accepted boundary extension."""

    valid_fraction, mean_support, mean_unc, sample_count = _sample_prediction_evidence(
        candidate.geometry,
        prediction,
    )
    path_quality = float(np.clip((mean_support + 1.0 - mean_unc) / 2.0, 0.0, 1.0))

    reason = dict(candidate.evidence)
    reason["algorithm"] = "tangent_boundary_raycast"
    reason["confidence"] = candidate.probability
    reason["edge_confidence"] = candidate.probability
    reason["path_quality"] = path_quality
    reason["valid_raster_fraction"] = valid_fraction
    reason["raster_sample_count"] = sample_count

    return RepairPath(
        candidate_id=candidate.candidate_id,
        geometry=candidate.geometry,
        total_cost=candidate.cost,
        mean_center_support=mean_support,
        mean_uncertainty=mean_unc,
        reason=reason,
        repair_type="boundary_extension",
    )
