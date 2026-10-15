"""Merge accepted segments, smooth conservatively, and create stable row records."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.interpolate import splev, splprep
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, snap, substring, unary_union

from rowgraphnet.domain import (
    CandidateConnection,
    ExtractedSegment,
    RepairPath,
    RowRecord,
    SpacingEstimate,
)
from rowgraphnet.geometry.global_optimizer import OptimizationResult


@dataclass(slots=True)
class RefinementResult:
    rows: list[RowRecord]
    repairs: list[RepairPath]
    uncertain_rows: list[RowRecord]


def _canonical_geometry_bytes(geometry: LineString) -> bytes:
    forward = [(round(x, 6), round(y, 6)) for x, y in geometry.coords]
    reverse = list(reversed(forward))
    return repr(min(forward, reverse)).encode("utf-8")


def _stable_row_id(geometry: LineString) -> str:
    return "row_" + hashlib.sha256(_canonical_geometry_bytes(geometry)).hexdigest()[:16]


def _axial_azimuth_deg(geometry: LineString) -> float:
    coords = np.asarray(geometry.coords, dtype=float)
    differences = np.diff(coords, axis=0)
    lengths = np.linalg.norm(differences, axis=1)
    valid = lengths > np.finfo(float).eps
    if not np.any(valid):
        return 0.0
    # Azimuth is clockwise from north.  Doubling makes it axial in [0, 180).
    azimuth = np.arctan2(differences[valid, 0], differences[valid, 1])
    sin2 = np.average(np.sin(2.0 * azimuth), weights=lengths[valid])
    cos2 = np.average(np.cos(2.0 * azimuth), weights=lengths[valid])
    return float(np.degrees(0.5 * np.arctan2(sin2, cos2)) % 180.0)


def _smooth_safely(geometry: LineString, maximum_deviation_m: float | None) -> LineString:
    if maximum_deviation_m is None or maximum_deviation_m <= 0:
        return geometry
    coords = np.asarray(geometry.coords, dtype=float)
    if len(coords) < 4:
        return geometry
    edge_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    positive = edge_lengths[edge_lengths > np.finfo(float).eps]
    if not len(positive):
        return geometry
    target_step = float(np.median(positive))
    parameter = np.r_[0.0, np.cumsum(edge_lengths)]
    if parameter[-1] <= 0:
        return geometry
    parameter /= parameter[-1]
    # Tentativa 1: B-spline via splprep
    try:
        spline, _ = splprep(
            [coords[:, 0], coords[:, 1]],
            u=parameter,
            s=len(coords) * maximum_deviation_m**2,
            k=min(3, len(coords) - 1),
        )
        sample_count = max(math.ceil(geometry.length / target_step) + 1, len(coords))
        sample_u = np.linspace(0.0, 1.0, sample_count)
        smooth_x, smooth_y = splev(sample_u, spline)
        smoothed = LineString(np.column_stack((smooth_x, smooth_y)))
        if (
            smoothed.is_valid
            and not smoothed.is_empty
            and smoothed.is_simple
            and smoothed.hausdorff_distance(geometry) <= maximum_deviation_m
        ):
            final = smoothed.simplify(maximum_deviation_m * 0.25, preserve_topology=True)
            if (
                isinstance(final, LineString)
                and final.is_valid
                and not final.is_empty
                and final.is_simple
                and final.hausdorff_distance(geometry) <= maximum_deviation_m
            ):
                return final
            return smoothed
    except (TypeError, ValueError):
        pass

    # Tentativa 2: Simplificação Douglas-Peucker + suavização Chaikin (2 iterações)
    # Garante matematicamente desvio máximo <= tolerância e elimina o degrau de pixels.
    tol = maximum_deviation_m * 0.5
    simp = geometry.simplify(tol, preserve_topology=True)
    if isinstance(simp, LineString) and simp.is_valid and not simp.is_empty and simp.is_simple:
        s_coords = np.array(simp.coords)
        if len(s_coords) >= 3:
            # Duas iterações de Chaikin para curvas mais suaves
            ch = s_coords.copy()
            for _ in range(2):
                new_coords = [ch[0]]
                for i in range(len(ch) - 1):
                    p0 = ch[i]
                    p1 = ch[i + 1]
                    new_coords.extend([0.75 * p0 + 0.25 * p1, 0.25 * p0 + 0.75 * p1])
                new_coords.append(ch[-1])
                ch = np.array(new_coords)
            ch_line = LineString(ch)
            if (
                ch_line.is_valid
                and not ch_line.is_empty
                and ch_line.is_simple
                and ch_line.hausdorff_distance(geometry) <= maximum_deviation_m
            ):
                # Simplificação final para remover vértices colineares redundantes
                final = ch_line.simplify(maximum_deviation_m * 0.25, preserve_topology=True)
                if (
                    isinstance(final, LineString)
                    and final.is_valid
                    and not final.is_empty
                    and final.is_simple
                    and final.hausdorff_distance(geometry) <= maximum_deviation_m
                ):
                    return final
                return ch_line
        if simp.hausdorff_distance(geometry) <= maximum_deviation_m:
            return simp

    return geometry


def _longest_line(geometry: object) -> tuple[LineString | None, bool]:
    if isinstance(geometry, LineString):
        return geometry, False
    if isinstance(geometry, MultiLineString):
        return None, True
    return None, bool(getattr(geometry, "geoms", ()))


def _clip_line(geometry: LineString, roi: BaseGeometry | None) -> tuple[LineString | None, bool]:
    if roi is None:
        return geometry, False
    return _longest_line(geometry.intersection(roi))


def _ambiguous_candidate_row(
    candidate: CandidateConnection,
    model_version: str,
    spacing: SpacingEstimate | None,
    maximum_spline_deviation_m: float | None = None,
) -> RowRecord:
    evidence = dict(candidate.evidence)
    evidence["candidate_id"] = candidate.candidate_id
    evidence["status"] = "ambiguous_connection"
    local_spacing = evidence.get("spacing_local_m")
    smoothed_geometry = _smooth_safely(candidate.geometry, maximum_spline_deviation_m)
    return RowRecord(
        row_id="hyp_" + candidate.candidate_id.removeprefix("edge_"),
        geometry=smoothed_geometry,
        confidence=float(candidate.probability),
        observed_ratio=0.0,
        repaired_ratio=1.0,
        spacing_local_m=(
            spacing.median_m
            if spacing is not None
            else float(local_spacing)
            if isinstance(local_spacing, (int, float))
            else None
        ),
        orientation_deg=_axial_azimuth_deg(smoothed_geometry),
        topology_status="ambiguous",
        repair_reason=json.dumps(evidence, sort_keys=True, ensure_ascii=False),
        model_version=model_version,
    )


def _snap_anchors(geometry: LineString, anchors: list[tuple[float, float]]) -> LineString:
    if not anchors:
        return geometry
    if len(anchors) > 2:
        return geometry
    coords = list(geometry.coords)
    if len(anchors) == 1:
        anchor = anchors[0]
        start_distance = Point(coords[0]).distance(Point(anchor))
        end_distance = Point(coords[-1]).distance(Point(anchor))
        coords[0 if start_distance <= end_distance else -1] = anchor
    else:
        first, second = anchors
        direct = Point(coords[0]).distance(Point(first)) + Point(coords[-1]).distance(Point(second))
        reverse = Point(coords[0]).distance(Point(second)) + Point(coords[-1]).distance(
            Point(first)
        )
        coords[0], coords[-1] = (first, second) if direct <= reverse else (second, first)
    return LineString(coords)


def _project_repairs_onto_final_row(
    geometry: LineString,
    repairs: list[RepairPath],
    maximum_alignment_distance_m: float | None,
) -> tuple[LineString, bool]:
    """Make exported synthesized pieces exact subcurves of the final row."""

    numerical_tolerance = max(
        float(np.spacing(max(geometry.length, 1.0))) * 64.0,
        np.finfo(float).eps,
    )
    if maximum_alignment_distance_m is not None and (
        not math.isfinite(maximum_alignment_distance_m) or maximum_alignment_distance_m < 0.0
    ):
        raise ValueError("maximum repair alignment distance must be finite and non-negative")
    tolerance = max(maximum_alignment_distance_m or 0.0, numerical_tolerance)
    for repair in repairs:
        start = Point(repair.geometry.coords[0])
        end = Point(repair.geometry.coords[-1])
        if start.distance(geometry) > tolerance or end.distance(geometry) > tolerance:
            return geometry, False
        start_distance = float(geometry.project(start))
        end_distance = float(geometry.project(end))
        if abs(start_distance - end_distance) <= numerical_tolerance:
            return geometry, False
        projected = substring(
            geometry,
            min(start_distance, end_distance),
            max(start_distance, end_distance),
        )
        if not isinstance(projected, LineString) or projected.is_empty or not projected.is_simple:
            return geometry, False
        snapped = snap(geometry, projected, 1e-4)
        if isinstance(snapped, LineString) and snapped.is_valid and snapped.is_simple:
            geometry = snapped
        repair.geometry = projected
        repair.reason["geometry_aligned_to_final_row"] = True
    return geometry, True


def _mark_row_ambiguous(row: RowRecord, reason: str) -> None:
    row.topology_status = "ambiguous"
    try:
        details = json.loads(row.repair_reason)
    except (json.JSONDecodeError, TypeError):
        details = {"previous_reason": row.repair_reason}
    if isinstance(details, list):
        details.append({"validation_reason": reason})
    elif isinstance(details, dict):
        details["validation_reason"] = reason
    row.repair_reason = json.dumps(details, sort_keys=True, ensure_ascii=False)


def _apply_collective_topology_validation(
    rows: list[RowRecord],
    roi: BaseGeometry | None,
) -> None:
    for row in rows:
        if (
            row.geometry.is_empty
            or not row.geometry.is_valid
            or not row.geometry.is_simple
            or row.geometry.length <= 0.0
        ):
            _mark_row_ambiguous(row, "invalid_or_non_simple_final_geometry")
        elif roi is not None and not roi.covers(row.geometry):
            _mark_row_ambiguous(row, "final_geometry_outside_analysis_roi")
    for index, first in enumerate(rows):
        for second in rows[index + 1 :]:
            if first.geometry.equals(second.geometry):
                _mark_row_ambiguous(first, f"duplicate_of_{second.row_id}")
                _mark_row_ambiguous(second, f"duplicate_of_{first.row_id}")
            elif first.geometry.crosses(second.geometry):
                _mark_row_ambiguous(first, f"crosses_{second.row_id}")
                _mark_row_ambiguous(second, f"crosses_{first.row_id}")
            elif first.geometry.overlaps(second.geometry):
                _mark_row_ambiguous(first, f"overlaps_{second.row_id}")
                _mark_row_ambiguous(second, f"overlaps_{first.row_id}")
            elif first.geometry.touches(second.geometry):
                _mark_row_ambiguous(first, f"touches_{second.row_id}")
                _mark_row_ambiguous(second, f"touches_{first.row_id}")


def refine_rows(
    segments: list[ExtractedSegment],
    optimization: OptimizationResult,
    repair_paths: list[RepairPath],
    model_version: str,
    spacing: SpacingEstimate | None = None,
    roi: BaseGeometry | None = None,
    maximum_spline_deviation_m: float | None = None,
) -> RefinementResult:
    """Create one final LineString per connected, topologically valid row."""

    segment_lookup = {item.segment_id: item for item in segments}
    path_lookup = {item.candidate_id: item for item in repair_paths}
    graph = nx.Graph()
    graph.add_nodes_from(sorted(segment_lookup))
    boundary_extensions_by_segment: dict[str, list[RepairPath]] = {}
    boundary_anchors_by_segment: dict[str, list[tuple[float, float]]] = {}

    for candidate in optimization.accepted:
        if candidate.is_boundary_extension:
            if candidate.candidate_id in path_lookup:
                boundary_extensions_by_segment.setdefault(candidate.segment_a, []).append(
                    path_lookup[candidate.candidate_id]
                )
            if candidate.boundary_intersection_xy is not None:
                boundary_anchors_by_segment.setdefault(candidate.segment_a, []).append(
                    candidate.boundary_intersection_xy
                )
        else:
            graph.add_edge(
                candidate.segment_a, candidate.segment_b, candidate_id=candidate.candidate_id
            )

    rows: list[RowRecord] = []
    disconnected_rows: list[RowRecord] = []
    accepted_repairs: list[RepairPath] = []
    for component in sorted(nx.connected_components(graph), key=lambda value: sorted(value)):
        observed_segments = [segment_lookup[identifier] for identifier in sorted(component)]
        component_edges = [
            data["candidate_id"]
            for source, target, data in graph.edges(component, data=True)
            if source in component and target in component
        ]
        component_internal_repairs = [
            path_lookup[item] for item in component_edges if item in path_lookup
        ]
        component_boundary_repairs = [
            repair
            for seg_id in component
            for repair in boundary_extensions_by_segment.get(seg_id, [])
        ]
        component_anchors = [
            anchor for seg_id in component for anchor in boundary_anchors_by_segment.get(seg_id, [])
        ]
        component_repairs = component_internal_repairs + component_boundary_repairs
        lines = [item.geometry for item in observed_segments] + [
            item.geometry for item in component_repairs
        ]
        merged = linemerge(unary_union(lines)) if len(lines) > 1 else lines[0]
        geometry, disconnected = _longest_line(merged)
        if disconnected:
            reason = json.dumps(
                {
                    "status": "disconnected_component_after_merge",
                    "candidate_ids": sorted(component_edges),
                    "segment_ids": sorted(component),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            disconnected_rows.extend(
                RowRecord(
                    row_id=_stable_row_id(item.geometry),
                    geometry=_smooth_safely(item.geometry, maximum_spline_deviation_m),
                    confidence=item.confidence,
                    observed_ratio=1.0,
                    repaired_ratio=0.0,
                    spacing_local_m=(
                        float(item.attrs["spacing_local_m"])
                        if isinstance(item.attrs.get("spacing_local_m"), (int, float))
                        else None
                    ),
                    orientation_deg=_axial_azimuth_deg(item.geometry),
                    topology_status="ambiguous",
                    repair_reason=reason,
                    model_version=model_version,
                )
                for item in observed_segments
            )
            continue
        if geometry is None or geometry.is_empty:
            continue
        geometry = _smooth_safely(geometry, maximum_spline_deviation_m)
        geometry = _snap_anchors(geometry, component_anchors)
        geometry, clipped_multiple = _clip_line(geometry, roi)
        if clipped_multiple:
            reason = json.dumps(
                {
                    "status": "multiple_parts_after_roi_clip",
                    "candidate_ids": sorted(component_edges),
                    "segment_ids": sorted(component),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            disconnected_rows.extend(
                RowRecord(
                    row_id=_stable_row_id(item.geometry),
                    geometry=_smooth_safely(item.geometry, maximum_spline_deviation_m),
                    confidence=item.confidence,
                    observed_ratio=1.0,
                    repaired_ratio=0.0,
                    spacing_local_m=(
                        float(item.attrs["spacing_local_m"])
                        if isinstance(item.attrs.get("spacing_local_m"), (int, float))
                        else None
                    ),
                    orientation_deg=_axial_azimuth_deg(item.geometry),
                    topology_status="ambiguous",
                    repair_reason=reason,
                    model_version=model_version,
                )
                for item in observed_segments
            )
            continue
        if geometry is None or geometry.is_empty or geometry.length <= 0:
            continue

        geometry, repairs_match_final_row = _project_repairs_onto_final_row(
            geometry,
            component_repairs,
            maximum_spline_deviation_m,
        )

        observed_length = sum(item.geometry.length for item in observed_segments)
        repaired_length = sum(item.geometry.length for item in component_repairs)
        accounted_length = observed_length + repaired_length
        if accounted_length <= 0:
            continue
        observed_ratio = float(np.clip(observed_length / accounted_length, 0.0, 1.0))
        repaired_ratio = float(np.clip(repaired_length / accounted_length, 0.0, 1.0))
        confidence_numerator = sum(
            item.confidence * item.geometry.length for item in observed_segments
        ) + sum(
            ((item.mean_center_support + 1.0 - item.mean_uncertainty) / 2.0) * item.geometry.length
            for item in component_repairs
        )
        confidence = float(np.clip(confidence_numerator / accounted_length, 0.0, 1.0))
        local_spacing_values = [
            float(item.attrs["spacing_local_m"])
            for item in observed_segments
            if isinstance(item.attrs.get("spacing_local_m"), (int, float))
        ]
        spacing_local_m = (
            spacing.median_m
            if spacing is not None
            else float(np.median(local_spacing_values))
            if local_spacing_values
            else None
        )
        touches_junction = any(
            bool(item.attrs.get("touches_junction")) for item in observed_segments
        )
        unresolved_open_end = any(
            bool(item.attrs.get("unresolved_open_endpoint")) for item in observed_segments
        )
        status = (
            "ambiguous"
            if touches_junction or unresolved_open_end or not repairs_match_final_row
            else "valid"
        )
        row_id = _stable_row_id(geometry)
        for repair in component_repairs:
            repair.reason.update(
                {
                    "row_id": row_id,
                    "row_confidence": confidence,
                    "model_version": model_version,
                }
            )
        reasons = [item.reason for item in component_repairs]
        if not repairs_match_final_row:
            reasons.append({"validation_reason": "repair_not_projectable_to_final_row"})
        row = RowRecord(
            row_id=row_id,
            geometry=geometry,
            confidence=confidence,
            observed_ratio=observed_ratio,
            repaired_ratio=repaired_ratio,
            spacing_local_m=spacing_local_m,
            orientation_deg=_axial_azimuth_deg(geometry),
            topology_status=status,
            repair_reason=json.dumps(reasons, sort_keys=True, ensure_ascii=False),
            model_version=model_version,
        )
        rows.append(row)
        accepted_repairs.extend(component_repairs)

    _apply_collective_topology_validation(rows, roi)
    uncertain_rows = [
        _ambiguous_candidate_row(candidate, model_version, spacing, maximum_spline_deviation_m)
        for candidate in optimization.ambiguous
    ]
    uncertain_rows.extend(disconnected_rows)
    uncertain_rows.extend([item for item in rows if item.topology_status != "valid"])
    valid_rows = [item for item in rows if item.topology_status == "valid"]
    valid_rows.sort(key=lambda item: item.row_id)
    uncertain_rows.sort(key=lambda item: item.row_id)
    valid_row_ids = {item.row_id for item in valid_rows}
    unique_repairs = {
        item.candidate_id: item
        for item in accepted_repairs
        if item.reason.get("row_id") in valid_row_ids
    }
    return RefinementResult(
        rows=valid_rows,
        repairs=[unique_repairs[key] for key in sorted(unique_repairs)],
        uncertain_rows=uncertain_rows,
    )
