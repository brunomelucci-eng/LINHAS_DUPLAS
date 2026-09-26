"""Constrained refinement and reference-guided repair for double crop rows.

This module is designed to run after vectorisation/deduplication and before the
final topology validation.  It addresses three recurring artefacts:

* local kinks created by noisy skeletons;
* crooked terminal sections close to the ROI boundary;
* broken/tortuous rows that can be repaired from a valid sister or neighbour.

The geometry of a valid reference is never copied wholesale.  Only the exact
interval corresponding to a target gap or bad local section is sampled,
offset to the target row, blended to the target endpoints, snapped to the
centre-probability ridge, and validated.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import atan2, degrees
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
from affine import Affine
from scipy.signal import savgol_filter
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.ops import substring, unary_union
from shapely.strtree import STRtree

from src.logging_utils import StageTimer
from .lineage import combine_record_source_raw_ids, parse_source_raw_ids
from .regularized_centerline_fairing import classify_refinement_source, is_true

logger = logging.getLogger(__name__)

_EPS = 1e-9


@dataclass(frozen=True)
class LineQuality:
    length_m: float
    chord_ratio: float
    max_turn_deg: float
    p95_turn_deg: float
    kink_fraction: float
    probability_median: float
    probability_p20: float
    quality_class: str


@dataclass(frozen=True)
class Track:
    track_id: int
    indices: Tuple[int, ...]
    bearing_deg: float
    total_length_m: float
    quality_class: str


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _cfg(config: dict) -> dict:
    return config.get("double_row_refinement", {})


def _is_true(value: Any) -> bool:
    """Compatibility helper with NaN-safe explicit truth semantics."""
    return is_true(value)


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _bearing(line: LineString) -> float:
    coords = np.asarray(line.coords, dtype=np.float64)
    if len(coords) < 2:
        return 0.0
    centered = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    v = vt[0]
    return float(degrees(atan2(v[1], v[0])) % 180.0)


def _resample(line: LineString, spacing_m: float) -> Tuple[np.ndarray, np.ndarray]:
    if line.is_empty or line.length <= _EPS:
        coords = np.asarray(line.coords, dtype=np.float64)
        return np.zeros(len(coords), dtype=np.float64), coords
    spacing_m = max(float(spacing_m), 0.02)
    count = max(2, int(np.ceil(line.length / spacing_m)) + 1)
    distances = np.linspace(0.0, line.length, count)
    vertices = np.asarray(line.coords, dtype=np.float64)[:, :2]
    segment_vectors = np.diff(vertices, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    indices = np.searchsorted(cumulative, distances, side="right") - 1
    indices = np.clip(indices, 0, len(segment_lengths) - 1)
    local = distances - cumulative[indices]
    fractions = np.divide(
        local,
        segment_lengths[indices],
        out=np.zeros_like(local),
        where=segment_lengths[indices] > _EPS,
    )
    sampled = vertices[indices] + fractions[:, None] * segment_vectors[indices]
    sampled[0] = vertices[0]
    sampled[-1] = vertices[-1]
    return distances, sampled


def _turn_angles(coords: np.ndarray, step: int = 1) -> np.ndarray:
    if len(coords) < 2 * step + 1:
        return np.zeros(len(coords), dtype=np.float64)
    turns = np.zeros(len(coords), dtype=np.float64)
    for i in range(step, len(coords) - step):
        a = coords[i] - coords[i - step]
        b = coords[i + step] - coords[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na <= _EPS or nb <= _EPS:
            continue
        cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        turns[i] = degrees(np.arccos(cosine))
    return turns


def _sample_probability_xy(
    points: np.ndarray,
    probability_raster: Optional[np.ndarray],
    transform: Optional[Affine],
) -> np.ndarray:
    if probability_raster is None or transform is None or len(points) == 0:
        return np.full(len(points), np.nan, dtype=np.float64)
    inv = ~transform
    h, w = probability_raster.shape
    values = np.zeros(len(points), dtype=np.float64)
    xy = np.asarray(points, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    col_f = inv.a * x + inv.b * y + inv.c
    row_f = inv.d * x + inv.e * y + inv.f
    valid = (
        (row_f >= 0.0)
        & (row_f < h - 1)
        & (col_f >= 0.0)
        & (col_f < w - 1)
    )
    if not np.any(valid):
        return values
    valid_indices = np.flatnonzero(valid)
    valid_rows = row_f[valid]
    valid_cols = col_f[valid]
    r0 = np.floor(valid_rows).astype(np.int64)
    c0 = np.floor(valid_cols).astype(np.int64)
    dr = valid_rows - r0
    dc = valid_cols - c0
    v00 = np.asarray(probability_raster[r0, c0], dtype=np.float64)
    v01 = np.asarray(probability_raster[r0, c0 + 1], dtype=np.float64)
    v10 = np.asarray(probability_raster[r0 + 1, c0], dtype=np.float64)
    v11 = np.asarray(probability_raster[r0 + 1, c0 + 1], dtype=np.float64)
    values[valid_indices] = (
        v00 * (1 - dr) * (1 - dc)
        + v01 * (1 - dr) * dc
        + v10 * dr * (1 - dc)
        + v11 * dr * dc
    )
    return values


def evaluate_line_quality(
    line: LineString,
    config: dict,
    probability_raster: Optional[np.ndarray] = None,
    transform: Optional[Affine] = None,
) -> LineQuality:
    """Classify a line using local angular stability and real probability."""
    cfg = _cfg(config)
    qcfg = cfg.get("quality", {})
    spacing = float(qcfg.get("sample_spacing_m", 0.25))
    _, coords = _resample(line, spacing)
    step = max(1, int(round(float(qcfg.get("turn_window_m", 0.75)) / max(spacing, 0.02))))
    turns = _turn_angles(coords, step=step)
    internal = turns[step:-step] if len(turns) > 2 * step else turns
    max_turn = float(np.max(internal)) if len(internal) else 0.0
    p95_turn = float(np.percentile(internal, 95)) if len(internal) else 0.0
    kink_threshold = float(qcfg.get("kink_threshold_deg", 12.0))
    kink_fraction = float(np.mean(internal > kink_threshold)) if len(internal) else 0.0

    chord = Point(line.coords[0]).distance(Point(line.coords[-1])) if len(line.coords) >= 2 else 0.0
    chord_ratio = float(line.length / max(chord, _EPS))
    probs = _sample_probability_xy(coords, probability_raster, transform)
    finite = probs[np.isfinite(probs)]
    prob_median = float(np.median(finite)) if len(finite) else float("nan")
    prob_p20 = float(np.percentile(finite, 20)) if len(finite) else float("nan")

    min_good_length = float(qcfg.get("min_good_length_m", 4.0))
    good_max_turn = float(qcfg.get("good_max_turn_deg", 10.0))
    good_p95_turn = float(qcfg.get("good_p95_turn_deg", 6.0))
    max_good_kink_fraction = float(qcfg.get("max_good_kink_fraction", 0.02))
    min_good_prob = float(qcfg.get("min_good_probability_median", 0.20))
    probability_ok = np.isnan(prob_median) or prob_median >= min_good_prob

    if (
        line.length >= min_good_length
        and line.is_valid
        and line.is_simple
        and max_turn <= good_max_turn
        and p95_turn <= good_p95_turn
        and kink_fraction <= max_good_kink_fraction
        and probability_ok
    ):
        quality_class = "GOOD"
    elif line.is_valid and line.length >= float(qcfg.get("min_repairable_length_m", 1.0)):
        quality_class = "SUSPECT"
    else:
        quality_class = "INVALID"

    return LineQuality(
        length_m=float(line.length),
        chord_ratio=chord_ratio,
        max_turn_deg=max_turn,
        p95_turn_deg=p95_turn,
        kink_fraction=kink_fraction,
        probability_median=prob_median,
        probability_p20=prob_p20,
        quality_class=quality_class,
    )


def _safe_savgol(values: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    if len(values) < 5:
        return values.copy()
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    window = max(5, window)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")


def _clip_displacement(original: np.ndarray, candidate: np.ndarray, max_deviation_m: float) -> np.ndarray:
    delta = candidate - original
    norms = np.linalg.norm(delta, axis=1)
    scale = np.ones_like(norms)
    mask = norms > max_deviation_m
    scale[mask] = max_deviation_m / np.maximum(norms[mask], _EPS)
    return original + delta * scale[:, None]


def _snap_to_probability_ridge(
    original: np.ndarray,
    candidate: np.ndarray,
    config: dict,
    probability_raster: Optional[np.ndarray],
    transform: Optional[Affine],
) -> np.ndarray:
    cfg = _cfg(config).get("center_mass_snap", {})
    if not cfg.get("enabled", True) or probability_raster is None or transform is None or len(candidate) < 3:
        return candidate
    radius = float(cfg.get("search_radius_m", 0.12))
    step_m = float(cfg.get("step_m", 0.02))
    min_probability = float(cfg.get("min_probability", 0.12))
    offsets = np.arange(-radius, radius + step_m * 0.5, step_m, dtype=np.float64)
    chosen_offsets = np.zeros(len(candidate), dtype=np.float64)

    tangents = candidate[2:] - candidate[:-2]
    tangent_norms = np.linalg.norm(tangents, axis=1)
    valid_tangents = tangent_norms > _EPS
    unit_tangents = np.divide(
        tangents,
        tangent_norms[:, None],
        out=np.zeros_like(tangents),
        where=valid_tangents[:, None],
    )
    normals = np.column_stack((-unit_tangents[:, 1], unit_tangents[:, 0]))
    probes = candidate[1:-1, None, :] + offsets[None, :, None] * normals[:, None, :]
    probs = _sample_probability_xy(
        probes.reshape(-1, 2), probability_raster, transform
    ).reshape(len(candidate) - 2, len(offsets))
    best_indices = np.nanargmax(probs, axis=1)
    row_indices = np.arange(len(best_indices))
    best_probabilities = probs[row_indices, best_indices]
    accepted = valid_tangents & (best_probabilities >= min_probability)
    interior_offsets = np.zeros(len(candidate) - 2, dtype=np.float64)
    interior_offsets[accepted] = offsets[best_indices[accepted]]
    chosen_offsets[1:-1] = interior_offsets

    smooth_window_m = float(cfg.get("offset_smoothing_window_m", 0.80))
    spacing_est = float(np.median(np.linalg.norm(np.diff(candidate, axis=0), axis=1)))
    window = max(5, int(round(smooth_window_m / max(spacing_est, 0.02))))
    if window % 2 == 0:
        window += 1
    chosen_offsets = _safe_savgol(chosen_offsets, window, polyorder=2)

    snapped = candidate.copy()
    snapped[1:-1] += chosen_offsets[1:-1, None] * normals

    max_total = float(cfg.get("max_total_shift_m", 0.12))
    return _clip_displacement(original, snapped, max_total)


def _smooth_line_center_constrained_cached(
    line: LineString,
    config: dict,
    probability_raster: Optional[np.ndarray] = None,
    transform: Optional[Affine] = None,
    quality: Optional[LineQuality] = None,
) -> Tuple[LineString, Dict[str, Any], Optional[LineQuality]]:
    """Smooth local corners without moving the row away from the cane mass."""
    cfg = _cfg(config).get("smoothing", {})
    if not cfg.get("enabled", True) or line.length <= _EPS:
        return line, {"smoothed": False, "smoothing_reason": "disabled"}, quality

    if quality is None:
        quality = evaluate_line_quality(line, config, probability_raster, transform)
    # Preserve lines that already pass the local angular and probability gates.
    # This is both safer and much faster on large talhões.
    if quality.quality_class == "GOOD":
        return line, {"smoothed": False, "smoothing_reason": "good_preserved"}, quality

    spacing = float(cfg.get("resample_spacing_m", 0.20))
    _, original = _resample(line, spacing)
    if len(original) < 5:
        return line, {"smoothed": False, "smoothing_reason": "too_short"}, quality

    window_m = float(cfg.get("window_m", 1.40))
    window = max(5, int(round(window_m / max(spacing, 0.02))))
    if window % 2 == 0:
        window += 1
    x = _safe_savgol(original[:, 0], window, polyorder=2)
    y = _safe_savgol(original[:, 1], window, polyorder=2)
    candidate = np.column_stack([x, y])
    candidate[0] = original[0]
    candidate[-1] = original[-1]

    max_deviation = float(cfg.get("max_deviation_m", 0.10))
    candidate = _clip_displacement(original, candidate, max_deviation)
    candidate = _snap_to_probability_ridge(original, candidate, config, probability_raster, transform)
    candidate[0] = original[0]
    candidate[-1] = original[-1]

    result = LineString(candidate)
    hausdorff = float(line.hausdorff_distance(result))
    unsafe = (
        result.is_empty
        or not result.is_valid
        or not result.is_simple
        or hausdorff > float(cfg.get("max_hausdorff_deviation_m", 0.12))
    )
    if unsafe:
        return line, {
            "smoothed": False,
            "smoothing_reason": "safety_reject",
            "smoothing_hausdorff_m": hausdorff,
        }, quality
    after = evaluate_line_quality(result, config, probability_raster, transform)
    if after.p95_turn_deg > quality.p95_turn_deg + float(cfg.get("allowed_p95_turn_regression_deg", 0.5)):
        return line, {"smoothed": False, "smoothing_reason": "angular_regression"}, quality
    return result, {
        "smoothed": True,
        "smoothing_reason": "accepted",
        "smoothing_hausdorff_m": hausdorff,
        "p95_turn_before_deg": quality.p95_turn_deg,
        "p95_turn_after_deg": after.p95_turn_deg,
    }, after


def smooth_line_center_constrained(
    line: LineString,
    config: dict,
    probability_raster: Optional[np.ndarray] = None,
    transform: Optional[Affine] = None,
) -> Tuple[LineString, Dict[str, Any]]:
    """Public compatibility wrapper for constrained smoothing."""
    result, metadata, _ = _smooth_line_center_constrained_cached(
        line,
        config,
        probability_raster,
        transform,
    )
    return result, metadata


def _endpoint_distance_to_boundary(line: LineString, boundary: LineString, at_start: bool) -> float:
    point = Point(line.coords[0] if at_start else line.coords[-1])
    return float(point.distance(boundary))


def _fit_terminal_direction(line: LineString, at_start: bool, replace_m: float, fit_window_m: float) -> Tuple[np.ndarray, np.ndarray]:
    length = float(line.length)
    if at_start:
        anchor_d = min(replace_m, length)
        fit_start = anchor_d
        fit_end = min(length, anchor_d + fit_window_m)
        anchor = np.asarray(line.interpolate(anchor_d).coords[0], dtype=np.float64)
        inner = np.asarray(line.interpolate(fit_end).coords[0], dtype=np.float64)
        direction = anchor - inner
    else:
        anchor_d = max(0.0, length - replace_m)
        fit_start = max(0.0, anchor_d - fit_window_m)
        fit_end = anchor_d
        anchor = np.asarray(line.interpolate(anchor_d).coords[0], dtype=np.float64)
        inner = np.asarray(line.interpolate(fit_start).coords[0], dtype=np.float64)
        direction = anchor - inner

    segment = substring(line, fit_start, fit_end)
    coords = np.asarray(segment.coords, dtype=np.float64) if segment.geom_type == "LineString" else np.vstack([anchor, inner])
    if len(coords) >= 2:
        centered = coords - coords.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pca = vt[0]
        if np.dot(pca, direction) < 0:
            pca = -pca
        direction = pca
    norm = np.linalg.norm(direction)
    if norm <= _EPS:
        return anchor, np.zeros(2, dtype=np.float64)
    return anchor, direction / norm


def _nearest_boundary_intersection(anchor: np.ndarray, direction: np.ndarray, boundary: LineString, max_length: float) -> Optional[Point]:
    ray = LineString([anchor, anchor + direction * max_length])
    intersection = ray.intersection(boundary)
    points: List[Point] = []
    if intersection.is_empty:
        return None
    if isinstance(intersection, Point):
        points = [intersection]
    elif isinstance(intersection, MultiPoint):
        points = list(intersection.geoms)
    elif isinstance(intersection, GeometryCollection):
        points = [g for g in intersection.geoms if isinstance(g, Point)]
    elif hasattr(intersection, "geoms"):
        points = [g for g in intersection.geoms if isinstance(g, Point)]
    if not points:
        return None
    anchor_pt = Point(anchor)
    positive = [p for p in points if p.distance(anchor_pt) > 1e-6]
    return min(positive, key=lambda p: p.distance(anchor_pt)) if positive else None


def _segment_probability_support(
    segment: LineString,
    probability_raster: Optional[np.ndarray],
    transform: Optional[Affine],
    threshold: float,
    spacing_m: float = 0.05,
) -> Tuple[float, float]:
    _, coords = _resample(segment, spacing_m)
    probs = _sample_probability_xy(coords, probability_raster, transform)
    finite = probs[np.isfinite(probs)]
    if len(finite) == 0:
        return 0.0, 0.0
    return float(np.median(finite)), float(np.mean(finite >= threshold))


def straighten_and_extend_terminals(
    line: LineString,
    roi_polygon: Polygon,
    config: dict,
    probability_raster: Optional[np.ndarray] = None,
    transform: Optional[Affine] = None,
    sister_support: bool = False,
) -> Tuple[LineString, Dict[str, Any]]:
    """Replace the last/first two metres by a stable straight segment to the ROI."""
    cfg = _cfg(config).get("terminal_extension", {})
    if not cfg.get("enabled", True) or line.length <= _EPS:
        return line, {"terminal_straightened": False, "terminal_extended_m": 0.0}

    boundary = roi_polygon.boundary
    trigger = float(cfg.get("trigger_distance_to_boundary_m", 2.0))
    replace_m = float(cfg.get("replace_terminal_m", 2.0))
    fit_window_m = float(cfg.get("direction_fit_window_m", 3.0))
    max_extension_m = float(cfg.get("max_extension_m", 2.5))
    min_prob = float(cfg.get("min_probability", 0.15))
    min_fraction = float(cfg.get("min_supported_fraction", 0.45))
    require_support = bool(cfg.get("require_probability_or_sister_support", True))

    result = line
    changed = False
    total_extended = 0.0
    for at_start in (True, False):
        if _endpoint_distance_to_boundary(result, boundary, at_start) > trigger:
            continue
        anchor, direction = _fit_terminal_direction(result, at_start, replace_m, fit_window_m)
        if np.linalg.norm(direction) <= _EPS:
            continue
        max_ray = replace_m + max_extension_m + trigger
        boundary_point = _nearest_boundary_intersection(anchor, direction, boundary, max_ray)
        if boundary_point is None:
            continue
        terminal = LineString([tuple(anchor), boundary_point.coords[0]])
        median_prob, supported_fraction = _segment_probability_support(
            terminal, probability_raster, transform, min_prob
        )
        supported = sister_support or supported_fraction >= min_fraction
        if require_support and not supported:
            continue

        if at_start:
            anchor_d = min(replace_m, result.length)
            remainder = substring(result, anchor_d, result.length)
            new_coords = [boundary_point.coords[0], tuple(anchor)] + list(remainder.coords)[1:]
            old_endpoint = Point(result.coords[0])
        else:
            anchor_d = max(0.0, result.length - replace_m)
            remainder = substring(result, 0.0, anchor_d)
            new_coords = list(remainder.coords)[:-1] + [tuple(anchor), boundary_point.coords[0]]
            old_endpoint = Point(result.coords[-1])

        candidate = LineString(new_coords)
        if candidate.is_valid and candidate.is_simple:
            result = candidate
            changed = True
            total_extended += float(old_endpoint.distance(boundary_point))

    return result, {
        "terminal_straightened": changed,
        "terminal_extended_m": total_extended,
    }


def _endpoint_candidates(line: LineString) -> Tuple[Point, Point]:
    return Point(line.coords[0]), Point(line.coords[-1])


def _connector_metrics(a: LineString, b: LineString) -> Optional[Tuple[float, float, float, bool, bool]]:
    endpoints_a = _endpoint_candidates(a)
    endpoints_b = _endpoint_candidates(b)
    best = None
    for ai, pa in enumerate(endpoints_a):
        for bi, pb in enumerate(endpoints_b):
            distance = pa.distance(pb)
            if best is None or distance < best[0]:
                best = (distance, ai == 0, bi == 0, pa, pb)
    if best is None:
        return None
    distance, a_start, b_start, pa, pb = best
    conn = np.asarray([pb.x - pa.x, pb.y - pa.y], dtype=np.float64)
    norm = np.linalg.norm(conn)
    if norm <= _EPS:
        return None
    conn /= norm

    def outward(line: LineString, at_start: bool) -> np.ndarray:
        coords = np.asarray(line.coords, dtype=np.float64)
        v = coords[0] - coords[min(1, len(coords) - 1)] if at_start else coords[-1] - coords[max(0, len(coords) - 2)]
        n = np.linalg.norm(v)
        return v / n if n > _EPS else v

    va = outward(a, a_start)
    vb = outward(b, b_start)
    angle_a = degrees(np.arccos(float(np.clip(np.dot(va, conn), -1.0, 1.0))))
    angle_b = degrees(np.arccos(float(np.clip(np.dot(vb, -conn), -1.0, 1.0))))
    return float(distance), float(angle_a), float(angle_b), bool(a_start), bool(b_start)


def build_row_tracks(lines: Sequence[LineString], config: dict, qualities: Sequence[LineQuality]) -> List[Track]:
    """Group fragments that are collinear parts of the same physical row."""
    cfg = _cfg(config).get("track_assignment", {})
    max_gap = float(cfg.get("max_same_track_gap_m", 6.0))
    max_angle = float(cfg.get("max_same_track_angle_deg", 8.0))
    max_connector_angle = float(cfg.get("max_connector_angle_deg", 10.0))
    max_lateral = float(cfg.get("max_lateral_offset_m", 0.18))
    n = len(lines)
    if n == 0:
        return []
    tree = STRtree(lines)
    uf = _UnionFind(n)
    bearings = [_bearing(line) for line in lines]

    for i, line in enumerate(lines):
        search = line.buffer(max_gap)
        for j_raw in tree.query(search, predicate="intersects"):
            j = int(j_raw)
            if j <= i:
                continue
            if _angle_diff_deg(bearings[i], bearings[j]) > max_angle:
                continue
            metrics = _connector_metrics(line, lines[j])
            if metrics is None:
                continue
            distance, angle_i, angle_j, _, _ = metrics
            if distance > max_gap or angle_i > max_connector_angle or angle_j > max_connector_angle:
                continue
            # The lateral mismatch is approximated by distance*sin(connector angle).
            lateral = distance * np.sin(np.radians(min(angle_i, angle_j)))
            if lateral > max_lateral:
                continue
            uf.union(i, j)

    components: Dict[int, List[int]] = {}
    for i in range(n):
        components.setdefault(uf.find(i), []).append(i)

    tracks: List[Track] = []
    for track_id, indices in enumerate(components.values(), start=1):
        total_length = float(sum(lines[i].length for i in indices))
        weights = np.asarray([max(lines[i].length, _EPS) for i in indices])
        doubled = np.radians(np.asarray([bearings[i] for i in indices]) * 2.0)
        bearing = float((degrees(0.5 * atan2(np.sum(weights * np.sin(doubled)), np.sum(weights * np.cos(doubled)))) % 180.0))
        quality_class = "GOOD" if len(indices) == 1 and qualities[indices[0]].quality_class == "GOOD" else "SUSPECT"
        tracks.append(Track(track_id, tuple(indices), bearing, total_length, quality_class))
    return tracks


def _project_interval(fragment: LineString, reference: LineString) -> Tuple[float, float, LineString]:
    p0 = reference.project(Point(fragment.coords[0]))
    p1 = reference.project(Point(fragment.coords[-1]))
    if p0 <= p1:
        return float(p0), float(p1), fragment
    return float(p1), float(p0), LineString(list(fragment.coords)[::-1])


def _signed_offset_to_reference(fragments: Sequence[LineString], reference: LineString, spacing_m: float = 0.25) -> float:
    offsets: List[float] = []
    for fragment in fragments:
        _, coords = _resample(fragment, spacing_m)
        for xy in coords:
            point = Point(xy)
            s = reference.project(point)
            ref_point = np.asarray(reference.interpolate(s).coords[0], dtype=np.float64)
            delta = xy - ref_point
            s0 = max(0.0, s - 0.10)
            s1 = min(reference.length, s + 0.10)
            tangent = np.asarray(reference.interpolate(s1).coords[0]) - np.asarray(reference.interpolate(s0).coords[0])
            norm = np.linalg.norm(tangent)
            if norm <= _EPS:
                continue
            tangent /= norm
            normal = np.array([-tangent[1], tangent[0]])
            offsets.append(float(np.dot(delta, normal)))
    if not offsets:
        return 0.0
    return float(np.median(offsets))


def _offset_reference_segment(reference_segment: LineString, offset_m: float, spacing_m: float) -> np.ndarray:
    _, coords = _resample(reference_segment, spacing_m)
    shifted = coords.copy()
    for i in range(len(coords)):
        if len(coords) == 1:
            tangent = np.array([1.0, 0.0])
        elif i == 0:
            tangent = coords[1] - coords[0]
        elif i == len(coords) - 1:
            tangent = coords[-1] - coords[-2]
        else:
            tangent = coords[i + 1] - coords[i - 1]
        norm = np.linalg.norm(tangent)
        if norm <= _EPS:
            continue
        tangent /= norm
        normal = np.array([-tangent[1], tangent[0]])
        shifted[i] += offset_m * normal
    return shifted


def _blend_guide_to_endpoints(guide: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> np.ndarray:
    if len(guide) < 2:
        return np.vstack([start_xy, end_xy])
    start_delta = start_xy - guide[0]
    end_delta = end_xy - guide[-1]
    t = np.linspace(0.0, 1.0, len(guide))[:, None]
    blended = guide + (1.0 - t) * start_delta + t * end_delta
    blended[0] = start_xy
    blended[-1] = end_xy
    return blended


def _concat_pieces(pieces: Sequence[Tuple[float, float, LineString]], join_tolerance_m: float = 0.15) -> LineString:
    ordered = sorted(pieces, key=lambda x: (x[0], x[1]))
    coords: List[Tuple[float, float]] = []
    for _, _, piece in ordered:
        pcoords = list(piece.coords)
        if not pcoords:
            continue
        if not coords:
            coords.extend(pcoords)
            continue
        if Point(coords[-1]).distance(Point(pcoords[0])) <= join_tolerance_m:
            coords.extend(pcoords[1:])
        else:
            coords.extend(pcoords)
    cleaned: List[Tuple[float, float]] = []
    for xy in coords:
        if not cleaned or Point(cleaned[-1]).distance(Point(xy)) > 1e-6:
            cleaned.append(tuple(xy))
    return LineString(cleaned) if len(cleaned) >= 2 else LineString()


def _reference_score(track: Track, reference: Track, lines: Sequence[LineString]) -> float:
    if reference.quality_class != "GOOD":
        return -np.inf
    angle = _angle_diff_deg(track.bearing_deg, reference.bearing_deg)
    if angle > 10.0:
        return -np.inf
    target_union = max((lines[i] for i in track.indices), key=lambda g: g.length)
    ref_line = lines[reference.indices[0]]
    distance = float(target_union.distance(ref_line))
    if distance > 4.0:
        return -np.inf
    return 1.0 - min(angle / 10.0, 1.0) - 0.10 * distance


def _select_reference_track(
    track: Track,
    tracks: Sequence[Track],
    lines: Sequence[LineString],
    pair_ids: Sequence[Any],
    config: dict,
) -> Optional[Track]:
    repair_cfg = _cfg(config).get("reference_guided_repair", {})
    require_same_pair = bool(repair_cfg.get("require_same_pair_id", True))
    allow_neighbour = bool(repair_cfg.get("allow_neighbor_reference", False))
    pair_values = {pair_ids[i] for i in track.indices if pair_ids[i] is not None and not (isinstance(pair_ids[i], float) and np.isnan(pair_ids[i]))}
    sister_candidates: List[Tuple[float, Track]] = []
    neighbour_candidates: List[Tuple[float, Track]] = []
    for ref in tracks:
        if ref.track_id == track.track_id or ref.quality_class != "GOOD":
            continue
        score = _reference_score(track, ref, lines)
        if not np.isfinite(score):
            continue
        ref_pair_values = {pair_ids[i] for i in ref.indices if pair_ids[i] is not None and not (isinstance(pair_ids[i], float) and np.isnan(pair_ids[i]))}
        if pair_values and pair_values.intersection(ref_pair_values):
            sister_candidates.append((score + 1.0, ref))
        else:
            neighbour_candidates.append((score, ref))
    if sister_candidates:
        return max(sister_candidates, key=lambda item: item[0])[1]
    if require_same_pair or not allow_neighbour:
        return None
    both_broken_cfg = _cfg(config).get("both_broken_repair", {})
    if not both_broken_cfg.get("enabled", True) or not both_broken_cfg.get(
        "use_good_neighbor_as_local_template", True
    ):
        return None
    return max(neighbour_candidates, key=lambda item: item[0])[1] if neighbour_candidates else None


def complete_track_from_reference(
    track: Track,
    reference_track: Track,
    lines: Sequence[LineString],
    config: dict,
    probability_raster: Optional[np.ndarray] = None,
    transform: Optional[Affine] = None,
) -> Tuple[Optional[LineString], Dict[str, Any]]:
    """Fill only missing intervals of a broken track from a valid reference."""
    cfg = _cfg(config).get("reference_guided_repair", {})
    if not cfg.get("enabled", True):
        return None, {"reference_repair": False, "repair_reason": "disabled"}
    reference = lines[reference_track.indices[0]]
    fragments = [lines[i] for i in track.indices]
    intervals = [_project_interval(fragment, reference) for fragment in fragments]
    intervals.sort(key=lambda item: item[0])
    if not intervals:
        return None, {"reference_repair": False, "repair_reason": "no_intervals"}

    offset = _signed_offset_to_reference(fragments, reference)
    if abs(offset) < float(cfg.get("min_reference_offset_m", 0.30)):
        return None, {"reference_repair": False, "repair_reason": "offset_too_small"}
    if abs(offset) > float(cfg.get("max_reference_distance_m", 4.0)):
        return None, {"reference_repair": False, "repair_reason": "reference_too_far"}

    max_gap = float(cfg.get("max_completion_gap_m", 6.0))
    min_gap = float(cfg.get("min_completion_gap_m", 0.20))
    guide_spacing = float(cfg.get("guide_spacing_m", 0.20))
    min_prob = float(cfg.get("min_probability_median", 0.12))
    min_fraction = float(cfg.get("min_probability_fraction", 0.45))
    pieces: List[Tuple[float, float, LineString]] = [(a, b, frag) for a, b, frag in intervals]
    repaired_m = 0.0
    accepted_gaps = 0
    unresolved_gaps = 0
    join_tolerance = float(cfg.get("join_tolerance_m", 0.20))

    for left, right in zip(intervals[:-1], intervals[1:]):
        left_end, right_start = left[1], right[0]
        gap = right_start - left_end
        if gap <= join_tolerance:
            continue
        if gap < min_gap or gap > max_gap:
            unresolved_gaps += 1
            continue
        ref_segment = substring(reference, left_end, right_start)
        if ref_segment.is_empty or ref_segment.geom_type != "LineString":
            unresolved_gaps += 1
            continue
        guide = _offset_reference_segment(ref_segment, offset, guide_spacing)
        start_xy = np.asarray(left[2].coords[-1], dtype=np.float64)
        end_xy = np.asarray(right[2].coords[0], dtype=np.float64)
        guide = _blend_guide_to_endpoints(guide, start_xy, end_xy)
        original_proxy = guide.copy()
        guide = _snap_to_probability_ridge(original_proxy, guide, config, probability_raster, transform)
        guide[0] = start_xy
        guide[-1] = end_xy
        fill = LineString(guide)
        median_prob, fraction = _segment_probability_support(fill, probability_raster, transform, min_prob)
        if probability_raster is not None and (median_prob < min_prob or fraction < min_fraction):
            unresolved_gaps += 1
            continue
        if not fill.is_valid or not fill.is_simple:
            unresolved_gaps += 1
            continue
        pieces.append((left_end, right_start, fill))
        repaired_m += float(fill.length)
        accepted_gaps += 1

    if accepted_gaps == 0:
        return None, {"reference_repair": False, "repair_reason": "no_gap_accepted"}
    if unresolved_gaps > 0:
        # Never create an implicit straight bridge over a gap that failed the
        # probability/geometric gates. Keep the original fragments instead.
        return None, {
            "reference_repair": False,
            "repair_reason": "unresolved_gap_remains",
            "unresolved_gaps": unresolved_gaps,
        }
    merged = _concat_pieces(pieces, join_tolerance_m=join_tolerance)
    if merged.is_empty or not merged.is_valid or not merged.is_simple:
        return None, {"reference_repair": False, "repair_reason": "merged_unsafe"}
    return merged, {
        "reference_repair": True,
        "repair_reason": "accepted",
        "reference_track_id": reference_track.track_id,
        "reference_offset_m": abs(offset),
        "reference_completed_m": repaired_m,
        "reference_gaps_completed": accepted_gaps,
    }


def _replace_kinked_target_from_reference(
    target: LineString,
    reference: LineString,
    config: dict,
    probability_raster: Optional[np.ndarray],
    transform: Optional[Affine],
) -> Tuple[LineString, Dict[str, Any]]:
    cfg = _cfg(config).get("reference_guided_repair", {})
    if not cfg.get("enabled", True):
        return target, {"kink_repair": False, "kink_repair_reason": "disabled"}
    spacing = float(cfg.get("kink_sample_spacing_m", 0.20))
    distances, coords = _resample(target, spacing)
    step = max(1, int(round(float(cfg.get("kink_turn_window_m", 0.60)) / max(spacing, 0.02))))
    turns = _turn_angles(coords, step=step)
    threshold = float(cfg.get("repair_kink_threshold_deg", 14.0))
    bad = turns > threshold
    if not np.any(bad):
        return target, {"kink_repair": False, "kink_repair_reason": "no_kink"}
    dilation_m = float(cfg.get("kink_repair_half_window_m", 0.80))
    dilation = max(1, int(round(dilation_m / max(spacing, 0.02))))
    expanded = bad.copy()
    for idx in np.where(bad)[0]:
        expanded[max(0, idx - dilation): min(len(expanded), idx + dilation + 1)] = True

    runs: List[Tuple[int, int]] = []
    start = None
    for i, value in enumerate(expanded):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(expanded) - 1))

    offset = _signed_offset_to_reference([target], reference)
    result = target
    repaired_m = 0.0
    # Apply from the end so target distance indices remain meaningful enough.
    for i0, i1 in reversed(runs):
        d0, d1 = float(distances[i0]), float(distances[i1])
        if d1 - d0 < 0.20 or d0 <= 0.05 or d1 >= target.length - 0.05:
            continue
        start_xy = np.asarray(target.interpolate(d0).coords[0])
        end_xy = np.asarray(target.interpolate(d1).coords[0])
        r0 = reference.project(Point(start_xy))
        r1 = reference.project(Point(end_xy))
        if abs(r1 - r0) < 0.20:
            continue
        if r1 < r0:
            r0, r1 = r1, r0
        ref_segment = substring(reference, r0, r1)
        guide = _offset_reference_segment(ref_segment, offset, spacing)
        guide = _blend_guide_to_endpoints(guide, start_xy, end_xy)
        guide = _snap_to_probability_ridge(guide.copy(), guide, config, probability_raster, transform)
        replacement = LineString(guide)
        before = substring(result, 0.0, d0)
        after = substring(result, d1, result.length)
        candidate = _concat_pieces([(0.0, d0, before), (d0, d1, replacement), (d1, result.length, after)])
        if candidate.is_valid and candidate.is_simple:
            result = candidate
            repaired_m += float(replacement.length)
    return result, {
        "kink_repair": repaired_m > 0,
        "kink_repaired_m": repaired_m,
        "kink_repair_reason": "accepted" if repaired_m > 0 else "no_interval_accepted",
    }


def _candidate_crosses_other(
    candidate: LineString,
    lines: Sequence[LineString],
    tree: STRtree,
    ignored_indices: Iterable[int],
    min_clearance_m: float = 0.0,
) -> bool:
    ignored = set(int(i) for i in ignored_indices)
    query_geom = candidate.buffer(max(min_clearance_m, 0.01))
    for raw in tree.query(query_geom, predicate="intersects"):
        idx = int(raw)
        if idx in ignored:
            continue
        other = lines[idx]
        if candidate.crosses(other):
            return True
        if min_clearance_m > 0 and candidate.distance(other) < min_clearance_m:
            return True
    return False


def _changed_debug_geometry(candidate: LineString, originals: Sequence[LineString]):
    """Return only newly introduced geometry for focused debug layers."""
    original_union = unary_union(list(originals))
    changed = candidate.difference(original_union.buffer(1e-6))
    return candidate if changed.is_empty else changed


def _safe_pair_id_values(gdf: gpd.GeoDataFrame) -> List[Any]:
    if "pair_id" not in gdf.columns:
        return [None] * len(gdf)
    return gdf["pair_id"].tolist()


def refine_double_rows(
    lines_gdf: gpd.GeoDataFrame,
    roi_polygon: Polygon,
    config: dict,
    probability_raster: Optional[np.ndarray] = None,
    transform: Optional[Affine] = None,
    collect_debug: bool = True,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Run constrained smoothing, local reference repair, and straight terminals.

    The function is deliberately conservative: uncertain repairs are rejected
    and surfaced in the debug GeoDataFrame instead of being forced.
    """
    cfg = _cfg(config)
    if not cfg.get("enabled", True) or lines_gdf.empty:
        return lines_gdf.copy(), gpd.GeoDataFrame(geometry=[], crs=lines_gdf.crs)

    result = lines_gdf.copy().reset_index(drop=True)
    lines = result.geometry.tolist()
    with StageTimer("refinement_quality_before"):
        qualities = [
            evaluate_line_quality(line, config, probability_raster, transform)
            for line in lines
        ]
    result["quality_before"] = [q.quality_class for q in qualities]
    result["max_turn_before_deg"] = [q.max_turn_deg for q in qualities]
    result["p95_turn_before_deg"] = [q.p95_turn_deg for q in qualities]

    debug_records: List[Dict[str, Any]] = []
    smoothed_lines: List[LineString] = []
    smooth_meta: List[Dict[str, Any]] = []
    quality_after_candidates: List[LineQuality] = []
    defer_smoothing = bool(
        config.get("final_centerline_fairing", {}).get("enabled", False)
    )
    with StageTimer("refinement_smoothing"):
        for idx, line in enumerate(lines):
            if defer_smoothing:
                smoothed = line
                meta = {
                    "smoothed": False,
                    "smoothing_reason": "deferred_to_regularized_fairing",
                }
                smoothed_quality = qualities[idx]
            else:
                smoothed, meta, smoothed_quality = _smooth_line_center_constrained_cached(
                    line,
                    config,
                    probability_raster,
                    transform,
                    quality=qualities[idx],
                )
            smoothed_lines.append(smoothed)
            smooth_meta.append(meta)
            quality_after_candidates.append(smoothed_quality or qualities[idx])
            if collect_debug and meta.get("smoothed"):
                debug_records.append({
                    "line_index": idx,
                    "action": "smoothing",
                    "debug_layer": "linhas_suavizadas",
                    **meta,
                    "geometry": smoothed,
                })
            elif collect_debug and meta.get("smoothing_reason") in {"safety_reject", "angular_regression"}:
                debug_records.append({
                    "line_index": idx,
                    "action": "smoothing_rejected",
                    "debug_layer": "linhas_rejeitadas",
                    **meta,
                    "geometry": line,
                })
    with StageTimer("refinement_quality_after"):
        qualities_after_smooth = list(quality_after_candidates)
    lines = smoothed_lines
    # Preserve the established GeoPackage schema order. Iterating a set here
    # made otherwise identical runs emit these columns in hash-dependent order.
    smooth_column_order = (
        "p95_turn_after_deg",
        "smoothing_hausdorff_m",
        "p95_turn_before_deg",
        "smoothing_reason",
        "smoothed",
    )
    present_smooth_keys = {key for meta in smooth_meta for key in meta}
    for key in smooth_column_order:
        if key in present_smooth_keys:
            result[key] = [meta.get(key) for meta in smooth_meta]

    with StageTimer("refinement_track_assignment"):
        tracks = build_row_tracks(lines, config, qualities_after_smooth)
    line_tree = STRtree(lines)
    pair_ids = _safe_pair_id_values(result)
    index_to_track = {index: track.track_id for track in tracks for index in track.indices}
    result["track_id"] = [index_to_track.get(i) for i in range(len(result))]

    # Broken tracks: use a valid sister first, otherwise a valid neighbouring row.
    replacements: Dict[int, Tuple[LineString, Dict[str, Any]]] = {}
    consumed_indices: set[int] = set()
    good_reference_tracks = [track for track in tracks if track.quality_class == "GOOD"]
    good_reference_geometries = [lines[track.indices[0]] for track in good_reference_tracks]
    good_reference_tree = STRtree(good_reference_geometries) if good_reference_geometries else None
    track_references: Dict[int, Track] = {}
    reference_queries = 0
    reference_candidates = 0
    reference_repair_attempts = 0
    reference_repairs_accepted = 0
    with StageTimer("refinement_reference_selection"):
        for track in tracks:
            if track.quality_class == "GOOD" or good_reference_tree is None:
                continue
            target_geometry = max((lines[index] for index in track.indices), key=lambda geometry: geometry.length)
            local_indices = good_reference_tree.query(
                target_geometry,
                predicate="dwithin",
                distance=4.0,
            )
            ordered_indices = sorted(int(index) for index in local_indices)
            local_references = [good_reference_tracks[index] for index in ordered_indices]
            reference_queries += 1
            reference_candidates += len(local_references)
            reference = _select_reference_track(
                track,
                local_references,
                lines,
                pair_ids,
                config,
            )
            if reference is not None:
                track_references[track.track_id] = reference

    with StageTimer("refinement_reference_repair"):
        for track in tracks:
            if track.quality_class == "GOOD":
                continue
            reference = track_references.get(track.track_id)
            if reference is None:
                continue
            reference_repair_attempts += 1
            merged, meta = complete_track_from_reference(
                track, reference, lines, config, probability_raster, transform
            )
            if merged is not None:
                clearance = float(cfg.get("reference_guided_repair", {}).get("min_neighbor_clearance_m", 0.35))
                if _candidate_crosses_other(merged, lines, line_tree, track.indices, clearance):
                    if collect_debug:
                        debug_records.append({
                            "line_index": max(track.indices, key=lambda i: lines[i].length),
                            "action": "gap_completion_rejected",
                            "debug_layer": "linhas_rejeitadas",
                            "repair_reason": "crossing_or_clearance_risk",
                            "geometry": merged,
                        })
                    continue
                primary = max(track.indices, key=lambda i: lines[i].length)
                replacements[primary] = (merged, meta)
                reference_repairs_accepted += 1
                consumed_indices.update(i for i in track.indices if i != primary)
                if collect_debug:
                    debug_records.append({
                        "line_index": primary,
                        "action": "gap_completion",
                        "debug_layer": "trechos_substituidos_pela_irma",
                        **meta,
                        "geometry": _changed_debug_geometry(
                            merged, [lines[index] for index in track.indices]
                        ),
                    })
            elif len(track.indices) == 1:
                idx = track.indices[0]
                ref_line = lines[reference.indices[0]]
                repaired, kink_meta = _replace_kinked_target_from_reference(
                    lines[idx], ref_line, config, probability_raster, transform
                )
                if kink_meta.get("kink_repair"):
                    clearance = float(cfg.get("reference_guided_repair", {}).get("min_neighbor_clearance_m", 0.35))
                    if _candidate_crosses_other(repaired, lines, line_tree, [idx], clearance):
                        if collect_debug:
                            debug_records.append({
                                "line_index": idx,
                                "action": "kink_repair_rejected",
                                "debug_layer": "linhas_rejeitadas",
                                "kink_repair_reason": "crossing_or_clearance_risk",
                                "geometry": repaired,
                            })
                        continue
                    replacements[idx] = (repaired, {**meta, **kink_meta, "reference_track_id": reference.track_id})
                    reference_repairs_accepted += 1
                    if collect_debug:
                        debug_records.append({
                            "line_index": idx,
                            "action": "kink_repair",
                            "debug_layer": "trechos_substituidos_pela_irma",
                            **kink_meta,
                            "geometry": _changed_debug_geometry(repaired, [lines[idx]]),
                        })

    logger.info(
        "Refinement references: queries=%d local_candidates=%d selected=%d "
        "good_tracks=%d repair_attempts=%d accepted=%d rejected=%d.",
        reference_queries,
        reference_candidates,
        len(track_references),
        len(good_reference_tracks),
        reference_repair_attempts,
        reference_repairs_accepted,
        reference_repair_attempts - reference_repairs_accepted,
    )

    updated_rows: List[Dict[str, Any]] = []
    updated_quality_cache: List[Optional[LineQuality]] = []
    for idx, row in result.iterrows():
        if idx in consumed_indices:
            continue
        record = row.to_dict()
        replacement = replacements.get(idx)
        line, meta = replacement if replacement is not None else (lines[idx], {})
        record.update(meta)
        if replacement is not None and "track_id" in result.columns:
            # A reference-guided geometry is supported by both the target
            # fragments and the sister/reference track.  Retaining both sides
            # makes the evidence chain auditable without changing the repair.
            evidence_track_ids = [record.get("track_id")]
            if meta.get("reference_track_id") is not None:
                evidence_track_ids.append(meta.get("reference_track_id"))
            evidence_rows = result.loc[result["track_id"].isin(evidence_track_ids)]
            lineage = combine_record_source_raw_ids(
                member for _, member in evidence_rows.iterrows()
            )
            if parse_source_raw_ids(lineage):
                record["source_raw_ids"] = lineage
        record["geometry"] = line
        updated_rows.append(record)
        updated_quality_cache.append(
            None if replacement is not None else qualities_after_smooth[idx]
        )
    result = gpd.GeoDataFrame(updated_rows, geometry="geometry", crs=lines_gdf.crs).reset_index(drop=True)

    # Terminal straightening is applied after reference repair. Pair support is
    # considered true when a confident pair annotation exists.
    final_lines: List[LineString] = []
    terminal_meta: List[Dict[str, Any]] = []
    pre_terminal_lines = result.geometry.tolist()
    pre_terminal_tree = STRtree(pre_terminal_lines)
    terminal_clearance = float(cfg.get("terminal_extension", {}).get("min_neighbor_clearance_m", 0.35))
    final_quality_cache: List[Optional[LineQuality]] = []
    with StageTimer("refinement_terminal_extension"):
        for row_index, row in result.iterrows():
            sister_support = bool(
                row.get("pair_status") == "valid_pair"
                and float(row.get("pair_confidence", 0.0) or 0.0)
                >= float(cfg.get("terminal_extension", {}).get("min_pair_confidence", 0.75))
            )
            refined, meta = straighten_and_extend_terminals(
                row.geometry,
                roi_polygon,
                config,
                probability_raster,
                transform,
                sister_support=sister_support,
            )
            if meta.get("terminal_straightened") and _candidate_crosses_other(
                refined, pre_terminal_lines, pre_terminal_tree, [row_index], terminal_clearance
            ):
                refined = row.geometry
                meta = {
                    "terminal_straightened": False,
                    "terminal_extended_m": 0.0,
                    "terminal_reason": "crossing_or_clearance_risk",
                }
                if collect_debug:
                    debug_records.append({
                        "line_index": row_index,
                        "action": "terminal_extension_rejected",
                        "debug_layer": "linhas_rejeitadas",
                        **meta,
                        "geometry": row.geometry,
                    })
            elif collect_debug and meta.get("terminal_straightened"):
                debug_records.append({
                    "line_index": row_index,
                    "action": "terminal_extension",
                    "debug_layer": "extensoes_terminais",
                    **meta,
                    "geometry": _changed_debug_geometry(refined, [row.geometry]),
                })
            final_lines.append(refined)
            terminal_meta.append(meta)
            final_quality_cache.append(
                None
                if meta.get("terminal_straightened")
                else updated_quality_cache[row_index]
            )
    result.geometry = final_lines
    terminal_column_order = (
        "terminal_extended_m",
        "terminal_reason",
        "terminal_straightened",
    )
    present_terminal_keys = {key for meta in terminal_meta for key in meta}
    for key in terminal_column_order:
        if key in present_terminal_keys:
            result[key] = [meta.get(key) for meta in terminal_meta]

    with StageTimer("refinement_final_quality"):
        final_qualities = [
            cached
            if cached is not None
            else evaluate_line_quality(line, config, probability_raster, transform)
            for line, cached in zip(final_lines, final_quality_cache)
        ]
    result["quality_after"] = [q.quality_class for q in final_qualities]
    result["max_turn_after_deg"] = [q.max_turn_deg for q in final_qualities]
    result["p95_turn_after_deg"] = [q.p95_turn_deg for q in final_qualities]
    result["review_required"] = [q.quality_class != "GOOD" for q in final_qualities]
    result["refinement_source"] = result.apply(classify_refinement_source, axis=1)

    debug_gdf = gpd.GeoDataFrame(debug_records, geometry="geometry", crs=lines_gdf.crs) if debug_records else gpd.GeoDataFrame(geometry=[], crs=lines_gdf.crs)
    quality_counts = {
        quality_class: sum(q.quality_class == quality_class for q in qualities)
        for quality_class in ("GOOD", "SUSPECT", "INVALID")
    }
    logger.info(
        "Double-row refinement: %d input rows -> %d output rows; quality=%s; "
        "tracks=%d good_tracks=%d; reference repairs=%d; terminal changes=%d; debug_records=%d.",
        len(lines_gdf),
        len(result),
        quality_counts,
        len(tracks),
        len(good_reference_tracks),
        int(result["reference_repair"].eq(True).sum()) if "reference_repair" in result else 0,
        int(result["terminal_straightened"].eq(True).sum()) if "terminal_straightened" in result else 0,
        len(debug_records),
    )
    return result, debug_gdf
