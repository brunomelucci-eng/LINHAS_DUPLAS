"""Regularized final fairing for double-row crop centerlines.

The routines in this module operate only on vector geometries and an existing
center-probability raster.  They deliberately avoid independent per-station
peak selection: lateral offsets are selected as one globally regularized path
and are then represented by a restricted cubic B-spline.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from math import ceil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from shapely import distance as shapely_distance, points as shapely_points
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.ops import substring
from shapely.strtree import STRtree


_EPS = 1e-9


@dataclass(frozen=True)
class AngularScaleMetrics:
    """Angular and curvature diagnostics measured at one physical scale."""

    window_m: float
    max_turn_deg: float
    p95_turn_deg: float
    curvature_energy: float
    curvature_variation: float
    lateral_inversions: int
    length_excess_ratio: float


@dataclass(frozen=True)
class MultiScaleAngularMetrics:
    """Short-, medium-, and long-scale angular diagnostics."""

    short: AngularScaleMetrics
    medium: AngularScaleMetrics
    long: AngularScaleMetrics


@dataclass(frozen=True)
class SerrationDiagnostics:
    """Pixel-scale diagnostics used by the final fail-closed output gate."""

    failed: bool
    score: float
    gsd_m: float
    sample_spacing_m: float
    micro_p95_turn_deg: float
    local_p95_turn_deg: float
    vertex_p95_turn_deg: float
    inversions_per_m: float
    vertex_density_per_m: float
    length_excess_ratio: float
    residual_p95_m: float


@dataclass(frozen=True)
class DynamicStationProfile:
    """Metric curvature classes and continuous per-station controls."""

    distances_m: np.ndarray
    turn_deg_per_m: np.ndarray
    raw_classes: np.ndarray
    stable_classes: np.ndarray
    sigma_m: np.ndarray
    output_spacing_m: np.ndarray


def _fairing_cfg(config: dict) -> dict:
    return config.get("final_centerline_fairing", {})


def _sampling_spacing(config: dict) -> float:
    cfg = _fairing_cfg(config).get("sampling", {})
    return max(float(cfg.get("longitudinal_spacing_m", 0.20)), 0.02)


def resample_line(line: LineString, spacing_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a LineString by arclength using vectorized segment lookup."""
    if line.is_empty or len(line.coords) < 2 or line.length <= _EPS:
        coords = np.asarray(line.coords, dtype=np.float64)
        return np.zeros(len(coords), dtype=np.float64), coords[:, :2]

    spacing_m = max(float(spacing_m), 0.02)
    count = max(2, int(ceil(line.length / spacing_m)) + 1)
    distances = np.linspace(0.0, float(line.length), count)
    vertices = np.asarray(line.coords, dtype=np.float64)[:, :2]
    vectors = np.diff(vertices, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    indices = np.searchsorted(cumulative, distances, side="right") - 1
    indices = np.clip(indices, 0, len(lengths) - 1)
    local = distances - cumulative[indices]
    fractions = np.divide(
        local,
        lengths[indices],
        out=np.zeros_like(local),
        where=lengths[indices] > _EPS,
    )
    sampled = vertices[indices] + fractions[:, None] * vectors[indices]
    sampled[0] = vertices[0]
    sampled[-1] = vertices[-1]
    return distances, sampled


def _safe_savgol(values: np.ndarray, window: int, polyorder: int = 3) -> np.ndarray:
    if len(values) < 5:
        return values.copy()
    window = max(polyorder + 2, int(window))
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window <= polyorder:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")


def _turn_components(coords: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return unsigned turns and signed curvature proxies without Python loops."""
    if len(coords) < 2 * step + 1:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    before = coords[step:-step] - coords[:-2 * step]
    after = coords[2 * step:] - coords[step:-step]
    norm_before = np.linalg.norm(before, axis=1)
    norm_after = np.linalg.norm(after, axis=1)
    valid = (norm_before > _EPS) & (norm_after > _EPS)
    dot = np.einsum("ij,ij->i", before, after)
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    signed = np.zeros(len(before), dtype=np.float64)
    signed[valid] = np.degrees(np.arctan2(cross[valid], dot[valid]))
    return np.abs(signed), signed


def _scale_metrics(
    coords: np.ndarray,
    spacing_m: float,
    window_m: float,
) -> AngularScaleMetrics:
    step = max(1, int(round(float(window_m) / max(spacing_m, 0.02))))
    turns, signed_turns = _turn_components(coords, step)
    max_turn = float(np.max(turns)) if len(turns) else 0.0
    p95_turn = float(np.percentile(turns, 95)) if len(turns) else 0.0
    curvature = np.radians(signed_turns) / max(2.0 * step * spacing_m, _EPS)
    curvature_energy = float(np.mean(curvature ** 2)) if len(curvature) else 0.0
    curvature_variation = (
        float(np.mean(np.abs(np.diff(curvature)))) if len(curvature) > 1 else 0.0
    )
    meaningful = np.sign(curvature[np.abs(curvature) > 1e-4])
    inversions = int(np.sum(meaningful[1:] * meaningful[:-1] < 0)) if len(meaningful) > 1 else 0

    smooth_window = max(5, int(round((2.0 * window_m) / max(spacing_m, 0.02))))
    smooth = np.column_stack(
        (
            _safe_savgol(coords[:, 0], smooth_window),
            _safe_savgol(coords[:, 1], smooth_window),
        )
    )
    raw_length = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
    smooth_length = float(np.linalg.norm(np.diff(smooth, axis=0), axis=1).sum())
    excess = max(0.0, raw_length / max(smooth_length, _EPS) - 1.0)
    return AngularScaleMetrics(
        window_m=float(window_m),
        max_turn_deg=max_turn,
        p95_turn_deg=p95_turn,
        curvature_energy=curvature_energy,
        curvature_variation=curvature_variation,
        lateral_inversions=inversions,
        length_excess_ratio=float(excess),
    )


def compute_multiscale_angular_metrics(
    line: LineString,
    config: dict,
) -> MultiScaleAngularMetrics:
    """Measure serration and curve preservation at three physical scales."""
    spacing = _sampling_spacing(config)
    _, coords = resample_line(line, spacing)
    metrics_cfg = _fairing_cfg(config).get("angular_metrics", {})
    short_window = float(metrics_cfg.get("short_window_m", 0.25))
    medium_window = float(metrics_cfg.get("medium_window_m", 0.60))
    long_window = float(metrics_cfg.get("long_window_m", 1.35))
    return MultiScaleAngularMetrics(
        short=_scale_metrics(coords, spacing, short_window),
        medium=_scale_metrics(coords, spacing, medium_window),
        long=_scale_metrics(coords, spacing, long_window),
    )


def _raster_gsd_m(transform: Optional[Affine], config: dict) -> float:
    """Return a rotation-safe pixel size, with a configured metric fallback."""
    serration_cfg = _fairing_cfg(config).get("micro_serration", {})
    fallback = max(float(serration_cfg.get("default_gsd_m", 0.04)), 0.005)
    if transform is None:
        return fallback
    sizes = np.asarray(
        (
            np.hypot(float(transform.a), float(transform.d)),
            np.hypot(float(transform.b), float(transform.e)),
        ),
        dtype=np.float64,
    )
    sizes = sizes[np.isfinite(sizes) & (sizes > _EPS)]
    return float(np.median(sizes)) if len(sizes) else fallback


def compute_serration_diagnostics(
    line: LineString,
    config: dict,
    transform: Optional[Affine] = None,
) -> SerrationDiagnostics:
    """Detect repeated raster stair-steps without mistaking a smooth arc for noise.

    The previous detector resampled every geometry at 0.20 m.  At the imagery
    resolutions used by this project that skips roughly five raster pixels and
    can hide the alternating horizontal/diagonal steps produced by skeletons.
    This detector derives its sampling and angular windows from the raster GSD,
    and combines oscillation with either angular, residual, or vertex-density
    evidence.  A coherent curve has a stable curvature sign and therefore does
    not trigger only because it is stored with many vertices.
    """
    serration_cfg = _fairing_cfg(config).get("micro_serration", {})
    gsd_m = _raster_gsd_m(transform, config)
    maximum_spacing = max(
        float(serration_cfg.get("max_sample_spacing_m", 0.05)), 0.02
    )
    sample_spacing = max(0.02, min(gsd_m, maximum_spacing))
    if line is None or line.is_empty or len(line.coords) < 2 or line.length <= _EPS:
        return SerrationDiagnostics(
            failed=False,
            score=0.0,
            gsd_m=gsd_m,
            sample_spacing_m=sample_spacing,
            micro_p95_turn_deg=0.0,
            local_p95_turn_deg=0.0,
            vertex_p95_turn_deg=0.0,
            inversions_per_m=0.0,
            vertex_density_per_m=0.0,
            length_excess_ratio=0.0,
            residual_p95_m=0.0,
        )

    _, sampled = resample_line(line, sample_spacing)
    window_pixels = serration_cfg.get("window_pixels", (3, 5, 9))
    windows = sorted(
        max(1, int(value)) for value in window_pixels
        if float(value) > 0.0
    )
    if not windows:
        windows = [3, 5, 9]
    scales = [
        _scale_metrics(sampled, sample_spacing, pixels * gsd_m)
        for pixels in windows
    ]
    micro = scales[0]
    local_p95 = max(metric.p95_turn_deg for metric in scales)
    length_excess = max(metric.length_excess_ratio for metric in scales)
    sampled_inversions = max(metric.lateral_inversions for metric in scales)

    vertices = np.asarray(line.coords, dtype=np.float64)[:, :2]
    vertex_turns, vertex_signed = _turn_components(vertices, 1)
    minimum_meaningful_turn = float(
        serration_cfg.get("min_meaningful_vertex_turn_deg", 0.50)
    )
    meaningful_signs = np.sign(
        vertex_signed[np.abs(vertex_signed) >= minimum_meaningful_turn]
    )
    vertex_inversions = (
        int(np.sum(meaningful_signs[1:] * meaningful_signs[:-1] < 0))
        if len(meaningful_signs) > 1
        else 0
    )
    vertex_p95 = float(np.percentile(vertex_turns, 95)) if len(vertex_turns) else 0.0
    inversions_per_m = float(max(sampled_inversions, vertex_inversions)) / max(
        float(line.length), _EPS
    )
    vertex_density = float(len(vertices)) / max(float(line.length), _EPS)

    residual_window = max(
        5,
        int(round(float(serration_cfg.get("residual_window_pixels", 9)) * gsd_m / sample_spacing)),
    )
    smooth = np.column_stack(
        (
            _safe_savgol(sampled[:, 0], residual_window),
            _safe_savgol(sampled[:, 1], residual_window),
        )
    )
    residuals = np.linalg.norm(sampled - smooth, axis=1)
    residual_p95 = float(np.percentile(residuals, 95)) if len(residuals) else 0.0

    angle_limit = max(float(serration_cfg.get("max_micro_p95_turn_deg", 2.50)), _EPS)
    vertex_angle_limit = max(
        float(serration_cfg.get("max_vertex_p95_turn_deg", 3.00)), _EPS
    )
    inversion_limit = max(
        float(serration_cfg.get("max_lateral_inversions_per_m", 0.75)), _EPS
    )
    density_limit = max(
        float(serration_cfg.get("max_vertex_density_per_m", 8.0)), _EPS
    )
    excess_limit = max(
        float(serration_cfg.get("max_length_excess_ratio", 0.0025)), _EPS
    )
    residual_limit = max(
        float(serration_cfg.get("max_residual_p95_gsd_ratio", 0.25)) * gsd_m,
        0.005,
    )
    angle_ratio = max(micro.p95_turn_deg / angle_limit, vertex_p95 / vertex_angle_limit)
    inversion_ratio = inversions_per_m / inversion_limit
    density_ratio = vertex_density / density_limit
    shape_ratio = max(length_excess / excess_limit, residual_p95 / residual_limit)
    # A numerically smooth dense curve can contain many tiny curvature-sign
    # reversals.  Inversions alone are therefore not evidence of visible
    # serration.  Require two independent signals for each failure channel:
    # repeated angular reversals, or raster-dense vertices plus measurable
    # residual/length excess.  This keeps true smooth arcs clean while still
    # rejecting the alternating horizontal/diagonal steps of a skeleton.
    has_angular_oscillation = angle_ratio >= 1.0 and inversion_ratio >= 1.0
    has_raster_shape = density_ratio >= 1.0 and shape_ratio >= 1.0
    oscillation_score = (
        float(np.sqrt(max(angle_ratio, 0.0) * max(inversion_ratio, 0.0)))
        if has_angular_oscillation
        else min(max(min(angle_ratio, inversion_ratio), 0.0), 1.0 - 1e-9)
    )
    raster_density_score = (
        float(
            np.sqrt(
                min(max(density_ratio, 0.0), 4.0)
                * min(max(shape_ratio, 0.0), 4.0)
            )
        )
        if has_raster_shape
        else min(max(min(density_ratio, shape_ratio), 0.0), 1.0 - 1e-9)
    )
    score = max(oscillation_score, raster_density_score)
    minimum_length = float(serration_cfg.get("min_line_length_m", 1.50))
    failed = bool(
        serration_cfg.get("enabled", True)
        and line.length >= minimum_length - 1e-9
        and score >= float(serration_cfg.get("max_score", 1.0)) - 1e-9
        and (has_angular_oscillation or has_raster_shape)
    )
    return SerrationDiagnostics(
        failed=failed,
        score=float(score),
        gsd_m=gsd_m,
        sample_spacing_m=sample_spacing,
        micro_p95_turn_deg=float(micro.p95_turn_deg),
        local_p95_turn_deg=float(local_p95),
        vertex_p95_turn_deg=float(vertex_p95),
        inversions_per_m=float(inversions_per_m),
        vertex_density_per_m=float(vertex_density),
        length_excess_ratio=float(length_excess),
        residual_p95_m=float(residual_p95),
    )


def annotate_serration_status(
    lines_gdf: gpd.GeoDataFrame,
    config: dict,
    transform: Optional[Affine] = None,
) -> gpd.GeoDataFrame:
    """Audit every final geometry and make unresolved serration fail closed."""
    result = lines_gdf.copy()
    diagnostics = [
        compute_serration_diagnostics(geometry, config, transform)
        for geometry in result.geometry
    ]
    result["serration_status"] = [
        "failed" if item.failed else "clean" for item in diagnostics
    ]
    result["serration_score"] = [item.score for item in diagnostics]
    if "serration_before" not in result.columns:
        result["serration_before"] = [item.score for item in diagnostics]
    else:
        result["serration_before"] = pd.to_numeric(
            result["serration_before"], errors="coerce"
        ).fillna(pd.Series([item.score for item in diagnostics], index=result.index))
    result["serration_after"] = [item.score for item in diagnostics]
    result["serration_micro_p95_deg"] = [
        item.micro_p95_turn_deg for item in diagnostics
    ]
    result["serration_local_p95_deg"] = [
        item.local_p95_turn_deg for item in diagnostics
    ]
    result["serration_vertex_p95_deg"] = [
        item.vertex_p95_turn_deg for item in diagnostics
    ]
    result["serration_inversions_per_m"] = [
        item.inversions_per_m for item in diagnostics
    ]
    result["serration_vertex_density_per_m"] = [
        item.vertex_density_per_m for item in diagnostics
    ]
    result["serration_length_excess_ratio"] = [
        item.length_excess_ratio for item in diagnostics
    ]
    result["serration_residual_p95_m"] = [
        item.residual_p95_m for item in diagnostics
    ]
    if "serration_repair_method" not in result.columns:
        fallback = result.get(
            "fairing_fallback_method", pd.Series(pd.NA, index=result.index)
        )
        result["serration_repair_method"] = fallback.fillna("none").astype(str)
    failed_mask = result["serration_status"].eq("failed")
    if "review_required" not in result.columns:
        result["review_required"] = False
    result.loc[failed_mask, "review_required"] = True
    return result


def _sample_probability_xy(
    points: np.ndarray,
    probability_raster: Optional[np.ndarray],
    transform: Optional[Affine],
) -> np.ndarray:
    """Vectorized bilinear sampling used by the fairing data term."""
    if probability_raster is None or transform is None or len(points) == 0:
        return np.full(len(points), np.nan, dtype=np.float64)
    if hasattr(probability_raster, "sample_bilinear"):
        return np.asarray(
            probability_raster.sample_bilinear(points, transform), dtype=np.float64
        )
    inv = ~transform
    points = np.asarray(points, dtype=np.float64)
    cols = inv.a * points[:, 0] + inv.b * points[:, 1] + inv.c
    rows = inv.d * points[:, 0] + inv.e * points[:, 1] + inv.f
    height, width = probability_raster.shape
    valid = (rows >= 0.0) & (rows < height - 1) & (cols >= 0.0) & (cols < width - 1)
    values = np.zeros(len(points), dtype=np.float64)
    if not np.any(valid):
        return values
    r = rows[valid]
    c = cols[valid]
    r0 = np.floor(r).astype(np.int64)
    c0 = np.floor(c).astype(np.int64)
    dr = r - r0
    dc = c - c0
    v00 = np.asarray(probability_raster[r0, c0], dtype=np.float64)
    v01 = np.asarray(probability_raster[r0, c0 + 1], dtype=np.float64)
    v10 = np.asarray(probability_raster[r0 + 1, c0], dtype=np.float64)
    v11 = np.asarray(probability_raster[r0 + 1, c0 + 1], dtype=np.float64)
    values[valid] = (
        v00 * (1.0 - dr) * (1.0 - dc)
        + v01 * (1.0 - dr) * dc
        + v10 * dr * (1.0 - dc)
        + v11 * dr * dc
    )
    return values


class RasterProbabilitySampler:
    """Block-cached raster accessor for post-processing very large debug TIFFs."""

    def __init__(self, dataset, max_cached_blocks: int = 128):
        self.dataset = dataset
        self.shape = (int(dataset.height), int(dataset.width))
        self.transform = dataset.transform
        self.block_height, self.block_width = dataset.block_shapes[0]
        self.max_cached_blocks = max(4, int(max_cached_blocks))
        self._cache: OrderedDict[Tuple[int, int], np.ndarray] = OrderedDict()

    def _block(self, block_row: int, block_col: int) -> Tuple[np.ndarray, int, int]:
        key = (int(block_row), int(block_col))
        cached = self._cache.get(key)
        row0 = key[0] * self.block_height
        col0 = key[1] * self.block_width
        if cached is None:
            from rasterio.windows import Window

            height = min(self.block_height + 1, self.shape[0] - row0)
            width = min(self.block_width + 1, self.shape[1] - col0)
            cached = self.dataset.read(
                1,
                window=Window(col0, row0, width, height),
            )
            self._cache[key] = cached
            if len(self._cache) > self.max_cached_blocks:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return cached, row0, col0

    def _values_at_indices(self, rows, cols) -> np.ndarray:
        rows_array, cols_array = np.broadcast_arrays(
            np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        )
        original_shape = rows_array.shape
        flat_rows = rows_array.ravel()
        flat_cols = cols_array.ravel()
        values = np.zeros(len(flat_rows), dtype=np.float64)
        valid = (
            (flat_rows >= 0)
            & (flat_rows < self.shape[0])
            & (flat_cols >= 0)
            & (flat_cols < self.shape[1])
        )
        valid_indices = np.flatnonzero(valid)
        groups: Dict[Tuple[int, int], List[int]] = {}
        for index in valid_indices:
            key = (
                int(flat_rows[index] // self.block_height),
                int(flat_cols[index] // self.block_width),
            )
            groups.setdefault(key, []).append(int(index))
        for (block_row, block_col), indices in groups.items():
            block, row0, col0 = self._block(block_row, block_col)
            index_array = np.asarray(indices, dtype=np.int64)
            values[index_array] = block[
                flat_rows[index_array] - row0,
                flat_cols[index_array] - col0,
            ]
        return values.reshape(original_shape)

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("RasterProbabilitySampler expects [rows, cols] indexing.")
        values = self._values_at_indices(key[0], key[1])
        return values.item() if values.ndim == 0 else values

    def sample_bilinear(
        self,
        points: np.ndarray,
        transform: Optional[Affine] = None,
    ) -> np.ndarray:
        transform = transform or self.transform
        inverse = ~transform
        points = np.asarray(points, dtype=np.float64)
        cols = inverse.a * points[:, 0] + inverse.b * points[:, 1] + inverse.c
        rows = inverse.d * points[:, 0] + inverse.e * points[:, 1] + inverse.f
        valid = (
            (rows >= 0.0)
            & (rows < self.shape[0] - 1)
            & (cols >= 0.0)
            & (cols < self.shape[1] - 1)
        )
        values = np.zeros(len(points), dtype=np.float64)
        if not np.any(valid):
            return values
        r, c = rows[valid], cols[valid]
        r0, c0 = np.floor(r).astype(np.int64), np.floor(c).astype(np.int64)
        dr, dc = r - r0, c - c0
        values[valid] = (
            self._values_at_indices(r0, c0) * (1.0 - dr) * (1.0 - dc)
            + self._values_at_indices(r0, c0 + 1) * (1.0 - dr) * dc
            + self._values_at_indices(r0 + 1, c0) * dr * (1.0 - dc)
            + self._values_at_indices(r0 + 1, c0 + 1) * dr * dc
        )
        return values


def _smooth_baseline(coords: np.ndarray, spacing_m: float, config: dict) -> np.ndarray:
    geometry_cfg = _fairing_cfg(config).get("geometry", {})
    window_m = float(geometry_cfg.get("baseline_window_m", 1.40))
    window = max(5, int(round(window_m / max(spacing_m, 0.02))))
    baseline = np.column_stack(
        (
            _safe_savgol(coords[:, 0], window),
            _safe_savgol(coords[:, 1], window),
        )
    )
    baseline[0] = coords[0]
    baseline[-1] = coords[-1]
    return baseline


def _tangents_and_normals(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tangents = np.empty_like(coords)
    tangents[0] = coords[1] - coords[0]
    tangents[-1] = coords[-1] - coords[-2]
    if len(coords) > 2:
        tangents[1:-1] = coords[2:] - coords[:-2]
    norms = np.linalg.norm(tangents, axis=1)
    tangents = np.divide(
        tangents,
        norms[:, None],
        out=np.tile(np.array([[1.0, 0.0]]), (len(coords), 1)),
        where=norms[:, None] > _EPS,
    )
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    return tangents, normals


def _lateral_offsets(config: dict) -> np.ndarray:
    sampling = _fairing_cfg(config).get("sampling", {})
    radius = max(float(sampling.get("lateral_search_radius_m", 0.12)), 0.0)
    step = max(float(sampling.get("lateral_step_m", 0.02)), 0.005)
    count = int(round((2.0 * radius) / step))
    offsets = np.linspace(-radius, radius, count + 1, dtype=np.float64)
    if not np.any(np.isclose(offsets, 0.0, atol=1e-12)):
        offsets = np.sort(np.append(offsets, 0.0))
    return offsets


def _adaptive_output_cfg(config: dict) -> dict:
    """Resolve the new adaptive schema with legacy output settings as fallback."""
    fairing = _fairing_cfg(config)
    legacy = fairing.get("output_sampling", {})
    legacy_curve = float(legacy.get("curve_spacing_m", 0.25))
    resolved = {
        "enabled": True,
        "straight_spacing_m": float(legacy.get("straight_spacing_m", 0.60)),
        "gentle_curve_spacing_m": legacy_curve,
        "curve_spacing_m": legacy_curve,
        "high_curvature_spacing_m": float(
            legacy.get("high_curvature_spacing_m", 0.18)
        ),
        "terminal_spacing_m": 0.25,
        "splice_spacing_m": 0.20,
        "straight_max_turn_deg_per_m": float(
            legacy.get("curve_turn_threshold_deg", 2.0)
        ),
        "gentle_curve_max_turn_deg_per_m": float(
            legacy.get("curve_turn_threshold_deg", 2.0)
        ),
        "curve_max_turn_deg_per_m": float(
            legacy.get("high_curvature_turn_threshold_deg", 7.0)
        ),
        "min_class_persistence_m": 0.80,
        "transition_blend_m": 0.50,
    }
    resolved.update(fairing.get("adaptive_output_sampling", {}))
    return resolved


def _signed_turn_rate(
    coords: np.ndarray,
    distances_m: np.ndarray,
    window_m: float,
) -> np.ndarray:
    """Estimate signed direction change per metre using a metric window."""
    count = len(coords)
    rates = np.zeros(count, dtype=np.float64)
    if count < 3 or distances_m[-1] <= _EPS:
        return rates
    half_window = max(float(window_m) * 0.5, _EPS)
    left = np.searchsorted(distances_m, distances_m - half_window, side="right") - 1
    right = np.searchsorted(distances_m, distances_m + half_window, side="left")
    left = np.clip(left, 0, count - 1)
    right = np.clip(right, 0, count - 1)
    indices = np.arange(count)
    before = coords - coords[left]
    after = coords[right] - coords
    before_norm = np.linalg.norm(before, axis=1)
    after_norm = np.linalg.norm(after, axis=1)
    valid = (
        (left < indices)
        & (right > indices)
        & (before_norm > _EPS)
        & (after_norm > _EPS)
    )
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    dot = np.einsum("ij,ij->i", before, after)
    angle = np.degrees(np.arctan2(cross, dot))
    tangent_separation = 0.5 * (distances_m[right] - distances_m[left])
    rates[valid] = angle[valid] / np.maximum(tangent_separation[valid], _EPS)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        rates = np.interp(indices, valid_indices, rates[valid_indices])
    return rates


def _persistent_turn_rate(
    coords: np.ndarray,
    distances_m: np.ndarray,
) -> np.ndarray:
    short = _signed_turn_rate(coords, distances_m, 0.50)
    long = _signed_turn_rate(coords, distances_m, 1.00)
    coherent = (short * long >= 0.0) | (np.abs(short) < 1e-9) | (np.abs(long) < 1e-9)
    return np.where(coherent, np.minimum(np.abs(short), np.abs(long)), 0.0)


def _class_runs(classes: np.ndarray) -> List[Tuple[int, int, int]]:
    if len(classes) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(classes) != 0) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(classes)]
    return [(int(start), int(end), int(classes[start])) for start, end in zip(starts, ends)]


def apply_class_hysteresis(
    raw_classes: np.ndarray,
    distances_m: np.ndarray,
    min_persistence_m: float,
) -> np.ndarray:
    """Remove curvature-class runs that do not persist for a metric distance."""
    stable = np.asarray(raw_classes, dtype=np.int8).copy()
    distances_m = np.asarray(distances_m, dtype=np.float64)
    if len(stable) < 2 or min_persistence_m <= 0.0:
        return stable
    positive_steps = np.diff(distances_m)
    positive_steps = positive_steps[positive_steps > _EPS]
    sample_width = float(np.median(positive_steps)) if len(positive_steps) else 0.0

    for _ in range(max(1, len(stable))):
        runs = _class_runs(stable)
        changed = False
        for run_index, (start, end, _) in enumerate(runs):
            run_length = float(distances_m[end - 1] - distances_m[start] + sample_width)
            if run_length + _EPS >= min_persistence_m or len(runs) == 1:
                continue
            left_run = runs[run_index - 1] if run_index > 0 else None
            right_run = runs[run_index + 1] if run_index + 1 < len(runs) else None
            if left_run is None:
                replacement = right_run[2]
            elif right_run is None:
                replacement = left_run[2]
            elif left_run[2] == right_run[2]:
                replacement = left_run[2]
            else:
                left_length = distances_m[left_run[1] - 1] - distances_m[left_run[0]] + sample_width
                right_length = distances_m[right_run[1] - 1] - distances_m[right_run[0]] + sample_width
                replacement = (
                    left_run[2]
                    if left_length > right_length + _EPS
                    else right_run[2]
                    if right_length > left_length + _EPS
                    else max(left_run[2], right_run[2])
                )
            stable[start:end] = replacement
            changed = True
            break
        if not changed:
            break
    return stable


def _blend_station_values(
    values: np.ndarray,
    distances_m: np.ndarray,
    transition_m: float,
) -> np.ndarray:
    if len(values) < 3 or transition_m <= 0.0:
        return np.asarray(values, dtype=np.float64).copy()
    positive_steps = np.diff(distances_m)
    positive_steps = positive_steps[positive_steps > _EPS]
    spacing = float(np.median(positive_steps)) if len(positive_steps) else transition_m
    sigma_samples = max(transition_m / max(spacing, 0.02) / 2.0, 0.0)
    if sigma_samples <= _EPS:
        return np.asarray(values, dtype=np.float64).copy()
    return gaussian_filter1d(
        np.asarray(values, dtype=np.float64),
        sigma=sigma_samples,
        mode="nearest",
        truncate=2.0,
    )


def _dynamic_profile_from_coords(
    coords: np.ndarray,
    distances_m: np.ndarray,
    config: dict,
    sister_line: Optional[LineString] = None,
) -> DynamicStationProfile:
    adaptive_cfg = _adaptive_output_cfg(config)
    smoothing_cfg = _fairing_cfg(config).get("offset_smoothing", {})
    turn_rate = _persistent_turn_rate(coords, distances_m)

    if sister_line is not None and not sister_line.is_empty and sister_line.length > _EPS:
        sister_distances, sister_coords = resample_line(
            sister_line, _sampling_spacing(config)
        )
        sister_baseline = _smooth_baseline(
            sister_coords, _sampling_spacing(config), config
        )
        sister_rate = _persistent_turn_rate(sister_baseline, sister_distances)
        own_fraction = distances_m / max(distances_m[-1], _EPS)
        sister_fraction = sister_distances / max(sister_distances[-1], _EPS)
        turn_rate = np.maximum(
            turn_rate,
            np.interp(own_fraction, sister_fraction, sister_rate),
        )

    thresholds = np.asarray(
        [
            float(adaptive_cfg.get("straight_max_turn_deg_per_m", 1.0)),
            float(adaptive_cfg.get("gentle_curve_max_turn_deg_per_m", 3.0)),
            float(adaptive_cfg.get("curve_max_turn_deg_per_m", 7.0)),
        ],
        dtype=np.float64,
    )
    thresholds = np.maximum.accumulate(thresholds)
    raw_classes = np.digitize(turn_rate, thresholds, right=True).astype(np.int8)
    stable_classes = apply_class_hysteresis(
        raw_classes,
        distances_m,
        float(adaptive_cfg.get("min_class_persistence_m", 0.80)),
    )
    sigma_by_class = np.asarray(
        [
            float(smoothing_cfg.get("straight_sigma_m", 0.70)),
            float(smoothing_cfg.get("gentle_curve_sigma_m", 0.45)),
            float(smoothing_cfg.get("curve_sigma_m", 0.28)),
            float(smoothing_cfg.get("high_curvature_sigma_m", 0.18)),
        ],
        dtype=np.float64,
    )
    spacing_by_class = np.asarray(
        [
            float(adaptive_cfg.get("straight_spacing_m", 1.00)),
            float(adaptive_cfg.get("gentle_curve_spacing_m", 0.60)),
            float(adaptive_cfg.get("curve_spacing_m", 0.35)),
            float(adaptive_cfg.get("high_curvature_spacing_m", 0.20)),
        ],
        dtype=np.float64,
    )
    transition = float(adaptive_cfg.get("transition_blend_m", 0.50))
    sigma_profile = _blend_station_values(
        sigma_by_class[stable_classes], distances_m, transition
    )
    spacing_profile = _blend_station_values(
        spacing_by_class[stable_classes], distances_m, transition
    )
    return DynamicStationProfile(
        distances_m=np.asarray(distances_m, dtype=np.float64),
        turn_deg_per_m=turn_rate,
        raw_classes=raw_classes,
        stable_classes=stable_classes,
        sigma_m=sigma_profile,
        output_spacing_m=spacing_profile,
    )


def compute_dynamic_station_profile(
    line: LineString,
    config: dict,
    sister_line: Optional[LineString] = None,
) -> DynamicStationProfile:
    """Build the metric curvature profile used by smoothing and output sampling."""
    distances, coords = resample_line(line, _sampling_spacing(config))
    baseline = _smooth_baseline(coords, _sampling_spacing(config), config)
    return _dynamic_profile_from_coords(
        baseline, distances, config, sister_line=sister_line
    )


def _centered_moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Smooth a one-dimensional signal with a centred, endpoint-safe window."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3 or window_samples <= 1:
        return values.copy()
    window_samples = min(int(window_samples), len(values))
    if window_samples % 2 == 0:
        window_samples = window_samples + 1 if window_samples < len(values) else window_samples - 1
    if window_samples <= 1:
        return values.copy()
    half = window_samples // 2
    padded = np.pad(values, (half, half), mode="edge")
    kernel = np.full(window_samples, 1.0 / window_samples, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def _variable_gaussian_offsets(
    values: np.ndarray,
    sigma_profile_m: np.ndarray,
    spacing_m: float,
    truncate: float,
    sigma_anchors_m: Sequence[float],
) -> np.ndarray:
    """Blend constant-sigma Gaussian results into a continuous sigma field."""
    anchors = np.unique(
        np.asarray(
            [max(float(value), 0.0) for value in sigma_anchors_m],
            dtype=np.float64,
        )
    )
    if len(anchors) == 0 or anchors[-1] <= _EPS:
        return values.copy()
    filtered = np.vstack(
        [
            gaussian_filter1d(
                values,
                sigma=max(anchor / max(spacing_m, 0.02), _EPS),
                mode="nearest",
                truncate=truncate,
            )
            for anchor in anchors
        ]
    )
    if len(anchors) == 1:
        return filtered[0]
    target = np.clip(np.asarray(sigma_profile_m, dtype=np.float64), anchors[0], anchors[-1])
    result = np.empty_like(values, dtype=np.float64)
    upper_indices = np.searchsorted(anchors, target, side="right")
    upper_indices = np.clip(upper_indices, 1, len(anchors) - 1)
    lower_indices = upper_indices - 1
    lower = anchors[lower_indices]
    upper = anchors[upper_indices]
    fraction = (target - lower) / np.maximum(upper - lower, _EPS)
    station_indices = np.arange(len(values))
    result[:] = (
        filtered[lower_indices, station_indices] * (1.0 - fraction)
        + filtered[upper_indices, station_indices] * fraction
    )
    return result


def smooth_lateral_offsets(
    selected_offsets: np.ndarray,
    baseline: np.ndarray,
    spacing_m: float,
    config: dict,
    sister_line: Optional[LineString] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Smooth Viterbi offsets without changing the longitudinal coordinate."""
    values = np.asarray(selected_offsets, dtype=np.float64)
    smoothing_cfg = _fairing_cfg(config).get("offset_smoothing", {})
    if not smoothing_cfg.get("enabled", False) or len(values) < 3:
        return values.copy(), {"offset_smoothing_method": "disabled"}

    method = str(smoothing_cfg.get("method", "gaussian")).strip().lower()
    if method == "moving_average":
        window_m = max(float(smoothing_cfg.get("moving_average_window_m", 0.80)), 0.0)
        window_samples = max(1, int(round(window_m / max(spacing_m, 0.02))))
        smoothed = _centered_moving_average(values, window_samples)
        smoothed[0], smoothed[-1] = values[0], values[-1]
        return smoothed, {
            "offset_smoothing_method": "moving_average",
            "offset_smoothing_window_m": window_m,
        }

    if method in {"gaussian", "dynamic_gaussian"}:
        truncate = max(float(smoothing_cfg.get("truncate", 3.0)), 0.5)
        dynamic_enabled = method == "dynamic_gaussian" or bool(
            smoothing_cfg.get("dynamic_sigma_enabled", True)
        )
        try:
            if dynamic_enabled:
                distances_m = np.arange(len(values), dtype=np.float64) * max(
                    spacing_m, 0.02
                )
                profile = _dynamic_profile_from_coords(
                    np.asarray(baseline, dtype=np.float64),
                    distances_m,
                    config,
                    sister_line=sister_line,
                )
                sigma_profile = profile.sigma_m
                anchors = (
                    smoothing_cfg.get("straight_sigma_m", 0.70),
                    smoothing_cfg.get("gentle_curve_sigma_m", 0.45),
                    smoothing_cfg.get("curve_sigma_m", 0.28),
                    smoothing_cfg.get("high_curvature_sigma_m", 0.18),
                )
                smoothed = _variable_gaussian_offsets(
                    values,
                    sigma_profile,
                    spacing_m,
                    truncate,
                    anchors,
                )
            else:
                fixed_sigma = max(
                    float(smoothing_cfg.get("gaussian_sigma_m", 0.45)), 0.0
                )
                sigma_profile = np.full(len(values), fixed_sigma, dtype=np.float64)
                smoothed = gaussian_filter1d(
                    values,
                    sigma=max(fixed_sigma / max(spacing_m, 0.02), _EPS),
                    mode="nearest",
                    truncate=truncate,
                )
            if not np.all(np.isfinite(smoothed)):
                raise ValueError("Gaussian offset smoothing produced non-finite values.")
            radius = float(
                _fairing_cfg(config)
                .get("sampling", {})
                .get("lateral_search_radius_m", 0.12)
            )
            smoothed = np.clip(smoothed, -abs(radius), abs(radius))
            smoothed[0], smoothed[-1] = values[0], values[-1]
            return smoothed, {
                "offset_smoothing_method": (
                    "gaussian_dynamic" if dynamic_enabled else "gaussian"
                ),
                "offset_smoothing_sigma_min_m": float(np.min(sigma_profile)),
                "offset_smoothing_sigma_max_m": float(np.max(sigma_profile)),
            }
        except (FloatingPointError, TypeError, ValueError):
            if str(smoothing_cfg.get("fallback_method", "moving_average")).lower() == "moving_average":
                window_m = max(
                    float(smoothing_cfg.get("moving_average_window_m", 0.80)), 0.0
                )
                window_samples = max(
                    1, int(round(window_m / max(spacing_m, 0.02)))
                )
                smoothed = _centered_moving_average(values, window_samples)
                smoothed[0], smoothed[-1] = values[0], values[-1]
                return smoothed, {
                    "offset_smoothing_method": "moving_average_fallback",
                    "offset_smoothing_window_m": window_m,
                }

    # Unknown methods remain a safe no-op; Viterbi and all gates stay active.
    return values.copy(), {"offset_smoothing_method": "disabled"}


def _pair_distance_cost(
    candidates: np.ndarray,
    original: np.ndarray,
    sister_line: Optional[LineString],
    target_spacing_m: Optional[float],
) -> Tuple[np.ndarray, Optional[float]]:
    if sister_line is None:
        return np.zeros(candidates.shape[:2], dtype=np.float64), target_spacing_m
    if target_spacing_m is None:
        original_distances = np.asarray(
            shapely_distance(shapely_points(original), sister_line), dtype=np.float64
        )
        target_spacing_m = float(np.median(original_distances))
    flat_points = shapely_points(candidates.reshape(-1, 2))
    distances = np.asarray(shapely_distance(flat_points, sister_line), dtype=np.float64)
    distances = distances.reshape(candidates.shape[:2])
    return (distances - target_spacing_m) ** 2, target_spacing_m


def select_regularized_offsets(
    original: np.ndarray,
    baseline: np.ndarray,
    normals: np.ndarray,
    config: dict,
    probability_raster: Optional[np.ndarray] = None,
    transform: Optional[Affine] = None,
    sister_line: Optional[LineString] = None,
    target_pair_spacing_m: Optional[float] = None,
    station_spacing_m: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select one globally coherent lateral-offset sequence using Viterbi DP."""
    offsets = _lateral_offsets(config)
    radius = max(float(np.max(np.abs(offsets))), 0.005)
    candidates = baseline[:, None, :] + normals[:, None, :] * offsets[None, :, None]
    probabilities = _sample_probability_xy(
        candidates.reshape(-1, 2), probability_raster, transform
    ).reshape(candidates.shape[:2])

    probability_cfg = _fairing_cfg(config).get("probability", {})
    if probability_raster is None or transform is None:
        probability_cost = np.zeros_like(probabilities)
    else:
        sigma = max(float(probability_cfg.get("lateral_blur_sigma_samples", 1.0)), 0.0)
        if sigma > 0:
            probabilities = gaussian_filter1d(probabilities, sigma=sigma, axis=1, mode="nearest")
        clipped = np.clip(probabilities, 1e-6, 1.0)
        probability_cost = -np.log(clipped)
        minimum = float(probability_cfg.get("min_probability", 0.12))
        probability_cost += np.maximum(0.0, minimum - probabilities) * 20.0

    regularization = _fairing_cfg(config).get("regularization", {})
    w_first = float(regularization.get("first_difference_weight", 4.0))
    w_second = float(regularization.get("second_difference_weight", 12.0))
    w_original = float(regularization.get("original_distance_weight", 2.0))
    w_pair = float(regularization.get("pair_spacing_weight", 3.0))

    original_offsets = np.einsum("ij,ij->i", original - baseline, normals)
    original_cost = ((offsets[None, :] - original_offsets[:, None]) / radius) ** 2
    pair_cost, resolved_pair_spacing = _pair_distance_cost(
        candidates, original, sister_line, target_pair_spacing_m
    )
    unary = probability_cost + w_original * original_cost + w_pair * pair_cost / (radius ** 2)

    geometry_cfg = _fairing_cfg(config).get("geometry", {})
    if geometry_cfg.get("preserve_endpoints", True):
        zero_index = int(np.argmin(np.abs(offsets)))
        unary[0, :] += 1e12
        unary[-1, :] += 1e12
        unary[0, zero_index] -= 1e12
        unary[-1, zero_index] -= 1e12

    count, states = unary.shape
    normalized = offsets / radius
    if count == 1:
        path = np.array([int(np.argmin(unary[0]))], dtype=np.int16)
    else:
        first_difference = (normalized[None, :] - normalized[:, None]) ** 2
        dp = unary[0, :, None] + unary[1, None, :] + w_first * first_difference
        back = np.full((count, states, states), -1, dtype=np.int16)
        a = normalized[:, None, None]
        b = normalized[None, :, None]
        c = normalized[None, None, :]
        transition_first = w_first * (c - b) ** 2
        transition_second = w_second * (c - 2.0 * b + a) ** 2
        for station in range(2, count):
            transition = dp[:, :, None] + transition_first + transition_second
            predecessor = np.argmin(transition, axis=0).astype(np.int16)
            best = np.take_along_axis(transition, predecessor[None, :, :], axis=0)[0]
            dp = best + unary[station][None, :]
            back[station] = predecessor

        previous, current = np.unravel_index(int(np.argmin(dp)), dp.shape)
        path = np.empty(count, dtype=np.int16)
        path[-2], path[-1] = previous, current
        for station in range(count - 1, 1, -1):
            predecessor = int(back[station, previous, current])
            path[station - 2] = predecessor
            current = previous
            previous = predecessor

    selected_offsets_raw = offsets[path]
    if station_spacing_m is None:
        segment_lengths = np.linalg.norm(np.diff(original, axis=0), axis=1)
        positive_lengths = segment_lengths[segment_lengths > _EPS]
        station_spacing_m = (
            float(np.median(positive_lengths))
            if len(positive_lengths)
            else _sampling_spacing(config)
        )
    selected_offsets, smoothing_meta = smooth_lateral_offsets(
        selected_offsets_raw,
        baseline,
        station_spacing_m,
        config,
        sister_line,
    )
    optimized = baseline + normals * selected_offsets[:, None]
    optimized[0] = original[0]
    optimized[-1] = original[-1]
    selected_probability = probabilities[np.arange(count), path]
    return optimized, {
        "selected_offsets_raw_m": selected_offsets_raw,
        "selected_offsets_m": selected_offsets,
        "selected_probability": selected_probability,
        "target_pair_spacing_m": resolved_pair_spacing,
        **smoothing_meta,
    }


def _restricted_bspline(
    points: np.ndarray,
    original: np.ndarray,
    distances: np.ndarray,
    config: dict,
) -> LineString:
    geometry_cfg = _fairing_cfg(config).get("geometry", {})
    max_shift = float(geometry_cfg.get("max_lateral_shift_m", 0.10))
    high_spacing = float(
        _adaptive_output_cfg(config).get("high_curvature_spacing_m", 0.20)
    )
    dense_spacing = max(0.04, min(high_spacing / 2.0, 0.10))
    dense_count = max(2, int(ceil(distances[-1] / dense_spacing)) + 1)
    dense_distances = np.linspace(0.0, distances[-1], dense_count)

    if distances[-1] <= _EPS:
        return LineString(original)
    if len(points) < 4:
        # A cubic spline is not appropriate for two or three stations, but the
        # safety projection below still expects the dense sampling grid. Keep
        # candidate and reference arrays aligned by interpolating linearly.
        dense = np.column_stack(
            (
                np.interp(dense_distances, distances, points[:, 0]),
                np.interp(dense_distances, distances, points[:, 1]),
            )
        )
    else:
        u = distances / distances[-1]
        smoothing = float(
            geometry_cfg.get("bspline_smoothing_per_station_m2", 0.0001)
        ) * len(points)
        try:
            tck, _ = splprep(
                [points[:, 0], points[:, 1]],
                u=u,
                s=smoothing,
                k=min(3, len(points) - 1),
            )
            dense_u = dense_distances / distances[-1]
            dense = np.column_stack(splev(dense_u, tck))
        except (TypeError, ValueError):
            dense = np.column_stack(
                (
                    np.interp(dense_distances, distances, points[:, 0]),
                    np.interp(dense_distances, distances, points[:, 1]),
                )
            )

    original_dense = np.column_stack(
        (
            np.interp(dense_distances, distances, original[:, 0]),
            np.interp(dense_distances, distances, original[:, 1]),
        )
    )
    displacement = dense - original_dense
    norms = np.linalg.norm(displacement, axis=1)
    scale = np.minimum(1.0, max_shift / np.maximum(norms, _EPS))
    dense = original_dense + displacement * scale[:, None]
    dense[0] = original[0]
    dense[-1] = original[-1]
    return LineString(dense)


def _legacy_adaptive_resample_line(line: LineString, config: dict) -> LineString:
    """Preserve the pre-dynamic three-point sampler for benchmark compatibility."""
    if line.is_empty or line.length <= _EPS:
        return line
    output_cfg = _fairing_cfg(config).get("output_sampling", {})
    straight_spacing = float(output_cfg.get("straight_spacing_m", 0.60))
    curve_spacing = float(output_cfg.get("curve_spacing_m", 0.25))
    high_spacing = float(output_cfg.get("high_curvature_spacing_m", 0.18))
    curve_threshold = float(output_cfg.get("curve_turn_threshold_deg", 2.0))
    high_threshold = float(output_cfg.get("high_curvature_turn_threshold_deg", 7.0))
    probe = max(curve_spacing, 0.20)

    distances = [0.0]
    current = 0.0
    while current < line.length - _EPS:
        left_d = max(0.0, current - probe)
        right_d = min(line.length, current + probe)
        left = np.asarray(line.interpolate(left_d).coords[0], dtype=np.float64)
        center = np.asarray(line.interpolate(current).coords[0], dtype=np.float64)
        right = np.asarray(line.interpolate(right_d).coords[0], dtype=np.float64)
        before = center - left
        after = right - center
        denom = np.linalg.norm(before) * np.linalg.norm(after)
        turn = 0.0
        if denom > _EPS:
            turn = float(
                np.degrees(
                    np.arccos(np.clip(np.dot(before, after) / denom, -1.0, 1.0))
                )
            )
        spacing = (
            high_spacing
            if turn >= high_threshold
            else curve_spacing
            if turn >= curve_threshold
            else straight_spacing
        )
        current = min(line.length, current + max(spacing, 0.05))
        if current - distances[-1] > _EPS:
            distances.append(current)
    coords = [line.interpolate(distance).coords[0] for distance in distances]
    coords[0] = line.coords[0]
    coords[-1] = line.coords[-1]
    return LineString(coords)


def _refine_distances_for_chord_error(
    line: LineString,
    distances_m: np.ndarray,
    maximum_error_m: float,
    probe_spacing_m: float,
    minimum_spacing_m: float = 0.05,
) -> np.ndarray:
    """Insert curve vertices until every output chord follows the dense line.

    Curvature classes choose a useful initial density, but a single chord may
    still cut a real curve by more than the geometry safety budget.  This audit
    samples the fitted dense curve and recursively splits only those chords.
    Straight portions remain at the requested 3 m spacing.
    """
    distances_m = np.unique(np.asarray(distances_m, dtype=np.float64))
    if (
        len(distances_m) < 2
        or maximum_error_m <= 0.0
        or line.length <= minimum_spacing_m
    ):
        return distances_m

    probe_distances, probe_coordinates = resample_line(
        line, max(float(probe_spacing_m), 0.02)
    )

    def interval_error(start_m: float, end_m: float) -> Tuple[float, float]:
        left = int(np.searchsorted(probe_distances, start_m, side="right"))
        right = int(np.searchsorted(probe_distances, end_m, side="left"))
        if right <= left:
            return 0.0, 0.5 * (start_m + end_m)
        points = probe_coordinates[left:right]
        start = np.asarray(line.interpolate(start_m).coords[0], dtype=np.float64)[:2]
        end = np.asarray(line.interpolate(end_m).coords[0], dtype=np.float64)[:2]
        chord = end - start
        chord_squared = float(np.dot(chord, chord))
        if chord_squared <= _EPS:
            errors = np.linalg.norm(points - start, axis=1)
        else:
            fractions = np.clip(
                ((points - start) @ chord) / chord_squared, 0.0, 1.0
            )
            projections = start + fractions[:, None] * chord
            errors = np.linalg.norm(points - projections, axis=1)
        maximum_index = int(np.argmax(errors))
        return float(errors[maximum_index]), float(
            probe_distances[left + maximum_index]
        )

    def refine_interval(start_m: float, end_m: float, depth: int = 0) -> List[float]:
        if depth >= 20 or end_m - start_m <= 2.0 * minimum_spacing_m:
            return [start_m, end_m]
        error_m, split_m = interval_error(start_m, end_m)
        if error_m <= maximum_error_m + 1e-9:
            return [start_m, end_m]
        if (
            split_m - start_m < minimum_spacing_m
            or end_m - split_m < minimum_spacing_m
        ):
            split_m = 0.5 * (start_m + end_m)
        left = refine_interval(start_m, split_m, depth + 1)
        right = refine_interval(split_m, end_m, depth + 1)
        return left[:-1] + right

    refined: List[float] = [float(distances_m[0])]
    for start_m, end_m in zip(distances_m[:-1], distances_m[1:]):
        interval = refine_interval(float(start_m), float(end_m))
        refined.extend(interval[1:])
    result = np.unique(np.asarray(refined, dtype=np.float64))
    result[0], result[-1] = 0.0, float(line.length)
    return result


def adaptive_resample_line(
    line: LineString,
    config: dict,
    sister_line: Optional[LineString] = None,
    protected_intervals_m: Optional[Sequence[Tuple[float, float, str]]] = None,
) -> LineString:
    """Resample by integrated metric curvature density with persistent classes."""
    if line.is_empty or line.length <= _EPS:
        return line
    adaptive_cfg = _adaptive_output_cfg(config)
    if not adaptive_cfg.get("enabled", True):
        return _legacy_adaptive_resample_line(line, config)

    profile = compute_dynamic_station_profile(line, config, sister_line=sister_line)
    spacing_profile = np.maximum(profile.output_spacing_m.copy(), 0.05)
    for start_m, end_m, interval_kind in protected_intervals_m or ():
        low, high = sorted((float(start_m), float(end_m)))
        configured_spacing = (
            float(adaptive_cfg.get("terminal_spacing_m", 0.25))
            if interval_kind == "terminal"
            else float(adaptive_cfg.get("splice_spacing_m", 0.20))
        )
        mask = (profile.distances_m >= low - _EPS) & (
            profile.distances_m <= high + _EPS
        )
        spacing_profile[mask] = np.minimum(
            spacing_profile[mask], configured_spacing
        )

    segment_lengths = np.diff(profile.distances_m)
    inverse_spacing = 1.0 / np.maximum(spacing_profile, 0.05)
    cumulative_density = np.r_[
        0.0,
        np.cumsum(
            segment_lengths
            * 0.5
            * (inverse_spacing[:-1] + inverse_spacing[1:])
        ),
    ]
    if cumulative_density[-1] <= _EPS:
        return LineString([line.coords[0], line.coords[-1]])
    density_targets = np.arange(
        0.0, np.floor(cumulative_density[-1]) + 1.0, dtype=np.float64
    )
    if (
        len(density_targets) == 0
        or cumulative_density[-1] - density_targets[-1] > 1e-7
    ):
        density_targets = np.r_[density_targets, cumulative_density[-1]]
    elif abs(cumulative_density[-1] - density_targets[-1]) <= 1e-7:
        density_targets[-1] = cumulative_density[-1]
    output_distances = np.interp(
        density_targets, cumulative_density, profile.distances_m
    )
    output_distances[0], output_distances[-1] = 0.0, line.length
    maximum_chord_error = max(
        float(adaptive_cfg.get("max_chord_error_m", 0.02)), 0.0
    )
    output_distances = _refine_distances_for_chord_error(
        line,
        output_distances,
        maximum_chord_error,
        float(adaptive_cfg.get("chord_error_probe_spacing_m", 0.05)),
    )
    coordinates = [line.interpolate(distance).coords[0] for distance in output_distances]
    coordinates[0], coordinates[-1] = line.coords[0], line.coords[-1]
    result = LineString(coordinates)
    # ``straight_spacing_m`` is a target density, not by itself a strict upper
    # bound when the density varies through a transition.  Segmentize makes the
    # user-facing 3 m contract explicit without changing the fitted geometry.
    maximum_spacing = max(
        float(
            adaptive_cfg.get(
                "max_vertex_spacing_m",
                adaptive_cfg.get("straight_spacing_m", 3.0),
            )
        ),
        0.05,
    )
    # The epsilon prevents a 3.0000000000000004 m floating-point segment from
    # being unnecessarily divided in half by GEOS.
    return result.segmentize(maximum_spacing + 1e-9)


def _probability_median(
    line: LineString,
    spacing_m: float,
    probability_raster: Optional[np.ndarray],
    transform: Optional[Affine],
) -> float:
    if probability_raster is None or transform is None:
        return float("nan")
    _, coords = resample_line(line, spacing_m)
    values = _sample_probability_xy(coords, probability_raster, transform)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float("nan")


def _conservative_serration_candidate(
    original_line: LineString,
    sigma_m: float,
    config: dict,
    sister_line: Optional[LineString],
) -> LineString:
    """Remove a residual coordinate staircase without moving the line wholesale.

    This fallback is deliberately independent of the rejected Viterbi/B-spline
    candidate.  It smooths X and Y along arclength, keeps both endpoints exact,
    and fades the correction in over the terminal metre.  The caller still has
    to pass the result through every normal geometric/probability safety gate.
    """
    acceptance = _fairing_cfg(config).get("acceptance", {})
    sampling_m = max(
        float(acceptance.get("conservative_gaussian_sampling_m", 0.10)),
        0.01,
    )
    taper_m = max(
        float(acceptance.get("conservative_gaussian_terminal_taper_m", 0.15)),
        0.01,
    )
    truncate = max(
        float(acceptance.get("conservative_gaussian_truncate", 3.0)),
        0.5,
    )
    distances, original_coords = resample_line(original_line, sampling_m)
    sigma_samples = max(float(sigma_m) / sampling_m, _EPS)
    smoothed_coords = np.column_stack(
        (
            gaussian_filter1d(
                original_coords[:, 0],
                sigma=sigma_samples,
                mode="nearest",
                truncate=truncate,
            ),
            gaussian_filter1d(
                original_coords[:, 1],
                sigma=sigma_samples,
                mode="nearest",
                truncate=truncate,
            ),
        )
    )
    edge_distance = np.minimum(distances, original_line.length - distances)
    taper_position = np.clip(edge_distance / taper_m, 0.0, 1.0)
    taper_weight = taper_position * taper_position * (3.0 - 2.0 * taper_position)
    candidate_coords = original_coords + taper_weight[:, None] * (
        smoothed_coords - original_coords
    )
    candidate_coords[0], candidate_coords[-1] = (
        original_coords[0],
        original_coords[-1],
    )
    return adaptive_resample_line(
        LineString(candidate_coords), config, sister_line=sister_line
    )


def _two_stage_gaussian_cfg(config: dict) -> dict:
    return _fairing_cfg(config).get("two_stage_gaussian", {})


def _angular_gaussian_candidate(
    line: LineString,
    config: dict,
    *,
    stage: str,
    transform: Optional[Affine] = None,
    sister_line: Optional[LineString] = None,
) -> Tuple[LineString, LineString, Dict[str, Any]]:
    """Smooth raster-scale lateral oscillation with curvature-dependent sigma.

    Filtering X/Y directly shrinks real curves.  Instead, this routine builds a
    persistent angular baseline, expresses the observed line as lateral offsets
    from that baseline, and filters only those offsets.  Straight stations can
    therefore use a wider Gaussian than high-curvature stations.
    """
    cfg = _two_stage_gaussian_cfg(config)
    if line.is_empty or len(line.coords) < 3 or line.length <= _EPS:
        return line, line, {
            "two_stage_gaussian_applied": False,
            "two_stage_gaussian_stage": stage,
            "two_stage_gaussian_reason": "empty_or_short",
        }

    gsd_m = _raster_gsd_m(transform, config)
    sampling_m = max(
        0.02,
        min(
            gsd_m,
            float(cfg.get("internal_sampling_max_m", 0.10)),
        ),
    )
    distances, coordinates = resample_line(line, sampling_m)
    baseline = _smooth_baseline(coordinates, sampling_m, config)
    _, normals = _tangents_and_normals(baseline)
    offsets = np.einsum("ij,ij->i", coordinates - baseline, normals)
    profile = _dynamic_profile_from_coords(
        baseline,
        distances,
        config,
        sister_line=sister_line,
    )

    stage_cfg = cfg.get(stage, {})
    sigma_by_class = np.asarray(
        [
            float(stage_cfg.get("straight_sigma_m", 0.35 if stage == "light" else 1.00)),
            float(stage_cfg.get("gentle_curve_sigma_m", 0.25 if stage == "light" else 0.70)),
            float(stage_cfg.get("curve_sigma_m", 0.16 if stage == "light" else 0.45)),
            float(stage_cfg.get("high_curvature_sigma_m", 0.10 if stage == "light" else 0.25)),
        ],
        dtype=np.float64,
    )
    sigma_by_class = np.maximum(sigma_by_class, 0.0)
    transition_m = float(cfg.get("transition_blend_m", 0.50))
    sigma_profile = _blend_station_values(
        sigma_by_class[profile.stable_classes],
        distances,
        transition_m,
    )
    smoothed_offsets = _variable_gaussian_offsets(
        offsets,
        sigma_profile,
        sampling_m,
        max(float(cfg.get("truncate", 3.0)), 0.5),
        sigma_by_class,
    )

    taper_m = max(float(cfg.get("terminal_taper_m", 1.00)), 0.01)
    edge_distance = np.minimum(distances, line.length - distances)
    taper_position = np.clip(edge_distance / taper_m, 0.0, 1.0)
    taper_weight = taper_position * taper_position * (3.0 - 2.0 * taper_position)
    filtered_offsets = offsets + taper_weight * (smoothed_offsets - offsets)
    candidate_coordinates = baseline + normals * filtered_offsets[:, None]
    candidate_coordinates[0], candidate_coordinates[-1] = (
        coordinates[0],
        coordinates[-1],
    )
    dense_candidate = LineString(candidate_coordinates)
    candidate = adaptive_resample_line(
        dense_candidate,
        config,
        sister_line=sister_line,
    )
    return dense_candidate, candidate, {
        "two_stage_gaussian_applied": not candidate.equals_exact(line, tolerance=1e-9),
        "two_stage_gaussian_stage": stage,
        "two_stage_gaussian_reason": "candidate",
        "two_stage_gaussian_sampling_m": sampling_m,
        "two_stage_gaussian_sigma_min_m": float(np.min(sigma_profile)),
        "two_stage_gaussian_sigma_max_m": float(np.max(sigma_profile)),
        "two_stage_gaussian_vertices_before": len(line.coords),
        "two_stage_gaussian_vertices_after": len(candidate.coords),
    }


def two_stage_gaussian_smooth_line(
    line: LineString,
    config: dict,
    transform: Optional[Affine] = None,
    sister_line: Optional[LineString] = None,
) -> Tuple[LineString, Dict[str, Any]]:
    """Apply a light pass to every line and an aggressive pass only if needed."""
    cfg = _two_stage_gaussian_cfg(config)
    before = compute_serration_diagnostics(line, config, transform)
    if not cfg.get("enabled", False):
        return line, {
            "two_stage_gaussian_applied": False,
            "two_stage_gaussian_stage": "disabled",
            "two_stage_gaussian_score_before": before.score,
            "two_stage_gaussian_score_after": before.score,
        }

    light_dense, light, metadata = _angular_gaussian_candidate(
        line,
        config,
        stage="light",
        transform=transform,
        sister_line=sister_line,
    )
    # Diagnose the dense filtered curve, not the 3 m export.  Otherwise sparse
    # output vertices could merely hide a staircase instead of removing it.
    light_diagnostics = compute_serration_diagnostics(
        light_dense, config, transform
    )
    result = light
    final_diagnostics = light_diagnostics
    aggressive_attempted = False
    if light_diagnostics.failed:
        aggressive_attempted = True
        aggressive_dense, result, aggressive_metadata = _angular_gaussian_candidate(
            light_dense,
            config,
            stage="aggressive",
            transform=transform,
            sister_line=sister_line,
        )
        metadata.update(aggressive_metadata)
        final_diagnostics = compute_serration_diagnostics(
            aggressive_dense, config, transform
        )

    metadata.update(
        {
            "two_stage_gaussian_applied": not result.equals_exact(
                line, tolerance=1e-9
            ),
            "two_stage_gaussian_stage": (
                "aggressive" if aggressive_attempted else "light"
            ),
            "two_stage_gaussian_aggressive_attempted": aggressive_attempted,
            "two_stage_gaussian_light_failed": light_diagnostics.failed,
            "two_stage_gaussian_score_before": before.score,
            "two_stage_gaussian_score_light": light_diagnostics.score,
            "two_stage_gaussian_score_after": final_diagnostics.score,
            "two_stage_gaussian_failed_after": final_diagnostics.failed,
            "two_stage_gaussian_micro_p95_after_deg": (
                final_diagnostics.micro_p95_turn_deg
            ),
            "two_stage_gaussian_vertex_p95_after_deg": (
                final_diagnostics.vertex_p95_turn_deg
            ),
            "two_stage_gaussian_inversions_after_per_m": (
                final_diagnostics.inversions_per_m
            ),
            "two_stage_gaussian_residual_p95_after_m": (
                final_diagnostics.residual_p95_m
            ),
            "two_stage_gaussian_length_excess_after": (
                final_diagnostics.length_excess_ratio
            ),
        }
    )
    return result, metadata


def _blend_candidate_toward_original(
    original_line: LineString,
    smooth_candidate: LineString,
    factor: float,
    config: dict,
    sister_line: Optional[LineString],
) -> LineString:
    """Blend a rejected smooth curve back toward the original by arclength."""
    distances, original_coords = resample_line(
        original_line, _sampling_spacing(config)
    )
    fractions = distances / max(distances[-1], _EPS)
    candidate_coords = np.asarray(
        [
            smooth_candidate.interpolate(float(value) * smooth_candidate.length).coords[0]
            for value in fractions
        ],
        dtype=np.float64,
    )[:, :2]
    blended_coords = original_coords + float(factor) * (
        candidate_coords - original_coords
    )
    blended_coords[0], blended_coords[-1] = original_coords[0], original_coords[-1]
    blended = LineString(blended_coords)
    return adaptive_resample_line(
        blended, config, sister_line=sister_line
    )


def _fairing_candidate_diagnostics(
    original_line: LineString,
    candidate: LineString,
    config: dict,
    probability_raster: Optional[np.ndarray],
    transform: Optional[Affine],
    spacing_m: float,
    before_metrics: MultiScaleAngularMetrics,
    before_probability: float,
    *,
    require_short_gain: bool = True,
) -> Dict[str, Any]:
    """Evaluate every local safety gate and return the first rejection reason."""
    diagnostics: Dict[str, Any] = {
        "reason": None,
        "hausdorff_m": float("nan"),
        "lateral_shift_m": float("nan"),
        "before_metrics": before_metrics,
        "after_metrics": None,
        "before_probability": before_probability,
        "after_probability": float("nan"),
        "short_reduction": float("nan"),
        "before_serration": None,
        "after_serration": None,
    }
    if candidate.is_empty or not candidate.is_valid:
        diagnostics["reason"] = "invalid_geometry"
        return diagnostics
    geometry_cfg = _fairing_cfg(config).get("geometry", {})
    if geometry_cfg.get("require_simple", True) and not candidate.is_simple:
        diagnostics["reason"] = "self_intersection"
        return diagnostics

    hausdorff = float(original_line.hausdorff_distance(candidate))
    _, candidate_samples = resample_line(candidate, min(spacing_m, 0.10))
    lateral_shift = float(
        np.max(
            shapely_distance(
                shapely_points(candidate_samples), original_line
            )
        )
    )
    after_metrics = compute_multiscale_angular_metrics(candidate, config)
    before_serration = compute_serration_diagnostics(
        original_line, config, transform
    )
    after_serration = compute_serration_diagnostics(candidate, config, transform)
    after_probability = _probability_median(
        candidate, spacing_m, probability_raster, transform
    )
    short_reduction = (
        0.0
        if before_metrics.short.p95_turn_deg <= _EPS
        else 1.0
        - after_metrics.short.p95_turn_deg / before_metrics.short.p95_turn_deg
    )
    diagnostics.update(
        {
            "hausdorff_m": hausdorff,
            "lateral_shift_m": lateral_shift,
            "after_metrics": after_metrics,
            "after_probability": after_probability,
            "short_reduction": short_reduction,
            "before_serration": before_serration,
            "after_serration": after_serration,
        }
    )

    max_hausdorff = float(geometry_cfg.get("max_hausdorff_m", 0.12))
    max_lateral = float(geometry_cfg.get("max_lateral_shift_m", 0.10))
    acceptance = _fairing_cfg(config).get("acceptance", {})
    medium_limit = max(
        before_metrics.medium.p95_turn_deg
        + float(acceptance.get("max_medium_p95_regression_deg", 0.50)),
        before_metrics.medium.p95_turn_deg
        * float(acceptance.get("max_medium_p95_ratio", 1.05)),
    )
    long_limit = max(
        before_metrics.long.p95_turn_deg
        + float(acceptance.get("max_long_p95_regression_deg", 1.00)),
        before_metrics.long.p95_turn_deg
        * float(acceptance.get("max_long_p95_ratio", 1.10)),
    )
    probability_cfg = _fairing_cfg(config).get("probability", {})
    max_loss = float(probability_cfg.get("max_median_probability_loss", 0.02))
    short_improved = (
        after_metrics.short.p95_turn_deg
        < before_metrics.short.p95_turn_deg - 1e-6
        or after_metrics.short.curvature_energy
        < before_metrics.short.curvature_energy - 1e-9
    )
    minimum_short_reduction = float(
        acceptance.get("min_short_p95_reduction_ratio", 0.0)
    )
    enforce_reduction_above = float(
        acceptance.get("enforce_min_reduction_above_short_p95_deg", 2.50)
    )

    if hausdorff > max_hausdorff + 1e-9:
        diagnostics["reason"] = "hausdorff_limit"
    elif lateral_shift > max_lateral + 1e-9:
        diagnostics["reason"] = "lateral_shift_limit"
    elif after_serration.failed:
        diagnostics["reason"] = "residual_serration"
    elif require_short_gain and not short_improved:
        diagnostics["reason"] = "no_short_scale_gain"
    elif (
        require_short_gain
        and before_metrics.short.p95_turn_deg >= enforce_reduction_above
        and short_reduction < minimum_short_reduction - 1e-9
    ):
        diagnostics["reason"] = "insufficient_short_scale_gain"
    elif after_metrics.medium.p95_turn_deg > medium_limit + 1e-9:
        diagnostics["reason"] = "medium_scale_regression"
    elif after_metrics.long.p95_turn_deg > long_limit + 1e-9:
        diagnostics["reason"] = "long_scale_regression"
    elif (
        np.isfinite(before_probability)
        and np.isfinite(after_probability)
        and after_probability < before_probability - max_loss - 1e-9
    ):
        diagnostics["reason"] = "probability_regression"
    return diagnostics


def fair_line_regularized(
    line: LineString,
    config: dict,
    probability_raster: Optional[np.ndarray] = None,
    transform: Optional[Affine] = None,
    sister_line: Optional[LineString] = None,
    target_pair_spacing_m: Optional[float] = None,
) -> Tuple[LineString, Dict[str, Any]]:
    """Fair one line and automatically revert when any safety gate fails."""
    if line.is_empty or len(line.coords) < 2 or line.length <= _EPS:
        return line, {"fairing_applied": False, "fairing_reason": "empty_or_short"}

    spacing = _sampling_spacing(config)
    distances, original = resample_line(line, spacing)
    baseline = _smooth_baseline(original, spacing, config)
    _, normals = _tangents_and_normals(baseline)
    optimized, viterbi_meta = select_regularized_offsets(
        original,
        baseline,
        normals,
        config,
        probability_raster,
        transform,
        sister_line,
        target_pair_spacing_m,
        spacing,
    )
    spline = _restricted_bspline(optimized, original, distances, config)
    candidate = adaptive_resample_line(
        spline, config, sister_line=sister_line
    )
    primary_candidate = candidate

    before_metrics = compute_multiscale_angular_metrics(line, config)
    before_probability = _probability_median(
        line, spacing, probability_raster, transform
    )
    diagnostics = _fairing_candidate_diagnostics(
        line,
        candidate,
        config,
        probability_raster,
        transform,
        spacing,
        before_metrics,
        before_probability,
    )
    initial_rejection = diagnostics["reason"]
    backoff_factor = 1.0
    fallback_method: Optional[str] = None
    fallback_sigma_m: Optional[float] = None
    acceptance = _fairing_cfg(config).get("acceptance", {})
    minimum_reduction = float(
        acceptance.get("min_backoff_short_p95_reduction_ratio", 0.20)
    )
    if initial_rejection is not None:
        sigma_values = acceptance.get(
            "conservative_gaussian_sigmas_m", (0.45, 0.40, 0.35, 0.30, 0.25)
        )
        for sigma_m in sigma_values:
            sigma_m = float(sigma_m)
            if sigma_m <= 0.0:
                continue
            conservative = _conservative_serration_candidate(
                line, sigma_m, config, sister_line
            )
            conservative_diagnostics = _fairing_candidate_diagnostics(
                line,
                conservative,
                config,
                probability_raster,
                transform,
                spacing,
                before_metrics,
                before_probability,
            )
            if (
                conservative_diagnostics["reason"] is None
                and conservative_diagnostics["short_reduction"]
                >= minimum_reduction - 1e-9
            ):
                candidate = conservative
                diagnostics = conservative_diagnostics
                fallback_method = "coordinate_gaussian"
                fallback_sigma_m = sigma_m
                break

    if (
        diagnostics["reason"] is not None
        and not primary_candidate.is_empty
        and primary_candidate.is_valid
    ):
        factors = acceptance.get(
            "backoff_factors", (0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10)
        )
        for factor in factors:
            blended = _blend_candidate_toward_original(
                line, primary_candidate, float(factor), config, sister_line
            )
            blended_diagnostics = _fairing_candidate_diagnostics(
                line,
                blended,
                config,
                probability_raster,
                transform,
                spacing,
                before_metrics,
                before_probability,
            )
            if (
                blended_diagnostics["reason"] is None
                and blended_diagnostics["short_reduction"]
                >= minimum_reduction - 1e-9
            ):
                candidate = blended
                diagnostics = blended_diagnostics
                backoff_factor = float(factor)
                fallback_method = "progressive_blend"
                break

    if diagnostics["reason"] is not None:
        before_serration = diagnostics.get("before_serration")
        after_serration = diagnostics.get("after_serration")
        return line, {
            "fairing_applied": False,
            "fairing_reason": diagnostics["reason"],
            "fairing_initial_rejection_reason": initial_rejection,
            "fairing_fallback_method": fallback_method,
            "fairing_fallback_sigma_m": fallback_sigma_m,
            "fairing_hausdorff_m": diagnostics["hausdorff_m"],
            "fairing_max_lateral_shift_m": diagnostics["lateral_shift_m"],
            "serration_before": (
                before_serration.score if before_serration is not None else None
            ),
            "serration_after": (
                after_serration.score if after_serration is not None else None
            ),
            "serration_repair_method": fallback_method or "reverted",
        }

    after_metrics = diagnostics["after_metrics"]
    maximum_lateral_shift = min(
        float(diagnostics["lateral_shift_m"]),
        float(
            _fairing_cfg(config)
            .get("geometry", {})
            .get("max_lateral_shift_m", 0.10)
        ),
    )
    candidate_coordinates = np.asarray(candidate.coords, dtype=np.float64)[:, :2]
    candidate_steps = np.linalg.norm(np.diff(candidate_coordinates, axis=0), axis=1)
    return candidate, {
        "fairing_applied": True,
        "fairing_reason": "accepted",
        "fairing_initial_rejection_reason": initial_rejection,
        "fairing_backoff_factor": backoff_factor,
        "fairing_fallback_method": fallback_method,
        "fairing_fallback_sigma_m": fallback_sigma_m,
        "fairing_hausdorff_m": diagnostics["hausdorff_m"],
        "fairing_max_lateral_shift_m": maximum_lateral_shift,
        "fairing_probability_before": before_probability,
        "fairing_probability_after": diagnostics["after_probability"],
        "fairing_short_p95_before_deg": before_metrics.short.p95_turn_deg,
        "fairing_short_p95_after_deg": after_metrics.short.p95_turn_deg,
        "fairing_short_p95_reduction": diagnostics["short_reduction"],
        "fairing_medium_p95_before_deg": before_metrics.medium.p95_turn_deg,
        "fairing_medium_p95_after_deg": after_metrics.medium.p95_turn_deg,
        "fairing_long_p95_before_deg": before_metrics.long.p95_turn_deg,
        "fairing_long_p95_after_deg": after_metrics.long.p95_turn_deg,
        "fairing_target_pair_spacing_m": viterbi_meta.get("target_pair_spacing_m"),
        "fairing_offset_smoothing_method": viterbi_meta.get(
            "offset_smoothing_method", "disabled"
        ),
        "fairing_sigma_min_m": viterbi_meta.get(
            "offset_smoothing_sigma_min_m"
        ),
        "fairing_sigma_max_m": viterbi_meta.get(
            "offset_smoothing_sigma_max_m"
        ),
        "fairing_output_spacing_median": (
            float(np.median(candidate_steps)) if len(candidate_steps) else 0.0
        ),
        "serration_before": diagnostics["before_serration"].score,
        "serration_after": diagnostics["after_serration"].score,
        "serration_repair_method": fallback_method or "regularized",
    }


def is_true(value: Any) -> bool:
    """Return truth only for explicit boolean/integer true; NaN is false."""
    if value is True or isinstance(value, np.bool_) and bool(value):
        return True
    if value is None or value is pd.NA:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        return False
    try:
        return bool(value == 1)
    except (TypeError, ValueError):
        return False


def classify_refinement_source(row: Any) -> str:
    """Classify provenance from explicit flags without treating NaN as true."""
    changes = []
    if is_true(row.get("reference_repair")):
        changes.append("reference_guided")
    if is_true(row.get("kink_repair")):
        changes.append("kink_repaired")
    if is_true(row.get("terminal_straightened")) or is_true(
        row.get("fairing_terminal_changed")
    ):
        changes.append("terminal_straightened")
    if is_true(row.get("smoothed")) or is_true(row.get("fairing_applied")):
        changes.append("smoothed")
    unique = list(dict.fromkeys(changes))
    if len(unique) > 1:
        return "mixed"
    return unique[0] if unique else "original"


def _unit_tangent(line: LineString, distance_m: float, half_window_m: float = 0.15) -> np.ndarray:
    before = max(0.0, float(distance_m) - half_window_m)
    after = min(float(line.length), float(distance_m) + half_window_m)
    p0 = np.asarray(line.interpolate(before).coords[0], dtype=np.float64)
    p1 = np.asarray(line.interpolate(after).coords[0], dtype=np.float64)
    tangent = p1 - p0
    norm = np.linalg.norm(tangent)
    return tangent / norm if norm > _EPS else np.array([1.0, 0.0], dtype=np.float64)


def _one_sided_tangent(
    line: LineString,
    distance_m: float,
    *,
    before: bool,
    window_m: float = 0.20,
) -> np.ndarray:
    if before:
        start, end = max(0.0, distance_m - window_m), distance_m
    else:
        start, end = distance_m, min(line.length, distance_m + window_m)
    p0 = np.asarray(line.interpolate(start).coords[0], dtype=np.float64)
    p1 = np.asarray(line.interpolate(end).coords[0], dtype=np.float64)
    tangent = p1 - p0
    norm = np.linalg.norm(tangent)
    return tangent / norm if norm > _EPS else _unit_tangent(line, distance_m)


def _hermite_transition(
    start: np.ndarray,
    end: np.ndarray,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    spacing_m: float,
) -> np.ndarray:
    distance = float(np.linalg.norm(end - start))
    count = max(5, int(ceil(distance / max(spacing_m, 0.05))) + 1)
    t = np.linspace(0.0, 1.0, count)
    h00 = 2.0 * t ** 3 - 3.0 * t ** 2 + 1.0
    h10 = t ** 3 - 2.0 * t ** 2 + t
    h01 = -2.0 * t ** 3 + 3.0 * t ** 2
    h11 = t ** 3 - t ** 2
    scale = max(distance, spacing_m)
    points = (
        h00[:, None] * start
        + h10[:, None] * start_tangent * scale
        + h01[:, None] * end
        + h11[:, None] * end_tangent * scale
    )
    points[0] = start
    points[-1] = end
    # Make the first and last polyline chords explicitly tangent-aligned. The
    # cubic is analytically C1, and these guards preserve that property after
    # discretisation into a LineString.
    tangent_leg = min(max(spacing_m * 0.5, 0.04), distance * 0.20)
    points[1] = start + start_tangent * tangent_leg
    points[-2] = end - end_tangent * tangent_leg
    return points


def _concat_coordinate_parts(parts: Sequence[np.ndarray]) -> LineString:
    coordinates: List[Tuple[float, float]] = []
    for part in parts:
        for point in np.asarray(part, dtype=np.float64):
            xy = (float(point[0]), float(point[1]))
            if not coordinates or Point(coordinates[-1]).distance(Point(xy)) > 1e-8:
                coordinates.append(xy)
    return LineString(coordinates) if len(coordinates) >= 2 else LineString()


def _sample_straight_segment(
    start: np.ndarray,
    end: np.ndarray,
    spacing_m: float,
) -> np.ndarray:
    length = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
    count = max(2, int(ceil(length / max(spacing_m, 0.05))) + 1)
    return np.linspace(start, end, count, dtype=np.float64)


def _substring_coords(line: LineString, start_m: float, end_m: float) -> np.ndarray:
    piece = substring(line, float(start_m), float(end_m))
    if piece.is_empty:
        return np.zeros((0, 2), dtype=np.float64)
    if isinstance(piece, Point):
        return np.asarray([piece.coords[0]], dtype=np.float64)
    return np.asarray(piece.coords, dtype=np.float64)[:, :2]


def repair_bad_interval_from_reference(
    target: LineString,
    reference: LineString,
    start_m: float,
    end_m: float,
    config: dict,
) -> Tuple[LineString, Dict[str, Any]]:
    """Replace only one bad interval using reference curvature and C1 splices."""
    start_m = max(0.0, float(start_m))
    end_m = min(float(target.length), float(end_m))
    if end_m - start_m <= 0.20:
        return target, {"splice_repaired": False, "splice_reason": "interval_too_short"}

    start = np.asarray(target.interpolate(start_m).coords[0], dtype=np.float64)
    end = np.asarray(target.interpolate(end_m).coords[0], dtype=np.float64)
    reference_start = float(reference.project(Point(start)))
    reference_end = float(reference.project(Point(end)))
    reverse_reference = reference_end < reference_start
    low, high = sorted((reference_start, reference_end))
    reference_piece = substring(reference, low, high)
    if reference_piece.is_empty or not isinstance(reference_piece, LineString):
        return target, {"splice_repaired": False, "splice_reason": "empty_reference_interval"}
    if reverse_reference:
        reference_piece = LineString(list(reference_piece.coords)[::-1])

    splice_cfg = _fairing_cfg(config).get("splice_blending", {})
    sampling_cfg = _fairing_cfg(config).get("sampling", {})
    spacing = max(float(sampling_cfg.get("longitudinal_spacing_m", 0.20)), 0.05)
    _, guide = resample_line(reference_piece, spacing)
    # Translate/warp the local template only enough to meet target endpoints.
    t = np.linspace(0.0, 1.0, len(guide))[:, None]
    guide = guide + (1.0 - t) * (start - guide[0]) + t * (end - guide[-1])
    guide[0], guide[-1] = start, end
    guide_line = LineString(guide)

    transition = min(
        float(splice_cfg.get("transition_length_m", 1.00)),
        guide_line.length * 0.35,
    )
    transition_spacing = min(
        spacing,
        float(_adaptive_output_cfg(config).get("splice_spacing_m", 0.20)),
    )
    left_end = np.asarray(guide_line.interpolate(transition).coords[0], dtype=np.float64)
    right_start = np.asarray(
        guide_line.interpolate(max(transition, guide_line.length - transition)).coords[0],
        dtype=np.float64,
    )
    target_start_tangent = _one_sided_tangent(target, start_m, before=True)
    target_end_tangent = _one_sided_tangent(target, end_m, before=False)
    guide_left_tangent = _unit_tangent(guide_line, transition)
    guide_right_tangent = _unit_tangent(guide_line, max(transition, guide_line.length - transition))
    left_blend = _hermite_transition(
        start, left_end, target_start_tangent, guide_left_tangent, transition_spacing
    )
    right_blend = _hermite_transition(
        right_start, end, guide_right_tangent, target_end_tangent, transition_spacing
    )
    guide_middle = _substring_coords(
        guide_line,
        transition,
        max(transition, guide_line.length - transition),
    )
    replacement = _concat_coordinate_parts((left_blend, guide_middle, right_blend))
    prefix = _substring_coords(target, 0.0, start_m)
    suffix = _substring_coords(target, end_m, target.length)
    candidate = _concat_coordinate_parts(
        (prefix, np.asarray(replacement.coords), suffix)
    )
    if candidate.is_empty or not candidate.is_valid or not candidate.is_simple:
        return target, {"splice_repaired": False, "splice_reason": "unsafe_geometry"}
    return candidate, {
        "splice_repaired": True,
        "splice_reason": "accepted",
        "splice_start_m": start_m,
        "splice_end_m": end_m,
        "splice_transition_m": transition,
    }


def _robust_outward_direction(
    line: LineString,
    anchor_m: float,
    fit_window_m: float,
) -> np.ndarray:
    start = max(0.0, anchor_m - fit_window_m)
    coords = _substring_coords(line, start, anchor_m)
    anchor = np.asarray(line.interpolate(anchor_m).coords[0], dtype=np.float64)
    inner = np.asarray(line.interpolate(start).coords[0], dtype=np.float64)
    expected = anchor - inner
    if len(coords) >= 2:
        centered = coords - coords.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        if np.dot(direction, expected) < 0:
            direction = -direction
    else:
        direction = expected
    norm = np.linalg.norm(direction)
    return direction / norm if norm > _EPS else np.zeros(2, dtype=np.float64)


def _boundary_intersection(
    anchor: np.ndarray,
    direction: np.ndarray,
    boundary,
    maximum_m: float,
) -> Optional[Point]:
    ray = LineString([anchor, anchor + direction * maximum_m])
    intersection = ray.intersection(boundary)
    if intersection.is_empty:
        return None
    if isinstance(intersection, Point):
        points = [intersection]
    elif isinstance(intersection, MultiPoint):
        points = list(intersection.geoms)
    elif isinstance(intersection, GeometryCollection) or hasattr(intersection, "geoms"):
        points = [geometry for geometry in intersection.geoms if isinstance(geometry, Point)]
    else:
        points = []
    anchor_point = Point(anchor)
    positive = [point for point in points if point.distance(anchor_point) > 1e-7]
    return min(positive, key=lambda point: point.distance(anchor_point)) if positive else None


def _straighten_end_c1(
    line: LineString,
    roi_polygon: Polygon,
    config: dict,
    sister_line: Optional[LineString],
    probability_raster: Optional[np.ndarray],
    transform: Optional[Affine],
) -> Tuple[LineString, Dict[str, Any]]:
    terminal_cfg = _fairing_cfg(config).get("terminal", {})
    trigger = float(terminal_cfg.get("trigger_distance_to_boundary_m", 2.0))
    endpoint = Point(line.coords[-1])
    if endpoint.distance(roi_polygon.boundary) > trigger:
        return line, {"fairing_terminal_changed": False, "fairing_terminal_reason": "far_from_boundary"}

    replace_m = min(float(terminal_cfg.get("replace_terminal_m", 2.0)), line.length)
    fit_window = float(terminal_cfg.get("direction_fit_window_m", 3.0))
    transition = min(float(terminal_cfg.get("transition_length_m", 0.50)), replace_m * 0.5)
    max_extension = float(terminal_cfg.get("max_extension_m", 2.50))
    anchor_m = max(0.0, line.length - replace_m)
    transition_start_m = max(0.0, anchor_m - transition)
    anchor = np.asarray(line.interpolate(anchor_m).coords[0], dtype=np.float64)
    direction = _robust_outward_direction(line, anchor_m, fit_window)
    if np.linalg.norm(direction) <= _EPS:
        return line, {"fairing_terminal_changed": False, "fairing_terminal_reason": "no_direction"}

    if sister_line is not None and terminal_cfg.get("use_sister_direction_when_available", True):
        sister_end = Point(sister_line.coords[-1])
        sister_start = Point(sister_line.coords[0])
        oriented_sister = sister_line
        if sister_start.distance(endpoint) < sister_end.distance(endpoint):
            oriented_sister = LineString(list(sister_line.coords)[::-1])
        sister_anchor = max(0.0, oriented_sister.length - replace_m)
        sister_direction = _robust_outward_direction(
            oriented_sister, sister_anchor, fit_window
        )
        if np.dot(sister_direction, direction) < 0:
            sister_direction = -sister_direction
        combined = direction + sister_direction
        if np.linalg.norm(combined) > _EPS:
            direction = combined / np.linalg.norm(combined)

    boundary_point = _boundary_intersection(
        anchor,
        direction,
        roi_polygon.boundary,
        replace_m + max_extension + trigger,
    )
    if boundary_point is None:
        return line, {"fairing_terminal_changed": False, "fairing_terminal_reason": "no_boundary_hit"}
    boundary_xy = np.asarray(boundary_point.coords[0], dtype=np.float64)
    extension = max(0.0, float(np.linalg.norm(boundary_xy - anchor)) - replace_m)
    endpoint_shift = float(endpoint.distance(boundary_point))
    if extension > max_extension + 1e-6 or endpoint_shift > max_extension + 1e-6:
        return line, {"fairing_terminal_changed": False, "fairing_terminal_reason": "extension_limit"}

    probability_cfg = _fairing_cfg(config).get("probability", {})
    minimum_probability = float(probability_cfg.get("min_probability", 0.12))
    straight = LineString([anchor, boundary_xy])
    straight_probability = _probability_median(
        straight, 0.10, probability_raster, transform
    )
    if np.isfinite(straight_probability) and straight_probability < minimum_probability:
        return line, {"fairing_terminal_changed": False, "fairing_terminal_reason": "low_probability"}

    blend_start = np.asarray(
        line.interpolate(transition_start_m).coords[0], dtype=np.float64
    )
    original_tangent = _unit_tangent(line, transition_start_m)
    blend = _hermite_transition(
        blend_start,
        anchor,
        original_tangent,
        direction,
        max(0.10, transition / 4.0),
    )
    prefix = _substring_coords(line, 0.0, transition_start_m)
    adaptive_cfg = _adaptive_output_cfg(config)
    if adaptive_cfg.get("enabled", True):
        straight_coords = _sample_straight_segment(
            anchor,
            boundary_xy,
            float(adaptive_cfg.get("terminal_spacing_m", 0.25)),
        )
    else:
        straight_coords = np.vstack((anchor, boundary_xy))
    candidate = _concat_coordinate_parts((prefix, blend, straight_coords))
    if candidate.is_empty or not candidate.is_valid or not candidate.is_simple:
        return line, {"fairing_terminal_changed": False, "fairing_terminal_reason": "unsafe_geometry"}
    return candidate, {
        "fairing_terminal_changed": True,
        "fairing_terminal_reason": "accepted",
        "fairing_terminal_extended_m": endpoint_shift,
        "fairing_terminal_transition_m": transition,
    }


def straighten_terminals_c1(
    line: LineString,
    roi_polygon: Polygon,
    config: dict,
    probability_raster: Optional[np.ndarray] = None,
    transform: Optional[Affine] = None,
    sister_line: Optional[LineString] = None,
) -> Tuple[LineString, Dict[str, Any]]:
    """Straighten both terminal two-metre sections with C1 transitions."""
    if not _fairing_cfg(config).get("terminal", {}).get("enabled", True):
        return line, {"fairing_terminal_changed": False, "fairing_terminal_reason": "disabled"}
    result, end_meta = _straighten_end_c1(
        line, roi_polygon, config, sister_line, probability_raster, transform
    )
    reversed_result = LineString(list(result.coords)[::-1])
    reversed_sister = (
        LineString(list(sister_line.coords)[::-1]) if sister_line is not None else None
    )
    reversed_result, start_meta = _straighten_end_c1(
        reversed_result,
        roi_polygon,
        config,
        reversed_sister,
        probability_raster,
        transform,
    )
    result = LineString(list(reversed_result.coords)[::-1])
    changed = is_true(end_meta.get("fairing_terminal_changed")) or is_true(
        start_meta.get("fairing_terminal_changed")
    )
    return result, {
        "fairing_terminal_changed": changed,
        "fairing_terminal_reason": (
            "accepted" if changed else f"start={start_meta.get('fairing_terminal_reason')};end={end_meta.get('fairing_terminal_reason')}"
        ),
        # The configured extension limit applies independently to each end;
        # report the greatest endpoint movement, not the sum of both ends.
        "fairing_terminal_extended_m": max(
            float(end_meta.get("fairing_terminal_extended_m", 0.0) or 0.0),
            float(start_meta.get("fairing_terminal_extended_m", 0.0) or 0.0),
        ),
    }


def _median_distance_to_line(line: LineString, other: LineString, spacing_m: float = 0.25) -> float:
    _, coords = resample_line(line, spacing_m)
    distances = np.asarray(
        shapely_distance(shapely_points(coords), other), dtype=np.float64
    )
    return float(np.median(distances)) if len(distances) else float("nan")


def _introduces_crossing(
    original: LineString,
    candidate: LineString,
    lines: Sequence[LineString],
    tree: STRtree,
    own_index: int,
) -> bool:
    for raw_index in tree.query(candidate, predicate="intersects"):
        index = int(raw_index)
        if index == own_index:
            continue
        other = lines[index]
        # Topology can regress through a crossing, a new touch/branch, or an
        # overlap. Revert any newly introduced network contact, not only the
        # strict Shapely ``crosses`` predicate.
        if candidate.intersects(other) and not original.intersects(other):
            return True
    return False


def _max_linear_length_within_roi(line: LineString, roi_polygon: Polygon) -> float:
    """Return the longest linear component retained by an ROI intersection."""
    if line is None or line.is_empty or roi_polygon is None or roi_polygon.is_empty:
        return 0.0
    clipped = line.intersection(roi_polygon)
    if clipped.is_empty:
        return 0.0
    if clipped.geom_type == "LineString":
        return float(clipped.length)
    if hasattr(clipped, "geoms"):
        lengths = [
            float(part.length)
            for part in clipped.geoms
            if part.geom_type == "LineString" and not part.is_empty
        ]
        return max(lengths, default=0.0)
    return 0.0


def _network_rejection_reason(
    original: LineString,
    candidate: LineString,
    originals: Sequence[LineString],
    tree: STRtree,
    own_index: int,
    sister: Optional[LineString],
    roi_polygon: Optional[Polygon],
    config: dict,
) -> Optional[str]:
    """Return a network/ROI rejection reason without mutating the candidate."""
    if _introduces_crossing(original, candidate, originals, tree, own_index):
        return "new_network_contact"
    if (
        roi_polygon is not None
        and not roi_polygon.buffer(1e-7).covers(candidate)
        and roi_polygon.buffer(1e-7).covers(original)
    ):
        return "outside_roi"
    if roi_polygon is not None:
        minimum_length = float(
            config.get("vector", {}).get("min_line_length_m", 3.0)
        )
        original_inside_length = _max_linear_length_within_roi(
            original, roi_polygon
        )
        candidate_inside_length = _max_linear_length_within_roi(
            candidate, roi_polygon
        )
        if (
            original_inside_length + _EPS >= minimum_length
            and candidate_inside_length + _EPS < minimum_length
        ):
            return "roi_clipped_too_short"
    fairing_cfg = _fairing_cfg(config)
    if sister is not None and fairing_cfg.get("geometry", {}).get(
        "preserve_pair_order", True
    ):
        before_spacing = _median_distance_to_line(original, sister)
        after_spacing = _median_distance_to_line(candidate, sister)
        maximum_change = float(
            fairing_cfg.get("geometry", {}).get(
                "max_pair_spacing_change_m", 0.10
            )
        )
        if abs(after_spacing - before_spacing) > maximum_change + 1e-9:
            return "pair_spacing_change"
    return None


def _pair_lookup(gdf: gpd.GeoDataFrame) -> Dict[int, int]:
    if "pair_id" not in gdf.columns:
        return {}
    groups: Dict[Any, List[int]] = {}
    for index, value in enumerate(gdf["pair_id"]):
        if value is None or value is pd.NA:
            continue
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            continue
        groups.setdefault(value, []).append(index)
    result: Dict[int, int] = {}
    for indices in groups.values():
        if len(indices) == 2:
            result[indices[0]] = indices[1]
            result[indices[1]] = indices[0]
    return result


def _good_line_serration_diagnostics(
    line: LineString, config: dict, transform: Optional[Affine] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Decide whether a nominally GOOD line is actually a repeated staircase."""
    override_cfg = _fairing_cfg(config).get("good_line_serration_override", {})
    if not bool(override_cfg.get("enabled", False)):
        return False, {"fairing_good_serration_override": False}

    serration = compute_serration_diagnostics(line, config, transform)
    should_override = serration.failed
    return should_override, {
        "fairing_good_serration_override": should_override,
        "fairing_good_short_p95_deg": serration.micro_p95_turn_deg,
        "fairing_good_lateral_inversions_per_m": serration.inversions_per_m,
        "serration_before": serration.score,
    }


def fair_centerlines_gdf(
    lines_gdf: gpd.GeoDataFrame,
    roi_polygon: Polygon,
    config: dict,
    probability_raster: Optional[np.ndarray] = None,
    transform: Optional[Affine] = None,
    collect_debug: bool = False,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Apply final fairing conservatively to a complete row network."""
    cfg = _fairing_cfg(config)
    if not cfg.get("enabled", False) or lines_gdf.empty:
        return lines_gdf.copy(), gpd.GeoDataFrame(geometry=[], crs=lines_gdf.crs)

    result = lines_gdf.copy().reset_index(drop=True)
    originals = result.geometry.tolist()
    tree = STRtree(originals)
    sister_by_index = _pair_lookup(result)
    preserve_good = bool(cfg.get("preserve_good_lines", True))
    geometries: List[LineString] = []
    metadata: List[Dict[str, Any]] = []
    debug_records: List[Dict[str, Any]] = []

    for index, row in result.iterrows():
        original = row.geometry
        sister_index = sister_by_index.get(index)
        sister = originals[sister_index] if sister_index is not None else None
        explicit_change = any(
            is_true(row.get(column))
            for column in (
                "reference_repair",
                "kink_repair",
                "terminal_straightened",
            )
        )
        quality = row.get("quality_after", row.get("quality_before"))
        preserve_candidate = preserve_good and quality == "GOOD" and not explicit_change
        good_override_meta: Dict[str, Any] = {}
        preserve_this_good = preserve_candidate
        if preserve_candidate:
            override_good, good_override_meta = _good_line_serration_diagnostics(
                original, config, transform
            )
            preserve_this_good = not override_good

        if preserve_this_good:
            candidate = original
            fair_meta: Dict[str, Any] = {
                "fairing_applied": False,
                "fairing_reason": "good_preserved",
                **good_override_meta,
            }
        else:
            target_spacing = (
                _median_distance_to_line(original, sister) if sister is not None else None
            )
            candidate, fair_meta = fair_line_regularized(
                original,
                config,
                probability_raster,
                transform,
                sister,
                target_spacing,
            )
            fair_meta.update(good_override_meta)

        gaussian_candidate, gaussian_meta = two_stage_gaussian_smooth_line(
            candidate,
            config,
            transform=transform,
            sister_line=sister,
        )
        if _two_stage_gaussian_cfg(config).get("enabled", False):
            spacing = _sampling_spacing(config)
            gaussian_diagnostics = _fairing_candidate_diagnostics(
                original,
                gaussian_candidate,
                config,
                probability_raster,
                transform,
                spacing,
                compute_multiscale_angular_metrics(original, config),
                _probability_median(
                    original, spacing, probability_raster, transform
                ),
                require_short_gain=False,
            )
            # The 3 m output sampling must never make a residual staircase look
            # clean.  The helper audits the internally dense curve and this gate
            # must agree before the exported candidate can be accepted.
            dense_serration_failed = bool(
                gaussian_meta.get("two_stage_gaussian_failed_after", False)
            )
            if (
                gaussian_diagnostics["reason"] is None
                and not dense_serration_failed
            ):
                candidate = gaussian_candidate
                fair_meta.update(gaussian_meta)
                fair_meta["fairing_applied"] = not candidate.equals_exact(
                    original, tolerance=1e-9
                )
                fair_meta["fairing_reason"] = (
                    "accepted_two_stage_gaussian_"
                    + str(gaussian_meta.get("two_stage_gaussian_stage", "light"))
                )
                fair_meta["fairing_hausdorff_m"] = gaussian_diagnostics[
                    "hausdorff_m"
                ]
                fair_meta["fairing_max_lateral_shift_m"] = gaussian_diagnostics[
                    "lateral_shift_m"
                ]
                fair_meta["serration_before"] = gaussian_diagnostics[
                    "before_serration"
                ].score
                fair_meta["serration_after"] = gaussian_diagnostics[
                    "after_serration"
                ].score
                fair_meta["serration_repair_method"] = (
                    "angular_gaussian_"
                    + str(gaussian_meta.get("two_stage_gaussian_stage", "light"))
                )
                preserve_this_good = False
            else:
                fair_meta.update(gaussian_meta)
                fair_meta["two_stage_gaussian_applied"] = False
                fair_meta["two_stage_gaussian_rejection_reason"] = (
                    "residual_serration_dense"
                    if dense_serration_failed
                    else gaussian_diagnostics["reason"]
                )

        fair_candidate = candidate
        if preserve_this_good:
            terminal_meta = {
                "fairing_terminal_changed": False,
                "fairing_terminal_reason": "good_preserved",
            }
        else:
            candidate, terminal_meta = straighten_terminals_c1(
                fair_candidate,
                roi_polygon,
                config,
                probability_raster,
                transform,
                sister,
            )
        combined_meta = {**fair_meta, **terminal_meta}
        rejected_reason = _network_rejection_reason(
            original,
            candidate,
            originals,
            tree,
            index,
            sister,
            roi_polygon,
            config,
        )
        if (
            rejected_reason is not None
            and is_true(terminal_meta.get("fairing_terminal_changed"))
            and is_true(fair_meta.get("fairing_applied"))
        ):
            fair_rejection = _network_rejection_reason(
                original,
                fair_candidate,
                originals,
                tree,
                index,
                sister,
                roi_polygon,
                config,
            )
            if fair_rejection is None:
                candidate = fair_candidate
                combined_meta["fairing_terminal_changed"] = False
                combined_meta["fairing_terminal_rejection_reason"] = rejected_reason
                combined_meta["fairing_terminal_reason"] = (
                    f"rejected_{rejected_reason}"
                )
                combined_meta["fairing_terminal_extended_m"] = 0.0
                rejected_reason = None
        if (
            rejected_reason == "pair_spacing_change"
            and sister is not None
            and is_true(fair_meta.get("fairing_applied"))
        ):
            # Independent smoothing can move one member of a valid double row
            # just beyond the spacing guard.  Retry progressively toward the
            # original while checking both the local serration gate and the
            # pair/network gates at every step.  This preserves the strict
            # 0.10 m limit instead of disabling it or accepting a bad arc.
            spacing = _sampling_spacing(config)
            before_metrics = compute_multiscale_angular_metrics(original, config)
            before_probability = _probability_median(
                original, spacing, probability_raster, transform
            )
            pair_factors = cfg.get("acceptance", {}).get(
                "pair_spacing_backoff_factors",
                (0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20),
            )
            for factor in pair_factors:
                pair_candidate = _blend_candidate_toward_original(
                    original,
                    candidate,
                    float(factor),
                    config,
                    sister,
                )
                local_diagnostics = _fairing_candidate_diagnostics(
                    original,
                    pair_candidate,
                    config,
                    probability_raster,
                    transform,
                    spacing,
                    before_metrics,
                    before_probability,
                )
                if local_diagnostics["reason"] is not None:
                    continue
                pair_rejection = _network_rejection_reason(
                    original,
                    pair_candidate,
                    originals,
                    tree,
                    index,
                    sister,
                    roi_polygon,
                    config,
                )
                if pair_rejection is not None:
                    continue
                candidate = pair_candidate
                rejected_reason = None
                combined_meta["fairing_backoff_factor"] = float(factor)
                combined_meta["fairing_fallback_method"] = "pair_spacing_backoff"
                combined_meta["fairing_reason"] = "accepted_pair_spacing_backoff"
                combined_meta["fairing_hausdorff_m"] = local_diagnostics[
                    "hausdorff_m"
                ]
                combined_meta["fairing_max_lateral_shift_m"] = local_diagnostics[
                    "lateral_shift_m"
                ]
                combined_meta["serration_after"] = local_diagnostics[
                    "after_serration"
                ].score
                combined_meta["serration_repair_method"] = "pair_spacing_backoff"
                break
        if rejected_reason is not None:
            candidate = original
            combined_meta["fairing_applied"] = False
            combined_meta["fairing_terminal_changed"] = False
            combined_meta["fairing_reason"] = rejected_reason

        geometries.append(candidate)
        metadata.append(combined_meta)
        if collect_debug and not candidate.equals_exact(original, tolerance=1e-9):
            debug_records.append(
                {
                    "line_index": index,
                    "action": "final_centerline_fairing",
                    **combined_meta,
                    "geometry": candidate,
                }
            )

    result.geometry = geometries
    column_order = (
        "fairing_applied",
        "fairing_reason",
        "fairing_initial_rejection_reason",
        "fairing_backoff_factor",
        "fairing_fallback_method",
        "fairing_fallback_sigma_m",
        "fairing_good_serration_override",
        "fairing_good_short_p95_deg",
        "fairing_good_lateral_inversions_per_m",
        "fairing_hausdorff_m",
        "fairing_max_lateral_shift_m",
        "fairing_probability_before",
        "fairing_probability_after",
        "fairing_short_p95_before_deg",
        "fairing_short_p95_after_deg",
        "fairing_short_p95_reduction",
        "fairing_medium_p95_before_deg",
        "fairing_medium_p95_after_deg",
        "fairing_long_p95_before_deg",
        "fairing_long_p95_after_deg",
        "fairing_target_pair_spacing_m",
        "fairing_offset_smoothing_method",
        "fairing_sigma_min_m",
        "fairing_sigma_max_m",
        "fairing_output_spacing_median",
        "serration_before",
        "serration_after",
        "serration_repair_method",
        "two_stage_gaussian_applied",
        "two_stage_gaussian_stage",
        "two_stage_gaussian_reason",
        "two_stage_gaussian_rejection_reason",
        "two_stage_gaussian_aggressive_attempted",
        "two_stage_gaussian_light_failed",
        "two_stage_gaussian_failed_after",
        "two_stage_gaussian_sampling_m",
        "two_stage_gaussian_sigma_min_m",
        "two_stage_gaussian_sigma_max_m",
        "two_stage_gaussian_score_before",
        "two_stage_gaussian_score_light",
        "two_stage_gaussian_score_after",
        "two_stage_gaussian_micro_p95_after_deg",
        "two_stage_gaussian_vertex_p95_after_deg",
        "two_stage_gaussian_inversions_after_per_m",
        "two_stage_gaussian_residual_p95_after_m",
        "two_stage_gaussian_length_excess_after",
        "two_stage_gaussian_vertices_before",
        "two_stage_gaussian_vertices_after",
        "fairing_terminal_changed",
        "fairing_terminal_reason",
        "fairing_terminal_rejection_reason",
        "fairing_terminal_extended_m",
    )
    present = {key for item in metadata for key in item}
    for column in column_order:
        if column in present:
            result[column] = [item.get(column) for item in metadata]

    changed_indices = [
        index
        for index, (before, after) in enumerate(zip(originals, geometries))
        if not after.equals_exact(before, tolerance=1e-9)
    ]
    if changed_indices and "quality_after" in result.columns:
        # Local import avoids a module-import cycle while keeping the existing
        # quality classification authoritative after geometry changes.
        from .double_row_refinement import evaluate_line_quality

        for index in changed_indices:
            quality = evaluate_line_quality(
                geometries[index], config, probability_raster, transform
            )
            result.at[index, "quality_after"] = quality.quality_class
            if "max_turn_after_deg" in result.columns:
                result.at[index, "max_turn_after_deg"] = quality.max_turn_deg
            if "p95_turn_after_deg" in result.columns:
                result.at[index, "p95_turn_after_deg"] = quality.p95_turn_deg
            if "review_required" in result.columns:
                result.at[index, "review_required"] = quality.quality_class != "GOOD"
    result["refinement_source"] = result.apply(classify_refinement_source, axis=1)

    debug = (
        gpd.GeoDataFrame(debug_records, geometry="geometry", crs=result.crs)
        if debug_records
        else gpd.GeoDataFrame(geometry=[], crs=result.crs)
    )
    return result, debug
