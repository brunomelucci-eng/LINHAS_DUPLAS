"""Conservative conversion of continuous center evidence into row segments."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable

import numpy as np
from affine import Affine
from pyproj import CRS
from shapely.geometry import LineString
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize

from rowgraphnet.domain import ExtractedSegment, RasterPrediction

Pixel = tuple[int, int]


def _is_metric_crs(crs: CRS) -> bool:
    if not crs.is_projected:
        return False
    axis_info = crs.axis_info
    return bool(axis_info) and all(
        axis.unit_name.lower() in {"metre", "meter"} for axis in axis_info
    )


def adaptive_hysteresis(
    probability: np.ndarray,
    valid_mask: np.ndarray | None = None,
    low_quantile: float = 0.90,
    high_quantile: float = 0.98,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Apply hysteresis with thresholds estimated from the current scene.

    Quantiles are model-calibration parameters, never physical row rules.  A
    caller may replace them with values selected on a validation set.
    """

    values = np.asarray(probability, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("probability must be a 2-D array")
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("expected 0 <= low_quantile < high_quantile <= 1")
    valid = np.isfinite(values)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must match probability")
        valid &= mask
    if not np.any(valid):
        return np.zeros(values.shape, dtype=bool), (float("nan"), float("nan"))

    # Para imagens grandes, subamostrar para estimar quantis sem duplicar centenas de milhões de floats
    if values.size > 1_000_000:
        step = max(1, int(math.sqrt(values.size / 500_000)))
        sub_valid = valid[::step, ::step]
        sample = values[::step, ::step][sub_valid]
        if len(sample) == 0:
            sample = values[valid]
    else:
        sample = values[valid]

    low = float(np.quantile(sample, low_quantile))
    high = float(np.quantile(sample, high_quantile))
    # Quantized or mostly empty maps can yield equal quantiles.  Separate the
    # thresholds using unique evidence levels instead of an arbitrary value.
    if high <= low:
        unique = np.unique(sample)
        if len(unique) == 1:
            if unique[0] <= 0:
                return np.zeros(values.shape, dtype=bool), (low, high)
            high = float(np.nextafter(np.float32(unique[0]), np.float32(-np.inf)))
            low = high
        else:
            # ``apply_hysteresis_threshold`` uses a strict comparison for the
            # strong seed.  Move the common plateau threshold by one ULP so
            # the observed maximum remains a seed without introducing a
            # physical epsilon or admitting the lower quantized level.
            high = float(np.nextafter(np.float32(unique[-1]), np.float32(-np.inf)))
            low = high
    elif high >= float(np.max(sample)):
        high = float(np.nextafter(np.float32(high), np.float32(-np.inf)))
    masked_input = np.where(valid, values, np.float32(-np.inf))
    binary = apply_hysteresis_threshold(masked_input, low, high)
    del masked_input
    binary &= valid
    return binary, (low, high)


def _pixel_neighbors(pixel: Pixel, skeleton: np.ndarray) -> list[Pixel]:
    row, col = pixel
    result: list[Pixel] = []
    row_min, row_max = max(0, row - 1), min(skeleton.shape[0], row + 2)
    col_min, col_max = max(0, col - 1), min(skeleton.shape[1], col + 2)
    for rr in range(row_min, row_max):
        for cc in range(col_min, col_max):
            if (rr, cc) != pixel and skeleton[rr, cc]:
                result.append((rr, cc))
    return result


def _trace_skeleton(skeleton: np.ndarray) -> tuple[list[list[Pixel]], set[Pixel]]:
    pixels = [tuple(value) for value in np.argwhere(skeleton)]
    neighbors = {pixel: _pixel_neighbors(pixel, skeleton) for pixel in pixels}
    nodes = {pixel for pixel, adjacent in neighbors.items() if len(adjacent) != 2}
    uncertain_nodes = {pixel for pixel, adjacent in neighbors.items() if len(adjacent) > 2}
    visited_edges: set[frozenset[Pixel]] = set()
    chains: list[list[Pixel]] = []

    def walk(start: Pixel, nxt: Pixel) -> list[Pixel]:
        chain = [start]
        previous, current = start, nxt
        visited_edges.add(frozenset((previous, current)))
        while True:
            chain.append(current)
            if current in nodes and current != start:
                break
            candidates = [item for item in neighbors[current] if item != previous]
            if not candidates:
                break
            following = candidates[0]
            edge = frozenset((current, following))
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            previous, current = current, following
        return chain

    for node in sorted(nodes):
        for adjacent in sorted(neighbors[node]):
            edge = frozenset((node, adjacent))
            if edge not in visited_edges:
                chain = walk(node, adjacent)
                if len(chain) >= 2:
                    chains.append(chain)

    # A closed component has degree two everywhere and therefore no natural
    # endpoint.  Trace it once and mark it uncertain for downstream review.
    for pixel in sorted(pixels):
        for adjacent in sorted(neighbors[pixel]):
            edge = frozenset((pixel, adjacent))
            if edge not in visited_edges:
                chain = walk(pixel, adjacent)
                if len(chain) >= 2:
                    chains.append(chain)
                    uncertain_nodes.update(chain)
    return chains, uncertain_nodes


def _world_xy(transform: Affine, pixel: Pixel) -> tuple[float, float]:
    row, col = pixel
    x, y = transform * (col + 0.5, row + 0.5)
    return float(x), float(y)


def _stable_segment_id(coords: Iterable[tuple[float, float]]) -> str:
    rounded = [(round(x, 6), round(y, 6)) for x, y in coords]
    forward = repr(rounded)
    reverse = repr(list(reversed(rounded)))
    canonical = min(forward, reverse).encode("utf-8")
    return "seg_" + hashlib.sha256(canonical).hexdigest()[:16]


def _local_maximum(values: np.ndarray, pixel: Pixel, radius: int) -> float:
    row, col = pixel
    row_slice = slice(max(row - radius, 0), min(row + radius + 1, values.shape[0]))
    col_slice = slice(max(col - radius, 0), min(col + radius + 1, values.shape[1]))
    return float(np.nanmax(values[row_slice, col_slice]))


def extract_segments(
    prediction: RasterPrediction,
    low_quantile: float = 0.90,
    high_quantile: float = 0.98,
    minimum_pixels: int | None = None,
) -> tuple[list[ExtractedSegment], dict[str, float | int]]:
    """Extract observed, unbranched centerline segments.

    ``minimum_pixels`` is a raster sampling guard.  If omitted, isolated
    one-pixel components are rejected while every measurable chain is kept.
    """

    if not _is_metric_crs(CRS.from_user_input(prediction.crs)):
        raise ValueError("segment extraction requires a projected CRS with metre axes")
    binary, thresholds = adaptive_hysteresis(
        prediction.center,
        prediction.valid_mask,
        low_quantile,
        high_quantile,
    )
    skeleton = skeletonize(binary)
    chains, uncertain_nodes = _trace_skeleton(skeleton)
    if minimum_pixels is None:
        # The average foreground width is inferred from this very mask.  Tiny
        # skeleton spurs shorter than that width are rasterization artifacts,
        # not measurable row segments.
        inferred_width = binary.sum() / max(int(skeleton.sum()), 1)
        # A chain shorter than two of its own measured ridge widths cannot
        # establish a stable tangent and is retained only as raster evidence,
        # not promoted to a vector segment.
        minimum = max(math.ceil(2.0 * inferred_width), 2)
    else:
        minimum = max(int(minimum_pixels), 2)
    retained_chains = [chain for chain in chains if len(chain) >= minimum]
    endpoint_radius = max(math.ceil(minimum / 2.0), 1)
    endpoint_incidence = Counter(
        endpoint for chain in retained_chains for endpoint in (chain[0], chain[-1])
    )
    segments: list[ExtractedSegment] = []
    threshold_low, threshold_high = thresholds

    for chain in retained_chains:
        coords = [_world_xy(prediction.transform, pixel) for pixel in chain]
        # Repeated world coordinates can only arise from a degenerate affine.
        if len(set(coords)) < 2:
            continue
        geometry = LineString(coords)
        values = np.array([prediction.center[pixel] for pixel in chain], dtype=float)
        uncertainties = np.array([prediction.uncertainty[pixel] for pixel in chain], dtype=float)
        endpoints = (
            _local_maximum(prediction.endpoints, chain[0], endpoint_radius),
            _local_maximum(prediction.endpoints, chain[-1], endpoint_radius),
        )
        segments.append(
            ExtractedSegment(
                segment_id=_stable_segment_id(coords),
                geometry=geometry,
                confidence=float(np.nanmean(values)),
                observed_length_m=float(geometry.length),
                mean_uncertainty=float(np.nanmean(uncertainties)),
                endpoint_confidence=endpoints,
                attrs={
                    "touches_junction": bool(
                        endpoint_incidence[chain[0]] > 2
                        or endpoint_incidence[chain[-1]] > 2
                        or chain[0] == chain[-1]
                    ),
                    "pixel_count": len(chain),
                },
            )
        )
    segments.sort(key=lambda item: item.segment_id)
    diagnostics: dict[str, float | int] = {
        "low_threshold": threshold_low,
        "high_threshold": threshold_high,
        "foreground_pixels": int(binary.sum()),
        "skeleton_pixels": int(skeleton.sum()),
        "segments": len(segments),
        "junction_pixels": len(uncertain_nodes),
        "minimum_segment_pixels": minimum,
    }
    return segments, diagnostics


def segment_endpoint_index(
    segments: Iterable[ExtractedSegment],
) -> dict[tuple[str, int], tuple[float, float]]:
    """Return a deterministic lookup used by candidate generation."""

    result: dict[tuple[str, int], tuple[float, float]] = {}
    for segment in segments:
        points = segment.endpoints
        result[(segment.segment_id, 0)] = (points[0].x, points[0].y)
        result[(segment.segment_id, 1)] = (points[1].x, points[1].y)
    return result
