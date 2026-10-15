"""Deterministic georeferenced scenes for geometry and integration tests.

Synthetic scenes validate plumbing and topology; they are not evidence of
agricultural accuracy and must never be reported as real model validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import CRS
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import LineString, box
from skimage.draw import line as raster_line

from rowgraphnet.domain import RasterPrediction


@dataclass(slots=True)
class SyntheticScene:
    image: np.ndarray
    prediction: RasterPrediction
    ground_truth_rows: list[LineString]
    roi: object
    full_center: np.ndarray
    observed_center: np.ndarray
    gap_mask: np.ndarray


def _densify_polyline(
    x: np.ndarray, y: np.ndarray, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    pixels: list[tuple[int, int]] = []
    for index in range(len(x) - 1):
        rr, cc = raster_line(
            round(y[index]),
            round(x[index]),
            round(y[index + 1]),
            round(x[index + 1]),
        )
        for row, col in zip(rr, cc, strict=True):
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                pixel = (int(row), int(col))
                if not pixels or pixels[-1] != pixel:
                    pixels.append(pixel)
    return pixels


def _to_world(transform: Affine, pixels: list[tuple[int, int]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for row, col in pixels:
        x, y = transform * (col + 0.5, row + 0.5)
        result.append((float(x), float(y)))
    return result


def generate_synthetic_scene(
    kind: str = "s_curve",
    shape: tuple[int, int] = (192, 256),
    pixel_size_m: float = 0.10,
    spacing_px: float = 24.0,
    gap_length_spacing_fraction: float = 0.45,
    with_gaps: bool = True,
    with_false_linear_object: bool = False,
    with_nodata: bool = False,
    seed: int = 2026,
) -> SyntheticScene:
    """Create oracle-like multitask maps for controlled end-to-end tests."""

    if kind not in {"straight", "curve", "s_curve", "variable_spacing", "two_regimes"}:
        raise ValueError(f"unsupported synthetic scene kind: {kind}")
    if min(shape) < 32 or pixel_size_m <= 0 or spacing_px <= 2 or gap_length_spacing_fraction <= 0:
        raise ValueError("shape, pixel size, and spacing must define a measurable scene")
    rng = np.random.default_rng(seed)
    height, width = shape
    transform = Affine(pixel_size_m, 0.0, 500_000.0, 0.0, -pixel_size_m, 7_400_000.0)
    crs = CRS.from_epsg(31983)
    margin = max(round(spacing_px), 4)
    x_values = np.arange(margin, width - margin, dtype=float)
    base_rows = np.arange(margin, height - margin, spacing_px, dtype=float)
    full_binary = np.zeros(shape, dtype=bool)
    sin_accumulator = np.zeros(shape, dtype=float)
    cos_accumulator = np.zeros(shape, dtype=float)
    orientation_weight = np.zeros(shape, dtype=float)
    ground_truth: list[LineString] = []
    row_pixels: list[list[tuple[int, int]]] = []

    for row_index, base in enumerate(base_rows):
        phase_x = (x_values - x_values.min()) / max(float(np.ptp(x_values)), 1.0)
        if kind == "straight":
            y_values = np.full_like(x_values, base)
        elif kind == "curve":
            y_values = base + 0.00045 * (x_values - width / 2.0) ** 2
        elif kind == "s_curve":
            y_values = base + spacing_px * 0.18 * np.sin(2.0 * np.pi * phase_x)
        elif kind == "variable_spacing":
            scale = 1.0 + 0.25 * (phase_x - 0.5)
            y_values = margin + row_index * spacing_px * scale
        else:  # two_regimes
            half = len(base_rows) // 2
            if row_index < half:
                y_values = np.full_like(x_values, base * 0.70)
                x_local = x_values[x_values < width * 0.48]
                y_values = y_values[: len(x_local)]
            else:
                x_local = x_values[x_values > width * 0.52]
                y_values = base + 0.32 * (x_local - width * 0.52)
            pixels = _densify_polyline(x_local, y_values, shape)
            if len(pixels) < 2:
                continue
            slopes = np.gradient(y_values, x_local)
            row_pixels.append(pixels)
            for pixel in pixels:
                full_binary[pixel] = True
                theta = np.arctan(float(np.interp(pixel[1], x_local, slopes)))
                sin_accumulator[pixel] += np.sin(2.0 * theta)
                cos_accumulator[pixel] += np.cos(2.0 * theta)
                orientation_weight[pixel] += 1.0
            ground_truth.append(LineString(_to_world(transform, pixels)))
            continue

        pixels = _densify_polyline(x_values, y_values, shape)
        if len(pixels) < 2:
            continue
        slopes = np.gradient(y_values, x_values)
        row_pixels.append(pixels)
        for pixel in pixels:
            full_binary[pixel] = True
            theta = np.arctan(float(np.interp(pixel[1], x_values, slopes)))
            sin_accumulator[pixel] += np.sin(2.0 * theta)
            cos_accumulator[pixel] += np.cos(2.0 * theta)
            orientation_weight[pixel] += 1.0
        ground_truth.append(LineString(_to_world(transform, pixels)))

    full_center = gaussian_filter(full_binary.astype(float), sigma=0.8)
    if full_center.max() > 0:
        full_center /= full_center.max()
    gap_mask = np.zeros(shape, dtype=bool)
    endpoints = np.zeros(shape, dtype=float)
    if with_gaps:
        gap_width = max(round(spacing_px * gap_length_spacing_fraction), 2)
        for row_index, pixels in enumerate(row_pixels):
            columns = [pixel[1] for pixel in pixels]
            relative_position = 0.42 + 0.14 * ((row_index % 3) / 2.0)
            center_col = round(min(columns) + relative_position * (max(columns) - min(columns)))
            selected = [pixel for pixel in pixels if abs(pixel[1] - center_col) <= gap_width // 2]
            for pixel in selected:
                gap_mask[pixel] = True
        gap_mask = gaussian_filter(gap_mask.astype(float), sigma=1.0) > 0.05
    for pixels in row_pixels:
        endpoints[pixels[0]] = 1.0
        endpoints[pixels[-1]] = 1.0
    endpoints = gaussian_filter(endpoints, sigma=1.0)
    if endpoints.max() > 0:
        endpoints /= endpoints.max()

    observed_center = full_center.copy()
    observed_center[gap_mask] *= 0.02
    weak_noise = gaussian_filter(rng.random(shape), sigma=3.0)
    weak_noise = weak_noise / max(float(weak_noise.max()), np.finfo(float).eps)
    observed_center = np.clip(observed_center * 0.94 + weak_noise * 0.02, 0.0, 1.0)

    if with_false_linear_object:
        false_row = int(np.clip(height / 2 + spacing_px / 2, 0, height - 1))
        false_mask = np.zeros(shape, dtype=float)
        false_mask[false_row, margin : width - margin] = 1.0
        false_response = gaussian_filter(false_mask, sigma=1.2)
        false_response /= max(float(false_response.max()), np.finfo(float).eps)
        observed_center = np.maximum(observed_center, false_response * 0.38)

    # Propagate the closest row tangent to background pixels.  Confidence will
    # be low there, but A* still has a well-defined direction through a gap.
    _, nearest = distance_transform_edt(~full_binary, return_indices=True)
    row_sin = np.divide(
        sin_accumulator,
        np.maximum(orientation_weight, 1.0),
    )
    row_cos = np.divide(
        cos_accumulator,
        np.maximum(orientation_weight, 1.0),
    )
    sin2 = row_sin[nearest[0], nearest[1]]
    cos2 = row_cos[nearest[0], nearest[1]]
    norm = np.maximum(np.hypot(sin2, cos2), np.finfo(float).eps)
    sin2 /= norm
    cos2 /= norm

    band = gaussian_filter(full_binary.astype(float), sigma=max(spacing_px * 0.10, 1.0))
    band /= max(float(band.max()), np.finfo(float).eps)
    band[gap_mask] *= 0.20
    distance_px = distance_transform_edt(~full_binary)
    distance_m = distance_px * pixel_size_m
    uncertainty = np.clip(0.10 + 0.65 * gap_mask + 0.20 * (1.0 - band), 0.0, 1.0)

    valid_mask = np.ones(shape, dtype=bool)
    if with_nodata:
        nodata_height = max(height // 8, 1)
        nodata_width = max(width // 10, 1)
        valid_mask[:nodata_height, :nodata_width] = False
        observed_center[~valid_mask] = 0.0
        band[~valid_mask] = 0.0
        uncertainty[~valid_mask] = 1.0

    soil = np.array([0.48, 0.34, 0.22], dtype=np.float32)[:, None, None]
    vegetation = np.array([0.16, 0.55, 0.13], dtype=np.float32)[:, None, None]
    image = soil * (1.0 - band[None]) + vegetation * band[None]
    image += rng.normal(0.0, 0.025, size=image.shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    image[:, ~valid_mask] = 0.0

    prediction = RasterPrediction(
        center=observed_center.astype(np.float32),
        band=band.astype(np.float32),
        orientation_sin2=sin2.astype(np.float32),
        orientation_cos2=cos2.astype(np.float32),
        endpoints=endpoints.astype(np.float32),
        distance=distance_m.astype(np.float32),
        uncertainty=uncertainty.astype(np.float32),
        transform=transform,
        crs=crs,
        valid_mask=valid_mask,
    )
    left, bottom, right, top = (
        transform.c,
        transform.f + transform.e * height,
        transform.c + transform.a * width,
        transform.f,
    )
    return SyntheticScene(
        image=image,
        prediction=prediction,
        ground_truth_rows=ground_truth,
        roi=box(left, bottom, right, top),
        full_center=full_center.astype(np.float32),
        observed_center=observed_center.astype(np.float32),
        gap_mask=gap_mask,
    )
