"""Deterministic overlapping windows and normalized Hann mosaicking."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from rasterio.windows import Window


@dataclass(frozen=True, slots=True)
class TileWindow:
    """Integer pixel window with a stable row-major identifier."""

    tile_id: int
    row_off: int
    col_off: int
    height: int
    width: int

    @property
    def row_slice(self) -> slice:
        return slice(self.row_off, self.row_off + self.height)

    @property
    def col_slice(self) -> slice:
        return slice(self.col_off, self.col_off + self.width)

    @property
    def slices(self) -> tuple[slice, slice]:
        return self.row_slice, self.col_slice

    def as_rasterio_window(self) -> Window:
        return Window(self.col_off, self.row_off, self.width, self.height)


def _pair(value: int | tuple[int, int], *, name: str, allow_zero: bool) -> tuple[int, int]:
    values = (value, value) if isinstance(value, int) and not isinstance(value, bool) else value
    if not isinstance(values, tuple) or len(values) != 2:
        raise ValueError(f"{name} must be an integer or a (height, width) tuple")
    if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in values):
        raise ValueError(f"{name} values must be integers")
    normalized = int(values[0]), int(values[1])
    lower_bound = 0 if allow_zero else 1
    if any(item < lower_bound for item in normalized):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} values must be {qualifier}")
    return normalized


def _axis_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile >= length:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile + 1, step))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def generate_tiles(
    image_shape: tuple[int, int],
    tile_size: int | tuple[int, int],
    overlap: int | tuple[int, int] = 0,
) -> list[TileWindow]:
    """Generate complete, in-bounds coverage with deterministic overlap.

    The final tile on each axis is anchored to the raster boundary.  This
    avoids tiny remainder tiles; its overlap may consequently be larger than
    requested, but never smaller.
    """

    height, width = _pair(image_shape, name="image_shape", allow_zero=False)
    tile_height, tile_width = _pair(tile_size, name="tile_size", allow_zero=False)
    overlap_height, overlap_width = _pair(overlap, name="overlap", allow_zero=True)
    if overlap_height >= tile_height or overlap_width >= tile_width:
        raise ValueError("overlap must be smaller than tile_size on both axes")

    row_starts = _axis_starts(height, tile_height, overlap_height)
    col_starts = _axis_starts(width, tile_width, overlap_width)
    windows: list[TileWindow] = []
    for row_off in row_starts:
        for col_off in col_starts:
            windows.append(
                TileWindow(
                    tile_id=len(windows),
                    row_off=row_off,
                    col_off=col_off,
                    height=min(tile_height, height - row_off),
                    width=min(tile_width, width - col_off),
                )
            )
    return windows


def _pixel_centred_hann(length: int) -> np.ndarray:
    # Sampling between the mathematical endpoints keeps every raster pixel at
    # positive weight, including outer image boundaries and 1-pixel tiles.
    positions = (np.arange(length, dtype=np.float64) + 0.5) / length
    weights = np.sin(np.pi * positions) ** 2
    return weights / weights.max()


def hann_blend_weights(
    shape: tuple[int, int],
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Return a strictly-positive, symmetric, separable 2-D Hann window."""

    height, width = _pair(shape, name="shape", allow_zero=False)
    weights = np.outer(_pixel_centred_hann(height), _pixel_centred_hann(width))
    return weights.astype(dtype, copy=False)


class HannBlender:
    """Accumulate 2-D or channel-first tiles and normalize their Hann weights."""

    def __init__(self, output_shape: tuple[int, ...], *, dtype: np.dtype | type = np.float64):
        if len(output_shape) not in {2, 3} or any(
            isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1
            for size in output_shape
        ):
            raise ValueError("output_shape must be (height, width) or (channels, height, width)")
        self.output_shape = tuple(int(size) for size in output_shape)
        self._sum = np.zeros(self.output_shape, dtype=dtype)
        self._weight_sum = np.zeros(self.output_shape[-2:], dtype=dtype)

    @property
    def weight_sum(self) -> np.ndarray:
        """Current spatial weight sum, exposed read-only by convention."""

        return self._weight_sum

    def add(
        self,
        tile: np.ndarray,
        window: TileWindow,
        *,
        weights: np.ndarray | None = None,
    ) -> None:
        """Add one tile at ``window`` after strict shape/bounds validation."""

        array = np.asarray(tile)
        if window.height < 1 or window.width < 1:
            raise ValueError("tile window dimensions must be positive")
        expected = (*self.output_shape[:-2], window.height, window.width)
        if array.shape != expected:
            raise ValueError(f"tile has shape {array.shape}; expected {expected}")
        out_height, out_width = self.output_shape[-2:]
        if (
            window.row_off < 0
            or window.col_off < 0
            or window.row_off + window.height > out_height
            or window.col_off + window.width > out_width
        ):
            raise ValueError("tile window lies outside output_shape")
        spatial_weights = (
            hann_blend_weights((window.height, window.width), dtype=self._sum.dtype)
            if weights is None
            else np.asarray(weights, dtype=self._sum.dtype)
        )
        if spatial_weights.shape != (window.height, window.width):
            raise ValueError("weights must match the tile's spatial shape")
        if not np.isfinite(spatial_weights).all() or np.any(spatial_weights <= 0.0):
            raise ValueError("weights must be finite and strictly positive")
        row_slice, col_slice = window.slices
        self._sum[..., row_slice, col_slice] += array * spatial_weights
        self._weight_sum[row_slice, col_slice] += spatial_weights

    def finalize(self, *, dtype: np.dtype | type = np.float32) -> np.ndarray:
        """Return the normalized mosaic, failing if any output pixel is uncovered."""

        if np.any(self._weight_sum <= 0.0):
            uncovered = int(np.count_nonzero(self._weight_sum <= 0.0))
            raise ValueError(f"cannot finalize mosaic: {uncovered} pixels are uncovered")
        result = self._sum / self._weight_sum
        return result.astype(dtype, copy=False)


def blend_tiles(
    tiles: Iterable[tuple[TileWindow, np.ndarray]],
    output_shape: tuple[int, ...],
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Convenience wrapper for normalized Hann mosaicking."""

    blender = HannBlender(output_shape)
    for window, tile in tiles:
        blender.add(tile, window)
    return blender.finalize(dtype=dtype)
