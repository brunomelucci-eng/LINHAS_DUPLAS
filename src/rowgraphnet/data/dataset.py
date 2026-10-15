"""Lazy geospatial training dataset for the RowGraphNet raster heads.

Scenes are never materialized as a reprojected full raster. Each sample is
read through a metric :class:`rasterio.vrt.WarpedVRT`, and annotations are
rasterized only for the requested tile. Memory therefore stays bounded by the
tile size even for very large orthomosaics.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.errors
import torch
from affine import Affine
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import box, mapping
from torch.utils.data import DataLoader, Dataset

from rowgraphnet.data.contracts import DataContractError, ValidatedScene, validate_scene
from rowgraphnet.data.manifest import SceneManifest, load_scene_manifest
from rowgraphnet.data.normalization import InputNormalization
from rowgraphnet.data.rasterize import RasterizedAnnotations, rasterize_row_annotations

warnings.filterwarnings("ignore", category=rasterio.errors.NodataShadowWarning)


@dataclass(frozen=True, slots=True)
class _MetricGrid:
    transform: Affine
    width: int
    height: int


@dataclass(slots=True)
class _LazyScene:
    manifest: SceneManifest
    validated: ValidatedScene
    grid: _MetricGrid
    annotations: gpd.GeoDataFrame
    has_visibility_labels: bool


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _metric_grid(scene: ValidatedScene) -> _MetricGrid:
    """Return the deterministic north-up square-pixel grid for a scene ROI."""

    resolution = float(scene.raster.gsd.mean_m)
    if not math.isfinite(resolution) or resolution <= 0:
        raise DataContractError("cannot build metric grid from a non-positive GSD")
    minimum_x, minimum_y, maximum_x, maximum_y = scene.roi_in_metric_crs.bounds
    left = math.floor(minimum_x / resolution) * resolution
    bottom = math.floor(minimum_y / resolution) * resolution
    right = math.ceil(maximum_x / resolution) * resolution
    top = math.ceil(maximum_y / resolution) * resolution
    width = max(math.ceil((right - left) / resolution), 1)
    height = max(math.ceil((top - bottom) / resolution), 1)
    return _MetricGrid(
        transform=Affine(resolution, 0.0, left, 0.0, -resolution, top),
        width=width,
        height=height,
    )


def _tile_starts(length: int, tile_size: int, stride: int) -> tuple[int, ...]:
    """Cover an axis completely, anchoring the final tile at its far edge."""

    if length <= tile_size:
        return (0,)
    final = length - tile_size
    starts = list(range(0, final + 1, stride))
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def _read_vector(path: Path, layer: str | None) -> gpd.GeoDataFrame:
    kwargs = {"layer": layer} if layer is not None else {}
    try:
        frame = gpd.read_file(path, **kwargs)
    except Exception as exc:
        label = f" layer {layer!r}" if layer is not None else ""
        raise DataContractError(f"could not read {path}{label}") from exc
    if frame.crs is None:
        raise DataContractError(f"vector input must define a CRS: {path}")
    return frame


def _normalization_mapping(
    normalization: Mapping[str, Any] | None,
    normalize_scale: float | None,
) -> dict[str, Any]:
    if normalization is not None and normalize_scale is not None:
        raise ValueError("pass normalization or normalize_scale, not both")
    if normalization is not None:
        result = dict(normalization)
        if result.get("scale_integer_to_unit", True) and result.get("scale") is None:
            raise ValueError(
                "lazy WarpedVRT tiles are float32; normalization must provide an explicit "
                "scale (for uint8, use 1/255) and set scale_integer_to_unit=false"
            )
    else:
        scale = 255.0 if normalize_scale is None else float(normalize_scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("normalize_scale must be finite and positive")
        # WarpedVRT emits float32, so the scale must be explicit. Relying on
        # scale_integer_to_unit would make training and inference disagree.
        result = {"scale": 1.0 / scale, "scale_integer_to_unit": False}
    InputNormalization.from_metadata(result)
    return result


def compute_axial_orientation_maps(
    center_mask: np.ndarray,
    valid_mask: np.ndarray,
    search_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute local ``sin(2 theta)``/``cos(2 theta)`` centerline targets."""

    _positive_integer(search_radius, name="search_radius")
    if center_mask.ndim != 2 or valid_mask.shape != center_mask.shape:
        raise ValueError("center_mask and valid_mask must be equally shaped 2-D arrays")
    height, width = center_mask.shape
    sin2 = np.zeros((height, width), dtype=np.float32)
    cos2 = np.zeros((height, width), dtype=np.float32)
    orient_valid = np.zeros((1, height, width), dtype=bool)

    y_indices, x_indices = np.nonzero(center_mask & valid_mask)
    if len(y_indices) == 0:
        return np.stack([sin2, cos2], axis=0), orient_valid

    for y, x in zip(y_indices, x_indices, strict=True):
        y_min, y_max = max(0, y - search_radius), min(height, y + search_radius + 1)
        x_min, x_max = max(0, x - search_radius), min(width, x + search_radius + 1)
        patch = center_mask[y_min:y_max, x_min:x_max]
        py, px = np.nonzero(patch)
        if len(py) < 2:
            continue
        coordinates = np.stack((px, py), axis=1).astype(np.float32)
        coordinates -= coordinates.mean(axis=0)
        _, eigenvectors = np.linalg.eigh(coordinates.T @ coordinates)
        dx, dy = eigenvectors[:, -1]
        theta = np.arctan2(dy, dx)
        sin2[y, x] = np.sin(2.0 * theta)
        cos2[y, x] = np.cos(2.0 * theta)
        orient_valid[0, y, x] = True
    return np.stack([sin2, cos2], axis=0), orient_valid


def compute_band_and_distance_targets(
    center_complete: np.ndarray,
    gsd_m: float,
    band_width_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute optional proxy-band and unbiased local distance supervision.

    A centerline does not by itself annotate the visible cultivated strip.  The
    proxy band is therefore disabled unless its metric width is explicitly
    supplied by the dataset contract.  Distance pixels are supervised only
    when the nearest center is provably inside this tile: a nearest center
    farther away than the closest tile edge could actually lie outside the
    window and would yield a truncated distance-transform target.
    """
    if not math.isfinite(float(gsd_m)) or gsd_m <= 0:
        raise ValueError("gsd_m must be finite and positive")
    if band_width_m is not None and (
        not math.isfinite(float(band_width_m)) or band_width_m <= 0
    ):
        raise ValueError("band_width_m must be finite and positive when provided")
    full_binary = center_complete > 0
    if not full_binary.any():
        empty_float = np.zeros(center_complete.shape, dtype=np.float32)
        empty_valid = np.zeros(center_complete.shape, dtype=bool)
        return empty_float, empty_valid, empty_float.copy(), empty_valid.copy()

    distance_px = distance_transform_edt(~full_binary)
    distance_m = (distance_px * float(gsd_m)).astype(np.float32)
    height, width = full_binary.shape
    yy, xx = np.indices((height, width))
    edge_distance_px = np.minimum.reduce((yy + 1, xx + 1, height - yy, width - xx))
    distance_valid = distance_px < edge_distance_px

    if band_width_m is None:
        band = np.zeros(center_complete.shape, dtype=np.float32)
        band_valid = np.zeros(center_complete.shape, dtype=bool)
    else:
        radius_px = (float(band_width_m) / 2.0) / float(gsd_m)
        band = (distance_px <= radius_px).astype(np.float32)
        band_valid = distance_valid.copy()
    return band, band_valid, distance_m, distance_valid


def compute_true_endpoints_target(
    frame: gpd.GeoDataFrame,
    tile_transform: Affine,
    shape: tuple[int, int],
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Extract physical row endpoints within tile window, excluding cut boundaries."""
    endpoints = np.zeros(shape, dtype=np.float32)
    if frame.empty:
        return endpoints

    inv_transform = ~tile_transform
    height, width = shape

    for geom in frame.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = [geom] if geom.geom_type == "LineString" else (
            geom.geoms if geom.geom_type == "MultiLineString" else []
        )
        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            for pt in (coords[0], coords[-1]):
                col, row = inv_transform * (pt[0], pt[1])
                c, r = round(col), round(row)
                if 0 <= r < height and 0 <= c < width and valid_mask[r, c]:
                    endpoints[r, c] = 1.0

    if np.any(endpoints > 0):
        endpoints = gaussian_filter(endpoints, sigma=1.0)
        max_val = float(endpoints.max())
        if max_val > 0:
            endpoints /= max_val
    return endpoints.astype(np.float32, copy=False)


class RowGraphNetDataset(Dataset[dict[str, Any]]):
    """Windowed training samples backed by source rasters and vector labels."""

    def __init__(
        self,
        manifests: Sequence[SceneManifest | str | Path],
        *,
        tile_size: int = 512,
        stride: int = 256,
        working_dir: str | Path = "data_set/_working",
        split: str = "train",
        augment: bool | None = None,
        min_valid_pixels: int = 100,
        normalize_scale: float | None = None,
        normalization: Mapping[str, Any] | None = None,
        centerline_buffer_m: float | None = None,
        proxy_band_width_m: float | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.tile_size = _positive_integer(tile_size, name="tile_size")
        self.stride = _positive_integer(stride, name="stride")
        if self.stride > self.tile_size:
            raise ValueError("stride must not exceed tile_size because that leaves coverage gaps")
        if isinstance(min_valid_pixels, bool) or not isinstance(min_valid_pixels, int):
            raise ValueError("min_valid_pixels must be an integer")
        if min_valid_pixels < 0 or min_valid_pixels > tile_size * tile_size:
            raise ValueError("min_valid_pixels must be between zero and tile_size squared")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")

        # Kept as a compatibility/provenance attribute. Lazy VRT reading
        # deliberately creates no full-scene working raster here.
        self.working_dir = Path(working_dir).resolve()
        self.split = split
        self.augment = augment if augment is not None else split == "train"
        self.min_valid_pixels = min_valid_pixels
        self.normalization = _normalization_mapping(normalization, normalize_scale)
        self._input_normalization = InputNormalization.from_metadata(self.normalization)
        if centerline_buffer_m is not None and (
            not math.isfinite(float(centerline_buffer_m)) or centerline_buffer_m <= 0
        ):
            raise ValueError("centerline_buffer_m must be finite and positive")
        self.centerline_buffer_m = centerline_buffer_m
        if proxy_band_width_m is not None and (
            not math.isfinite(float(proxy_band_width_m)) or proxy_band_width_m <= 0
        ):
            raise ValueError("proxy_band_width_m must be finite and positive")
        self.proxy_band_width_m = proxy_band_width_m
        self.seed = seed
        self.epoch = 0

        self.scenes = [
            load_scene_manifest(item) if isinstance(item, (str, Path)) else item
            for item in manifests
        ]
        self.loaded_scenes: list[_LazyScene] = []
        self.patch_index: list[tuple[int, int, int]] = []
        self._prepare_and_index_scenes()

    def set_epoch(self, epoch: int) -> None:
        """Select deterministic per-sample augmentation for one epoch."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def _prepare_and_index_scenes(self) -> None:
        for scene_index, manifest in enumerate(self.scenes):
            roi = _read_vector(manifest.roi, manifest.roi_layer)
            validated = validate_scene(
                manifest.image,
                roi,
                bands=manifest.bands,
                invalid_mask_path=manifest.ignore.path,
            )
            grid = _metric_grid(validated)
            annotations = _read_vector(manifest.annotations, manifest.annotations_layer).to_crs(
                validated.raster.metric_crs
            )
            if not annotations.empty and not annotations.geometry.is_valid.all():
                from shapely import make_valid

                annotations["geometry"] = [
                    make_valid(g) if (g is not None and not g.is_valid) else g
                    for g in annotations.geometry
                ]
            valid_metric = (
                annotations.geometry.notna()
                & (~annotations.geometry.is_empty)
                & annotations.geometry.is_valid
                & (annotations.geometry.length > 0)
            )
            annotations = annotations.loc[valid_metric].copy()
            scene = _LazyScene(
                manifest=manifest,
                validated=validated,
                grid=grid,
                annotations=annotations,
                has_visibility_labels=manifest.visibility.field in annotations.columns,
            )
            self.loaded_scenes.append(scene)

            roi_geom = scene.validated.roi_in_metric_crs
            pixel_area = max(abs(grid.transform.a * grid.transform.e), 1e-6)
            min_area = self.min_valid_pixels * pixel_area

            x_starts = _tile_starts(grid.width, self.tile_size, self.stride)
            y_starts = _tile_starts(grid.height, self.tile_size, self.stride)
            total_candidates = len(x_starts) * len(y_starts)

            accepted_count = 0
            with ExitStack() as stack:
                source = stack.enter_context(rasterio.open(manifest.image))
                validity_vrt = stack.enter_context(
                    WarpedVRT(source, **self._vrt_options(scene, categorical=True))
                )
                alpha_indexes = [
                    index + 1
                    for index, interpretation in enumerate(source.colorinterp)
                    if interpretation == ColorInterp.alpha
                ]
                external_vrt = None
                if scene.validated.invalid_mask_path is not None:
                    external_source = stack.enter_context(
                        rasterio.open(scene.validated.invalid_mask_path)
                    )
                    external_vrt = stack.enter_context(
                        WarpedVRT(
                            external_source,
                            **self._vrt_options(scene, categorical=True),
                        )
                    )

                for y in y_starts:
                    for x in x_starts:
                        window = self._window(scene, y, x)
                        tile_box = box(*window_bounds(window, grid.transform))
                        if not roi_geom.intersects(tile_box):
                            continue
                        if not roi_geom.contains(tile_box):
                            inter_area = roi_geom.intersection(tile_box).area
                            if inter_area < min_area:
                                continue

                        band_masks = validity_vrt.read_masks(manifest.bands, window=window)
                        valid = np.all(band_masks > 0, axis=0)
                        if alpha_indexes:
                            alpha = validity_vrt.read(alpha_indexes[0], window=window)
                            valid &= alpha > 0
                        if not roi_geom.contains(tile_box):
                            tile_transform = window_transform(window, grid.transform)
                            valid &= geometry_mask(
                                [mapping(roi_geom)],
                                out_shape=valid.shape,
                                transform=tile_transform,
                                invert=True,
                            )
                        if external_vrt is not None:
                            external = external_vrt.read(1, window=window, masked=True)
                            valid &= (~np.ma.getmaskarray(external)) & (
                                np.asarray(external.filled(1)) == 0
                            )
                        if int(np.count_nonzero(valid)) >= self.min_valid_pixels:
                            self.patch_index.append((scene_index, y, x))
                            accepted_count += 1

            print(
                f"[{self.split.capitalize()} Indexer] Scene {scene_index+1}/{len(self.scenes)} ({manifest.scene_id}): "
                f"{accepted_count} valid patches (from {total_candidates} grid cells)",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.patch_index)

    def _window(self, scene: _LazyScene, y: int, x: int) -> Window:
        return Window(
            col_off=x,
            row_off=y,
            width=min(self.tile_size, scene.grid.width - x),
            height=min(self.tile_size, scene.grid.height - y),
        )

    def _vrt_options(self, scene: _LazyScene, *, categorical: bool = False) -> dict[str, Any]:
        options: dict[str, Any] = {
            "crs": scene.validated.raster.metric_crs,
            "transform": scene.grid.transform,
            "width": scene.grid.width,
            "height": scene.grid.height,
            "resampling": Resampling.nearest if categorical else Resampling.bilinear,
        }
        if not categorical:
            # Do not set a destination nodata value here.  Doing so shadows a
            # source alpha band in GDAL/Rasterio.  Alpha is read explicitly
            # below and combined with nodata/read masks.
            options.update(dtype="float32")
        return options

    def _read_image_and_validity(
        self,
        scene: _LazyScene,
        window: Window,
        *,
        normalize: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        manifest = scene.manifest
        with rasterio.open(manifest.image) as source, WarpedVRT(
            source, **self._vrt_options(scene)
        ) as vrt:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", rasterio.errors.NodataShadowWarning)
                masked = vrt.read(manifest.bands, window=window, masked=True)
            alpha_indexes = [
                index + 1
                for index, interpretation in enumerate(source.colorinterp)
                if interpretation == ColorInterp.alpha
            ]
            if alpha_indexes:
                # A separate nearest-neighbour VRT prevents interpolation from
                # turning transparent pixels into partially valid RGB data.
                with WarpedVRT(source, **self._vrt_options(scene, categorical=True)) as mask_vrt:
                    alpha = mask_vrt.read(alpha_indexes[0], window=window, masked=True)
                alpha_valid = (~np.ma.getmaskarray(alpha)) & (np.asarray(alpha.filled(0)) > 0)
            else:
                alpha_valid = np.ones((int(window.height), int(window.width)), dtype=bool)
        image = np.asarray(masked.filled(0), dtype=np.float32)
        invalid = np.ma.getmaskarray(masked).any(axis=0) | (~alpha_valid)
        invalid |= ~np.isfinite(image).all(axis=0)

        tile_transform = window_transform(window, scene.grid.transform)
        roi_valid = geometry_mask(
            [mapping(scene.validated.roi_in_metric_crs)],
            out_shape=(int(window.height), int(window.width)),
            transform=tile_transform,
            invert=True,
        )
        valid = (~invalid) & roi_valid

        mask_path = scene.validated.invalid_mask_path
        if mask_path is not None:
            with rasterio.open(mask_path) as source_mask, WarpedVRT(
                source_mask, **self._vrt_options(scene, categorical=True)
            ) as mask_vrt:
                external = mask_vrt.read(1, window=window, masked=True)
            external_invalid = np.ma.getmaskarray(external) | (
                np.asarray(external.filled(1)) != 0
            )
            valid &= ~external_invalid

        image[:, ~valid] = 0.0
        if normalize:
            image = self._input_normalization.apply(image)
            image[:, ~valid] = 0.0
        return image, valid

    def _tile_annotations(
        self,
        scene: _LazyScene,
        window: Window,
        valid_mask: np.ndarray,
    ) -> tuple[RasterizedAnnotations, gpd.GeoDataFrame, Affine]:
        tile_transform = window_transform(window, scene.grid.transform)
        bounds = window_bounds(window, scene.grid.transform)
        # Include geometries outside the tile whose rasterized target buffer
        # still overlaps it. ``centerline_buffer_m`` is a per-side radius;
        # when omitted, the rasterizer falls back to one-GSD total width.
        query_margin_m = max(
            scene.validated.raster.gsd.mean_m,
            self.centerline_buffer_m or 0.0,
        )
        query = box(*bounds).buffer(query_margin_m)
        frame = scene.annotations
        if not frame.empty:
            indexes = frame.sindex.query(query, predicate="intersects")
            frame = frame.iloc[np.asarray(indexes, dtype=np.int64)]
        annotations = rasterize_row_annotations(
            frame,
            out_shape=valid_mask.shape,
            transform=tile_transform,
            raster_crs=scene.validated.raster.metric_crs,
            row_id_field=scene.manifest.row_id_field,
            visibility_field=scene.manifest.visibility.field,
            occluded_values=scene.manifest.visibility.occluded_values,
            ignore_values=scene.manifest.ignore.values,
            centerline_width_m=(
                None
                if self.centerline_buffer_m is None
                else 2.0 * self.centerline_buffer_m
            ),
            metric_crs=scene.validated.raster.metric_crs,
            valid_mask=valid_mask,
        )
        return annotations, frame, tile_transform

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_index, y, x = self.patch_index[index]
        scene = self.loaded_scenes[scene_index]
        window = self._window(scene, y, x)
        image, raster_valid = self._read_image_and_validity(scene, window, normalize=True)
        annotations, frame, tile_transform = self._tile_annotations(scene, window, raster_valid)
        raw = annotations.center_loss_targets()
        orientation, orientation_valid = compute_axial_orientation_maps(
            annotations.center_observed,
            annotations.center_observed_valid_mask,
        )
        band, band_valid, distance_m, distance_valid = compute_band_and_distance_targets(
            annotations.center_complete,
            scene.validated.raster.gsd.mean_m,
            self.proxy_band_width_m,
        )
        endpoints = compute_true_endpoints_target(
            frame,
            tile_transform,
            raster_valid.shape,
            raster_valid,
        )

        pad_y = self.tile_size - int(window.height)
        pad_x = self.tile_size - int(window.width)

        def pad_2d(array: np.ndarray, value: Any = 0) -> np.ndarray:
            if pad_y == 0 and pad_x == 0:
                return array
            return np.pad(
                array,
                ((0, pad_y), (0, pad_x)),
                mode="constant",
                constant_values=value,
            )

        def pad_chw(array: np.ndarray, value: Any = 0) -> np.ndarray:
            if pad_y == 0 and pad_x == 0:
                return array
            return np.pad(
                array,
                ((0, 0), (0, pad_y), (0, pad_x)),
                mode="constant",
                constant_values=value,
            )

        image = pad_chw(image).astype(np.float32, copy=False)
        orientation = pad_chw(orientation).astype(np.float32, copy=False)
        orientation_valid = pad_chw(orientation_valid).astype(np.float32, copy=False)
        target_arrays: dict[str, np.ndarray] = {
            "band": pad_2d(band).astype(np.float32)[None],
            "band_valid_mask": pad_2d(band_valid & raw["valid_mask"]).astype(np.float32)[None],
            "center_observed": pad_2d(raw["center_observed"]).astype(np.float32)[None],
            "center_complete": pad_2d(raw["center_complete"]).astype(np.float32)[None],
            "center_observed_valid_mask": pad_2d(
                raw["center_observed_valid_mask"]
            ).astype(np.float32)[None],
            "center_complete_valid_mask": pad_2d(
                raw["center_complete_valid_mask"]
            ).astype(np.float32)[None],
            "occluded_mask": pad_2d(raw["occluded_mask"]).astype(np.float32)[None],
            "valid_mask": pad_2d(raw["valid_mask"]).astype(np.float32)[None],
            "row_id_mask": pad_2d(raw["row_id_mask"]).astype(np.int64)[None],
            "orientation": orientation,
            "orientation_valid_mask": orientation_valid,
            "endpoints": pad_2d(endpoints).astype(np.float32)[None],
            # Endpoint negatives are meaningful only when the annotation
            # schema explicitly distinguishes visibility.  Legacy centerline
            # files often end at an ROI clip, which is not a physical row end.
            "endpoints_valid_mask": pad_2d(
                raw["valid_mask"] if scene.has_visibility_labels else np.zeros_like(raw["valid_mask"])
            ).astype(np.float32)[None],
            "distance": pad_2d(distance_m).astype(np.float32)[None],
            "distance_valid_mask": pad_2d(distance_valid & raw["valid_mask"]).astype(np.float32)[None],
            # The refiner learns complete-only pieces from this explicit map.
            "repair": pad_2d(raw["occluded_mask"]).astype(np.float32)[None],
            "repair_valid_mask": pad_2d(
                raw["center_complete_valid_mask"]
                if scene.has_visibility_labels
                else np.zeros_like(raw["center_complete_valid_mask"])
            ).astype(np.float32)[None],
        }

        if self.augment:
            # The transform depends only on (seed, epoch, index), not worker
            # order or prior RNG consumption. Resume therefore reproduces it.
            generator = np.random.default_rng(
                np.random.SeedSequence((self.seed, self.epoch, index))
            )
            if generator.random() > 0.5:
                image = np.flip(image, axis=2).copy()
                target_arrays = {
                    name: np.flip(value, axis=2).copy() for name, value in target_arrays.items()
                }
                target_arrays["orientation"][0] *= -1.0
            if generator.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                target_arrays = {
                    name: np.flip(value, axis=1).copy() for name, value in target_arrays.items()
                }
                target_arrays["orientation"][0] *= -1.0

        targets = {name: torch.from_numpy(value) for name, value in target_arrays.items()}
        return {
            "image": torch.from_numpy(image),
            "targets": targets,
            "scene_id": scene.manifest.scene_id,
            "window": torch.tensor((x, y, int(window.width), int(window.height))),
        }


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


class RowGraphNetDataModule:
    """Deterministic train/validation loaders over lazy scene datasets."""

    def __init__(
        self,
        train_manifests: Sequence[SceneManifest | str | Path],
        val_manifests: Sequence[SceneManifest | str | Path],
        *,
        tile_size: int = 512,
        stride: int = 256,
        batch_size: int = 4,
        num_workers: int = 0,
        working_dir: str | Path = "data_set/_working",
        min_valid_pixels: int = 100,
        normalization: Mapping[str, Any] | None = None,
        centerline_buffer_m: float | None = None,
        proxy_band_width_m: float | None = None,
        pin_memory: bool | None = None,
        seed: int = 0,
    ) -> None:
        self.batch_size = _positive_integer(batch_size, name="batch_size")
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        self.seed = seed
        self.train_dataset = RowGraphNetDataset(
            train_manifests,
            tile_size=tile_size,
            stride=stride,
            working_dir=working_dir,
            split="train",
            augment=True,
            min_valid_pixels=min_valid_pixels,
            normalization=normalization,
            centerline_buffer_m=centerline_buffer_m,
            proxy_band_width_m=proxy_band_width_m,
            seed=seed,
        )
        self.val_dataset = RowGraphNetDataset(
            val_manifests,
            tile_size=tile_size,
            stride=stride,
            working_dir=working_dir,
            split="validation",
            augment=False,
            min_valid_pixels=min_valid_pixels,
            normalization=normalization,
            centerline_buffer_m=centerline_buffer_m,
            proxy_band_width_m=proxy_band_width_m,
            seed=seed,
        )

    def train_dataloader(self, *, epoch: int = 0) -> DataLoader:
        self.train_dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(self.seed + epoch)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=_seed_worker,
            generator=generator,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=_seed_worker,
            generator=generator,
            persistent_workers=self.num_workers > 0,
        )


__all__ = [
    "RowGraphNetDataModule",
    "RowGraphNetDataset",
    "compute_axial_orientation_maps",
    "compute_band_and_distance_targets",
    "compute_true_endpoints_target",
]
