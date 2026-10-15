"""Create the north-up, square-pixel metric grid used by geometry stages."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject

from rowgraphnet.data.contracts import DataContractError, ValidatedScene, validate_scene


@dataclass(frozen=True, slots=True)
class PreparedMetricScene:
    """A cropped metric working raster and its source-band provenance."""

    source: ValidatedScene
    working: ValidatedScene
    raster_path: Path
    invalid_mask_path: Path | None
    source_band_indexes: tuple[int, ...]


def _target_grid(scene: ValidatedScene) -> tuple[Affine, int, int]:
    resolution = scene.raster.gsd.mean_m
    if not math.isfinite(resolution) or resolution <= 0:
        raise DataContractError("cannot build metric grid from a non-positive GSD")
    minimum_x, minimum_y, maximum_x, maximum_y = scene.roi_in_metric_crs.bounds
    left = math.floor(minimum_x / resolution) * resolution
    bottom = math.floor(minimum_y / resolution) * resolution
    right = math.ceil(maximum_x / resolution) * resolution
    top = math.ceil(maximum_y / resolution) * resolution
    width = max(math.ceil((right - left) / resolution), 1)
    height = max(math.ceil((top - bottom) / resolution), 1)
    return Affine(resolution, 0.0, left, 0.0, -resolution, top), width, height


def _warp_invalid_mask(
    scene: ValidatedScene,
    output_path: Path,
    transform: Affine,
    width: int,
    height: int,
) -> Path | None:
    if scene.invalid_mask_path is None:
        return None
    destination = output_path.with_name(f"{output_path.stem}_invalid_mask.tif")
    temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    with rasterio.open(scene.invalid_mask_path) as source:
        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": 1,
            "dtype": "uint8",
            "crs": scene.raster.metric_crs,
            "transform": transform,
            "nodata": 1,
            "compress": "deflate",
        }
        with rasterio.open(temporary, "w", **profile) as target:
            reproject(
                source=rasterio.band(source, 1),
                destination=rasterio.band(target, 1),
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=transform,
                dst_crs=scene.raster.metric_crs,
                dst_nodata=1,
                resampling=Resampling.nearest,
                init_dest_nodata=True,
            )
    temporary.replace(destination)
    return destination


def prepare_metric_scene(
    scene: ValidatedScene,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> PreparedMetricScene:
    """Warp selected source bands into the mandatory metric working grid.

    The output is cropped to the ROI envelope, north-up, and uses square pixels
    whose resolution is measured from the source affine at the ROI.  Continuous
    imagery uses bilinear resampling; a categorical invalid mask uses nearest.
    """

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    invalid_destination = destination.with_name(f"{destination.stem}_invalid_mask.tif")
    if not overwrite and (destination.exists() or invalid_destination.exists()):
        raise FileExistsError(f"metric working output already exists: {destination}")
    transform, width, height = _target_grid(scene)
    temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    source_bands = scene.raster.bands
    try:
        with rasterio.open(scene.raster.path) as source:
            profile = {
                "driver": "GTiff",
                "width": width,
                "height": height,
                "count": len(source_bands),
                "dtype": "float32",
                "crs": scene.raster.metric_crs,
                "transform": transform,
                "nodata": np.nan,
                "compress": "deflate",
                "predictor": 3,
                "BIGTIFF": "IF_SAFER",
            }
            with rasterio.open(temporary, "w", **profile) as target:
                target.update_tags(
                    source_raster=str(scene.raster.path),
                    source_band_indexes=",".join(map(str, source_bands)),
                )
                for output_index, source_index in enumerate(source_bands, start=1):
                    reproject(
                        source=rasterio.band(source, source_index),
                        destination=rasterio.band(target, output_index),
                        src_transform=source.transform,
                        src_crs=source.crs,
                        src_nodata=source.nodatavals[source_index - 1],
                        dst_transform=transform,
                        dst_crs=scene.raster.metric_crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                        init_dest_nodata=True,
                    )
                    description = source.descriptions[source_index - 1]
                    if description:
                        target.set_band_description(output_index, description)
        temporary.replace(destination)
        metric_mask = _warp_invalid_mask(scene, destination, transform, width, height)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    working = validate_scene(
        destination,
        scene.roi_in_metric_crs,
        roi_crs=scene.raster.metric_crs,
        bands=tuple(range(1, len(source_bands) + 1)),
        invalid_mask_path=metric_mask,
    )
    return PreparedMetricScene(
        source=scene,
        working=working,
        raster_path=destination,
        invalid_mask_path=metric_mask,
        source_band_indexes=source_bands,
    )
