"""GeoTIFF export for continuous RowGraphNet prediction maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS

from rowgraphnet.domain import RasterPrediction

RASTER_FILENAMES: dict[str, str] = {
    "center": "center_probability.tif",
    "band": "band_probability.tif",
    "orientation_sin2": "orientation_sin2.tif",
    "orientation_cos2": "orientation_cos2.tif",
    "endpoints": "endpoint_probability.tif",
    "distance": "distance_to_center.tif",
    "uncertainty": "uncertainty.tif",
}

PROBABILITY_MAPS = frozenset({"center", "band", "endpoints"})


def _validate_prediction(prediction: RasterPrediction) -> np.ndarray:
    try:
        CRS.from_user_input(prediction.crs)
    except Exception as error:
        raise ValueError("prediction must contain a valid CRS") from error
    determinant = prediction.transform.a * prediction.transform.e - (
        prediction.transform.b * prediction.transform.d
    )
    if not np.isfinite(determinant) or abs(determinant) <= np.finfo(float).eps:
        raise ValueError("prediction transform must be finite and invertible")
    if prediction.valid_mask is None:
        valid = np.ones(prediction.center.shape, dtype=bool)
    else:
        valid = np.asarray(prediction.valid_mask, dtype=bool)
    for name in RASTER_FILENAMES:
        values = np.asarray(getattr(prediction, name))
        if np.any(valid & ~np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values in valid pixels")
        if name in PROBABILITY_MAPS and np.any(valid & ((values < 0.0) | (values > 1.0))):
            raise ValueError(f"{name} values in valid pixels must be within [0, 1]")
    return valid


def write_continuous_geotiff(
    path: str | Path,
    values: np.ndarray,
    *,
    transform: Affine,
    crs: CRS | str | int,
    valid_mask: np.ndarray | None = None,
    variable_name: str,
) -> Path:
    """Write one floating-point map without thresholding or quantization."""

    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if valid_mask is None:
        valid = np.ones(array.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != array.shape:
            raise ValueError("valid_mask must match values")
    if np.any(valid & ~np.isfinite(array)):
        raise ValueError("values contain non-finite data in valid pixels")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = np.asarray(array, dtype=np.float32).copy()
    output[~valid] = np.nan
    with rasterio.open(
        destination,
        "w",
        driver="GTiff",
        width=output.shape[1],
        height=output.shape[0],
        count=1,
        dtype="float32",
        crs=CRS.from_user_input(crs),
        transform=transform,
        nodata=np.nan,
        compress="deflate",
        predictor=3,
    ) as dataset:
        dataset.write(output, 1)
        dataset.set_band_description(1, variable_name)
        dataset.update_tags(
            ROWGRAPHNET_SCHEMA="1.0",
            VARIABLE=variable_name,
            CONTINUOUS="true",
        )
    return destination


def write_prediction_rasters(
    output_dir: str | Path,
    prediction: RasterPrediction,
) -> dict[str, Path]:
    """Write every continuous head using fixed, provenance-friendly names."""

    valid = _validate_prediction(prediction)
    output = Path(output_dir)
    paths: dict[str, Path] = {}
    for name, filename in RASTER_FILENAMES.items():
        paths[name] = write_continuous_geotiff(
            output / filename,
            getattr(prediction, name),
            transform=prediction.transform,
            crs=prediction.crs,
            valid_mask=valid,
            variable_name=name,
        )
    return paths


export_prediction_rasters = write_prediction_rasters
