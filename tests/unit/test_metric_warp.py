from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from rowgraphnet.data import is_metric_crs, prepare_metric_scene, validate_scene


def test_geographic_scene_is_warped_to_square_metric_grid(tmp_path) -> None:
    source_path = tmp_path / "geographic.tif"
    transform = from_origin(-48.0, -20.0, 0.00001, 0.00001)
    data = np.arange(3 * 32 * 32, dtype=np.uint16).reshape(3, 32, 32)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=3,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(data)

    roi = box(-47.9999, -20.00022, -47.9997, -20.00002)
    source = validate_scene(source_path, roi, roi_crs="EPSG:4326", bands=(1, 2, 3))
    prepared = prepare_metric_scene(source, tmp_path / "metric.tif")

    assert prepared.source_band_indexes == (1, 2, 3)
    assert is_metric_crs(prepared.working.raster.crs)
    assert prepared.working.raster.crs == source.raster.metric_crs
    affine = prepared.working.raster.transform
    assert affine.b == 0.0
    assert affine.d == 0.0
    assert math.isclose(abs(affine.a), abs(affine.e), rel_tol=1e-12)
    assert prepared.working.roi_in_metric_crs.area > 0
    with rasterio.open(prepared.raster_path) as warped:
        assert warped.count == 3
        assert warped.dtypes == ("float32", "float32", "float32")
        assert warped.tags()["source_band_indexes"] == "1,2,3"
