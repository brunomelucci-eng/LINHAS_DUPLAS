from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon, box

from rowgraphnet.data import (
    DataContractError,
    is_metric_crs,
    rasterize_row_annotations,
    validate_raster,
    validate_scene,
)


def _write_raster(path, *, crs, transform, width=40, height=30, count=3):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(np.ones((count, height, width), dtype=np.uint8))


def test_validate_scene_selects_local_metric_crs_and_metric_gsd(tmp_path):
    raster_path = tmp_path / "geographic.tif"
    transform = from_origin(-47.0, -15.0, 0.00001, 0.00001)
    _write_raster(raster_path, crs="EPSG:4326", transform=transform)
    roi = gpd.GeoDataFrame(
        geometry=[box(-46.9999, -15.0002, -46.9997, -15.00005)],
        crs="EPSG:4326",
    )

    scene = validate_scene(raster_path, roi, bands=[1, 2, 3])

    assert scene.raster.crs == CRS.from_epsg(4326)
    assert scene.raster.metric_crs == CRS.from_epsg(32723)
    assert is_metric_crs(scene.raster.metric_crs)
    assert 0.9 < scene.raster.gsd.x_m < 1.3
    assert 0.9 < scene.raster.gsd.y_m < 1.3
    assert not scene.roi_was_clipped
    assert scene.roi_in_metric_crs.area > 0


def test_validate_raster_preserves_rotated_affine_basis_lengths(tmp_path):
    raster_path = tmp_path / "metric.tif"
    transform = rasterio.Affine(0.3, 0.0, 500_000.0, 0.0, -0.5, 7_400_000.0)
    _write_raster(raster_path, crs="EPSG:32723", transform=transform)

    contract = validate_raster(raster_path)

    assert contract.metric_crs == CRS.from_epsg(32723)
    assert contract.gsd.x_m == pytest.approx(0.3)
    assert contract.gsd.y_m == pytest.approx(0.5)
    assert contract.bands == (1, 2, 3)


@pytest.mark.parametrize("bands", [[], [0], [4], [1, 1], [True, 2]])
def test_validate_raster_rejects_invalid_band_selection(tmp_path, bands):
    raster_path = tmp_path / "bands.tif"
    _write_raster(
        raster_path,
        crs="EPSG:32723",
        transform=from_origin(500_000, 7_400_000, 1, 1),
    )

    with pytest.raises(DataContractError, match=r"band|Band"):
        validate_raster(raster_path, bands=bands)


def test_validate_scene_rejects_missing_crs_and_non_overlapping_roi(tmp_path):
    missing_crs = tmp_path / "missing_crs.tif"
    _write_raster(missing_crs, crs=None, transform=from_origin(0, 30, 1, 1))
    with pytest.raises(DataContractError, match="CRS"):
        validate_scene(missing_crs, box(0, 0, 1, 1), roi_crs="EPSG:3857")

    raster_path = tmp_path / "metric.tif"
    _write_raster(raster_path, crs="EPSG:3857", transform=from_origin(0, 30, 1, 1))
    with pytest.raises(DataContractError, match="overlap"):
        validate_scene(raster_path, box(100, 100, 110, 110), roi_crs="EPSG:3857")


def test_invalid_mask_must_be_on_exactly_the_same_grid(tmp_path):
    raster_path = tmp_path / "image.tif"
    mask_path = tmp_path / "mask.tif"
    transform = from_origin(500_000, 7_400_000, 1, 1)
    _write_raster(raster_path, crs="EPSG:32723", transform=transform)
    _write_raster(
        mask_path,
        crs="EPSG:32723",
        transform=transform,
        width=39,
        height=30,
        count=1,
    )

    with pytest.raises(DataContractError, match="dimensions"):
        validate_scene(
            raster_path,
            box(500_001, 7_399_971, 500_039, 7_399_999),
            roi_crs="EPSG:32723",
            invalid_mask_path=mask_path,
        )


def test_rasterized_annotations_preserve_row_identity_occlusion_and_ignore():
    transform = from_origin(0, 20, 1, 1)
    annotations = gpd.GeoDataFrame(
        {
            "row_id": ["row-b", "row-a", "row-a", "row-a", None],
            "visibility": ["observed", "observed", "occluded", "observed", "ignore"],
        },
        geometry=[
            LineString([(1, 10), (19, 10)]),
            LineString([(1, 15), (8, 15)]),
            LineString([(8, 15), (12, 15)]),
            LineString([(12, 15), (19, 15)]),
            Polygon([(4, 14), (6, 14), (6, 16), (4, 16)]),
        ],
        crs="EPSG:3857",
    )

    targets = rasterize_row_annotations(
        annotations,
        out_shape=(20, 20),
        transform=transform,
        raster_crs="EPSG:3857",
        centerline_width_m=1.0,
    )

    assert targets.row_id_to_label == {"row-a": 1, "row-b": 2}
    observed_rc = rasterio.transform.rowcol(transform, 2, 15)
    occluded_rc = rasterio.transform.rowcol(transform, 10, 15)
    ignored_rc = rasterio.transform.rowcol(transform, 5, 15)
    row_b_rc = rasterio.transform.rowcol(transform, 10, 10)
    assert targets.observed_mask[observed_rc]
    assert targets.row_id_mask[observed_rc] == 1
    assert targets.occluded_mask[occluded_rc]
    assert targets.center_mask[occluded_rc]
    assert targets.row_id_mask[occluded_rc] == 1
    assert targets.row_id_mask[row_b_rc] == 2
    assert targets.ignore_mask[ignored_rc]
    assert not targets.valid_mask[ignored_rc]
    assert not targets.center_mask[ignored_rc]
    assert targets.row_id_mask[ignored_rc] == 0


def test_metric_target_width_scales_with_raster_gsd():
    annotations = gpd.GeoDataFrame(
        {"row_id": ["row-1"], "visibility": ["observed"]},
        geometry=[LineString([(2, 10), (18, 10)])],
        crs="EPSG:3857",
    )
    coarse = rasterize_row_annotations(
        annotations,
        out_shape=(20, 20),
        transform=from_origin(0, 20, 1, 1),
        raster_crs="EPSG:3857",
        centerline_width_m=2,
    )
    fine = rasterize_row_annotations(
        annotations,
        out_shape=(40, 40),
        transform=from_origin(0, 20, 0.5, 0.5),
        raster_crs="EPSG:3857",
        centerline_width_m=2,
    )

    coarse_column = coarse.center_mask[:, 10].sum()
    fine_column = fine.center_mask[:, 20].sum()
    assert fine_column > coarse_column
    assert fine.target_width_m == coarse.target_width_m == 2


def test_rasterization_rejects_non_metric_workspace():
    annotations = gpd.GeoDataFrame(
        {"row_id": ["row-1"]},
        geometry=[LineString([(-47.0, -15.0), (-46.999, -15.0)])],
        crs="EPSG:4326",
    )

    with pytest.raises(DataContractError, match="metric_crs"):
        rasterize_row_annotations(
            annotations,
            out_shape=(20, 20),
            transform=from_origin(-47.0, -15.0, 0.0001, 0.0001),
            raster_crs="EPSG:4326",
            metric_crs="EPSG:4326",
        )
