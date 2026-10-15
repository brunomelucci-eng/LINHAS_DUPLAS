from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from rowgraphnet.data import DataContractError, rasterize_row_annotations


def test_center_targets_separate_observed_occluded_and_invalid_pixels() -> None:
    transform = from_origin(0, 20, 1, 1)
    annotations = gpd.GeoDataFrame(
        {
            "row_id": ["row-1", "row-1", "row-1"],
            "visibility": ["observed", "occluded", "observed"],
        },
        geometry=[
            LineString([(1, 10), (7, 10)]),
            LineString([(7, 10), (12, 10)]),
            LineString([(12, 10), (19, 10)]),
        ],
        crs="EPSG:3857",
    )
    raster_valid = np.ones((20, 20), dtype=bool)
    invalid_rc = rasterio.transform.rowcol(transform, 16, 10)
    raster_valid[invalid_rc] = False

    targets = rasterize_row_annotations(
        annotations,
        out_shape=(20, 20),
        transform=transform,
        raster_crs="EPSG:3857",
        centerline_width_m=1.0,
        valid_mask=raster_valid,
    )

    observed_rc = rasterio.transform.rowcol(transform, 3, 10)
    occluded_rc = rasterio.transform.rowcol(transform, 9, 10)
    assert targets.center_observed[observed_rc]
    assert targets.center_complete[observed_rc]
    assert targets.center_observed_valid_mask[observed_rc]
    assert targets.center_complete_valid_mask[observed_rc]

    assert not targets.center_observed[occluded_rc]
    assert targets.center_complete[occluded_rc]
    assert not targets.center_observed_valid_mask[occluded_rc]
    assert targets.center_complete_valid_mask[occluded_rc]
    assert targets.row_id_mask[occluded_rc] == 1

    assert not targets.valid_mask[invalid_rc]
    assert not targets.center_observed[invalid_rc]
    assert not targets.center_complete[invalid_rc]
    assert targets.row_id_mask[invalid_rc] == 0

    canonical = targets.center_loss_targets()
    assert canonical["center_observed"] is targets.observed_mask
    assert canonical["center_complete"] is targets.center_mask
    np.testing.assert_array_equal(
        canonical["center_observed_valid_mask"],
        targets.valid_mask & ~targets.occluded_mask,
    )


def test_rasterization_rejects_valid_mask_on_a_different_grid() -> None:
    annotations = gpd.GeoDataFrame(
        {"row_id": ["row-1"]},
        geometry=[LineString([(1, 10), (19, 10)])],
        crs="EPSG:3857",
    )

    with pytest.raises(DataContractError, match="valid_mask shape"):
        rasterize_row_annotations(
            annotations,
            out_shape=(20, 20),
            transform=from_origin(0, 20, 1, 1),
            raster_crs="EPSG:3857",
            valid_mask=np.ones((19, 20), dtype=bool),
        )
