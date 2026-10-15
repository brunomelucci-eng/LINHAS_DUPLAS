"""Unit tests for RowGraphNetDataset and RowGraphNetDataModule."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine
from rasterio.enums import ColorInterp
from shapely.geometry import box

from rowgraphnet.data.dataset import RowGraphNetDataset, compute_band_and_distance_targets
from rowgraphnet.data.manifest import IgnoreSpec, SceneManifest, VisibilitySpec


def _create_synthetic_scene(
    tmp_dir: Path, scene_id: str = "test_001", *, with_alpha: bool = False
) -> SceneManifest:
    img_path = tmp_dir / f"{scene_id}_img.tif"
    roi_path = tmp_dir / f"{scene_id}_roi.gpkg"
    anno_path = tmp_dir / f"{scene_id}_rows.geojson"

    # Create 3-band raster in metric UTM
    transform = Affine(0.1, 0.0, 500000.0, 0.0, -0.1, 7500000.0)
    data = (np.random.rand(3, 100, 100) * 255).astype(np.uint8)
    if with_alpha:
        alpha = np.full((1, 100, 100), 255, dtype=np.uint8)
        alpha[:, :10, :10] = 0
        data = np.concatenate((data, alpha), axis=0)

    profile = {
        "driver": "GTiff",
        "height": 100,
        "width": 100,
        "count": int(data.shape[0]),
        "dtype": "uint8",
        "crs": "EPSG:32723",
        "transform": transform,
    }
    with rasterio.open(img_path, "w", **profile) as dst:
        dst.write(data)
        if with_alpha:
            dst.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )

    import geopandas as gpd
    from shapely.geometry import LineString

    roi_poly = box(500000.0, 7499990.0, 500010.0, 7500000.0)
    gdf_roi = gpd.GeoDataFrame({"geometry": [roi_poly]}, crs="EPSG:32723")
    gdf_roi.to_file(roi_path, driver="GPKG")

    line = LineString([(500001.0, 7499999.0), (500009.0, 7499991.0)])
    gdf_lines = gpd.GeoDataFrame(
        {"row_id": ["row_1"], "visibility": ["observed"], "geometry": [line]},
        crs="EPSG:32723",
    )
    gdf_lines.to_file(anno_path, driver="GeoJSON")

    manifest = SceneManifest(
        schema_version=1,
        scene_id=scene_id,
        group_id=f"group_{scene_id}",
        split="train",
        image=img_path,
        bands=(1, 2, 3),
        roi=roi_path,
        annotations=anno_path,
        visibility=VisibilitySpec(field="visibility", observed_values=("observed",), occluded_values=("occluded",)),
        ignore=IgnoreSpec(field="visibility", values=("ignore",)),
        sensor={"sensor": "synthetic"},
        row_id_field="row_id",
    )
    return manifest


def test_rowgraphnet_dataset_creation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        manifest = _create_synthetic_scene(tmp_dir)

        dataset = RowGraphNetDataset(
            manifests=[manifest],
            tile_size=64,
            stride=32,
            working_dir=tmp_dir / "_working",
            split="train",
            min_valid_pixels=1,
        )

        assert len(dataset) > 0
        sample = dataset[0]

        assert "image" in sample
        assert "targets" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape == (3, 64, 64)

        targets = sample["targets"]
        assert "center_observed" in targets
        assert "center_complete" in targets
        assert "row_id_mask" in targets
        assert "valid_mask" in targets
        assert "band" in targets
        assert "endpoints" in targets
        assert "distance" in targets
        assert targets["center_observed"].shape == (1, 64, 64)
        assert targets["band"].shape == (1, 64, 64)
        assert targets["endpoints"].shape == (1, 64, 64)
        assert targets["distance"].shape == (1, 64, 64)
        assert torch.isfinite(targets["distance"]).all()
        assert float(targets["distance"].min()) >= 0.0


def test_alpha_band_is_preserved_as_invalidity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dataset = RowGraphNetDataset(
            [_create_synthetic_scene(Path(tmp), with_alpha=True)],
            tile_size=64,
            stride=64,
            split="validation",
            min_valid_pixels=1,
        )
        sample = dataset[0]
        assert sample["targets"]["valid_mask"][0, 0, 0].item() == 0.0
        assert torch.count_nonzero(sample["image"][:, :10, :10]).item() == 0


def test_missing_center_has_no_arbitrary_distance_or_band_supervision() -> None:
    center = np.zeros((16, 16), dtype=np.uint8)
    band, band_valid, distance, distance_valid = compute_band_and_distance_targets(
        center, gsd_m=0.1
    )
    assert not band.any()
    assert not band_valid.any()
    assert not distance.any()
    assert not distance_valid.any()


def test_centerline_buffer_is_per_side_and_reaches_rasterizer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dataset = RowGraphNetDataset(
            [_create_synthetic_scene(Path(tmp))],
            tile_size=64,
            stride=64,
            split="validation",
            min_valid_pixels=1,
            centerline_buffer_m=0.10,
        )
        scene_index, y, x = dataset.patch_index[0]
        scene = dataset.loaded_scenes[scene_index]
        window = dataset._window(scene, y, x)
        _, valid = dataset._read_image_and_validity(scene, window, normalize=False)
        annotations, _, _ = dataset._tile_annotations(scene, window, valid)

        assert annotations.target_width_m == pytest.approx(0.20)


@pytest.mark.parametrize("value", [0.0, -0.10, float("inf"), float("nan")])
def test_centerline_buffer_rejects_invalid_radius(value: float) -> None:
    with (
        tempfile.TemporaryDirectory() as tmp,
        pytest.raises(ValueError, match="centerline_buffer_m"),
    ):
        RowGraphNetDataset(
            [_create_synthetic_scene(Path(tmp))],
            centerline_buffer_m=value,
        )
