import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import LineString, Polygon

from scripts import refine_existing_vectors
from src.postprocessing.output_products import (
    create_output_layout,
    persist_two_stage_vector_outputs,
)
from src.postprocessing.postprocess_pipeline import run_postprocessing


STABLE_FINAL_COLUMNS = [
    "final_line_id",
    "run_id",
    "track_id",
    "pair_id",
    "pair_side",
    "source_raw_ids",
    "source_type",
    "was_merged",
    "was_repaired",
    "was_reconstructed",
    "was_smoothed",
    "was_terminal_extended",
    "mean_probability",
    "median_probability",
    "pair_status",
    "topology_status",
    "review_required",
    "postprocess_version",
    "geometry",
]


def _config():
    return {
        "outputs": {
            "always_save_post_inference": True,
            "always_save_postprocessing": True,
            "overwrite_existing": False,
            "save_lineage": True,
            "save_comparison_report": True,
        },
        "postprocess_pipeline": {
            "enabled": True,
            "version": "test-v1",
            "prevector_cleanup": {"enabled": False},
            "track_merge": {"enabled": True},
            "fail_on_unmatched": True,
            "fail_on_crossing": True,
        },
        "postprocessing": {"double_row_validation": {"enabled": False}},
        "deduplication": {"enabled": False},
        "double_row_refinement": {"enabled": False},
        "final_centerline_fairing": {"enabled": False},
        "line_extension": {"enabled": False},
        "vector": {"min_line_length_m": 0.1},
        "metadata": {"probability_sample_spacing_m": 0.10},
        "crs": {"export_epsg": 3857},
    }


def _write_probability(path, values, transform):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
    ) as destination:
        destination.write(values, 1)


def test_integrated_and_standalone_postprocess_are_equivalent(tmp_path, monkeypatch):
    config = _config()
    vectorized = gpd.GeoDataFrame(
        {
            "track_id": [10, 11],
            "pair_id": [1, 1],
            "pair_status": ["valid_pair", "valid_pair"],
        },
        geometry=[
            LineString([(1.0, 0.0), (9.0, 0.0)]),
            LineString([(1.0, 0.9), (9.0, 0.9)]),
        ],
        crs="EPSG:3857",
    )
    roi = gpd.GeoDataFrame(
        geometry=[Polygon([(0, -1), (10, -1), (10, 2), (0, 2)])],
        crs=vectorized.crs,
    )
    transform = Affine.translation(0.0, 3.0) * Affine.scale(0.10, -0.10)
    probability = np.linspace(0.55, 0.95, 50 * 120, dtype=np.float32).reshape(50, 120)
    probability_path = tmp_path / "probability_center.tif"
    roi_path = tmp_path / "roi.geojson"
    standalone_path = tmp_path / "standalone__pos_processamento.gpkg"
    _write_probability(probability_path, probability, transform)
    roi.to_file(roi_path, driver="GeoJSON")

    layout = create_output_layout(tmp_path / "integrated", "talhao.gpkg", config)
    persist_two_stage_vector_outputs(
        vectorized,
        probability,
        transform,
        roi,
        config,
        layout=layout,
        run_id="run_equivalence",
        checkpoint="best.pt",
        postprocess_fn=run_postprocessing,
        export_epsg=3857,
    )

    monkeypatch.setattr(refine_existing_vectors, "load_config", lambda _path: config)
    assert refine_existing_vectors.main(
        [
            "--config",
            "unused-test-config.yaml",
            "--input-vectors",
            str(layout.post_inference_path),
            "--probability-raster",
            str(probability_path),
            "--roi",
            str(roi_path),
            "--output",
            str(standalone_path),
        ]
    ) == 0

    integrated = gpd.read_file(layout.postprocessing_path, layer="final_lines")
    standalone = gpd.read_file(standalone_path, layer="final_lines")
    assert len(integrated) == len(standalone) == 2
    assert set(integrated["run_id"]) == set(standalone["run_id"]) == {
        "run_equivalence"
    }
    assert set(STABLE_FINAL_COLUMNS) <= set(integrated.columns)
    assert set(STABLE_FINAL_COLUMNS) <= set(standalone.columns)
    integrated = integrated[STABLE_FINAL_COLUMNS].sort_values("final_line_id").reset_index(drop=True)
    standalone = standalone[STABLE_FINAL_COLUMNS].sort_values("final_line_id").reset_index(drop=True)
    assert_geodataframe_equal(
        integrated,
        standalone,
        check_dtype=False,
        check_like=True,
        check_geom_type=True,
        check_crs=True,
    )
