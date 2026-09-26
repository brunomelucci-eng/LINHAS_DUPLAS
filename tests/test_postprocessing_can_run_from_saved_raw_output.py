import hashlib

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import LineString, Polygon

from scripts import refine_existing_vectors
from src.postprocessing.output_products import (
    prepare_post_inference_lines,
    write_post_inference_product,
)


def _config():
    return {
        "outputs": {"overwrite_existing": False},
        "postprocess_pipeline": {
            "enabled": True,
            "version": "test-v1",
            "prevector_cleanup": {"enabled": False},
            "track_merge": {"enabled": True},
            "fail_on_unmatched": False,
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


def test_postprocessing_can_run_from_saved_raw_output(tmp_path, monkeypatch):
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
    probability = np.linspace(0.6, 0.95, 50 * 120, dtype=np.float32).reshape(50, 120)

    raw = prepare_post_inference_lines(
        vectorized,
        probability,
        transform,
        roi,
        config,
        run_id="run_standalone",
        checkpoint="best.pt",
        created_at="2026-07-20T12:00:00+00:00",
    )
    raw_path = tmp_path / "talhao__pos_inferencia.gpkg"
    probability_path = tmp_path / "probability_center.tif"
    roi_path = tmp_path / "roi.geojson"
    output_path = tmp_path / "talhao__pos_processamento.gpkg"
    write_post_inference_product(raw, raw_path)
    _write_probability(probability_path, probability, transform)
    roi.to_file(roi_path, driver="GeoJSON")
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    monkeypatch.setattr(refine_existing_vectors, "load_config", lambda _path: config)
    return_code = refine_existing_vectors.main(
        [
            "--config",
            "unused-test-config.yaml",
            "--input-vectors",
            str(raw_path),
            "--probability-raster",
            str(probability_path),
            "--roi",
            str(roi_path),
            "--output",
            str(output_path),
        ]
    )

    assert return_code == 0
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_hash
    assert "final_lines" in set(gpd.list_layers(output_path)["name"].astype(str))
    final = gpd.read_file(output_path, layer="final_lines")
    assert len(final) == 2
    assert set(final["run_id"]) == {"run_standalone"}
    assert final["final_line_id"].is_unique
    assert final["source_raw_ids"].str.contains("raw_").all()
    assert final.crs == raw.crs
