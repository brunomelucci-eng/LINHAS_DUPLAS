from dataclasses import replace
import hashlib

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from src.postprocessing.output_products import (
    create_output_layout,
    persist_two_stage_vector_outputs,
    prepare_final_lines,
    prepare_post_inference_lines,
    write_compatibility_vector_product,
    write_post_inference_product,
    write_postprocess_product,
)
from src.postprocessing.postprocess_pipeline import PostprocessResult


def _config():
    return {
        "outputs": {
            "always_save_post_inference": True,
            "always_save_postprocessing": True,
            "overwrite_existing": False,
            "save_lineage": True,
            "save_comparison_report": True,
        }
    }


def _empty_result(final):
    empty = gpd.GeoDataFrame(geometry=[], crs=final.crs)
    return PostprocessResult(
        final_lines=final,
        raw_lines=final.copy(),
        repaired_segments=empty.copy(),
        reconstructed_pairs=empty.copy(),
        terminal_extensions=empty.copy(),
        rejected_lines=empty.copy(),
        manual_review=empty.copy(),
        metrics={"status": "completed"},
    )


def test_postprocessing_never_overwrites_raw_output(tmp_path):
    config = _config()
    lines = gpd.GeoDataFrame(
        geometry=[LineString([(1.0, 0.0), (9.0, 0.0)])], crs="EPSG:3857"
    )
    roi = gpd.GeoDataFrame(
        geometry=[Polygon([(0, -1), (10, -1), (10, 1), (0, 1)])],
        crs=lines.crs,
    )
    raw = prepare_post_inference_lines(
        lines,
        None,
        None,
        roi,
        config,
        run_id="run_collision",
        checkpoint="best.pt",
        created_at="2026-07-20T12:00:00+00:00",
    )
    raw_path = tmp_path / "talhao__pos_inferencia.gpkg"
    write_post_inference_product(raw, raw_path)
    before_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    candidate = raw.copy()
    candidate["pair_status"] = "unmatched"
    candidate["topology_status"] = "valid"
    candidate["postprocess_version"] = "test-v1"
    final = prepare_final_lines(
        candidate,
        run_id="run_collision",
        created_at="2026-07-20T12:00:00+00:00",
        valid_raw_line_ids=raw["raw_line_id"].tolist(),
    )
    result = _empty_result(final)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_postprocess_product(result, raw_path, final_lines=final, overwrite=False)

    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == before_hash
    assert gpd.list_layers(raw_path)["name"].tolist() == ["post_inference_lines"]
    on_disk = gpd.read_file(raw_path, layer="post_inference_lines")
    assert on_disk.geometry.to_wkb().tolist() == raw.geometry.to_wkb().tolist()

    layout = create_output_layout(tmp_path / "run", "talhao.gpkg", config)
    colliding_layout = replace(layout, postprocessing_path=layout.post_inference_path)
    with pytest.raises(ValueError, match="paths must differ"):
        persist_two_stage_vector_outputs(
            lines,
            None,
            None,
            roi,
            config,
            layout=colliding_layout,
            run_id="run_collision",
            checkpoint="best.pt",
            postprocess_fn=lambda *_args, **_kwargs: result,
        )
    assert not layout.post_inference_path.exists()

    alias_path = tmp_path / "legacy_alias.gpkg"
    write_compatibility_vector_product(final, alias_path)
    alias_hash = hashlib.sha256(alias_path.read_bytes()).hexdigest()
    assert gpd.list_layers(alias_path)["name"].tolist() == ["predicted_rows"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_compatibility_vector_product(final, alias_path, overwrite=False)
    assert hashlib.sha256(alias_path.read_bytes()).hexdigest() == alias_hash
    assert not list(tmp_path.glob("*.partial-*"))
