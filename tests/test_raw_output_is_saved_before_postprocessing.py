import geopandas as gpd
from shapely.geometry import LineString, Polygon

import src.postprocessing.output_products as output_products
from src.postprocessing.output_products import (
    create_output_layout,
    persist_two_stage_vector_outputs,
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


def test_raw_output_is_saved_before_postprocessing(tmp_path, monkeypatch):
    config = _config()
    lines = gpd.GeoDataFrame(
        geometry=[LineString([(1.0, 0.0), (9.0, 0.0)])], crs="EPSG:3857"
    )
    roi = gpd.GeoDataFrame(
        geometry=[Polygon([(0, -1), (10, -1), (10, 1), (0, 1)])],
        crs=lines.crs,
    )
    layout = create_output_layout(tmp_path / "run", "talhao.gpkg", config)
    events = []

    real_raw_writer = output_products.write_post_inference_product
    real_final_writer = output_products.write_postprocess_product

    def raw_writer(*args, **kwargs):
        written = real_raw_writer(*args, **kwargs)
        events.append("raw_written")
        return written

    def final_writer(*args, **kwargs):
        written = real_final_writer(*args, **kwargs)
        events.append("final_written")
        return written

    def postprocess(raw_lines, *_args, **_kwargs):
        assert layout.post_inference_path.is_file()
        saved = gpd.read_file(
            layout.post_inference_path, layer="post_inference_lines"
        )
        assert saved["raw_line_id"].tolist() == raw_lines["raw_line_id"].tolist()
        assert saved.geometry.to_wkb().tolist() == raw_lines.geometry.to_wkb().tolist()
        events.append("postprocess")
        final = raw_lines.copy()
        final["pair_status"] = "unmatched"
        final["topology_status"] = "valid"
        final["postprocess_version"] = "test-v1"
        empty = gpd.GeoDataFrame(geometry=[], crs=raw_lines.crs)
        return PostprocessResult(
            final_lines=final,
            raw_lines=raw_lines.copy(),
            repaired_segments=empty.copy(),
            reconstructed_pairs=empty.copy(),
            terminal_extensions=empty.copy(),
            rejected_lines=empty.copy(),
            manual_review=empty.copy(),
            metrics={"status": "manual_review"},
        )

    monkeypatch.setattr(output_products, "write_post_inference_product", raw_writer)
    monkeypatch.setattr(output_products, "write_postprocess_product", final_writer)

    persist_two_stage_vector_outputs(
        lines,
        None,
        None,
        roi,
        config,
        layout=layout,
        run_id="run_order",
        checkpoint="best.pt",
        postprocess_fn=postprocess,
    )

    assert events == ["raw_written", "postprocess", "final_written"]
    assert layout.postprocessing_path.is_file()
