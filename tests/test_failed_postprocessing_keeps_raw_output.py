import json

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from src.postprocessing.output_products import (
    create_output_layout,
    persist_two_stage_vector_outputs,
)


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


def test_failed_postprocessing_keeps_raw_output(tmp_path):
    config = _config()
    lines = gpd.GeoDataFrame(
        geometry=[LineString([(1.0, 0.0), (9.0, 0.0)])], crs="EPSG:3857"
    )
    roi = gpd.GeoDataFrame(
        geometry=[Polygon([(0, -1), (10, -1), (10, 1), (0, 1)])],
        crs=lines.crs,
    )
    layout = create_output_layout(tmp_path / "run", "talhao.gpkg", config)

    def fail_after_raw_is_visible(raw_lines, *_args, **_kwargs):
        assert layout.post_inference_path.is_file()
        saved = gpd.read_file(
            layout.post_inference_path, layer="post_inference_lines"
        )
        assert saved["raw_line_id"].tolist() == raw_lines["raw_line_id"].tolist()
        raise RuntimeError("falha controlada no pos-processamento")

    with pytest.raises(RuntimeError, match="falha controlada"):
        persist_two_stage_vector_outputs(
            lines,
            None,
            None,
            roi,
            config,
            layout=layout,
            run_id="run_failure",
            checkpoint="best.pt",
            postprocess_fn=fail_after_raw_is_visible,
        )

    assert layout.post_inference_path.is_file()
    assert not layout.postprocessing_path.exists()
    raw = gpd.read_file(layout.post_inference_path, layer="post_inference_lines")
    assert len(raw) == 1
    assert set(raw["run_id"]) == {"run_failure"}
    assert set(raw["source_stage"]) == {"post_inference"}
    assert layout.manifest_path.is_file()
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "postprocessing_failed"
    assert manifest["post_inference"]["complete"] is True
    assert manifest["postprocessing"]["complete"] is False
    assert manifest["error_type"] == "RuntimeError"
    assert "falha controlada" in manifest["error"]
