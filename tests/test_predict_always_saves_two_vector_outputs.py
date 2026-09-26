import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import LineString, Polygon

from src.postprocessing.output_products import (
    create_output_layout,
    persist_two_stage_vector_outputs,
)
from src.postprocessing.postprocess_pipeline import PostprocessResult


RAW_REQUIRED = {
    "run_id",
    "raw_line_id",
    "source_stage",
    "checkpoint",
    "config_hash",
    "length_m",
    "mean_probability",
    "median_probability",
    "geometry_status",
    "created_at",
}
FINAL_REQUIRED = {
    "run_id",
    "final_line_id",
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
    "postprocess_status",
    "postprocess_version",
    "created_at",
}


def _config():
    return {
        "outputs": {
            "always_save_post_inference": True,
            "always_save_postprocessing": True,
            "overwrite_existing": False,
            "save_lineage": True,
            "save_comparison_report": True,
            "naming": {
                "post_inference_suffix": "__pos_inferencia",
                "postprocessing_suffix": "__pos_processamento",
            },
        },
        "metadata": {"probability_sample_spacing_m": 0.10},
    }


def _inputs():
    lines = gpd.GeoDataFrame(
        geometry=[
            LineString([(1.0, 0.0), (9.0, 0.0)]),
            LineString([(1.0, 0.9), (9.0, 0.9)]),
        ],
        crs="EPSG:3857",
    )
    roi = gpd.GeoDataFrame(
        geometry=[Polygon([(0, -1), (10, -1), (10, 2), (0, 2)])],
        crs=lines.crs,
    )
    probability = np.full((50, 120), 0.8, dtype=np.float32)
    transform = Affine.translation(0.0, 3.0) * Affine.scale(0.10, -0.10)
    return lines, roi, probability, transform


def _identity_postprocess(lines, *_args, **_kwargs):
    final = lines.copy(deep=True)
    final["track_id"] = [10, 11]
    final["pair_id"] = 1
    final["pair_status"] = "valid_pair"
    final["topology_status"] = "valid"
    final["review_required"] = False
    final["postprocess_version"] = "test-v1"
    empty = gpd.GeoDataFrame(geometry=[], crs=lines.crs)
    return PostprocessResult(
        final_lines=final,
        raw_lines=lines.copy(deep=True),
        repaired_segments=empty.copy(),
        reconstructed_pairs=empty.copy(),
        terminal_extensions=empty.copy(),
        rejected_lines=empty.copy(),
        manual_review=empty.copy(),
        metrics={"status": "completed"},
    )


def test_predict_always_saves_two_vector_outputs(tmp_path):
    config = _config()
    lines, roi, probability, transform = _inputs()
    layout = create_output_layout(
        tmp_path / "run_test",
        "talhao__pos_processamento.gpkg",
        config,
    )

    persisted = persist_two_stage_vector_outputs(
        lines,
        probability,
        transform,
        roi,
        config,
        layout=layout,
        run_id="run_test_001",
        checkpoint="best.pt",
        postprocess_fn=_identity_postprocess,
    )

    assert layout.post_inference_path.name == "talhao__pos_inferencia.gpkg"
    assert layout.postprocessing_path.name == "talhao__pos_processamento.gpkg"
    assert layout.post_inference_path != layout.postprocessing_path
    assert layout.post_inference_path.is_file()
    assert layout.postprocessing_path.is_file()
    assert gpd.list_layers(layout.post_inference_path)["name"].tolist() == [
        "post_inference_lines"
    ]
    assert set(gpd.list_layers(layout.postprocessing_path)["name"].astype(str)) == {
        "final_lines",
        "approved_lines",
        "production_lines",
        "manual_review",
        "rejected_lines",
        "repaired_segments",
        "reconstructed_pairs",
        "terminal_extensions",
    }

    raw = gpd.read_file(layout.post_inference_path, layer="post_inference_lines")
    final = gpd.read_file(layout.postprocessing_path, layer="final_lines")
    approved = gpd.read_file(layout.postprocessing_path, layer="approved_lines")
    production = gpd.read_file(layout.postprocessing_path, layer="production_lines")
    assert RAW_REQUIRED <= set(raw.columns)
    assert FINAL_REQUIRED <= set(final.columns)
    assert set(raw["run_id"]) == {"run_test_001"}
    assert set(final["run_id"]) == {"run_test_001"}
    assert set(raw["source_stage"]) == {"post_inference"}
    assert set(final["source_stage"]) == {"postprocessing"}
    assert raw.crs == final.crs == lines.crs
    assert persisted.raw_lines.geometry.to_wkb().tolist() == raw.geometry.to_wkb().tolist()
    assert persisted.final_lines.geometry.to_wkb().tolist() == final.geometry.to_wkb().tolist()
    assert set(approved["postprocess_status"]) == {"approved"}
    assert production.geometry.to_wkb().tolist() == approved.geometry.to_wkb().tolist()

    for mandatory_flag in (
        "always_save_post_inference",
        "always_save_postprocessing",
        "save_lineage",
        "save_comparison_report",
    ):
        invalid = _config()
        invalid["outputs"][mandatory_flag] = False
        with pytest.raises(ValueError):
            create_output_layout(tmp_path / mandatory_flag, "talhao.gpkg", invalid)


def test_postprocessing_product_exists_when_every_candidate_is_rejected(tmp_path):
    config = _config()
    lines, roi, probability, transform = _inputs()
    layout = create_output_layout(tmp_path / "run_rejected", "talhao.gpkg", config)

    def reject_all(raw_lines, *_args, **_kwargs):
        final = raw_lines.iloc[0:0].copy()
        empty = gpd.GeoDataFrame(geometry=[], crs=raw_lines.crs)
        rejected = raw_lines.copy(deep=True)
        rejected["rejection_reason"] = "controlled_test_rejection"
        return PostprocessResult(
            final_lines=final,
            raw_lines=raw_lines.copy(deep=True),
            repaired_segments=empty.copy(),
            reconstructed_pairs=empty.copy(),
            terminal_extensions=empty.copy(),
            rejected_lines=rejected,
            manual_review=empty.copy(),
            metrics={"status": "manual_review"},
        )

    persist_two_stage_vector_outputs(
        lines,
        probability,
        transform,
        roi,
        config,
        layout=layout,
        run_id="run_rejected",
        checkpoint="best.pt",
        postprocess_fn=reject_all,
    )

    assert layout.post_inference_path.is_file()
    assert layout.postprocessing_path.is_file()
    assert gpd.read_file(layout.postprocessing_path, layer="final_lines").empty
    rejected = gpd.read_file(layout.postprocessing_path, layer="rejected_lines")
    assert len(rejected) == len(lines)
