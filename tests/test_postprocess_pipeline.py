from dataclasses import fields

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import LineString, Polygon

import src.postprocessing.postprocess_pipeline as pipeline_module
from src.postprocessing.postprocess_pipeline import (
    PostprocessResult,
    run_postprocessing,
)


def _pipeline_config(enabled=True):
    return {
        "postprocess_pipeline": {
            "enabled": enabled,
            "version": "test-v1",
            "fail_on_unmatched": True,
            "fail_on_crossing": True,
        },
        "postprocessing": {
            "double_row_validation": {
                "enabled": True,
                "mode": "annotate",
                "intra_pair_min_m": 0.85,
                "intra_pair_max_m": 1.00,
                "inter_pair_min_m": 1.15,
                "inter_pair_max_m": 1.40,
                "tolerance_m": 0.08,
                "max_angle_difference_deg": 8.0,
                "min_longitudinal_overlap_ratio": 0.50,
            }
        },
        "deduplication": {"enabled": True},
        "double_row_refinement": {"enabled": False},
        "final_centerline_fairing": {"enabled": False},
        "line_extension": {"enabled": False},
        "vector": {"min_line_length_m": 1.0},
        "metadata": {"probability_sample_spacing_m": 0.10},
    }


def _inputs():
    lines = gpd.GeoDataFrame(
        {"input_name": ["left", "right"]},
        geometry=[
            LineString([(1.0, 0.0), (9.0, 0.0)]),
            LineString([(1.0, 0.9), (9.0, 0.9)]),
        ],
        crs="EPSG:3857",
    )
    roi = gpd.GeoDataFrame(
        geometry=[Polygon([(0, -2), (10, -2), (10, 3), (0, 3)])],
        crs=lines.crs,
    )
    probability = np.ones((100, 200), dtype=np.float32)
    transform = Affine.translation(0.0, 3.0) * Affine.scale(0.05, -0.05)
    return lines, roi, probability, transform


def test_run_postprocessing_returns_contract_without_mutating_raw():
    lines, roi, probability, transform = _inputs()
    original_columns = lines.columns.tolist()
    original_wkb = lines.geometry.to_wkb().tolist()

    result = run_postprocessing(
        lines, probability, transform, roi, _pipeline_config(), debug=True
    )

    assert isinstance(result, PostprocessResult)
    assert [field.name for field in fields(PostprocessResult)] == [
        "final_lines",
        "raw_lines",
        "repaired_segments",
        "reconstructed_pairs",
        "terminal_extensions",
        "rejected_lines",
        "manual_review",
        "metrics",
    ]
    for name in [field.name for field in fields(PostprocessResult)[:-1]]:
        frame = getattr(result, name)
        assert isinstance(frame, gpd.GeoDataFrame)
        assert frame.crs == lines.crs
    assert lines.columns.tolist() == original_columns
    assert lines.geometry.to_wkb().tolist() == original_wkb
    assert result.raw_lines.geometry.to_wkb().tolist() == original_wkb
    assert len(result.final_lines) == 2
    assert result.metrics["production_gate_passed"] is True


def test_disabled_pipeline_is_exact_geometry_noop(monkeypatch):
    lines, roi, probability, transform = _inputs()
    original_wkb = lines.geometry.to_wkb().tolist()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A disabled pipeline must not run geometry stages.")

    monkeypatch.setattr(pipeline_module, "remove_duplicate_lines", forbidden)
    monkeypatch.setattr(pipeline_module, "validate_double_rows", forbidden)
    monkeypatch.setattr(pipeline_module, "refine_double_rows", forbidden)
    monkeypatch.setattr(pipeline_module, "fair_centerlines_gdf", forbidden)

    result = run_postprocessing(
        lines,
        probability,
        transform,
        roi,
        _pipeline_config(enabled=False),
    )

    assert result.metrics["status"] == "disabled"
    assert result.final_lines.geometry.to_wkb().tolist() == original_wkb
    assert result.final_lines.columns.tolist() == lines.columns.tolist()
    assert result.repaired_segments.empty
    assert result.reconstructed_pairs.empty
    assert result.terminal_extensions.empty
    assert result.rejected_lines.empty


def test_second_pass_is_exactly_idempotent_and_has_provenance():
    lines, roi, probability, transform = _inputs()
    config = _pipeline_config()

    first = run_postprocessing(lines, probability, transform, roi, config)
    second = run_postprocessing(
        first.final_lines.copy(), probability, transform, roi, config
    )

    assert second.metrics["status"] == "already_processed"
    assert second.final_lines.geometry.to_wkb().tolist() == (
        first.final_lines.geometry.to_wkb().tolist()
    )
    assert second.final_lines.columns.tolist() == first.final_lines.columns.tolist()
    for column in (
        "postprocess_version",
        "postprocess_status",
        "source_type",
        "was_repaired",
        "was_reconstructed",
        "was_terminal_extended",
    ):
        assert column in first.final_lines.columns
        assert first.final_lines[column].notna().all()
    assert first.final_lines["was_repaired"].dtype == bool
    assert first.final_lines["was_reconstructed"].dtype == bool
    assert first.final_lines["was_terminal_extended"].dtype == bool
    assert second.repaired_segments.empty
    assert second.reconstructed_pairs.empty
    assert second.terminal_extensions.empty


def test_predict_and_refine_delegate_to_the_same_engine():
    from scripts import predict, refine_existing_vectors

    assert predict.run_postprocessing is pipeline_module.run_postprocessing
    assert refine_existing_vectors.run_postprocessing is pipeline_module.run_postprocessing
    predict_source = open("scripts/predict.py", encoding="utf-8").read()
    refine_source = open(
        "scripts/refine_existing_vectors.py", encoding="utf-8"
    ).read()
    assert predict_source.count("postprocess_fn=run_postprocessing") == 1
    assert refine_source.count("postprocess_result = run_postprocessing(") == 1
    for source in (predict_source, refine_source):
        for forbidden in (
            "refine_double_rows(",
            "fair_centerlines_gdf(",
            "validate_topology(",
            "validate_double_rows(",
        ):
            assert forbidden not in source


def test_final_fairing_is_the_only_enabled_terminal_owner(monkeypatch):
    lines, roi, probability, transform = _inputs()
    config = _pipeline_config()
    config["double_row_refinement"] = {
        "enabled": True,
        "terminal_extension": {"enabled": True},
    }
    config["final_centerline_fairing"] = {
        "enabled": True,
        "terminal": {"enabled": True},
    }
    observed = {}

    def refinement_stub(frame, _roi, stage_config, **_kwargs):
        observed["legacy_terminal_enabled"] = stage_config[
            "double_row_refinement"
        ]["terminal_extension"]["enabled"]
        return frame.copy(), gpd.GeoDataFrame(geometry=[], crs=frame.crs)

    def fairing_stub(frame, _roi, stage_config, **_kwargs):
        observed["final_terminal_enabled"] = stage_config[
            "final_centerline_fairing"
        ]["terminal"]["enabled"]
        return frame.copy(), gpd.GeoDataFrame(geometry=[], crs=frame.crs)

    monkeypatch.setattr(pipeline_module, "refine_double_rows", refinement_stub)
    monkeypatch.setattr(pipeline_module, "fair_centerlines_gdf", fairing_stub)

    run_postprocessing(lines, probability, transform, roi, config)

    assert observed == {
        "legacy_terminal_enabled": False,
        "final_terminal_enabled": True,
    }
    assert config["double_row_refinement"]["terminal_extension"]["enabled"] is True


def test_connected_fragments_merge_only_by_track_id():
    lines = gpd.GeoDataFrame(
        {"track_id": [7, 7]},
        geometry=[
            LineString([(1.0, 0.0), (5.0, 0.0)]),
            LineString([(5.0, 0.0), (9.0, 0.0)]),
        ],
        crs="EPSG:3857",
    )
    _, roi, probability, transform = _inputs()

    result = run_postprocessing(
        lines, probability, transform, roi, _pipeline_config()
    )

    assert len(result.final_lines) == 1
    assert result.final_lines.iloc[0]["track_id"] == 7
    assert result.final_lines.iloc[0]["track_merged"] == True
    assert result.final_lines.iloc[0]["track_fragment_count"] == 2
    # A safe consolidation is a merge, not a geometric repair.  The persisted
    # final product exposes these as independent provenance flags.
    assert result.final_lines.iloc[0]["was_repaired"] == False
    assert result.final_lines.geometry.iloc[0].length == 8.0
    assert result.metrics["production_gate_passed"] is False
    assert len(result.manual_review) == 1


def test_crossings_are_rejected_and_exported_for_audit():
    lines = gpd.GeoDataFrame(
        geometry=[
            LineString([(1.0, 0.0), (9.0, 0.0)]),
            LineString([(5.0, -2.0), (5.0, 2.0)]),
        ],
        crs="EPSG:3857",
    )
    _, roi, probability, transform = _inputs()

    result = run_postprocessing(
        lines, probability, transform, roi, _pipeline_config(), debug=True
    )

    assert result.final_lines.empty
    assert len(result.rejected_lines) == 2
    assert result.rejected_lines["has_crossing"].all()
    assert set(result.rejected_lines["postprocess_rejection_stage"]) == {"topology"}
    assert result.metrics["crossing_count"] == 2
    assert result.metrics["production_gate_passed"] is False
    assert result.metrics["production_gate_reasons"] == ("crossings",)


def test_unmatched_line_is_not_silently_reconstructed():
    lines = gpd.GeoDataFrame(
        geometry=[LineString([(1.0, 0.0), (9.0, 0.0)])],
        crs="EPSG:3857",
    )
    _, roi, probability, transform = _inputs()

    result = run_postprocessing(
        lines, probability, transform, roi, _pipeline_config()
    )

    assert len(result.final_lines) == 1
    assert result.reconstructed_pairs.empty
    assert len(result.manual_review) == 1
    assert result.final_lines.iloc[0]["postprocess_status"] == "manual_review"
    assert result.metrics["unmatched_count"] == 1
    assert result.metrics["production_gate_passed"] is False
    assert result.metrics["production_gate_reasons"] == ("unmatched_lines",)
