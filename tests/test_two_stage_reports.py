import hashlib
import json

import pytest

from src.postprocessing.output_products import create_output_layout, write_reports


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


def test_reports_and_manifest_are_complete_and_never_partially_overwritten(tmp_path):
    layout = create_output_layout(tmp_path / "run_report", "talhao.gpkg", _config())
    report = {
        "run_id": "run_report",
        "before_post_inference": {"line_count": 3, "total_length_m": 12.0},
        "after_postprocessing": {"line_count": 2, "total_length_m": 11.5},
        "changes": {
            "maximum_directed_distance_to_lineage_m": 0.04,
            "rejected_lines": 1,
        },
        "timing_seconds": {"inference": 1.25, "postprocessing": 0.75},
        "postprocess_metrics": {"production_gate_passed": True},
    }
    manifest = {
        "run_id": "run_report",
        "status": "completed",
        "post_inference": {
            "path": "vectors/talhao__pos_inferencia.gpkg",
            "complete": True,
        },
        "postprocessing": {
            "path": "vectors/talhao__pos_processamento.gpkg",
            "complete": True,
        },
    }

    write_reports(layout, report, manifest=manifest)

    assert json.loads(layout.postprocess_report_path.read_text(encoding="utf-8")) == report
    html = layout.comparison_report_path.read_text(encoding="utf-8")
    assert "ANTES: pos-inferencia / DEPOIS: pos-processamento" in html
    assert "line_count" in html
    assert "maximum_directed_distance_to_lineage_m" in html
    assert "postprocessing" in html
    assert "production_gate_passed" in html
    assert json.loads(layout.manifest_path.read_text(encoding="utf-8")) == manifest
    assert not list(layout.run_dir.rglob("*.partial-*"))

    protected = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            layout.postprocess_report_path,
            layout.comparison_report_path,
            layout.manifest_path,
        )
    }
    with pytest.raises(FileExistsError):
        write_reports(layout, report, manifest=manifest, overwrite=False)
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected
    } == protected
