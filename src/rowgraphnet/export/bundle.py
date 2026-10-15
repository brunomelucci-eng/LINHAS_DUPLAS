"""High-level export of one complete RowGraphNet inference bundle."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from rowgraphnet.domain import RasterPrediction, RepairPath, RowRecord
from rowgraphnet.export.geopackage import write_output_geopackages
from rowgraphnet.export.rasters import write_prediction_rasters
from rowgraphnet.export.report import REPORT_FILENAME, write_quality_report


def export_inference_bundle(
    output_dir: str | Path,
    prediction: RasterPrediction,
    *,
    rows: Iterable[RowRecord] = (),
    repairs: Iterable[RepairPath] = (),
    uncertain_rows: Iterable[RowRecord] = (),
    run_id: str,
    model_version: str,
    metrics: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    configuration: Mapping[str, Any] | None = None,
    input_files: Mapping[str, str | Path] | None = None,
    created_at: datetime | None = None,
) -> dict[str, Path]:
    """Write vectors, continuous rasters, and their fixed-schema report."""

    output = Path(output_dir)
    final_rows = tuple(rows)
    uncertain = tuple(uncertain_rows)
    mismatched_versions = sorted(
        {
            row.model_version
            for row in (*final_rows, *uncertain)
            if row.model_version != model_version
        }
    )
    if mismatched_versions:
        versions = ", ".join(mismatched_versions)
        raise ValueError(
            f"row model versions do not match bundle version {model_version}: {versions}"
        )

    vector_paths = write_output_geopackages(
        output,
        final_rows,
        repairs,
        uncertain,
        prediction.crs,
        model_version=model_version,
    )
    raster_paths = write_prediction_rasters(output, prediction)
    artifact_paths = {
        "linhas_simples": vector_paths["lines"],
        "trechos_reconstruidos": vector_paths["repairs"],
        "linhas_incertas": vector_paths["uncertain"],
        **{f"raster_{name}": path for name, path in raster_paths.items()},
    }
    report_path = write_quality_report(
        output / REPORT_FILENAME,
        run_id=run_id,
        model_version=model_version,
        crs=prediction.crs,
        metrics=metrics,
        provenance=provenance,
        configuration=configuration,
        input_files=input_files,
        output_files=artifact_paths,
        created_at=created_at,
    )
    return {**artifact_paths, "quality_report": report_path}


export_bundle = export_inference_bundle
