"""Metrics and reports used by the three-ROI diagnostic workflow."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy import ndimage

from src.postprocessing.skeleton_graph import skeleton_to_graph_and_branches


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _branch_length_m(branch: list[tuple[int, int]], transform) -> float:
    if len(branch) < 2:
        return 0.0
    points = np.asarray(branch, dtype=np.float64)
    drow = np.diff(points[:, 0])
    dcol = np.diff(points[:, 1])
    dx = transform.a * dcol + transform.b * drow
    dy = transform.d * dcol + transform.e * drow
    return float(np.hypot(dx, dy).sum())


def build_roi_diagnostic_report(
    probability_path: str | Path,
    cleaned_mask_path: str | Path,
    skeleton_path: str | Path,
    config: dict,
    *,
    mosaic_weights_path: str | Path | None = None,
    checkpoint: str | None = None,
    orthomosaic: str | None = None,
    roi: str | None = None,
) -> dict[str, Any]:
    """Calculate the mandatory diagnostic metrics from pipeline artefacts."""
    high = float(config.get("postprocessing", {}).get("center_threshold_high", 0.42))
    low = float(config.get("postprocessing", {}).get("center_threshold_low", 0.22))

    with rasterio.open(probability_path) as src:
        probability = src.read(1).astype(np.float32, copy=False)
        raster_valid = src.read_masks(1) != 0

    if mosaic_weights_path is not None:
        with rasterio.open(mosaic_weights_path) as src:
            weights = src.read(1)
        if weights.shape != probability.shape:
            raise ValueError("Mosaic weights and center probability must have identical shapes.")
        valid = (weights > 0) & np.isfinite(probability)
    else:
        valid = raster_valid & np.isfinite(probability)
    probability_values = probability[valid].astype(np.float64, copy=False)

    with rasterio.open(cleaned_mask_path) as src:
        cleaned_mask = src.read(1) != 0

    with rasterio.open(skeleton_path) as src:
        skeleton = (src.read(1) != 0).astype(np.uint8)
        transform = src.transform

    if probability_values.size == 0:
        raise ValueError("The center-probability raster has no valid pixels.")
    if probability.shape != cleaned_mask.shape or probability.shape != skeleton.shape:
        raise ValueError("Probability, cleaned mask, and skeleton must have identical shapes.")

    _, connected_components = ndimage.label(
        cleaned_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    branches, junctions, endpoints = skeleton_to_graph_and_branches(skeleton, config)
    lengths = np.asarray(
        [_branch_length_m(branch, transform) for branch in branches],
        dtype=np.float64,
    )
    total_branch_length_m = float(lengths.sum()) if lengths.size else 0.0

    probability_metrics = {
        "mean": _finite(np.mean(probability_values)),
        "median": _finite(np.median(probability_values)),
        "p20": _finite(np.percentile(probability_values, 20)),
        "fraction_above_low_threshold": float(np.mean(probability_values >= low)),
        "fraction_above_high_threshold": float(np.mean(probability_values >= high)),
        "low_threshold": low,
        "high_threshold": high,
        "valid_pixels": int(probability_values.size),
    }
    topology_metrics = {
        "connected_components": int(connected_components),
        "skeleton_pixels": int(np.count_nonzero(skeleton)),
        "endpoints": int(len(endpoints)),
        "junction_pixels": int(len(junctions)),
        "branches": int(len(branches)),
        "median_branch_length_m": _finite(np.median(lengths)) if lengths.size else None,
        "branches_shorter_than_1m_fraction": (
            float(np.mean(lengths < 1.0)) if lengths.size else None
        ),
        "total_branch_length_m": total_branch_length_m,
        "endpoints_per_100m": (
            float(len(endpoints) * 100.0 / total_branch_length_m)
            if total_branch_length_m > 0
            else None
        ),
    }
    return {
        "schema_version": 1,
        "inputs": {
            "checkpoint": checkpoint,
            "orthomosaic": orthomosaic,
            "roi": roi,
        },
        "probability": probability_metrics,
        "topology_before_vectorization": topology_metrics,
    }


def write_diagnostic_reports(report: dict[str, Any], output_dir: str | Path) -> None:
    """Write machine-readable JSON and a compact, self-contained HTML report."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "diagnostic_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    rows: list[str] = []
    for section_name in ("probability", "topology_before_vectorization"):
        section = report[section_name]
        for key, value in section.items():
            shown = "n/a" if value is None else value
            rows.append(
                "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    html.escape(section_name),
                    html.escape(str(key)),
                    html.escape(str(shown)),
                )
            )
    document = """<!doctype html>
<html lang=\"pt-BR\"><head><meta charset=\"utf-8\">
<title>Diagnóstico de ROI</title>
<style>body{font-family:system-ui;margin:2rem;color:#17202a}table{border-collapse:collapse}
th,td{padding:.45rem .7rem;border:1px solid #ccd1d1;text-align:left}th{background:#eef2f3}</style>
</head><body><h1>Diagnóstico de ROI</h1>
<table><thead><tr><th>Seção</th><th>Métrica</th><th>Valor</th></tr></thead><tbody>
""" + "\n".join(rows) + "\n</tbody></table></body></html>\n"
    (output / "diagnostic_report.html").write_text(document, encoding="utf-8")
