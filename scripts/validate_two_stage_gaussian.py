"""Validate the two-stage Gaussian smoother on a small vector fixture.

The input is never overwritten.  All metric operations run in a projected CRS,
and the output GeoPackage stores both the original lines and the candidates so
they can be compared directly in QGIS.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

from affine import Affine
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import distance as shapely_distance
from shapely import points as shapely_points

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.postprocessing.regularized_centerline_fairing import (
    compute_serration_diagnostics,
    resample_line,
    two_stage_gaussian_smooth_line,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Testa a suavizacao Gaussiana em duas etapas sem alterar a entrada."
    )
    parser.add_argument("--input", required=True, help="GeoPackage com LineStrings.")
    parser.add_argument("--output", required=True, help="Novo GeoPackage de auditoria.")
    parser.add_argument("--layer", help="Camada de entrada; autodetectada se omitida.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--metric-epsg", type=int, help="CRS metrico de processamento.")
    parser.add_argument("--gsd-m", type=float, help="GSD usado pelo detector.")
    return parser


def _select_layer(path: str, requested: Optional[str]) -> str:
    layers = gpd.list_layers(path)
    if requested:
        if requested not in set(layers["name"].astype(str)):
            raise ValueError(f"Camada inexistente: {requested}")
        return requested
    line_layers = layers.loc[
        layers["geometry_type"].astype(str).str.contains("LineString", na=False),
        "name",
    ].astype(str)
    if len(line_layers) != 1:
        raise ValueError("Informe --layer quando houver zero ou varias camadas de linha.")
    return str(line_layers.iloc[0])


def _metric_crs(lines: gpd.GeoDataFrame, requested_epsg: Optional[int]) -> Any:
    if requested_epsg is not None:
        return f"EPSG:{requested_epsg}"
    if "crs_epsg" in lines.columns:
        values = pd.to_numeric(lines["crs_epsg"], errors="coerce").dropna()
        values = values.loc[values.between(2000, 40000)].astype(int).unique()
        if len(values) == 1:
            return f"EPSG:{int(values[0])}"
    if lines.crs is not None and lines.crs.is_projected:
        return lines.crs
    estimated = lines.estimate_utm_crs()
    if estimated is None:
        raise ValueError("Nao foi possivel inferir um CRS metrico; use --metric-epsg.")
    return estimated


def _maximum_segment_m(line) -> float:
    coordinates = np.asarray(line.coords, dtype=np.float64)[:, :2]
    if len(coordinates) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(coordinates, axis=0), axis=1).max())


def _maximum_lateral_shift_m(original, candidate) -> float:
    _, coordinates = resample_line(candidate, 0.10)
    if len(coordinates) == 0:
        return 0.0
    return float(
        np.max(shapely_distance(shapely_points(coordinates), original))
    )


def _diagnostic_fields(prefix: str, diagnostics) -> dict[str, Any]:
    return {
        f"{prefix}_failed": bool(diagnostics.failed),
        f"{prefix}_score": float(diagnostics.score),
        f"{prefix}_micro_p95_deg": float(diagnostics.micro_p95_turn_deg),
        f"{prefix}_vertex_p95_deg": float(diagnostics.vertex_p95_turn_deg),
        f"{prefix}_inversions_m": float(diagnostics.inversions_per_m),
        f"{prefix}_residual_p95_m": float(diagnostics.residual_p95_m),
        f"{prefix}_length_excess": float(diagnostics.length_excess_ratio),
    }


def _new_contacts(originals: Sequence[Any], candidates: Sequence[Any]) -> set[int]:
    affected: set[int] = set()
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            had_contact = originals[left].intersects(originals[right])
            has_contact = candidates[left].intersects(candidates[right])
            if has_contact and not had_contact:
                affected.update((left, right))
    return affected


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if input_path == output_path:
        raise ValueError("A saida deve ser diferente da entrada.")
    if output_path.exists():
        raise FileExistsError(f"Recusando sobrescrever: {output_path}")

    layer = _select_layer(str(input_path), args.layer)
    source = gpd.read_file(input_path, layer=layer)
    if source.empty or not source.geom_type.eq("LineString").all():
        raise ValueError("A camada deve conter LineStrings nao vazias.")
    if source.crs is None:
        raise ValueError("A camada de entrada nao possui CRS.")

    processing_crs = _metric_crs(source, args.metric_epsg)
    metric = source.to_crs(processing_crs).reset_index(drop=True)
    config = load_config(args.config)
    if args.gsd_m is not None:
        config["final_centerline_fairing"]["micro_serration"][
            "default_gsd_m"
        ] = float(args.gsd_m)
    gsd_m = float(
        config["final_centerline_fairing"]["micro_serration"].get(
            "default_gsd_m", 0.04
        )
    )
    transform = Affine.scale(gsd_m, -gsd_m)

    candidates = []
    records: list[dict[str, Any]] = []
    for index, original in enumerate(metric.geometry):
        candidate, metadata = two_stage_gaussian_smooth_line(
            original, config, transform=transform
        )
        before = compute_serration_diagnostics(original, config, transform)
        after = compute_serration_diagnostics(candidate, config, transform)
        endpoint_shift = max(
            float(np.linalg.norm(np.asarray(candidate.coords[0]) - np.asarray(original.coords[0]))),
            float(np.linalg.norm(np.asarray(candidate.coords[-1]) - np.asarray(original.coords[-1]))),
        )
        hausdorff = float(original.hausdorff_distance(candidate))
        lateral_shift = _maximum_lateral_shift_m(original, candidate)
        length_change_ratio = float(candidate.length / max(original.length, 1e-12) - 1.0)
        candidates.append(candidate)
        record = {
            "source_index": index,
            "source_row_id": str(metric.iloc[index].get("row_id", index)),
            "gaussian_stage": metadata.get("two_stage_gaussian_stage"),
            "aggressive_attempted": bool(
                metadata.get("two_stage_gaussian_aggressive_attempted", False)
            ),
            "dense_failed_after": bool(
                metadata.get("two_stage_gaussian_failed_after", False)
            ),
            "dense_score_light": float(
                metadata.get("two_stage_gaussian_score_light", np.nan)
            ),
            "dense_score_after": float(
                metadata.get("two_stage_gaussian_score_after", np.nan)
            ),
            "hausdorff_m": hausdorff,
            "max_lateral_shift_m": lateral_shift,
            "endpoint_shift_m": endpoint_shift,
            "max_segment_m": _maximum_segment_m(candidate),
            "length_before_m": float(original.length),
            "length_after_m": float(candidate.length),
            "length_change_ratio": length_change_ratio,
            "vertices_before": len(original.coords),
            "vertices_after": len(candidate.coords),
            "is_valid": bool(candidate.is_valid),
            "is_simple": bool(candidate.is_simple),
            **_diagnostic_fields("before", before),
            **_diagnostic_fields("after", after),
        }
        records.append(record)

    contact_indices = _new_contacts(list(metric.geometry), candidates)
    geometry_cfg = config["final_centerline_fairing"].get("geometry", {})
    max_hausdorff = float(geometry_cfg.get("max_hausdorff_m", 0.12))
    max_lateral = float(geometry_cfg.get("max_lateral_shift_m", 0.10))
    max_segment = float(
        config["final_centerline_fairing"]
        .get("adaptive_output_sampling", {})
        .get("max_vertex_spacing_m", 3.0)
    )
    for index, record in enumerate(records):
        reasons = []
        if record["dense_failed_after"]:
            reasons.append("residual_serration_dense")
        if record["hausdorff_m"] > max_hausdorff + 1e-9:
            reasons.append("hausdorff_limit")
        if record["max_lateral_shift_m"] > max_lateral + 1e-9:
            reasons.append("lateral_shift_limit")
        if record["max_segment_m"] > max_segment + 2e-9:
            reasons.append("max_segment_limit")
        if record["endpoint_shift_m"] > 1e-8:
            reasons.append("endpoint_shift")
        if not record["is_valid"] or not record["is_simple"]:
            reasons.append("invalid_geometry")
        if index in contact_indices:
            reasons.append("new_network_contact")
        record["new_network_contact"] = index in contact_indices
        record["accepted_preview"] = not reasons
        record["rejection_reason"] = ",".join(reasons)

    audit_metric = gpd.GeoDataFrame(records, geometry=candidates, crs=processing_crs)
    audit_export = audit_metric.to_crs(source.crs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source.to_file(output_path, layer="original_input", driver="GPKG")
    audit_export.to_file(
        output_path, layer="gaussian_candidates", driver="GPKG", mode="a"
    )

    report = {
        "input": str(input_path),
        "input_layer": layer,
        "output": str(output_path),
        "input_crs": str(source.crs),
        "processing_crs": str(processing_crs),
        "gsd_m": gsd_m,
        "line_count": len(records),
        "accepted_count": int(sum(item["accepted_preview"] for item in records)),
        "aggressive_count": int(sum(item["aggressive_attempted"] for item in records)),
        "new_network_contact_count": len(contact_indices),
        "lines": records,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
