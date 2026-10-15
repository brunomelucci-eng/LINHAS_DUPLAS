"""GeoPackage writers with stable schemas, including empty output layers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from rowgraphnet.domain import RepairPath, RowRecord
from rowgraphnet.validation import validate_repair_paths, validate_row_records

LINES_FILENAME = "linhas_simples.gpkg"
REPAIRS_FILENAME = "trechos_reconstruidos.gpkg"
UNCERTAIN_FILENAME = "linhas_incertas.gpkg"

LINES_LAYER = "linhas_simples"
REPAIRS_LAYER = "trechos_reconstruidos"
UNCERTAIN_LAYER = "linhas_incertas"

ROW_COLUMNS: tuple[str, ...] = (
    "row_id",
    "confidence",
    "observed_ratio",
    "repaired_ratio",
    "spacing_local_m",
    "orientation_deg",
    "topology_status",
    "repair_reason",
    "model_version",
)

REPAIR_COLUMNS: tuple[str, ...] = (
    "repair_id",
    "row_id",
    "repair_type",
    "confidence",
    "total_cost",
    "mean_center_support",
    "mean_uncertainty",
    "repair_reason",
    "model_version",
)


def _crs(value: CRS | str | int) -> CRS:
    result = CRS.from_user_input(value)
    if not result:
        raise ValueError("a valid CRS is required for GeoPackage export")
    return result


def _canonical_json(value: Mapping[str, Any]) -> str:
    def default(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"unsupported JSON value: {type(item).__name__}")

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=default,
    )


def _series(values: list[Any], dtype: str) -> pd.Series:
    return pd.Series(values, dtype=dtype)


def _row_frame(records: tuple[RowRecord, ...], crs: CRS) -> gpd.GeoDataFrame:
    data = {
        "row_id": _series([record.row_id for record in records], "string"),
        "confidence": _series([record.confidence for record in records], "float64"),
        "observed_ratio": _series([record.observed_ratio for record in records], "float64"),
        "repaired_ratio": _series([record.repaired_ratio for record in records], "float64"),
        "spacing_local_m": _series(
            [record.spacing_local_m for record in records],
            "float64",
        ),
        "orientation_deg": _series([record.orientation_deg for record in records], "float64"),
        "topology_status": _series([record.topology_status for record in records], "string"),
        "repair_reason": _series([record.repair_reason for record in records], "string"),
        "model_version": _series([record.model_version for record in records], "string"),
    }
    geometry = gpd.GeoSeries([record.geometry for record in records], crs=crs)
    return gpd.GeoDataFrame(data, geometry=geometry, crs=crs)


def _repair_probability(path: RepairPath) -> float:
    value = path.reason.get("edge_confidence", path.reason.get("confidence"))
    # The path dataclass predates a dedicated confidence field.  Reuse the
    # same evidence summary employed by vector refinement until a calibrated
    # edge probability is provided in ``reason``.
    probability = (
        float(value)
        if value is not None
        else (path.mean_center_support + 1.0 - path.mean_uncertainty) / 2.0
    )
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("repair reason confidence must be within [0, 1]")
    return probability


def _repair_frame(
    paths: tuple[RepairPath, ...],
    crs: CRS,
    model_version: str,
) -> gpd.GeoDataFrame:
    data = {
        "repair_id": _series([path.candidate_id for path in paths], "string"),
        "row_id": _series([path.reason.get("row_id") for path in paths], "string"),
        "repair_type": _series(
            [
                str(
                    path.reason.get(
                        "repair_type", getattr(path, "repair_type", "internal_gap_repair")
                    )
                )
                for path in paths
            ],
            "string",
        ),
        "confidence": _series([_repair_probability(path) for path in paths], "float64"),
        "total_cost": _series([path.total_cost for path in paths], "float64"),
        "mean_center_support": _series(
            [path.mean_center_support for path in paths],
            "float64",
        ),
        "mean_uncertainty": _series([path.mean_uncertainty for path in paths], "float64"),
        "repair_reason": _series([_canonical_json(path.reason) for path in paths], "string"),
        "model_version": _series([model_version for _ in paths], "string"),
    }
    geometry = gpd.GeoSeries([path.geometry for path in paths], crs=crs)
    return gpd.GeoDataFrame(data, geometry=geometry, crs=crs)


def _write_frame(frame: gpd.GeoDataFrame, path: Path, layer: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"rowgraphnet_schema": "1.0", "geometry_type": "LineString"}
    try:
        import pyogrio

        pyogrio.write_dataframe(
            frame,
            path,
            layer=layer,
            driver="GPKG",
            geometry_type="LineString",
            append=False,
            dataset_metadata=metadata,
            layer_metadata=metadata,
        )
    except ImportError:
        frame.to_file(path, layer=layer, driver="GPKG", index=False, mode="w")
    return path


def write_rows_geopackage(
    path: str | Path,
    rows: Iterable[RowRecord],
    crs: CRS | str | int,
    *,
    layer: str = LINES_LAYER,
) -> Path:
    """Write final or uncertain rows using the mandatory stable schema."""

    records = validate_row_records(rows)
    return _write_frame(_row_frame(records, _crs(crs)), Path(path), layer)


def write_repairs_geopackage(
    path: str | Path,
    repairs: Iterable[RepairPath],
    crs: CRS | str | int,
    *,
    model_version: str,
    layer: str = REPAIRS_LAYER,
) -> Path:
    """Write accepted reconstructed paths and their auditable evidence."""

    if not model_version or not model_version.strip():
        raise ValueError("model_version must not be empty")
    paths = validate_repair_paths(repairs)
    return _write_frame(_repair_frame(paths, _crs(crs), model_version), Path(path), layer)


def write_output_geopackages(
    output_dir: str | Path,
    rows: Iterable[RowRecord],
    repairs: Iterable[RepairPath],
    uncertain_rows: Iterable[RowRecord],
    crs: CRS | str | int,
    *,
    model_version: str,
) -> dict[str, Path]:
    """Write the three required GeoPackages and enforce disjoint row IDs."""

    output = Path(output_dir)
    final_records = validate_row_records(rows)
    uncertain_records = validate_row_records(uncertain_rows)
    repair_records = validate_repair_paths(repairs)
    duplicated = {item.row_id for item in final_records} & {
        item.row_id for item in uncertain_records
    }
    if duplicated:
        duplicate_text = ", ".join(sorted(duplicated))
        raise ValueError(f"row IDs cannot occur in final and uncertain outputs: {duplicate_text}")
    invalid_final = [item.row_id for item in final_records if item.topology_status != "valid"]
    if invalid_final:
        raise ValueError("linhas_simples accepts only topology_status='valid'")
    invalid_uncertain = [
        item.row_id for item in uncertain_records if item.topology_status == "valid"
    ]
    if invalid_uncertain:
        raise ValueError("linhas_incertas accepts only ambiguous or rejected rows")
    final_row_ids = {item.row_id for item in final_records}
    orphan_repairs = [
        item.candidate_id
        for item in repair_records
        if not isinstance(item.reason.get("row_id"), str)
        or item.reason.get("row_id") not in final_row_ids
    ]
    if orphan_repairs:
        raise ValueError(
            "every reconstructed path must reference a valid final row_id; orphan repairs: "
            + ", ".join(sorted(orphan_repairs))
        )

    crs_value = _crs(crs)
    return {
        "lines": write_rows_geopackage(
            output / LINES_FILENAME,
            final_records,
            crs_value,
            layer=LINES_LAYER,
        ),
        "repairs": write_repairs_geopackage(
            output / REPAIRS_FILENAME,
            repair_records,
            crs_value,
            model_version=model_version,
            layer=REPAIRS_LAYER,
        ),
        "uncertain": write_rows_geopackage(
            output / UNCERTAIN_FILENAME,
            uncertain_records,
            crs_value,
            layer=UNCERTAIN_LAYER,
        ),
    }
