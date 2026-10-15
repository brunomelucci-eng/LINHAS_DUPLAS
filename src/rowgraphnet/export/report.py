"""Fixed-schema JSON quality report and cryptographic artifact hashes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS

from rowgraphnet import __version__

REPORT_FILENAME = "quality_report.json"
REPORT_SCHEMA_VERSION = "rowgraphnet.quality-report/1.0"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 hash without loading large rasters in RAM."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, CRS):
        return value.to_wkt()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash JSON content using canonical key ordering and UTF-8 encoding."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _artifact_manifest(files: Mapping[str, str | Path] | None) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for name, raw_path in sorted((files or {}).items()):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"artifact does not exist: {path}")
        manifest[str(name)] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return manifest


def _crs_metadata(value: CRS | str | int) -> dict[str, Any]:
    crs = CRS.from_user_input(value)
    authority = crs.to_authority()
    axis_units = sorted(
        {
            axis.unit_name
            for axis in crs.axis_info
            if axis.unit_name is not None and axis.unit_name.strip()
        }
    )
    return {
        "name": crs.name,
        "authority": (
            {"name": authority[0], "code": authority[1]} if authority is not None else None
        ),
        "is_projected": crs.is_projected,
        "axis_units": axis_units,
        "wkt": crs.to_wkt(),
    }


def write_quality_report(
    path: str | Path,
    *,
    run_id: str,
    model_version: str,
    crs: CRS | str | int,
    metrics: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    configuration: Mapping[str, Any] | None = None,
    input_files: Mapping[str, str | Path] | None = None,
    output_files: Mapping[str, str | Path] | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Write a strict report containing CRS, provenance, and SHA-256 hashes."""

    if not run_id or not run_id.strip():
        raise ValueError("run_id must not be empty")
    if not model_version or not model_version.strip():
        raise ValueError("model_version must not be empty")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    if configuration is not None and not isinstance(configuration, Mapping):
        raise TypeError("configuration must be a mapping")
    if metrics is not None and not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")

    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    timestamp = timestamp.astimezone(timezone.utc)
    configuration_value = _json_value(configuration or {})
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run": {
            "run_id": run_id,
            "model_version": model_version,
            "created_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
        },
        "crs": _crs_metadata(crs),
        "provenance": {
            "software": {"name": "rowgraphnet", "version": __version__},
            "configuration": configuration_value,
            "configuration_sha256": sha256_json(configuration_value),
            "details": _json_value(provenance or {}),
        },
        "artifacts": {
            "inputs": _artifact_manifest(input_files),
            "outputs": _artifact_manifest(output_files),
        },
        "metrics": _json_value(metrics or {}),
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(destination)
    return destination
