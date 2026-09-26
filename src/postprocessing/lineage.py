"""Canonical lineage encoding for post-processed crop-row vectors.

GeoPackage attributes cannot safely store Python lists.  ``source_raw_ids`` is
therefore persisted as a compact JSON array.  Readers remain deliberately
backward compatible with the scalar and comma-separated formats emitted by
older revisions of the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from typing import Any, Tuple

import pandas as pd


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _iter_source_raw_ids(value: Any):
    if _is_missing(value):
        return
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return
        if text.startswith("[") or text.startswith('"'):
            try:
                decoded = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                decoded = None
            if isinstance(decoded, (list, tuple, set)):
                for item in decoded:
                    yield from _iter_source_raw_ids(item)
                return
            if isinstance(decoded, str):
                yield from _iter_source_raw_ids(decoded)
                return
        # Legacy GeoPackages stored either one scalar ID or a CSV string.
        for item in text.split(","):
            identifier = item.strip()
            if identifier:
                yield identifier
        return
    if isinstance(value, Mapping):
        return
    if isinstance(value, Iterable):
        for item in value:
            yield from _iter_source_raw_ids(item)
        return
    identifier = str(value).strip()
    if identifier:
        yield identifier


def parse_source_raw_ids(value: Any) -> Tuple[str, ...]:
    """Return deterministic, unique source IDs from JSON, CSV, or a scalar."""
    return tuple(sorted(set(_iter_source_raw_ids(value) or ())))


def serialize_source_raw_ids(*values: Any) -> str:
    """Encode one or more lineage values as deterministic compact JSON."""
    identifiers = set()
    for value in values:
        identifiers.update(parse_source_raw_ids(value))
    return json.dumps(
        sorted(identifiers),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def source_raw_ids_from_record(record: Mapping[str, Any]) -> Tuple[str, ...]:
    """Read lineage from a record, falling back to its singular raw ID."""
    identifiers = parse_source_raw_ids(record.get("source_raw_ids"))
    if identifiers:
        return identifiers
    return parse_source_raw_ids(record.get("raw_line_id"))


def combine_record_source_raw_ids(records: Iterable[Mapping[str, Any]]) -> str:
    """Combine lineage from records into the canonical GeoPackage value."""
    return serialize_source_raw_ids(
        *(source_raw_ids_from_record(record) for record in records)
    )


__all__ = [
    "combine_record_source_raw_ids",
    "parse_source_raw_ids",
    "serialize_source_raw_ids",
    "source_raw_ids_from_record",
]
