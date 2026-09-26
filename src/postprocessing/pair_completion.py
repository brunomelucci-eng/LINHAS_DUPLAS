"""Conservative pair-completion gate.

Missing sisters are never synthesised from spacing alone.  This module keeps
the integration point explicit and routes incomplete/unsafe pairs to manual
review until a candidate has probability, lateral-order, spacing, and crossing
evidence strong enough to be accepted by a future reconstruction strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import geopandas as gpd
import pandas as pd


@dataclass(frozen=True)
class PairCompletionResult:
    lines: gpd.GeoDataFrame
    reconstructed_pairs: gpd.GeoDataFrame
    manual_review: gpd.GeoDataFrame
    metrics: Mapping[str, Any]


def _empty(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def audit_pair_completion(
    lines_gdf: gpd.GeoDataFrame,
    config: dict,
) -> PairCompletionResult:
    """Enforce pair cardinality without silently inventing a missing row."""
    lines = lines_gdf.copy()
    if lines.empty:
        empty = _empty(lines.crs)
        return PairCompletionResult(lines, empty.copy(), empty.copy(), {
            "valid_pair_line_count": 0,
            "incomplete_pair_line_count": 0,
            "reconstructed_pair_line_count": 0,
        })

    pair_status = (
        lines["pair_status"]
        if "pair_status" in lines.columns
        else pd.Series("unmatched", index=lines.index)
    )
    pair_ids = (
        lines["pair_id"]
        if "pair_id" in lines.columns
        else pd.Series(pd.NA, index=lines.index, dtype="object")
    )
    counts = pair_ids.dropna().value_counts()
    cardinality_ok = pair_ids.map(counts).eq(2).fillna(False)
    valid_mask = pair_status.eq("valid_pair") & cardinality_ok
    review = lines.loc[~valid_mask].copy()
    if not review.empty:
        review["pair_completion_reason"] = "insufficient_reconstruction_evidence"

    # Reconstruction intentionally remains empty: the existing codebase has no
    # validated candidate generator that satisfies every center-mass guard.
    reconstructed = _empty(lines.crs)
    return PairCompletionResult(
        lines=lines,
        reconstructed_pairs=reconstructed,
        manual_review=review,
        metrics={
            "valid_pair_line_count": int(valid_mask.sum()),
            "incomplete_pair_line_count": int((~valid_mask).sum()),
            "reconstructed_pair_line_count": 0,
            "unsafe_reconstruction_policy": str(
                config.get("pair_completion", {}).get(
                    "on_insufficient_evidence", "manual_review"
                )
            ),
        },
    )


__all__ = ["PairCompletionResult", "audit_pair_completion"]
