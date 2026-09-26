"""Canonical adapter for the single final C1 terminal correction."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from affine import Affine
from shapely.geometry import LineString, Polygon

from .regularized_centerline_fairing import straighten_terminals_c1


def straighten_terminals_once(
    line: LineString,
    roi_polygon: Polygon,
    config: dict,
    probability_raster: Optional[np.ndarray] = None,
    transform: Optional[Affine] = None,
    sister_line: Optional[LineString] = None,
) -> Tuple[LineString, dict]:
    """Delegate to the approved C1 implementation; never run a second fairing."""
    return straighten_terminals_c1(
        line,
        roi_polygon,
        config,
        probability_raster,
        transform,
        sister_line,
    )


__all__ = ["straighten_terminals_once"]
