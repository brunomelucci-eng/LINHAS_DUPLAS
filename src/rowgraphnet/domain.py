"""Typed contracts shared by inference, geometry, and export.

The dataclasses in this module deliberately contain no model-specific state.
They make the boundary between learned evidence and deterministic geometry
explicit, which keeps every accepted repair auditable and reversible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from affine import Affine
from pyproj import CRS
from shapely.geometry import LineString, Point

TopologyStatus = Literal["valid", "ambiguous", "rejected"]


@dataclass(slots=True)
class RasterPrediction:
    """Continuous model evidence in the source raster grid."""

    center: np.ndarray
    band: np.ndarray
    orientation_sin2: np.ndarray
    orientation_cos2: np.ndarray
    endpoints: np.ndarray
    distance: np.ndarray
    uncertainty: np.ndarray  # non-negative aleatoric scale; not necessarily a probability
    transform: Affine
    crs: CRS
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        center = np.asarray(self.center)
        expected = center.shape
        if len(expected) != 2:
            raise ValueError("prediction maps must be two-dimensional")
        names = (
            "band",
            "orientation_sin2",
            "orientation_cos2",
            "endpoints",
            "distance",
            "uncertainty",
        )
        for name in names:
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{name} has shape {value.shape}; expected {expected}")
        if self.valid_mask is not None and self.valid_mask.shape != expected:
            raise ValueError("valid_mask must use the same grid as center")
        valid = (
            np.ones(expected, dtype=bool)
            if self.valid_mask is None
            else np.asarray(self.valid_mask, dtype=bool)
        )
        probability_names = ("center", "band", "endpoints")
        for name in probability_names:
            values = np.asarray(getattr(self, name))
            if np.any(valid & ~np.isfinite(values)):
                raise ValueError(f"{name} must be finite in valid pixels")
            if np.any(valid & ((values < 0.0) | (values > 1.0))):
                raise ValueError(f"{name} must lie in [0, 1] in valid pixels")
        for name in ("orientation_sin2", "orientation_cos2"):
            values = np.asarray(getattr(self, name))
            if np.any(valid & ~np.isfinite(values)):
                raise ValueError(f"{name} must be finite in valid pixels")
        distance = np.asarray(self.distance)
        if np.any(valid & (~np.isfinite(distance) | (distance < 0.0))):
            raise ValueError("distance must be finite and non-negative in valid pixels")
        uncertainty = np.asarray(self.uncertainty)
        if np.any(valid & (~np.isfinite(uncertainty) | (uncertainty < 0.0))):
            raise ValueError("uncertainty must be finite and non-negative in valid pixels")
        affine_values = np.asarray(tuple(self.transform)[:6], dtype=np.float64)
        if not np.isfinite(affine_values).all() or abs(self.transform.determinant) <= np.finfo(
            float
        ).eps:
            raise ValueError("prediction transform must be finite and invertible")
        try:
            CRS.from_user_input(self.crs)
        except Exception as exc:
            raise ValueError("prediction CRS must be valid") from exc


@dataclass(slots=True, frozen=True)
class SpacingEstimate:
    """Local posterior summary for row spacing."""

    median_m: float
    mad_m: float
    entropy: float
    confidence: float
    samples_m: tuple[float, ...] = ()
    position_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.median_m) or self.median_m <= 0.0:
            raise ValueError("spacing median must be finite and positive")
        if not np.isfinite(self.mad_m) or self.mad_m < 0.0:
            raise ValueError("spacing MAD must be finite and non-negative")
        for name in ("entropy", "confidence"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"spacing {name} must lie in [0, 1]")
        if any(not np.isfinite(value) or value <= 0.0 for value in self.samples_m):
            raise ValueError("spacing samples must be finite and positive")
        if self.position_xy is not None and (
            len(self.position_xy) != 2 or not np.isfinite(self.position_xy).all()
        ):
            raise ValueError("spacing position must contain two finite coordinates")


@dataclass(slots=True)
class ExtractedSegment:
    """Observed skeleton segment with two oriented endpoints."""

    segment_id: str
    geometry: LineString
    confidence: float
    observed_length_m: float
    mean_uncertainty: float
    endpoint_confidence: tuple[float, float] = (0.0, 0.0)
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segment_id.strip():
            raise ValueError("segment_id must not be empty")
        _validate_linestring(self.geometry, "segment")
        _validate_unit_interval(self.confidence, "segment confidence")
        if not np.isfinite(self.observed_length_m) or self.observed_length_m <= 0.0:
            raise ValueError("observed segment length must be finite and positive")
        if not np.isfinite(self.mean_uncertainty) or self.mean_uncertainty < 0.0:
            raise ValueError("segment uncertainty must be finite and non-negative")
        if len(self.endpoint_confidence) != 2:
            raise ValueError("endpoint_confidence must contain exactly two values")
        for value in self.endpoint_confidence:
            _validate_unit_interval(value, "endpoint confidence")

    @property
    def endpoints(self) -> tuple[Point, Point]:
        coords = list(self.geometry.coords)
        return Point(coords[0]), Point(coords[-1])


@dataclass(slots=True)
class CandidateConnection:
    """A scored, but not yet accepted, continuation hypothesis."""

    candidate_id: str
    segment_a: str
    side_a: int
    segment_b: str
    side_b: int
    geometry: LineString
    probability: float
    cost: float
    evidence: dict[str, float | str | bool]
    ambiguous: bool = False
    is_boundary_extension: bool = False
    boundary_intersection_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.segment_a.strip():
            raise ValueError("candidate and source segment IDs must not be empty")
        if self.side_a not in (0, 1):
            raise ValueError("candidate source endpoint side must be zero or one")
        if self.is_boundary_extension:
            if self.side_b != -1 or self.segment_b:
                raise ValueError("boundary extensions must use an empty target and side -1")
            if self.boundary_intersection_xy is None or (
                len(self.boundary_intersection_xy) != 2
                or not np.isfinite(self.boundary_intersection_xy).all()
            ):
                raise ValueError("boundary extension intersection must contain two finite values")
        elif not self.segment_b.strip() or self.side_b not in (0, 1):
            raise ValueError("internal candidates require a target segment and endpoint side")
        _validate_linestring(self.geometry, "candidate")
        _validate_unit_interval(self.probability, "candidate probability")
        if not np.isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("candidate cost must be finite and non-negative")

    @property
    def endpoint_keys(self) -> tuple[tuple[str, int], ...]:
        if self.is_boundary_extension:
            return ((self.segment_a, self.side_a),)
        return ((self.segment_a, self.side_a), (self.segment_b, self.side_b))


@dataclass(slots=True)
class RepairPath:
    """A raster-supported path produced for a candidate connection."""

    candidate_id: str
    geometry: LineString
    total_cost: float
    mean_center_support: float
    mean_uncertainty: float
    reason: dict[str, Any]
    repair_type: str = "internal_gap_repair"

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("repair candidate_id must not be empty")
        _validate_linestring(self.geometry, "repair")
        if not np.isfinite(self.total_cost) or self.total_cost < 0.0:
            raise ValueError("repair cost must be finite and non-negative")
        _validate_unit_interval(self.mean_center_support, "repair center support")
        _validate_unit_interval(self.mean_uncertainty, "repair uncertainty")
        if self.repair_type not in {"internal_gap_repair", "boundary_extension"}:
            raise ValueError("repair_type is not recognized")


@dataclass(slots=True)
class RowRecord:
    """Final row feature and the mandatory output attributes."""

    row_id: str
    geometry: LineString
    confidence: float
    observed_ratio: float
    repaired_ratio: float
    spacing_local_m: float | None
    orientation_deg: float
    topology_status: TopologyStatus
    repair_reason: str
    model_version: str

    def __post_init__(self) -> None:
        if not self.row_id.strip() or not self.model_version.strip():
            raise ValueError("row_id and model_version must not be empty")
        _validate_linestring(self.geometry, "row")
        _validate_unit_interval(self.confidence, "row confidence")
        _validate_unit_interval(self.observed_ratio, "observed ratio")
        _validate_unit_interval(self.repaired_ratio, "repaired ratio")
        if self.observed_ratio + self.repaired_ratio > 1.0 + 1e-6:
            raise ValueError("observed and repaired ratios cannot exceed one in total")
        if self.spacing_local_m is not None and (
            not np.isfinite(self.spacing_local_m) or self.spacing_local_m <= 0.0
        ):
            raise ValueError("local spacing must be finite and positive or None")
        if np.isfinite(self.orientation_deg):
            deg = float(self.orientation_deg % 180.0)
            if deg >= 180.0 or abs(deg - 180.0) < 1e-7:
                deg = 0.0
            self.orientation_deg = deg
        if not np.isfinite(self.orientation_deg) or not 0.0 <= self.orientation_deg < 180.0:
            raise ValueError("row axial orientation must lie in [0, 180)")
        if self.topology_status not in {"valid", "ambiguous", "rejected"}:
            raise ValueError("row topology_status is not recognized")


def _validate_linestring(geometry: LineString, label: str) -> None:
    if (
        not isinstance(geometry, LineString)
        or geometry.is_empty
        or not geometry.is_valid
        or not geometry.is_simple
        or not np.isfinite(geometry.length)
        or geometry.length <= 0.0
    ):
        raise ValueError(f"{label} geometry must be a valid, simple, non-empty LineString")


def _validate_unit_interval(value: float, label: str) -> None:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and lie in [0, 1]")
