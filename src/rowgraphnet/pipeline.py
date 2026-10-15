"""End-to-end deterministic post-processing of RowGraphNet raster evidence."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from pyproj import CRS
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from rowgraphnet.data import is_metric_crs
from rowgraphnet.domain import (
    CandidateConnection,
    ExtractedSegment,
    RasterPrediction,
    RepairPath,
    SpacingEstimate,
)
from rowgraphnet.geometry.boundary_extension import (
    BoundaryExtensionConfig,
    create_boundary_repair_path,
    generate_boundary_extension_candidates,
)
from rowgraphnet.geometry.candidate_graph import (
    generate_candidates,
    mark_ambiguous_endpoint_choices,
)
from rowgraphnet.geometry.global_optimizer import OptimizationResult, optimize_connections
from rowgraphnet.geometry.orientation_field import (
    FusedOrientationField,
    OrientationRegime,
    discover_orientation_regimes,
    fuse_orientation_fields,
    orientation_field_to_world,
    structure_tensor_orientation,
)
from rowgraphnet.geometry.path_repair import PathCostWeights, repair_candidate_path
from rowgraphnet.geometry.segment_extractor import extract_segments
from rowgraphnet.geometry.spacing_estimator import SpacingField, estimate_spacing
from rowgraphnet.geometry.vector_refinement import RefinementResult, refine_rows


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """Validation-calibratable, dimensionless post-processing controls."""

    low_threshold_quantile: float = 0.90
    high_threshold_quantile: float = 0.98
    structure_sigma_width_fraction: float = 0.5
    maximum_orientation_regimes: int = 4
    orientation_confidence_quantile: float = 0.5
    spacing_reliability_threshold: float = 0.25
    candidate_ambiguity_entropy: float = 1.0
    candidate_ambiguity_probability_margin: float | None = 0.05
    minimum_connection_probability: float = 0.5
    path_evidence_weight: float = 0.25
    path_background_weight: float = 1.0
    path_orientation_weight: float = 1.0
    path_curvature_weight: float = 1.0
    path_uncertainty_weight: float = 1.0
    maximum_spline_deviation_pixels: float = 3.0
    true_endpoint_probability: float = 0.5
    abstain_on_ambiguous: bool = True
    enable_boundary_extension: bool = True
    max_boundary_distance_factor: float = 2.5
    max_boundary_absolute_distance_m: float | None = None
    boundary_min_terminal_vertices: int = 3
    boundary_min_observed_length_spacing_factor: float = 0.5
    boundary_terminal_length_spacing_factor: float = 1.5
    boundary_max_tangent_angular_dispersion_deg: float = 25.0
    boundary_max_tangent_eigenvalue_ratio: float = 0.35
    boundary_min_orientation_alignment: float = 0.4
    boundary_safe_corridor_fraction: float = 0.35
    boundary_raster_evidence_weight: float = 0.0
    boundary_minimum_valid_fraction: float = 1.0

    def __post_init__(self) -> None:
        quantiles = (
            self.low_threshold_quantile,
            self.high_threshold_quantile,
            self.orientation_confidence_quantile,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in quantiles):
            raise ValueError("all post-processing quantiles must lie in [0, 1]")
        if self.low_threshold_quantile >= self.high_threshold_quantile:
            raise ValueError("low threshold quantile must be below the high threshold quantile")
        if (
            isinstance(self.maximum_orientation_regimes, bool)
            or not isinstance(self.maximum_orientation_regimes, int)
            or self.maximum_orientation_regimes < 1
        ):
            raise ValueError("maximum_orientation_regimes must be a positive integer")
        if (
            not math.isfinite(self.structure_sigma_width_fraction)
            or self.structure_sigma_width_fraction <= 0
        ):
            raise ValueError("structure tensor scale fraction must be positive and finite")
        if not math.isfinite(self.spacing_reliability_threshold) or not (
            0 <= self.spacing_reliability_threshold <= 1
        ):
            raise ValueError("spacing reliability threshold must lie in [0, 1]")
        if not math.isfinite(self.candidate_ambiguity_entropy) or not (
            0 <= self.candidate_ambiguity_entropy <= 1
        ):
            raise ValueError("candidate ambiguity entropy must lie in [0, 1]")
        if self.candidate_ambiguity_probability_margin is not None and not (
            math.isfinite(self.candidate_ambiguity_probability_margin)
            and 0 <= self.candidate_ambiguity_probability_margin <= 1
        ):
            raise ValueError("candidate ambiguity probability margin must lie in [0, 1]")
        if not math.isfinite(self.minimum_connection_probability) or not (
            0 < self.minimum_connection_probability < 1
        ):
            raise ValueError("connection probability must lie strictly inside (0, 1)")
        if not math.isfinite(self.path_evidence_weight) or not 0 <= self.path_evidence_weight <= 1:
            raise ValueError("path evidence weight must lie in [0, 1]")
        if not math.isfinite(self.true_endpoint_probability) or not (
            0 <= self.true_endpoint_probability <= 1
        ):
            raise ValueError("true endpoint probability must lie in [0, 1]")
        if (
            not math.isfinite(self.maximum_spline_deviation_pixels)
            or self.maximum_spline_deviation_pixels < 0
        ):
            raise ValueError("spline deviation must be finite and non-negative")
        if not math.isfinite(self.max_boundary_distance_factor) or (
            self.max_boundary_distance_factor <= 0
        ):
            raise ValueError("boundary distance factor must be positive and finite")
        PathCostWeights(
            background=self.path_background_weight,
            orientation=self.path_orientation_weight,
            curvature=self.path_curvature_weight,
            uncertainty=self.path_uncertainty_weight,
        ).normalized()
        BoundaryExtensionConfig(
            enabled=self.enable_boundary_extension,
            max_boundary_distance_factor=self.max_boundary_distance_factor,
            max_absolute_distance_m=self.max_boundary_absolute_distance_m,
            min_terminal_vertices=self.boundary_min_terminal_vertices,
            min_observed_length_spacing_factor=self.boundary_min_observed_length_spacing_factor,
            terminal_length_spacing_factor=self.boundary_terminal_length_spacing_factor,
            max_tangent_angular_dispersion_deg=(
                self.boundary_max_tangent_angular_dispersion_deg
            ),
            max_tangent_eigenvalue_ratio=self.boundary_max_tangent_eigenvalue_ratio,
            min_orientation_alignment=self.boundary_min_orientation_alignment,
            safe_corridor_fraction=self.boundary_safe_corridor_fraction,
            true_endpoint_probability_threshold=self.true_endpoint_probability,
            minimum_probability=self.minimum_connection_probability,
            raster_evidence_weight=self.boundary_raster_evidence_weight,
            minimum_valid_fraction=self.boundary_minimum_valid_fraction,
        )


@dataclass(slots=True)
class PostprocessResult:
    refinement: RefinementResult
    segments: list[ExtractedSegment]
    candidates: list[CandidateConnection]
    repair_paths: list[RepairPath]
    optimization: OptimizationResult
    orientation_field: FusedOrientationField
    orientation_regimes: list[OrientationRegime]
    spacing_by_regime: dict[int, SpacingField]
    diagnostics: dict[str, Any]


def _pixel_size_m(prediction: RasterPrediction) -> float:
    x_size = math.hypot(prediction.transform.a, prediction.transform.d)
    y_size = math.hypot(prediction.transform.b, prediction.transform.e)
    if min(x_size, y_size) <= 0:
        raise ValueError("prediction transform has a non-positive pixel basis")
    return float(math.sqrt(x_size * y_size))


def _structure_sigma(prediction: RasterPrediction, config: PostprocessConfig) -> float:
    valid = (
        np.asarray(prediction.valid_mask, dtype=bool).copy()
        if prediction.valid_mask is not None
        else np.ones_like(prediction.center, dtype=bool)
    )
    valid &= np.isfinite(prediction.band) & np.isfinite(prediction.center)
    if not np.any(valid):
        return 1.0
    band_mass = float(np.clip(prediction.band[valid], 0.0, 1.0).sum())
    center_mass = float(np.clip(prediction.center[valid], 0.0, 1.0).sum())
    inferred_width = max(band_mass / max(center_mass, np.finfo(float).eps), 1.0)
    return max(inferred_width * config.structure_sigma_width_fraction, 0.5)


def _pixel_index(prediction: RasterPrediction, x: float, y: float) -> tuple[int, int] | None:
    col, row = (~prediction.transform) * (x, y)
    row_index, col_index = round(row - 0.5), round(col - 0.5)
    if not (
        0 <= row_index < prediction.center.shape[0] and 0 <= col_index < prediction.center.shape[1]
    ):
        return None
    return row_index, col_index


def _orientation_sampler(
    prediction: RasterPrediction,
    field: FusedOrientationField,
):
    def sample(x: float, y: float) -> tuple[float, float]:
        index = _pixel_index(prediction, x, y)
        if index is None:
            return 0.0, 0.0
        return float(field.theta_rad[index]), float(field.confidence[index])

    return sample


def _evidence_sampler(prediction: RasterPrediction, pixel_size: float):
    def sample(line: LineString) -> float:
        count = max(math.ceil(line.length / pixel_size) + 1, 2)
        values: list[float] = []
        for fraction in np.linspace(0.0, 1.0, count):
            point = line.interpolate(float(fraction), normalized=True)
            index = _pixel_index(prediction, point.x, point.y)
            if index is not None and (
                prediction.valid_mask is None or prediction.valid_mask[index]
            ):
                values.append(float(prediction.center[index]))
        return float(np.mean(values)) if values else 0.0

    return sample


def _path_intersects_unrelated_segment(
    path: LineString,
    candidate: CandidateConnection,
    segments: list[ExtractedSegment],
) -> bool:
    source_ids = {candidate.segment_a, candidate.segment_b}
    return any(
        segment.segment_id not in source_ids and path.intersects(segment.geometry)
        for segment in segments
    )


def _segment_regime(
    segment: ExtractedSegment,
    prediction: RasterPrediction,
    label_map: np.ndarray,
) -> int | None:
    labels: list[int] = []
    coords = list(segment.geometry.coords)
    step = max(len(coords) // 64, 1)
    for x, y in coords[::step]:
        index = _pixel_index(prediction, x, y)
        if index is not None and label_map[index] >= 0:
            labels.append(int(label_map[index]))
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def _pooled_spacing(
    regimes: list[OrientationRegime],
    fields: dict[int, SpacingField],
) -> SpacingEstimate | None:
    regime_weights = {item.regime_id: item.weight for item in regimes}
    available = [
        (regime_id, field.global_estimate)
        for regime_id, field in fields.items()
        if field.reliable and field.global_estimate is not None
    ]
    if not available:
        return None
    values = np.array([item.median_m for _, item in available if item is not None])
    weights = np.array(
        [
            max(regime_weights.get(regime_id, 0.0) * item.confidence, np.finfo(float).eps)
            for regime_id, item in available
            if item is not None
        ]
    )
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    median = float(values[order[np.searchsorted(cumulative, cumulative[-1] / 2.0)]])
    deviations = np.abs(values - median)
    deviation_order = np.argsort(deviations)
    deviation_cumulative = np.cumsum(weights[deviation_order])
    mad = float(
        deviations[
            deviation_order[np.searchsorted(deviation_cumulative, deviation_cumulative[-1] / 2.0)]
        ]
    )
    normalized_weights = weights / weights.sum()
    return SpacingEstimate(
        median_m=median,
        mad_m=mad,
        entropy=float(
            np.sum(
                normalized_weights
                * np.array([item.entropy for _, item in available if item is not None])
            )
        ),
        confidence=float(
            np.sum(
                normalized_weights
                * np.array([item.confidence for _, item in available if item is not None])
            )
        ),
        # Component-level raw samples are not exchangeable when the ROI has
        # distinct orientation sectors.  The pooled posterior therefore keeps
        # its robust center/spread rather than inflating search radii with them.
        samples_m=(median,),
    )


def postprocess_prediction(
    prediction: RasterPrediction,
    *,
    model_version: str,
    roi: BaseGeometry | None = None,
    field_boundary: BaseGeometry | None = None,
    config: PostprocessConfig | None = None,
    path_cost_weights: PathCostWeights | None = None,
    boundary_extension_config: BoundaryExtensionConfig | None = None,
) -> PostprocessResult:
    """Reconstruct rows from continuous evidence and abstain when unsupported."""

    if not model_version.strip():
        raise ValueError("model_version must not be empty")
    if not is_metric_crs(CRS.from_user_input(prediction.crs)):
        raise ValueError("post-processing requires the metric working grid")
    settings = config or PostprocessConfig()
    effective_field_boundary = field_boundary
    pixel_size = _pixel_size_m(prediction)
    sigma = _structure_sigma(prediction, settings)
    tensor = structure_tensor_orientation(
        prediction.center,
        sigma_px=sigma,
        valid_mask=prediction.valid_mask,
    )
    fused = fuse_orientation_fields(
        prediction.orientation_sin2,
        prediction.orientation_cos2,
        prediction.uncertainty,
        tensor,
        valid_mask=prediction.valid_mask,
    )
    world_fused = orientation_field_to_world(fused, prediction.transform)
    regimes, regime_map = discover_orientation_regimes(
        fused,
        evidence=prediction.center,
        max_regimes=settings.maximum_orientation_regimes,
        min_confidence_quantile=settings.orientation_confidence_quantile,
    )
    segments, extraction_diagnostics = extract_segments(
        prediction,
        low_quantile=settings.low_threshold_quantile,
        high_quantile=settings.high_threshold_quantile,
    )

    spacing_by_regime: dict[int, SpacingField] = {}
    regime_lookup = {item.regime_id: item for item in regimes}
    for regime in regimes:
        mask = regime_map == regime.regime_id
        if prediction.valid_mask is not None:
            mask &= prediction.valid_mask
        evidence_slice = np.where(mask, prediction.center, 0.0)
        spacing_by_regime[regime.regime_id] = estimate_spacing(
            evidence_slice,
            regime.orientation_rad,
            pixel_size_m=pixel_size,
            reliability_threshold=settings.spacing_reliability_threshold,
        )

    unassigned_segments: list[str] = []
    for segment in segments:
        assigned_regime = _segment_regime(segment, prediction, regime_map)
        regime_id = assigned_regime if assigned_regime is not None else -1
        segment.attrs["orientation_regime"] = regime_id
        if regime_id < 0 or regime_id not in spacing_by_regime:
            unassigned_segments.append(segment.segment_id)
            segment.attrs["spacing_reliable"] = False
            segment.attrs["touches_junction"] = True
            continue
        spacing_field = spacing_by_regime[regime_id]
        spacing_estimate = spacing_field.global_estimate
        segment.attrs["spacing_reliable"] = spacing_field.reliable
        if spacing_estimate is not None:
            segment.attrs["spacing_local_m"] = spacing_estimate.median_m
        if not spacing_field.reliable:
            segment.attrs["touches_junction"] = True

    all_candidates: list[CandidateConnection] = []
    spacing_for_candidate: dict[str, SpacingEstimate] = {}
    orientation_sample = _orientation_sampler(prediction, world_fused)
    evidence_sample = _evidence_sampler(prediction, pixel_size)
    pooled_spacing = _pooled_spacing(regimes, spacing_by_regime)
    eligible_segments = [
        segment for segment in segments if bool(segment.attrs.get("spacing_reliable"))
    ]
    if pooled_spacing is not None:
        all_candidates = generate_candidates(
            eligible_segments,
            pooled_spacing,
            orientation_sampler=orientation_sample,
            evidence_sampler=evidence_sample,
            ambiguity_entropy=settings.candidate_ambiguity_entropy,
            ambiguity_probability_margin=settings.candidate_ambiguity_probability_margin,
        )
        segment_lookup = {item.segment_id: item for item in segments}
        for candidate in all_candidates:
            regime_a = segment_lookup[candidate.segment_a].attrs.get("orientation_regime")
            regime_b = segment_lookup[candidate.segment_b].attrs.get("orientation_regime")
            local_spacing = pooled_spacing
            if regime_a == regime_b and isinstance(regime_a, int):
                field = spacing_by_regime.get(regime_a)
                if field is not None and field.reliable and field.global_estimate is not None:
                    local_spacing = field.global_estimate
            candidate.evidence["orientation_regime_a"] = regime_a
            candidate.evidence["orientation_regime_b"] = regime_b
            if regime_a != regime_b:
                first_regime = regime_lookup.get(regime_a) if isinstance(regime_a, int) else None
                second_regime = regime_lookup.get(regime_b) if isinstance(regime_b, int) else None
                if first_regime is None or second_regime is None:
                    candidate.probability = 0.0
                    candidate.cost = float("inf")
                    candidate.evidence["abstention_reason"] = "unresolved_orientation_regime"
                else:
                    delta = first_regime.orientation_rad - second_regime.orientation_rad
                    regime_alignment = float((1.0 + math.cos(2.0 * delta)) / 2.0)
                    transition_support = float(
                        min(
                            float(candidate.evidence.get("facing", 0.0)),
                            float(candidate.evidence.get("tangent", 0.0)),
                            float(candidate.evidence.get("orientation", 0.0)),
                        )
                    )
                    candidate.probability *= math.sqrt(regime_alignment * transition_support)
                    candidate.cost = float(
                        -math.log(max(candidate.probability, np.finfo(float).eps))
                    )
                    candidate.evidence["orientation_regime_transition"] = True
                    candidate.evidence["regime_alignment"] = regime_alignment
                    candidate.evidence["regime_transition_support"] = transition_support
            candidate.evidence["spacing_local_m"] = local_spacing.median_m
            spacing_for_candidate[candidate.candidate_id] = local_spacing

    repair_paths: list[RepairPath] = []
    effective_path_cost_weights = path_cost_weights or PathCostWeights(
        background=settings.path_background_weight,
        orientation=settings.path_orientation_weight,
        curvature=settings.path_curvature_weight,
        uncertainty=settings.path_uncertainty_weight,
    )
    for candidate in all_candidates:
        path = repair_candidate_path(
            candidate,
            prediction,
            spacing_for_candidate[candidate.candidate_id],
            weights=effective_path_cost_weights,
            orientation_field=world_fused,
        )
        if path is None:
            candidate.ambiguous = candidate.ambiguous or (
                candidate.probability >= settings.minimum_connection_probability
            )
            candidate.evidence["abstention_reason"] = "no_feasible_raster_path"
        elif _path_intersects_unrelated_segment(path.geometry, candidate, segments):
            candidate.ambiguous = candidate.ambiguous or (
                candidate.probability >= settings.minimum_connection_probability
            )
            candidate.evidence["abstention_reason"] = "path_intersects_unrelated_observed_segment"
        else:
            candidate.geometry = path.geometry
            path_quality = float(path.reason.get("path_quality", 0.0))
            proposal_probability = candidate.probability
            candidate.probability *= (
                np.clip(path_quality, 0.0, 1.0) ** settings.path_evidence_weight
            )
            candidate.cost = float(-math.log(max(candidate.probability, np.finfo(float).eps)))
            candidate.evidence["proposal_probability"] = proposal_probability
            candidate.evidence["path_quality"] = path_quality
            candidate.evidence["edge_confidence"] = candidate.probability
            path.reason["edge_confidence"] = candidate.probability
            path.reason["probability_includes_path_quality"] = True
            repair_paths.append(path)

    # Phase 2: Unilateral boundary extensions to physical field boundary
    if (
        effective_field_boundary is not None
        and settings.enable_boundary_extension
        and pooled_spacing is not None
    ):
        b_config = boundary_extension_config or BoundaryExtensionConfig(
            enabled=settings.enable_boundary_extension,
            max_boundary_distance_factor=settings.max_boundary_distance_factor,
            max_absolute_distance_m=settings.max_boundary_absolute_distance_m,
            min_terminal_vertices=settings.boundary_min_terminal_vertices,
            min_observed_length_spacing_factor=(
                settings.boundary_min_observed_length_spacing_factor
            ),
            terminal_length_spacing_factor=settings.boundary_terminal_length_spacing_factor,
            max_tangent_angular_dispersion_deg=(
                settings.boundary_max_tangent_angular_dispersion_deg
            ),
            max_tangent_eigenvalue_ratio=settings.boundary_max_tangent_eigenvalue_ratio,
            min_orientation_alignment=settings.boundary_min_orientation_alignment,
            safe_corridor_fraction=settings.boundary_safe_corridor_fraction,
            true_endpoint_probability_threshold=settings.true_endpoint_probability,
            minimum_probability=settings.minimum_connection_probability,
            raster_evidence_weight=settings.boundary_raster_evidence_weight,
            minimum_valid_fraction=settings.boundary_minimum_valid_fraction,
        )
        b_candidates = generate_boundary_extension_candidates(
            eligible_segments,
            prediction,
            pooled_spacing,
            effective_field_boundary,
            b_config,
            world_orientation=world_fused,
        )
        for b_cand in b_candidates:
            b_path = create_boundary_repair_path(b_cand, prediction)
            all_candidates.append(b_cand)
            repair_paths.append(b_path)

    mark_ambiguous_endpoint_choices(
        all_candidates,
        ambiguity_entropy=settings.candidate_ambiguity_entropy,
        ambiguity_probability_margin=settings.candidate_ambiguity_probability_margin,
    )

    optimization = optimize_connections(
        all_candidates,
        repair_paths,
        minimum_probability=settings.minimum_connection_probability,
        abstain_on_ambiguous=settings.abstain_on_ambiguous,
        path_evidence_weight=settings.path_evidence_weight,
    )
    connected_endpoints = {
        endpoint for candidate in optimization.accepted for endpoint in candidate.endpoint_keys
    }
    roi_boundary = roi.boundary if roi is not None else None
    field_edge = effective_field_boundary.boundary if effective_field_boundary is not None else None
    boundary_tolerance = 0.5 * max(
        math.hypot(prediction.transform.a, prediction.transform.d),
        math.hypot(prediction.transform.b, prediction.transform.e),
    )
    for segment in segments:
        unresolved = False
        for side, point in enumerate(segment.endpoints):
            if (segment.segment_id, side) in connected_endpoints:
                continue
            reaches_analysis_boundary = (
                roi_boundary is not None and point.distance(roi_boundary) <= boundary_tolerance
            ) or (field_edge is not None and point.distance(field_edge) <= boundary_tolerance)
            looks_like_true_end = (
                segment.endpoint_confidence[side] >= settings.true_endpoint_probability
            )
            if not reaches_analysis_boundary and not looks_like_true_end:
                unresolved = True
        segment.attrs["unresolved_open_endpoint"] = unresolved
    refinement = refine_rows(
        segments,
        optimization,
        repair_paths,
        model_version=model_version,
        spacing=None,
        roi=roi if roi is not None else effective_field_boundary,
        maximum_spline_deviation_m=settings.maximum_spline_deviation_pixels * pixel_size,
    )
    diagnostics: dict[str, Any] = {
        "configuration": asdict(settings),
        "pixel_size_m": pixel_size,
        "structure_sigma_px": sigma,
        "orientation_coordinate_system": "world_crs_xy_after_full_affine_linear_transform",
        "extraction": extraction_diagnostics,
        "orientation_regimes": [asdict(item) for item in regimes],
        "spacing": {
            str(regime_id): {
                "reliable": field.reliable,
                "global": asdict(field.global_estimate) if field.global_estimate else None,
            }
            for regime_id, field in spacing_by_regime.items()
        },
        "unassigned_segments": len(unassigned_segments),
        "candidate_count": len(all_candidates),
        "candidate_internal_count": sum(1 for c in all_candidates if not c.is_boundary_extension),
        "candidate_boundary_extension_count": sum(
            1 for c in all_candidates if c.is_boundary_extension
        ),
        "feasible_path_count": len(repair_paths),
        "accepted_connection_count": len(optimization.accepted),
        "accepted_internal_repair_count": sum(
            1 for c in optimization.accepted if not c.is_boundary_extension
        ),
        "accepted_boundary_extension_count": sum(
            1 for c in optimization.accepted if c.is_boundary_extension
        ),
        "ambiguous_connection_count": len(optimization.ambiguous),
        "optimizer_exact": optimization.exact,
        "optimizer_objective": optimization.objective,
        "valid_row_count": len(refinement.rows),
        "uncertain_row_count": len(refinement.uncertain_rows),
    }
    return PostprocessResult(
        refinement=refinement,
        segments=segments,
        candidates=all_candidates,
        repair_paths=repair_paths,
        optimization=optimization,
        orientation_field=fused,
        orientation_regimes=regimes,
        spacing_by_regime=spacing_by_regime,
        diagnostics=diagnostics,
    )
