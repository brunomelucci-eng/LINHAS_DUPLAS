"""Generate auditable endpoint-to-endpoint continuation hypotheses."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Iterable

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from rowgraphnet.domain import CandidateConnection, ExtractedSegment, SpacingEstimate

OrientationSampler = Callable[[float, float], tuple[float, float]]
EvidenceSampler = Callable[[LineString], float]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).eps:
        return np.zeros(2, dtype=float)
    return vector / norm


def _outward_tangent(segment: ExtractedSegment, side: int, support_length: float) -> np.ndarray:
    coords = np.asarray(segment.geometry.coords, dtype=float)
    if side not in (0, 1):
        raise ValueError("endpoint side must be zero or one")
    if side == 1:
        coords = coords[::-1]
    origin = coords[0]
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
    usable = np.flatnonzero(distances <= support_length)
    count = max(int(usable[-1] + 1) if len(usable) else 2, 2)
    local = coords[: min(count, len(coords))]
    centered = local - local.mean(axis=0)
    if len(local) >= 3 and np.any(centered):
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        inward = vh[0]
        if np.dot(inward, local[-1] - origin) < 0:
            inward = -inward
    else:
        inward = local[-1] - origin
    return -_unit(inward)


def _spacing_scale(spacing: SpacingEstimate) -> float:
    samples = np.asarray(spacing.samples_m, dtype=float)
    sample_spread = float(np.std(samples)) if len(samples) > 1 else 0.0
    uncertainty_floor = spacing.median_m * max(1.0 - spacing.confidence, spacing.entropy) * 0.25
    return max(
        1.4826 * spacing.mad_m,
        sample_spread,
        uncertainty_floor,
        np.spacing(spacing.median_m),
    )


def _search_radius(spacing: SpacingEstimate) -> float:
    samples = np.asarray(spacing.samples_m, dtype=float)
    upper = float(np.quantile(samples, 0.95)) if len(samples) else spacing.median_m
    measured_spread = max(
        1.4826 * spacing.mad_m, float(np.std(samples)) if len(samples) > 1 else 0.0
    )
    # A robust upper confidence envelope is permitted to grow only from
    # measured spacing variation.  High entropy never makes repair more aggressive.
    return max(upper + 3.0 * measured_spread, spacing.median_m)


def _crosses_observed(
    connector: LineString,
    source_a: str,
    source_b: str,
    segments: Iterable[ExtractedSegment],
    spatial_tree: Any = None,
) -> bool:
    if spatial_tree is not None and isinstance(segments, (list, tuple)):
        indices = spatial_tree.query(connector)
        for idx in indices:
            seg = segments[idx]
            if seg.segment_id in {source_a, source_b}:
                continue
            if connector.intersects(seg.geometry):
                return True
        return False
    for segment in segments:
        if segment.segment_id in {source_a, source_b}:
            continue
        if connector.intersects(segment.geometry):
            return True
    return False


def _candidate_id(segment_a: str, side_a: int, segment_b: str, side_b: int) -> str:
    endpoints = sorted(((segment_a, side_a), (segment_b, side_b)))
    digest = hashlib.sha256(repr(endpoints).encode("utf-8")).hexdigest()[:16]
    return f"edge_{digest}"


def _normalized_entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    values = np.asarray(probabilities, dtype=float)
    values = np.clip(values, np.finfo(float).eps, None)
    values /= values.sum()
    entropy = -float(np.sum(values * np.log(values)))
    return entropy / math.log(len(values))


def mark_ambiguous_endpoint_choices(
    candidates: Iterable[CandidateConnection],
    *,
    ambiguity_entropy: float,
    ambiguity_probability_margin: float | None,
) -> None:
    """Mark competing endpoint choices, including mixed repair types.

    This function is intentionally separate from candidate generation so a
    boundary extension and an internal gap repair compete under the same
    abstention rule.
    """

    if not math.isfinite(ambiguity_entropy) or not 0.0 <= ambiguity_entropy <= 1.0:
        raise ValueError("ambiguity_entropy must lie in [0, 1]")
    if ambiguity_probability_margin is not None and not (
        math.isfinite(ambiguity_probability_margin)
        and 0.0 <= ambiguity_probability_margin <= 1.0
    ):
        raise ValueError("ambiguity_probability_margin must lie in [0, 1] or be None")
    materialized = list(candidates)
    for candidate in materialized:
        had_choice_mark = "choice_entropy" in candidate.evidence
        candidate.evidence.pop("choice_entropy", None)
        candidate.evidence.pop("best_second_probability_margin", None)
        if had_choice_mark and "abstention_reason" not in candidate.evidence:
            candidate.ambiguous = False
    by_endpoint: dict[tuple[str, int], list[CandidateConnection]] = defaultdict(list)
    for candidate in materialized:
        for key in candidate.endpoint_keys:
            by_endpoint[key].append(candidate)
    for group in by_endpoint.values():
        entropy = _normalized_entropy([item.probability for item in group])
        ordered = sorted((item.probability for item in group), reverse=True)
        best_second_margin = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
        is_near_tie = (
            ambiguity_probability_margin is not None
            and len(ordered) > 1
            and best_second_margin <= ambiguity_probability_margin
        )
        if entropy >= ambiguity_entropy or is_near_tie:
            for item in group:
                item.ambiguous = True
                item.evidence["choice_entropy"] = entropy
                item.evidence["best_second_probability_margin"] = best_second_margin


def generate_candidates(
    segments: list[ExtractedSegment],
    spacing: SpacingEstimate,
    orientation_sampler: OrientationSampler | None = None,
    evidence_sampler: EvidenceSampler | None = None,
    ambiguity_entropy: float = 1.0,
    ambiguity_probability_margin: float | None = 0.05,
    search_radius_m: float | None = None,
) -> list[CandidateConnection]:
    """Propose plausible connections without accepting any of them.

    Candidate probability is a transparent combination of endpoint facing,
    tangent compatibility, transverse-corridor consistency, learned local
    orientation, endpoint evidence, and continuous center support.
    """

    if spacing.median_m <= 0:
        raise ValueError("spacing median must be positive")
    if not math.isfinite(ambiguity_entropy) or not 0.0 <= ambiguity_entropy <= 1.0:
        raise ValueError("ambiguity_entropy must lie in [0, 1]")
    if ambiguity_probability_margin is not None and not (
        math.isfinite(ambiguity_probability_margin)
        and 0.0 <= ambiguity_probability_margin <= 1.0
    ):
        raise ValueError("ambiguity_probability_margin must lie in [0, 1] or be None")
    radius = _search_radius(spacing) if search_radius_m is None else float(search_radius_m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("search radius must be positive and finite")
    tangent_support = min(radius, spacing.median_m)
    from shapely import STRtree

    segments_tree = STRtree([s.geometry for s in segments]) if segments else None
    endpoints: list[tuple[ExtractedSegment, int, Point, np.ndarray]] = []
    for segment in sorted(segments, key=lambda item: item.segment_id):
        points = segment.endpoints
        for side in (0, 1):
            endpoints.append(
                (segment, side, points[side], _outward_tangent(segment, side, tangent_support))
            )

    candidates: list[CandidateConnection] = []
    corridor_scale = _spacing_scale(spacing)
    pts = np.array([(p.x, p.y) for _, _, p, _ in endpoints], dtype=np.float64)
    if len(pts) > 0:
        kdtree = cKDTree(pts)
        candidate_pairs = sorted(kdtree.query_pairs(radius))
    else:
        candidate_pairs = []

    for index_a, index_b in candidate_pairs:
        segment_a, side_a, point_a, tangent_a = endpoints[index_a]
        segment_b, side_b, point_b, tangent_b = endpoints[index_b]
        if segment_a.segment_id == segment_b.segment_id:
            continue
        delta = np.array((point_b.x - point_a.x, point_b.y - point_a.y), dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance <= np.finfo(float).eps or distance > radius:
            continue
        direction = delta / distance
        facing_a = max(float(np.dot(tangent_a, direction)), 0.0)
        facing_b = max(float(np.dot(tangent_b, -direction)), 0.0)
        facing_score = math.sqrt(facing_a * facing_b)
        if facing_score <= 0:
            continue

        tangent_alignment = float(
            (1.0 + np.cos(2.0 * np.arccos(np.clip(np.dot(tangent_a, -tangent_b), -1, 1)))) / 2.0
        )
        lateral_a = abs(float(tangent_a[0] * delta[1] - tangent_a[1] * delta[0]))
        lateral_b = abs(float(tangent_b[0] * -delta[1] - tangent_b[1] * -delta[0]))
        lateral = (lateral_a + lateral_b) / 2.0
        corridor_score = float(np.exp(-0.5 * (lateral / corridor_scale) ** 2))
        distance_score = float(np.exp(-distance / radius))

        midpoint = ((point_a.x + point_b.x) / 2.0, (point_a.y + point_b.y) / 2.0)
        if orientation_sampler is None:
            orientation_score = tangent_alignment
            orientation_confidence = 0.0
        else:
            theta, orientation_confidence = orientation_sampler(*midpoint)
            connector_theta = math.atan2(delta[1], delta[0])
            orientation_score = float((1.0 + math.cos(2.0 * (connector_theta - theta))) / 2.0)
            orientation_score = orientation_score * np.clip(
                orientation_confidence, 0.0, 1.0
            ) + tangent_alignment * (1.0 - np.clip(orientation_confidence, 0.0, 1.0))

        connector = LineString((point_a, point_b))
        if _crosses_observed(
            connector,
            segment_a.segment_id,
            segment_b.segment_id,
            segments,
            spatial_tree=segments_tree,
        ):
            continue
        # The endpoint head represents evidence of a *true physical end*.
        # A continuation is plausible when neither open skeleton tip looks
        # like a true end; ROI/image cut handling occurs downstream.
        endpoint_score = float(
            np.sqrt(
                max(1.0 - segment_a.endpoint_confidence[side_a], 0.0)
                * max(1.0 - segment_b.endpoint_confidence[side_b], 0.0)
            )
        )
        visual_support = (
            float(np.clip(evidence_sampler(connector), 0.0, 1.0))
            if evidence_sampler is not None
            else float(np.sqrt(max(segment_a.confidence, 0.0) * max(segment_b.confidence, 0.0)))
        )
        scores = np.array(
            [
                facing_score,
                tangent_alignment,
                corridor_score,
                distance_score,
                orientation_score,
                visual_support,
            ],
            dtype=float,
        )
        # True-end evidence is a veto likelihood rather than one weak vote
        # among many geometric terms.  This prevents an otherwise pretty
        # continuation from overriding a confidently observed physical end.
        probability = float(
            np.mean(np.clip(scores, 0.0, 1.0)) * np.clip(endpoint_score, 0.0, 1.0)
        )
        cost = float(-np.log(np.clip(probability, np.finfo(float).eps, 1.0)))
        candidates.append(
            CandidateConnection(
                candidate_id=_candidate_id(
                    segment_a.segment_id,
                    side_a,
                    segment_b.segment_id,
                    side_b,
                ),
                segment_a=segment_a.segment_id,
                side_a=side_a,
                segment_b=segment_b.segment_id,
                side_b=side_b,
                geometry=connector,
                probability=probability,
                cost=cost,
                evidence={
                    "facing": facing_score,
                    "tangent": tangent_alignment,
                    "corridor": corridor_score,
                    "distance": distance_score,
                    "orientation": orientation_score,
                    "orientation_confidence": float(orientation_confidence),
                    "not_true_endpoint": endpoint_score,
                    "visual_support": visual_support,
                    "gap_m": distance,
                    "search_radius_m": radius,
                },
            )
        )

    # Ambiguity is assessed locally at each endpoint.  We retain the hypotheses
    # for review but the optimizer can abstain from selecting them.
    mark_ambiguous_endpoint_choices(
        candidates,
        ambiguity_entropy=ambiguity_entropy,
        ambiguity_probability_margin=ambiguity_probability_margin,
    )

    candidates.sort(key=lambda item: item.candidate_id)
    return candidates
