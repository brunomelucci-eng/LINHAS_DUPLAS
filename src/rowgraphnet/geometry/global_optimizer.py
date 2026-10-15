"""Deterministic global selection of mutually compatible repairs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree

from rowgraphnet.domain import CandidateConnection, RepairPath


@dataclass(slots=True)
class OptimizationResult:
    accepted: list[CandidateConnection]
    rejected: list[CandidateConnection]
    ambiguous: list[CandidateConnection]
    objective: float
    exact: bool


def _connection_geometry(
    candidate: CandidateConnection,
    paths: dict[str, RepairPath],
) -> LineString:
    path = paths.get(candidate.candidate_id)
    return path.geometry if path is not None else candidate.geometry


def _geometries_conflict(first: LineString, second: LineString) -> bool:
    # Endpoint ownership is handled before this call.  Any remaining contact
    # joins two distinct row hypotheses and is therefore a conflict, including
    # two extensions converging to the same field-boundary vertex.
    return first.intersects(second)


def _conflict_pairs(
    candidates: list[CandidateConnection],
    paths: dict[str, RepairPath],
) -> set[frozenset[str]]:
    conflicts: set[frozenset[str]] = set()
    if not candidates:
        return conflicts

    # 1. Endpoint collisions (O(N) via index map instead of O(N^2))
    endpoint_to_candidates: dict[Any, list[str]] = {}
    for candidate in candidates:
        for ep in candidate.endpoint_keys:
            endpoint_to_candidates.setdefault(ep, []).append(candidate.candidate_id)

    for cand_ids in endpoint_to_candidates.values():
        if len(cand_ids) > 1:
            for i in range(len(cand_ids)):
                id_i = cand_ids[i]
                for j in range(i + 1, len(cand_ids)):
                    conflicts.add(frozenset((id_i, cand_ids[j])))

    # 2. Geometric intersections (O(N log N) via STRtree instead of O(N^2))
    geoms = [_connection_geometry(c, paths) for c in candidates]
    tree = STRtree(geoms)
    intersections = tree.query(geoms, predicate="intersects")
    for i, j in zip(intersections[0], intersections[1]):
        if i < j:
            conflicts.add(frozenset((candidates[i].candidate_id, candidates[j].candidate_id)))

    return conflicts


def _would_make_cycle(
    adjacency: dict[str, set[str]],
    source: str,
    target: str,
) -> bool:
    if source == target:
        return True
    stack = [source]
    seen: set[str] = set()
    while stack:
        node = stack.pop()
        if node == target:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, ()))
    return False


def _posterior_probability(
    candidate: CandidateConnection,
    paths: dict[str, RepairPath],
    path_evidence_weight: float,
) -> float:
    probability = float(candidate.probability)
    path = paths.get(candidate.candidate_id)
    if (
        path is not None
        and not candidate.is_boundary_extension
        and not bool(path.reason.get("probability_includes_path_quality"))
    ):
        value = path.reason.get("path_quality")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            # Path evidence may only reduce the proposal probability.  It can
            # never rescue a geometrically weak candidate.  The exponent is a
            # validation-calibratable contribution rather than a hidden vote.
            path_quality = min(max(float(value), 0.0), 1.0)
            probability *= path_quality**path_evidence_weight
    return min(max(probability, 1e-12), 1.0 - 1e-12)


def optimize_connections(
    candidates: list[CandidateConnection],
    repair_paths: list[RepairPath] | None = None,
    minimum_probability: float = 0.5,
    abstain_on_ambiguous: bool = True,
    max_exact_candidates: int = 48,
    path_evidence_weight: float = 0.25,
) -> OptimizationResult:
    """Maximize posterior log-odds under row topology constraints.

    The exact solver is a branch-and-bound weighted set-packing search.  It
    enforces one continuation per endpoint, no repair crossings, and no cycles.
    Oversized components use the same deterministic ordering in a conservative
    greedy fallback and are identified by ``result.exact == False``.
    """

    if not 0 < minimum_probability < 1:
        raise ValueError("minimum_probability must lie strictly between zero and one")
    if not 0.0 <= path_evidence_weight <= 1.0:
        raise ValueError("path_evidence_weight must lie in [0, 1]")
    if (
        isinstance(max_exact_candidates, bool)
        or not isinstance(max_exact_candidates, int)
        or max_exact_candidates < 0
    ):
        raise ValueError("max_exact_candidates must be a non-negative integer")
    candidate_ids = [item.candidate_id for item in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")
    for candidate in candidates:
        if not np.isfinite(candidate.probability) or not 0.0 <= candidate.probability <= 1.0:
            raise ValueError("candidate probabilities must be finite and lie in [0, 1]")
        if (
            candidate.geometry.is_empty
            or not candidate.geometry.is_valid
            or not candidate.geometry.is_simple
            or candidate.geometry.length <= 0.0
        ):
            raise ValueError("candidate geometries must be valid, simple, and non-empty")
    path_items = list(repair_paths or [])
    path_ids = [path.candidate_id for path in path_items]
    if len(path_ids) != len(set(path_ids)):
        raise ValueError("repair path candidate IDs must be unique")
    orphan_paths = sorted(set(path_ids) - set(candidate_ids))
    if orphan_paths:
        raise ValueError(f"repair paths reference unknown candidates: {orphan_paths}")
    path_lookup = {path.candidate_id: path for path in path_items}
    paths_required = repair_paths is not None
    ambiguous = sorted(
        [item for item in candidates if item.ambiguous and abstain_on_ambiguous],
        key=lambda item: item.candidate_id,
    )
    ambiguity_ids = {item.candidate_id for item in ambiguous}
    eligible: list[CandidateConnection] = []
    probabilities: dict[str, float] = {}
    utilities: dict[str, float] = {}
    threshold_odds = math.log(minimum_probability / (1.0 - minimum_probability))
    for candidate in candidates:
        if paths_required and candidate.candidate_id not in path_lookup:
            probabilities[candidate.candidate_id] = 0.0
            utilities[candidate.candidate_id] = -math.inf
            continue
        probability = _posterior_probability(candidate, path_lookup, path_evidence_weight)
        probabilities[candidate.candidate_id] = probability
        utility = math.log(probability / (1.0 - probability)) - threshold_odds
        utilities[candidate.candidate_id] = utility
        if candidate.candidate_id not in ambiguity_ids and utility > 0:
            eligible.append(candidate)
    eligible.sort(key=lambda item: (-utilities[item.candidate_id], item.candidate_id))
    conflicts = _conflict_pairs(eligible, path_lookup)
    conflict_map: dict[str, set[str]] = {}
    for pair in conflicts:
        u, v = tuple(pair)
        conflict_map.setdefault(u, set()).add(v)
        conflict_map.setdefault(v, set()).add(u)

    best_selection: list[CandidateConnection] = []
    best_objective = 0.0

    def compatible(
        candidate: CandidateConnection,
        selected_ids: set[str],
        adjacency: dict[str, set[str]],
    ) -> bool:
        cand_conflicts = conflict_map.get(candidate.candidate_id)
        if cand_conflicts is not None and not cand_conflicts.isdisjoint(selected_ids):
            return False
        if candidate.is_boundary_extension:
            return True
        return not _would_make_cycle(adjacency, candidate.segment_a, candidate.segment_b)

    exact = len(eligible) <= max_exact_candidates
    if exact:
        suffix_bound = [0.0] * (len(eligible) + 1)
        for index in range(len(eligible) - 1, -1, -1):
            suffix_bound[index] = suffix_bound[index + 1] + utilities[eligible[index].candidate_id]

        def search(
            index: int,
            selected: list[CandidateConnection],
            selected_ids: set[str],
            adjacency: dict[str, set[str]],
            objective: float,
        ) -> None:
            nonlocal best_selection, best_objective
            if objective + suffix_bound[index] < best_objective - 1e-12:
                return
            if index == len(eligible):
                selected_key = tuple(item.candidate_id for item in selected)
                best_key = tuple(item.candidate_id for item in best_selection)
                if objective > best_objective + 1e-12 or (
                    abs(objective - best_objective) <= 1e-12 and selected_key < best_key
                ):
                    best_objective = objective
                    best_selection = selected.copy()
                return
            candidate = eligible[index]
            if compatible(candidate, selected_ids, adjacency):
                if not candidate.is_boundary_extension:
                    next_adjacency = {key: value.copy() for key, value in adjacency.items()}
                    next_adjacency.setdefault(candidate.segment_a, set()).add(candidate.segment_b)
                    next_adjacency.setdefault(candidate.segment_b, set()).add(candidate.segment_a)
                else:
                    next_adjacency = adjacency
                selected.append(candidate)
                selected_ids.add(candidate.candidate_id)
                search(
                    index + 1,
                    selected,
                    selected_ids,
                    next_adjacency,
                    objective + utilities[candidate.candidate_id],
                )
                selected_ids.remove(candidate.candidate_id)
                selected.pop()
            search(index + 1, selected, selected_ids, adjacency, objective)

        search(0, [], set(), {}, 0.0)
    else:
        adjacency: dict[str, set[str]] = {}
        selected_ids: set[str] = set()
        for candidate in eligible:
            if compatible(candidate, selected_ids, adjacency):
                best_selection.append(candidate)
                selected_ids.add(candidate.candidate_id)
                if not candidate.is_boundary_extension:
                    adjacency.setdefault(candidate.segment_a, set()).add(candidate.segment_b)
                    adjacency.setdefault(candidate.segment_b, set()).add(candidate.segment_a)
                best_objective += utilities[candidate.candidate_id]

    accepted_ids = {item.candidate_id for item in best_selection}
    rejected = sorted(
        [
            item
            for item in candidates
            if item.candidate_id not in accepted_ids and item.candidate_id not in ambiguity_ids
        ],
        key=lambda item: item.candidate_id,
    )
    return OptimizationResult(
        accepted=sorted(best_selection, key=lambda item: item.candidate_id),
        rejected=rejected,
        ambiguous=ambiguous,
        objective=float(best_objective),
        exact=exact,
    )
