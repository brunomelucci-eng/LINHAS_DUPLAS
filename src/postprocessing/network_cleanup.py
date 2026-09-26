"""Conservative cleanup of branch networks before vector fitting."""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import DefaultDict, List, Sequence, Tuple

import networkx as nx
import numpy as np

Pixel = Tuple[int, int]
Branch = List[Pixel]
logger = logging.getLogger(__name__)


def _clean_points(branch: Sequence[Pixel]) -> Branch:
    result: Branch = []
    for point in branch:
        point = tuple(point)
        if not result or point != result[-1]:
            result.append(point)
    return result


def _length(branch: Sequence[Pixel], gsd: float) -> float:
    if len(branch) < 2:
        return 0.0
    points = np.asarray(branch, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() * gsd)


def _outward_tangent(branch: Sequence[Pixel], at_start: bool) -> np.ndarray:
    step = min(5, len(branch) - 1)
    if at_start:
        vector = np.asarray(branch[0], dtype=float) - np.asarray(branch[step], dtype=float)
    else:
        vector = np.asarray(branch[-1], dtype=float) - np.asarray(branch[-1 - step], dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def cleanup_network(branches: List[Branch], config: dict, gsd: float = 1.0) -> List[Branch]:
    """Keep dominant collinear paths and discard artificial branches/cycles."""
    cfg = config.get("network_cleanup", {})
    cleaned = [_clean_points(branch) for branch in branches]
    cleaned = [branch for branch in cleaned if len(branch) >= 2]
    if not cfg.get("enabled", True) or not cleaned:
        return cleaned

    max_angle = float(cfg.get("max_branch_angle_deg", 8.0))
    max_short = float(cfg.get("max_short_branch_m", 1.5))
    reject_degree = bool(cfg.get("reject_degree_gt_2", True))
    reject_cycles = bool(cfg.get("reject_short_cycles", True))
    max_cycle = float(cfg.get("max_cycle_perimeter_m", 5.0))

    lengths = [_length(branch, gsd) for branch in cleaned]
    incident: DefaultDict[Pixel, list] = defaultdict(list)
    for idx, branch in enumerate(cleaned):
        incident[branch[0]].append((idx, True, _outward_tangent(branch, True)))
        incident[branch[-1]].append((idx, False, _outward_tangent(branch, False)))

    remove = set()
    for node, members in incident.items():
        if len(members) <= 2:
            continue

        pair_scores = []
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                ia, _, ta = members[a]
                ib, _, tb = members[b]
                if ia == ib:
                    continue
                deviation = float(np.degrees(np.arccos(np.clip(-np.dot(ta, tb), -1.0, 1.0))))
                pair_scores.append((deviation, -(lengths[ia] + lengths[ib]), ia, ib))

        if not pair_scores:
            continue
        deviation, _, first, second = min(pair_scores)
        keep = {first, second} if deviation <= max_angle else {max((m[0] for m in members), key=lengths.__getitem__)}

        for idx, _, _ in members:
            if idx not in keep and (lengths[idx] <= max_short or reject_degree):
                remove.add(idx)
                logger.debug("Removing branch %d at degree-%d node %s.", idx, len(members), node)

    if reject_cycles:
        graph = nx.Graph()
        owners = {}
        for idx, branch in enumerate(cleaned):
            if idx in remove:
                continue
            a, b = branch[0], branch[-1]
            if a == b and lengths[idx] <= max_cycle:
                remove.add(idx)
                continue
            graph.add_edge(a, b, weight=lengths[idx])
            owners[frozenset((a, b))] = idx

        for cycle in nx.cycle_basis(graph):
            edges = list(zip(cycle, cycle[1:] + cycle[:1]))
            perimeter = sum(float(graph.edges[a, b]["weight"]) for a, b in edges)
            if perimeter <= max_cycle:
                candidates = [owners[frozenset((a, b))] for a, b in edges]
                remove.add(min(candidates, key=lengths.__getitem__))

    result = [branch for idx, branch in enumerate(cleaned) if idx not in remove]
    logger.info("Network cleanup: kept %d/%d branches; removed %d unsafe branches.", len(result), len(cleaned), len(remove))
    return result
