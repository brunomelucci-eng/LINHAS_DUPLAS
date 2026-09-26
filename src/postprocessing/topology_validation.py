"""Network-wide topology validation for reconstructed crop rows."""

from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)


def _hairpin_metrics(line: LineString, sample_spacing_m: float = 0.20) -> Tuple[float, float]:
    """Return endpoint/chord ratio and accumulated absolute turning in degrees."""
    if line is None or line.is_empty or line.length <= 1e-9:
        return 1.0, 0.0
    endpoint_distance = Point(line.coords[0]).distance(Point(line.coords[-1]))
    ratio = float(endpoint_distance / line.length)
    sample_count = max(3, int(np.ceil(line.length / max(sample_spacing_m, 0.02))) + 1)
    distances = np.linspace(0.0, line.length, sample_count)
    coordinates = np.asarray([line.interpolate(value).coords[0] for value in distances])
    vectors = np.diff(coordinates[:, :2], axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors[norms > 1e-9]
    if len(vectors) < 2:
        return ratio, 0.0
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    dots = np.clip(np.sum(vectors[:-1] * vectors[1:], axis=1), -1.0, 1.0)
    total_turn = float(np.degrees(np.arccos(dots)).sum())
    return ratio, total_turn


def _query_indices(tree: STRtree, geometry, geometries: Sequence[LineString]) -> Iterable[int]:
    """Support both Shapely 1.x (geometries) and 2.x (integer indices)."""
    result = tree.query(geometry)
    by_id = {id(item): idx for idx, item in enumerate(geometries)}
    for item in result:
        if hasattr(item, "item") and not hasattr(item, "geom_type"):
            yield int(item.item())
        elif isinstance(item, int):
            yield item
        else:
            idx = by_id.get(id(item))
            if idx is not None:
                yield idx


def _parts(geometry, kind: str):
    if geometry.is_empty:
        return
    if geometry.geom_type == kind:
        yield geometry
    elif isinstance(geometry, (GeometryCollection, MultiPoint, MultiLineString)):
        for part in geometry.geoms:
            yield from _parts(part, kind)


def _node_key(point: Point, tolerance: float) -> Tuple[int, int]:
    scale = max(tolerance, 1e-9)
    return (round(point.x / scale), round(point.y / scale))


def _is_endpoint(line: LineString, point: Point, tolerance: float) -> bool:
    return min(Point(line.coords[0]).distance(point), Point(line.coords[-1]).distance(point)) <= tolerance


def analyze_topology(lines: Sequence[LineString], config: dict) -> List[Dict[str, Any]]:
    """Analyze the complete line network and return one attribute record per line."""
    topo_cfg = config.get("topology_validation", {})
    tolerance = float(topo_cfg.get("node_snap_tolerance_m", 0.01))
    max_cycle = float(
        topo_cfg.get(
            "max_cycle_perimeter_m",
            config.get("network_cleanup", {}).get("max_cycle_perimeter_m", 5.0),
        )
    )
    reject_closed_loops = bool(topo_cfg.get("reject_closed_loops", True))
    reject_hairpins = bool(topo_cfg.get("reject_hairpins", True))
    hairpin_min_length = float(topo_cfg.get("hairpin_min_length_m", 1.0))
    hairpin_max_ratio = float(topo_cfg.get("hairpin_max_endpoint_ratio", 0.35))
    hairpin_min_turn = float(topo_cfg.get("hairpin_min_total_turn_deg", 120.0))
    hairpin_spacing = float(topo_cfg.get("hairpin_sample_spacing_m", 0.20))

    records: List[Dict[str, Any]] = [
        {
            "has_crossing": False,
            "node_degree_max": 0,
            "has_short_cycle": False,
            "has_branch": False,
            "has_overlap": False,
            "has_closed_loop": False,
            "has_hairpin": False,
            "endpoint_length_ratio": 1.0,
            "total_turn_deg": 0.0,
            "topology_status": "valid",
            "rejection_reason": "",
        }
        for _ in lines
    ]
    if not lines:
        return records

    # Points at which each input line must be split to calculate network degree/cycles.
    split_points: Dict[int, List[Point]] = {
        i: [Point(line.coords[0]), Point(line.coords[-1])]
        for i, line in enumerate(lines)
        if line is not None and not line.is_empty and len(line.coords) >= 2
    }
    usable_indices = list(split_points)
    usable = [lines[i] for i in usable_indices]
    tree = STRtree(usable) if usable else None
    local_to_original = {local: original for local, original in enumerate(usable_indices)}

    if tree is not None:
        for local_i, line_i in enumerate(usable):
            i = local_to_original[local_i]
            for local_j in _query_indices(tree, line_i, usable):
                if local_j <= local_i:
                    continue
                j = local_to_original[local_j]
                line_j = lines[j]
                intersection = line_i.intersection(line_j)
                if intersection.is_empty:
                    continue

                overlap_length = sum(part.length for part in _parts(intersection, "LineString"))
                if overlap_length > tolerance:
                    records[i]["has_overlap"] = records[j]["has_overlap"] = True

                for point in _parts(intersection, "Point"):
                    split_points[i].append(point)
                    split_points[j].append(point)
                    # A contact is a crossing/T/X whenever it is not an endpoint of both lines.
                    if not (_is_endpoint(line_i, point, tolerance) and _is_endpoint(line_j, point, tolerance)):
                        records[i]["has_crossing"] = records[j]["has_crossing"] = True

    graph = nx.Graph()
    edge_owners: Dict[frozenset, set] = defaultdict(set)
    node_owners: Dict[Tuple[int, int], set] = defaultdict(set)

    for i, line in enumerate(lines):
        if i not in split_points:
            continue
        distances = sorted({max(0.0, min(line.length, line.project(p))) for p in split_points[i]})
        for distance in distances:
            node_owners[_node_key(line.interpolate(distance), tolerance)].add(i)
        for start, end in zip(distances, distances[1:]):
            if end - start <= tolerance:
                continue
            a = _node_key(line.interpolate(start), tolerance)
            b = _node_key(line.interpolate(end), tolerance)
            graph.add_edge(a, b, weight=end - start)
            edge_owners[frozenset((a, b))].add(i)

    for node, degree in graph.degree:
        owners = node_owners.get(node, set())
        for idx in owners:
            records[idx]["node_degree_max"] = max(records[idx]["node_degree_max"], degree)
            if degree > 2:
                records[idx]["has_branch"] = True

    for cycle in nx.cycle_basis(graph):
        cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1]))
        perimeter = sum(float(graph.edges[a, b].get("weight", 0.0)) for a, b in cycle_edges)
        if perimeter > max_cycle:
            continue
        owners = set()
        for a, b in cycle_edges:
            owners.update(edge_owners.get(frozenset((a, b)), set()))
        for idx in owners:
            records[idx]["has_short_cycle"] = True

    for idx, (line, record) in enumerate(zip(lines, records)):
        reasons = []
        if line is None or line.is_empty:
            reasons.append("empty")
        elif not line.is_valid:
            reasons.append("invalid_geometry")
        elif not line.is_simple:
            reasons.append("self_intersection")
        if line is not None and not line.is_empty and len(line.coords) >= 2:
            endpoint_ratio, total_turn = _hairpin_metrics(line, hairpin_spacing)
            record["endpoint_length_ratio"] = endpoint_ratio
            record["total_turn_deg"] = total_turn
            record["has_closed_loop"] = bool(line.is_ring)
            record["has_hairpin"] = bool(
                line.length >= hairpin_min_length
                and endpoint_ratio <= hairpin_max_ratio
                and total_turn >= hairpin_min_turn
            )
        if reject_closed_loops and record["has_closed_loop"]:
            reasons.append("closed_loop")
        if reject_hairpins and record["has_hairpin"]:
            reasons.append("hairpin")
        if record["has_crossing"]:
            reasons.append("crossing")
        if record["has_branch"]:
            reasons.append("degree_gt_2")
        if record["has_short_cycle"]:
            reasons.append("short_cycle")
        if record["has_overlap"]:
            reasons.append("overlap")
        record["rejection_reason"] = ";".join(reasons)
        record["topology_status"] = "rejected" if reasons else "valid"

    return records


def validate_topology(
    lines: List[LineString],
    config: dict,
    return_attributes: bool = False,
):
    """Filter invalid network members, optionally returning their topology attributes."""
    min_len = float(config.get("vector", {}).get("min_line_length_m", 3.0))
    records = analyze_topology(lines, config)
    valid_lines: List[LineString] = []
    valid_records: List[Dict[str, Any]] = []

    for line, record in zip(lines, records):
        if line is None or line.is_empty or line.length < min_len:
            continue
        if record["topology_status"] != "valid":
            logger.warning("Rejected line topology: %s", record["rejection_reason"])
            continue
        valid_lines.append(line)
        valid_records.append(record)

    logger.info("Topology validation: kept %d valid lines from %d candidates.", len(valid_lines), len(lines))
    if return_attributes:
        return valid_lines, valid_records
    return valid_lines
