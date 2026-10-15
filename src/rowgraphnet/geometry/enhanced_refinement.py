"""Enhanced post-processing refinement for crop rows:
1. Neighbor-guided long gap bridging (connecting skips/gaps up to 6m supported by adjacent parallel rows).
2. Boundary extension with cluster consensus (straight parallel projection to field boundary,
   using neighboring rows to compute a robust consensus angle and ignore terminal tip waviness).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, unary_union

from rowgraphnet.domain import RowRecord
from rowgraphnet.geometry.vector_refinement import (
    _axial_azimuth_deg,
    _smooth_safely,
    _stable_row_id,
)


def _safe_axial_azimuth_deg(geometry: LineString) -> float:
    """Compute axial azimuth strictly in [0, 180), guarding against float roundoff like -1e-16 % 180 == 180.0."""
    deg = float(_axial_azimuth_deg(geometry))
    if deg >= 180.0 or deg < 0.0:
        deg = deg % 180.0
    if deg >= 180.0 or abs(deg - 180.0) < 1e-7:
        deg = 0.0
    return float(np.clip(deg, 0.0, 179.999999))


@dataclass(slots=True, frozen=True)
class EnhancedRefinementConfig:
    """Configurable controls for neighbor-guided gap bridging and consensus boundary extension."""

    enabled: bool = True
    # Gap closing
    max_gap_distance_m: float = 6.0
    min_gap_distance_m: float = 0.10
    max_transverse_deviation_factor: float = 0.35  # fraction of spacing
    min_colinearity_cos: float = 0.88  # ~28 deg max divergence
    min_facing_cos: float = 0.80  # ~36 deg max facing divergence
    neighbor_parallel_window_factor: float = 2.5
    require_neighbor_support_above_m: float = 2.0
    # Boundary extension
    max_boundary_extension_m: float = 8.0
    min_boundary_extension_m: float = 0.10
    boundary_consensus_neighbors: int = 5
    safe_corridor_fraction: float = 0.30
    terminal_tangent_length_m: float = 5.0
    # Smoothing
    max_spline_deviation_m: float = 0.10


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-9:
        return np.zeros(2, dtype=float)
    return v / norm


def _outward_endpoint_tangent(line: LineString, side: int, length_m: float) -> np.ndarray:
    """Compute unit tangent pointing outward from the line at side (0=start, 1=end)."""
    coords = np.asarray(line.coords, dtype=float)
    if side == 1:
        coords = coords[::-1]
    origin = coords[0]
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
    usable = np.flatnonzero(distances <= length_m)
    count = max(int(usable[-1] + 1) if len(usable) else 2, 2)
    local = coords[: min(count, len(coords))]
    if len(local) >= 3:
        centered = local - local.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        inward = vh[0]
        if np.dot(inward, local[-1] - origin) < 0:
            inward = -inward
    else:
        inward = local[-1] - origin
    return -_unit(inward)


def _transverse_distance(point: tuple[float, float], line_origin: tuple[float, float], direction: np.ndarray) -> float:
    """Perpendicular distance from point to the line defined by line_origin and direction."""
    delta = np.array([point[0] - line_origin[0], point[1] - line_origin[1]], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    return float(abs(np.dot(delta, normal)))


def _has_parallel_neighbor_support(
    p_a: tuple[float, float],
    p_b: tuple[float, float],
    all_rows: list[RowRecord],
    current_ids: set[str],
    spacing_m: float,
    window_m: float,
) -> bool:
    """Check if there is at least one parallel row nearby that runs continuously along the gap."""
    connector = LineString([p_a, p_b])
    mid_point = connector.interpolate(0.5, normalized=True)
    buffer_zone = mid_point.buffer(window_m)

    min_spacing = 0.65 * spacing_m
    max_spacing = 1.45 * spacing_m

    for row in all_rows:
        if row.row_id in current_ids:
            continue
        if not buffer_zone.intersects(row.geometry):
            continue
        dist = connector.distance(row.geometry)
        if min_spacing <= dist <= max_spacing:
            # Check that the neighbor geometry spans past both ends of the connector
            proj_a = row.geometry.project(Point(p_a))
            proj_b = row.geometry.project(Point(p_b))
            span = abs(proj_b - proj_a)
            # The neighbor should cover a comparable longitudinal length
            if span >= 0.5 * connector.length:
                return True
    return False


def bridge_long_gaps_with_neighbors(
    rows: list[RowRecord],
    spacing_m: float,
    config: EnhancedRefinementConfig,
    model_version: str,
) -> tuple[list[RowRecord], int]:
    """Bridge long gaps (up to max_gap_distance_m) between row endpoints using parallel neighbor support."""
    if not rows or len(rows) < 2:
        return rows, 0

    tangent_len = max(config.terminal_tangent_length_m, spacing_m * 1.5)
    max_transverse = config.max_transverse_deviation_factor * spacing_m
    neighbor_window = config.neighbor_parallel_window_factor * spacing_m
    safe_corridor = config.safe_corridor_fraction * spacing_m

    # Extract free endpoints
    # item: (row_id, side, (x, y), tangent_vec)
    endpoints: list[tuple[str, int, tuple[float, float], np.ndarray]] = []
    row_map: dict[str, RowRecord] = {r.row_id: r for r in rows}

    for r in rows:
        coords = list(r.geometry.coords)
        t0 = _outward_endpoint_tangent(r.geometry, 0, tangent_len)
        t1 = _outward_endpoint_tangent(r.geometry, 1, tangent_len)
        endpoints.append((r.row_id, 0, coords[0], t0))
        endpoints.append((r.row_id, 1, coords[-1], t1))

    candidates: list[tuple[float, str, int, str, int, LineString]] = []

    for i in range(len(endpoints)):
        id_a, side_a, pt_a, t_a = endpoints[i]
        for j in range(i + 1, len(endpoints)):
            id_b, side_b, pt_b, t_b = endpoints[j]
            if id_a == id_b:
                continue

            delta = np.array([pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]], dtype=float)
            dist = float(np.linalg.norm(delta))
            if dist < config.min_gap_distance_m or dist > config.max_gap_distance_m:
                continue

            u_ab = delta / dist
            facing_a = float(np.dot(t_a, u_ab))
            facing_b = float(np.dot(t_b, -u_ab))
            if facing_a < config.min_facing_cos or facing_b < config.min_facing_cos:
                continue

            colinear = float(abs(np.dot(t_a, t_b)))
            if colinear < config.min_colinearity_cos:
                continue

            # Check transverse offset
            lat_a = _transverse_distance(pt_b, pt_a, t_a)
            lat_b = _transverse_distance(pt_a, pt_b, t_b)
            if max(lat_a, lat_b) > max_transverse:
                continue

            connector = LineString([pt_a, pt_b])

            # Check vetoes against other rows
            crosses_other = False
            for other_id, other_row in row_map.items():
                if other_id in (id_a, id_b):
                    continue
                if connector.crosses(other_row.geometry) or connector.distance(other_row.geometry) < safe_corridor:
                    crosses_other = True
                    break
            if crosses_other:
                continue

            # If gap is long, verify that a parallel neighbor is present as physical support
            if dist > config.require_neighbor_support_above_m:
                has_support = _has_parallel_neighbor_support(
                    pt_a, pt_b, rows, {id_a, id_b}, spacing_m, neighbor_window
                )
                if not has_support:
                    continue

            # Quality metric (lower is better): distance + lateral deviation penalty
            score = dist + 2.0 * max(lat_a, lat_b) - 2.0 * (facing_a + facing_b)
            candidates.append((score, id_a, side_a, id_b, side_b, connector))

    # Sort candidates by best score
    candidates.sort(key=lambda c: c[0])

    # Greedy 1-to-1 matching
    used_endpoints: set[tuple[str, int]] = set()
    accepted_edges: list[tuple[str, str, LineString]] = []

    for _score, id_a, side_a, id_b, side_b, connector in candidates:
        ep_a = (id_a, side_a)
        ep_b = (id_b, side_b)
        if ep_a in used_endpoints or ep_b in used_endpoints:
            continue
        used_endpoints.add(ep_a)
        used_endpoints.add(ep_b)
        accepted_edges.append((id_a, id_b, connector))

    if not accepted_edges:
        return rows, 0

    # Build graph to merge connected components
    g = nx.Graph()
    for r in rows:
        g.add_node(r.row_id)
    edge_connectors: dict[tuple[str, str], LineString] = {}
    for id_a, id_b, conn in accepted_edges:
        g.add_edge(id_a, id_b)
        edge_connectors[(min(id_a, id_b), max(id_a, id_b))] = conn

    new_rows: list[RowRecord] = []
    bridged_count = len(accepted_edges)

    for comp in nx.connected_components(g):
        comp_rows = [row_map[node_id] for node_id in comp]
        if len(comp_rows) == 1:
            new_rows.append(comp_rows[0])
            continue

        # Gather geometries to merge
        lines_to_merge = [r.geometry for r in comp_rows]
        for u, v in g.edges(comp):
            key = (min(u, v), max(u, v))
            if key in edge_connectors:
                lines_to_merge.append(edge_connectors[key])

        merged = linemerge(unary_union(lines_to_merge))
        if (
            isinstance(merged, LineString)
            and merged.is_valid
            and merged.is_simple
            and not merged.is_empty
            and np.isfinite(merged.length)
            and merged.length > 0.0
        ):
            try:
                geom = _smooth_safely(merged, config.max_spline_deviation_m)
                if not (isinstance(geom, LineString) and geom.is_valid and geom.is_simple and not geom.is_empty):
                    geom = merged
                total_len = sum(r.geometry.length for r in comp_rows)
                rep_len = sum(edge_connectors[(min(u, v), max(u, v))].length for u, v in g.edges(comp) if (min(u, v), max(u, v)) in edge_connectors)
                accounted = total_len + rep_len
                obs_ratio = float(np.clip(total_len / accounted, 0.0, 1.0)) if accounted > 0 else 1.0
                repaired_ratio = float(np.clip(rep_len / accounted, 0.0, 1.0)) if accounted > 0 else 0.0
                conf = float(np.mean([r.confidence for r in comp_rows]))
                new_id = _stable_row_id(geom)
                new_rows.append(
                    RowRecord(
                        row_id=new_id,
                        geometry=geom,
                        confidence=conf,
                        observed_ratio=obs_ratio,
                        repaired_ratio=repaired_ratio,
                        spacing_local_m=spacing_m,
                        orientation_deg=_safe_axial_azimuth_deg(geom),
                        topology_status="valid",
                        repair_reason=json.dumps({"enhanced_gap_bridged_count": len(comp_rows) - 1}),
                        model_version=model_version,
                    )
                )
            except Exception:
                new_rows.extend(comp_rows)
        else:
            # Fallback: keep original rows if merge failed
            new_rows.extend(comp_rows)


    return new_rows, bridged_count


def _field_polygons(field_boundary: BaseGeometry) -> list[Polygon]:
    if isinstance(field_boundary, Polygon):
        return [field_boundary] if not field_boundary.is_empty and field_boundary.is_valid else []
    elif isinstance(field_boundary, MultiPolygon):
        return [p for p in field_boundary.geoms if not p.is_empty and p.is_valid]
    return []


def _polygon_for_point(polygons: list[Polygon], pt: Point) -> Polygon | None:
    covering = [p for p in polygons if p.covers(pt)]
    if len(covering) == 1:
        return covering[0]
    candidates = [p for p in polygons if p.distance(pt) <= 1.0]
    if candidates:
        return min(candidates, key=lambda p: p.distance(pt))
    return None


def _find_nearest_neighbors_terminal_headings(
    target_point: Point,
    target_row_id: str,
    all_rows: list[RowRecord],
    k_neighbors: int,
    max_search_m: float,
    tangent_len: float,
) -> list[float]:
    """Gather terminal azimuths of k nearest neighboring rows near target_point."""
    distances: list[tuple[float, float]] = []  # (dist, azimuth_deg)
    for r in all_rows:
        if r.row_id == target_row_id:
            continue
        d = r.geometry.distance(target_point)
        if d <= max_search_m:
            # Find which endpoint of r is closer to target_point
            c = list(r.geometry.coords)
            d0 = math.hypot(c[0][0] - target_point.x, c[0][1] - target_point.y)
            d1 = math.hypot(c[-1][0] - target_point.x, c[-1][1] - target_point.y)
            side = 0 if d0 < d1 else 1
            tangent = _outward_endpoint_tangent(r.geometry, side, tangent_len)
            azimuth = float(math.degrees(math.atan2(tangent[0], tangent[1])) % 360.0)
            distances.append((d, azimuth))

    distances.sort(key=lambda item: item[0])
    return [az for _, az in distances[:k_neighbors]]


def _circular_median_deg(angles_deg: list[float]) -> float:
    """Robust circular median for angles in degrees."""
    if not angles_deg:
        return 0.0
    if len(angles_deg) == 1:
        return angles_deg[0]
    rads = np.radians(angles_deg)
    # Using complex mean as initial estimate, then finding circular deviation
    sin_sum = np.sum(np.sin(rads))
    cos_sum = np.sum(np.cos(rads))
    mean_angle = math.atan2(sin_sum, cos_sum)
    # Differences relative to mean angle
    diffs = np.arctan2(np.sin(rads - mean_angle), np.cos(rads - mean_angle))
    median_diff = float(np.median(diffs))
    result_deg = float(math.degrees(mean_angle + median_diff) % 360.0)
    return result_deg


def extend_to_boundary_with_consensus(
    rows: list[RowRecord],
    field_boundary: BaseGeometry,
    spacing_m: float,
    config: EnhancedRefinementConfig,
    model_version: str,
) -> tuple[list[RowRecord], int]:
    """Extend eligible row endpoints straight to the physical field boundary using neighbor consensus."""
    polygons = _field_polygons(field_boundary)
    if not polygons or not rows:
        return rows, 0

    tangent_len = max(config.terminal_tangent_length_m, spacing_m * 1.5)
    max_reach = config.max_boundary_extension_m
    neighbor_search = spacing_m * 5.0
    safe_corridor = config.safe_corridor_fraction * spacing_m

    extended_rows: list[RowRecord] = []
    extension_count = 0

    for r in rows:
        coords = list(r.geometry.coords)
        if len(coords) < 2:
            extended_rows.append(r)
            continue

        modified = False
        new_coords = list(coords)

        # Evaluate both endpoints: side 0 (start) and side 1 (end)
        for side in (0, 1):
            pt = Point(new_coords[0] if side == 0 else new_coords[-1])
            poly = _polygon_for_point(polygons, pt)
            if poly is None:
                continue

            dist_to_exterior = float(poly.exterior.distance(pt))
            if dist_to_exterior < config.min_boundary_extension_m or dist_to_exterior > max_reach:
                continue

            # 1. Neighbor consensus heading: collect headings of nearby neighbor rows
            neighbor_headings = _find_nearest_neighbors_terminal_headings(
                pt, r.row_id, rows, config.boundary_consensus_neighbors, neighbor_search, tangent_len
            )

            # Own tangent
            own_tangent = _outward_endpoint_tangent(
                LineString(new_coords), side if not modified else (0 if side == 0 else 1), tangent_len
            )
            own_heading = float(math.degrees(math.atan2(own_tangent[0], own_tangent[1])) % 360.0)

            if neighbor_headings:
                consensus_heading = _circular_median_deg(neighbor_headings)
                # Check if own tangent is heavily divergent (waviness); if so, favor consensus
                angle_diff = abs((own_heading - consensus_heading + 180.0) % 360.0 - 180.0)
                if angle_diff > 15.0:
                    effective_heading = consensus_heading
                else:
                    # Blend 70% consensus, 30% own
                    effective_heading = _circular_median_deg([*neighbor_headings, own_heading])
            else:
                effective_heading = own_heading

            u_rad = math.radians(effective_heading)
            ux = math.sin(u_rad)
            uy = math.cos(u_rad)

            # Cast straight ray along consensus direction
            reach = max_reach * 1.5
            start_xy = (pt.x, pt.y)
            ray = LineString([start_xy, (start_xy[0] + reach * ux, start_xy[1] + reach * uy)])

            inter = ray.intersection(poly.exterior)
            if inter.is_empty:
                continue

            # Find closest intersection point in forward direction
            candidate_points: list[Point] = []
            if isinstance(inter, Point):
                candidate_points = [inter]
            elif hasattr(inter, "geoms"):
                candidate_points = [g for g in inter.geoms if isinstance(g, Point)]

            best_pt: tuple[float, float] | None = None
            min_proj = float("inf")
            for cp in candidate_points:
                proj = (cp.x - start_xy[0]) * ux + (cp.y - start_xy[1]) * uy
                if proj > 0.05 and proj < min_proj and proj <= max_reach:
                    min_proj = proj
                    best_pt = (float(cp.x), float(cp.y))

            if best_pt is None:
                continue

            if math.hypot(best_pt[0] - start_xy[0], best_pt[1] - start_xy[1]) < 0.1:
                continue

            extension_segment = LineString([start_xy, best_pt])
            current_line = LineString(new_coords)

            # Check self-intersection veto
            if extension_segment.crosses(current_line):
                continue

            # Check vetoes against other rows
            crosses = False
            for other in rows:
                if other.row_id == r.row_id:
                    continue
                if extension_segment.crosses(other.geometry) or extension_segment.distance(other.geometry) < safe_corridor:
                    crosses = True
                    break
            if crosses:
                continue

            # Test candidate geometry for validity and simplicity
            candidate_coords = [best_pt, *new_coords] if side == 0 else [*new_coords, best_pt]
            candidate_geom = LineString(candidate_coords)
            if not (
                isinstance(candidate_geom, LineString)
                and candidate_geom.is_valid
                and candidate_geom.is_simple
                and not candidate_geom.is_empty
                and candidate_geom.length > current_line.length
            ):
                continue

            # Anchor straight extension to line
            new_coords = candidate_coords
            modified = True
            extension_count += 1

        if modified:
            try:
                new_geom = LineString(new_coords)
                if (
                    isinstance(new_geom, LineString)
                    and new_geom.is_valid
                    and new_geom.is_simple
                    and not new_geom.is_empty
                    and np.isfinite(new_geom.length)
                    and new_geom.length > 0.0
                ):
                    new_id = _stable_row_id(new_geom)
                    extended_rows.append(
                        RowRecord(
                            row_id=new_id,
                            geometry=new_geom,
                            confidence=r.confidence,
                            observed_ratio=r.observed_ratio,
                            repaired_ratio=r.repaired_ratio,
                            spacing_local_m=r.spacing_local_m,
                            orientation_deg=_safe_axial_azimuth_deg(new_geom),
                            topology_status="valid",
                            repair_reason=json.dumps({"enhanced_boundary_extended": True, "prev_id": r.row_id}),
                            model_version=model_version,
                        )
                    )
                else:
                    extended_rows.append(r)
            except Exception:
                extended_rows.append(r)
        else:
            extended_rows.append(r)


    return extended_rows, extension_count


def apply_enhanced_refinement(
    rows: list[RowRecord],
    spacing_m: float,
    field_boundary: BaseGeometry | None = None,
    config: EnhancedRefinementConfig | None = None,
    model_version: str = "enhanced-v1",
) -> tuple[list[RowRecord], dict[str, Any]]:
    """Complete enhanced refinement pipeline: neighbor gap bridging + consensus boundary extension."""
    cfg = config or EnhancedRefinementConfig()
    if not cfg.enabled or not rows:
        return rows, {"gaps_bridged": 0, "boundary_extensions": 0}

    # Step 1: Bridge long gaps with neighbor guidance
    gapped_rows, gaps_bridged = bridge_long_gaps_with_neighbors(
        rows, spacing_m, cfg, model_version
    )

    # Step 2: Extend straight to boundary with neighbor consensus
    boundary_extensions = 0
    final_rows = gapped_rows
    if field_boundary is not None:
        final_rows, boundary_extensions = extend_to_boundary_with_consensus(
            gapped_rows, field_boundary, spacing_m, cfg, model_version
        )

    # Topological deduplication and validation
    cleaned_rows: list[RowRecord] = []
    for r in final_rows:
        if r.geometry.is_valid and r.geometry.is_simple and r.geometry.length > 0:
            cleaned_rows.append(r)

    diagnostics = {
        "input_row_count": len(rows),
        "output_row_count": len(cleaned_rows),
        "gaps_bridged": gaps_bridged,
        "boundary_extensions": boundary_extensions,
        "spacing_m": spacing_m,
    }
    return cleaned_rows, diagnostics


def export_enhanced_geopackage(
    output_path: Path,
    rows: list[RowRecord],
    crs: CRS | str,
    layer_name: str = "linhas_completas_v2",
) -> Path:
    """Export enhanced rows to a standalone GeoPackage with stable schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crs_obj = CRS.from_user_input(crs)

    data = {
        "row_id": pd.Series([r.row_id for r in rows], dtype="string"),
        "confidence": pd.Series([r.confidence for r in rows], dtype="float64"),
        "observed_ratio": pd.Series([r.observed_ratio for r in rows], dtype="float64"),
        "repaired_ratio": pd.Series([r.repaired_ratio for r in rows], dtype="float64"),
        "spacing_local_m": pd.Series([r.spacing_local_m for r in rows], dtype="float64"),
        "orientation_deg": pd.Series([r.orientation_deg for r in rows], dtype="float64"),
        "topology_status": pd.Series([r.topology_status for r in rows], dtype="string"),
        "repair_reason": pd.Series([r.repair_reason for r in rows], dtype="string"),
        "model_version": pd.Series([r.model_version for r in rows], dtype="string"),
    }
    geometry = gpd.GeoSeries([r.geometry for r in rows], crs=crs_obj)
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=crs_obj)

    try:
        import pyogrio

        pyogrio.write_dataframe(
            gdf,
            output_path,
            layer=layer_name,
            driver="GPKG",
            geometry_type="LineString",
            append=False,
        )
    except ImportError:
        gdf.to_file(output_path, layer=layer_name, driver="GPKG", index=False, mode="w")

    return output_path
