"""
gap_bridging.py — Bridge gaps between disconnected crop-row branches.

Bug fixes vs previous version
------------------------------
* P0-06a : Replaced unlimited `query_pairs(max_gap_m)` with per-endpoint
           k-nearest-neighbour search (`tree.query(k=max_candidates_per_endpoint+1)`).
           With 11 968 branches the old approach took ~11 minutes; the new one
           runs in seconds because the total number of pairs considered is
           bounded by  N_endpoints × max_candidates_per_endpoint.

* P0-06b : Corridor probability is now sampled in a single vectorised NumPy
           operation instead of a Python for-loop (significant speedup for
           large rasters with many pairs).

* P0-06c : Hard limit on total candidate pairs (`max_candidate_pairs`) prevents
           pathological cases.

* P0-06d : Progress logging every `progress_log_interval` endpoints so the
           user can see the process is alive.

* P0-06e : Stage timeout (`max_stage_minutes`) aborts gracefully with a warning
           instead of hanging indefinitely.

* Endpoint-bug fix : connection is stored on the CORRECT side of each endpoint
  (`start_i`/`start_j`) instead of the inverted `not start_i`/`not start_j`.
"""

import time
import numpy as np
import rasterio
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.strtree import STRtree
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def _tree_indices(tree: STRtree, geometry, geometries):
    """Yield indices from either the Shapely 1.x or 2.x STRtree API."""
    by_id = {id(item): idx for idx, item in enumerate(geometries)}
    for item in tree.query(geometry):
        if hasattr(item, "item") and not hasattr(item, "geom_type"):
            yield int(item.item())
        elif isinstance(item, int):
            yield item
        else:
            idx = by_id.get(id(item))
            if idx is not None:
                yield idx


# ---------------------------------------------------------------------------
# Helper: tangent vector at branch endpoint
# ---------------------------------------------------------------------------

def get_endpoint_tangent(
    branch: List[Tuple[int, int]],
    at_start: bool,
    transform: rasterio.Affine,
) -> np.ndarray:
    """
    Return the unit tangent vector at one endpoint of a branch (world coords).
    The vector points **outward** from the line end.
    """
    n    = len(branch)
    step = min(n - 1, 5)
    if step < 1:
        return np.array([1.0, 0.0])

    if at_start:
        p_end  = branch[0]
        p_prev = branch[step]
    else:
        p_end  = branch[-1]
        p_prev = branch[-1 - step]

    wx_end,  wy_end  = rasterio.transform.xy(transform, p_end[0],  p_end[1])
    wx_prev, wy_prev = rasterio.transform.xy(transform, p_prev[0], p_prev[1])

    v    = np.array([wx_end - wx_prev, wy_end - wy_prev])
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


# ---------------------------------------------------------------------------
# Helper: vectorised corridor probability sampling
# ---------------------------------------------------------------------------

def sample_corridor_probability_batch(
    p1_world: Tuple[float, float],
    p2_world: Tuple[float, float],
    center_prob: np.ndarray,
    transform: rasterio.Affine,
    n_samples: int = 16,
) -> float:
    """
    Sample mean centerline probability along the segment p1 → p2.

    All (row, col) coordinates are computed in one NumPy operation instead of
    calling rasterio.transform.rowcol() inside a Python loop.
    """
    v    = np.array([p2_world[0] - p1_world[0], p2_world[1] - p1_world[1]])
    dist = float(np.linalg.norm(v))

    if dist < 1e-3:
        r, c = rasterio.transform.rowcol(transform, p1_world[0], p1_world[1])
        r = int(np.clip(r, 0, center_prob.shape[0] - 1))
        c = int(np.clip(c, 0, center_prob.shape[1] - 1))
        return float(center_prob[r, c])

    n   = max(2, n_samples)
    ts  = np.linspace(0.0, 1.0, n)
    wxs = p1_world[0] + ts * v[0]
    wys = p1_world[1] + ts * v[1]

    # Inverse affine: (col, row) = inv_transform * (x, y)
    inv = ~transform
    px  = inv.a * wxs + inv.b * wys + inv.c   # col (x direction)
    py  = inv.d * wxs + inv.e * wys + inv.f   # row (y direction)

    rows = np.clip(py.astype(np.int32), 0, center_prob.shape[0] - 1)
    cols = np.clip(px.astype(np.int32), 0, center_prob.shape[1] - 1)

    return float(center_prob[rows, cols].mean())


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def bridge_gaps(
    branches: List[List[Tuple[int, int]]],
    center_prob: np.ndarray,
    transform: rasterio.Affine,
    gsd: float,
    config: dict,
) -> List[List[Tuple[int, int]]]:
    """
    Bridge gaps between endpoints of crop-row branches.

    Uses k-nearest-neighbour search with hard limits to avoid the O(N²)
    explosion that occurred with unlimited `query_pairs`.
    """
    gap_cfg = config.get('gap_bridging', {})
    if not gap_cfg.get('enabled', True) or not branches:
        return branches

    # Orientation/double-row probability maps are not part of this function's API.
    for key in ("use_orientation_map", "use_double_row_context"):
        if gap_cfg.get(key, False):
            logger.warning(
                f"gap_bridging.{key} is set to true in configurations but is not yet implemented. "
                "Ignoring for this run."
            )

    max_gap_m          = gap_cfg.get('max_gap_m',               4.0)
    max_angle_deg      = gap_cfg.get('max_angle_deg', gap_cfg.get('max_angle_difference_deg', 25.0))
    max_offset_m       = gap_cfg.get('max_lateral_offset_m',     0.25)
    min_prob           = gap_cfg.get('min_corridor_probability',  0.12)
    k_neighbours       = gap_cfg.get('max_candidates_per_endpoint', 8)
    max_total_pairs    = gap_cfg.get('max_candidate_pairs',     150_000)
    n_corridor_samples = gap_cfg.get('corridor_samples',           16)
    log_interval       = gap_cfg.get('progress_log_interval',    1_000)
    max_minutes        = gap_cfg.get('max_stage_minutes',            8)
    timeout_policy     = gap_cfg.get('timeout_policy',           'fail')
    require_facing     = gap_cfg.get('require_facing_endpoints', True)
    reject_crossings   = gap_cfg.get(
        'reject_crossing_connections', gap_cfg.get('forbid_crossings', True)
    )
    reject_neighbors   = gap_cfg.get('reject_neighbor_row_connections', True)
    reject_degree      = gap_cfg.get('reject_degree_gt_2', True)
    max_per_endpoint   = int(gap_cfg.get('max_connections_per_endpoint', 1))

    cos_thresh = np.cos(np.radians(max_angle_deg))
    t_start    = time.monotonic()
    deadline   = t_start + max_minutes * 60.0

    # ------------------------------------------------------------------
    # 1. Extract endpoints
    # ------------------------------------------------------------------
    endpoints: List[Tuple[int, bool, Tuple[float, float], np.ndarray]] = []
    for idx, branch in enumerate(branches):
        if len(branch) < 2:
            continue
        w_s = rasterio.transform.xy(transform, branch[0][0],  branch[0][1])
        w_e = rasterio.transform.xy(transform, branch[-1][0], branch[-1][1])
        t_s = get_endpoint_tangent(branch, at_start=True,  transform=transform)
        t_e = get_endpoint_tangent(branch, at_start=False, transform=transform)
        endpoints.append((idx, True,  w_s, t_s))
        endpoints.append((idx, False, w_e, t_e))

    n_ep = len(endpoints)
    if n_ep == 0:
        return branches

    coords = np.array([ep[2] for ep in endpoints], dtype=np.float64)
    tree   = cKDTree(coords)

    branch_lines = []
    for branch in branches:
        rows = [point[0] for point in branch]
        cols = [point[1] for point in branch]
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        branch_lines.append(LineString(zip(xs, ys)))
    branch_tree = STRtree(branch_lines) if reject_crossings else None

    # ------------------------------------------------------------------
    # 2. Build candidate list using k-NN instead of all-pairs query
    # ------------------------------------------------------------------
    candidates = []
    seen_pairs = set()  # prevent duplicate connections (BUG P1-06)
    k_query = min(k_neighbours + 1, n_ep)
    # Query all endpoints in one compiled SciPy call. Rows remain in endpoint
    # order and neighbours remain distance-sorted, so candidate ordering and
    # every downstream acceptance criterion are unchanged.
    all_dists_knn, all_indices_knn = tree.query(coords, k=k_query)
    if k_query == 1:
        all_dists_knn = all_dists_knn[:, None]
        all_indices_knn = all_indices_knn[:, None]

    for i in range(n_ep):
        if time.monotonic() > deadline:
            # Handle timeout policy (BUG P1-08)
            msg = (
                f"gap_bridging: stage timeout after {max_minutes:.1f} minutes. "
                f"Processed {i}/{n_ep} endpoints."
            )
            if timeout_policy == 'fail':
                raise RuntimeError(f"{msg} Aborting execution as per timeout_policy='fail'.")
            else:
                logger.warning(f"{msg} Continuing with partial results.")
                break

        if i % log_interval == 0 and i > 0:
            elapsed = time.monotonic() - t_start
            logger.info(
                "gap_bridging progress: %d/%d endpoints, %d candidates, %.1fs elapsed.",
                i, n_ep, len(candidates), elapsed,
            )

        idx_i, is_start_i, w_i, t_i = endpoints[i]

        for dist, j in zip(all_dists_knn[i, 1:], all_indices_knn[i, 1:]):
            # First neighbour is the point itself (j==i, dist==0).
            if dist > max_gap_m:
                break   # results are sorted; no point continuing

            # Prevent duplicate pairs (BUG P1-06)
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            idx_j, is_start_j, w_j, t_j = endpoints[j]

            if idx_i == idx_j:
                continue   # same branch

            gap_vec  = np.array([w_j[0] - w_i[0], w_j[1] - w_i[1]])
            dist_m   = float(np.linalg.norm(gap_vec))
            if dist_m < 1e-3:
                continue

            gap_unit = gap_vec / dist_m

            # Angular alignment and facing endpoints. Outward tangents must aim
            # toward each other; this rejects side-to-side neighbouring rows.
            dot_i = float(np.dot(t_i,  gap_unit))
            dot_j = float(np.dot(t_j, -gap_unit))
            if require_facing and (dot_i < cos_thresh or dot_j < cos_thresh):
                continue
            if reject_neighbors and float(np.dot(t_i, -t_j)) < cos_thresh:
                continue

            # Lateral offset
            n_i      = np.array([-t_i[1], t_i[0]])
            n_j      = np.array([-t_j[1], t_j[0]])
            offset_i = abs(float(np.dot(gap_vec, n_i)))
            offset_j = abs(float(np.dot(gap_vec, n_j)))
            if offset_i > max_offset_m or offset_j > max_offset_m:
                continue

            connection_geometry = LineString([w_i, w_j])
            if branch_tree is not None:
                crosses_network = False
                for branch_idx in _tree_indices(branch_tree, connection_geometry, branch_lines):
                    if branch_idx in (idx_i, idx_j):
                        continue
                    if connection_geometry.intersects(branch_lines[branch_idx]):
                        crosses_network = True
                        break
                if crosses_network:
                    continue

            # Corridor probability (vectorised)
            mean_p = sample_corridor_probability_batch(
                w_i, w_j, center_prob, transform, n_samples=n_corridor_samples
            )
            if mean_p < min_prob:
                continue

            score = dot_i + dot_j - (offset_i + offset_j) / max(max_offset_m, 1e-6) + mean_p
            candidates.append({
                'score':      score,
                'endpoint_i': i,
                'endpoint_j': j,
                'branch_i':   idx_i,
                'branch_j':   idx_j,
                'is_start_i': is_start_i,
                'is_start_j': is_start_j,
                'geometry':   connection_geometry,
            })

    elapsed = time.monotonic() - t_start
    logger.info(
        "gap_bridging: built %d raw candidates from %d endpoints in %.1fs.",
        len(candidates), n_ep, elapsed,
    )

    # --- Sort and apply max limit (BUG P1-07) -------------------------------
    candidates.sort(key=lambda x: x['score'], reverse=True)
    if len(candidates) > max_total_pairs:
        logger.warning(
            f"gap_bridging: candidates count ({len(candidates)}) exceeded max_candidate_pairs ({max_total_pairs}). "
            f"Truncating to keep the top candidates."
        )
        candidates = candidates[:max_total_pairs]

    # ------------------------------------------------------------------
    # 3. Greedy selection
    # ------------------------------------------------------------------
    candidates.sort(key=lambda x: x['score'], reverse=True)

    endpoint_connections: Dict[int, int] = {}
    accepted_geometries = []
    parent = list(range(len(branches)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[ry] = rx
        return True

    connections: List[Tuple[int, int, bool, bool]] = []

    for cand in candidates:
        ep_i, ep_j = cand['endpoint_i'], cand['endpoint_j']
        if endpoint_connections.get(ep_i, 0) >= max_per_endpoint or endpoint_connections.get(ep_j, 0) >= max_per_endpoint:
            continue
        bi, bj = cand['branch_i'], cand['branch_j']
        if find(bi) == find(bj):
            continue   # would create a cycle
        if reject_crossings and any(cand['geometry'].crosses(other) for other in accepted_geometries):
            continue
        if reject_degree and max_per_endpoint > 1:
            # Crop-row paths are chains: accepting more than one edge at an
            # endpoint would create a degree > 2 node.
            if endpoint_connections.get(ep_i, 0) or endpoint_connections.get(ep_j, 0):
                continue
        union(bi, bj)
        endpoint_connections[ep_i] = endpoint_connections.get(ep_i, 0) + 1
        endpoint_connections[ep_j] = endpoint_connections.get(ep_j, 0) + 1
        accepted_geometries.append(cand['geometry'])
        connections.append((bi, bj, cand['is_start_i'], cand['is_start_j']))

    # ------------------------------------------------------------------
    # 4. Reassemble merged branches
    # ------------------------------------------------------------------
    branch_graph: Dict[int, Dict] = {
        i: {'pixels': branches[i].copy(), 'neighbors': {}}
        for i in range(len(branches))
    }

    for bi, bj, start_i, start_j in connections:
        # Store connection on the CORRECT side (fix for previous not-start bug)
        branch_graph[bi]['neighbors'][start_i] = (bj, start_j)
        branch_graph[bj]['neighbors'][start_j] = (bi, start_i)

    visited_branches: set = set()
    merged_branches:  List[List[Tuple[int, int]]] = []

    for i in range(len(branches)):
        if i in visited_branches:
            continue

        visited_branches.add(i)
        start_nbrs = branch_graph[i]['neighbors']

        # Trace backwards (True side = start of branch i)
        left_pixels: List[Tuple[int, int]] = []
        next_conn = start_nbrs.get(True)
        while next_conn is not None:
            nb, nb_side = next_conn
            if nb in visited_branches:
                break
            visited_branches.add(nb)
            pix = branch_graph[nb]['pixels']
            # If we enter from the end (False) we keep pixel order; from start → reverse
            left_pixels = (pix if not nb_side else list(reversed(pix))) + left_pixels
            next_conn = branch_graph[nb]['neighbors'].get(not nb_side)

        # Trace forwards (False side = end of branch i)
        right_pixels: List[Tuple[int, int]] = branch_graph[i]['pixels'].copy()
        next_conn = start_nbrs.get(False)
        while next_conn is not None:
            nb, nb_side = next_conn
            if nb in visited_branches:
                break
            visited_branches.add(nb)
            pix = branch_graph[nb]['pixels']
            # Enter from start (True) → keep order; from end (False) → reverse
            right_pixels = right_pixels + (pix if nb_side else list(reversed(pix)))
            next_conn = branch_graph[nb]['neighbors'].get(not nb_side)

        full_pixels = left_pixels + right_pixels
        merged_branches.append(full_pixels)

    logger.info(
        "gap_bridging: accepted %d connections, merged %d branches -> %d lines (%.1fs total).",
        len(connections), len(branches), len(merged_branches),
        time.monotonic() - t_start,
    )
    return merged_branches
