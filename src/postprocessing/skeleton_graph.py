"""
Memory-safe skeleton branch extraction.

The production path (`skeleton_to_graph_and_branches`) does not build a global
NetworkX graph. It scans the skeleton in row blocks, identifies endpoints and
junctions, and traces branches directly over the binary raster while storing
visited undirected edges in one uint8 value per raster pixel.

The legacy `skeleton_to_graph` and `extract_branches` functions are retained for
small unit tests and backwards compatibility only.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - project normally depends on OpenCV
    cv2 = None

logger = logging.getLogger(__name__)

Pixel = Tuple[int, int]
Branch = List[Pixel]

# Eight-connected neighbourhood.
_NEIGHBOURS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# Store every undirected edge once using the owner pixel and one of four bits.
# Canonical directions are E, SW, S, SE.
_EDGE_BITS = {
    (0, 1): np.uint8(1),
    (1, -1): np.uint8(2),
    (1, 0): np.uint8(4),
    (1, 1): np.uint8(8),
}


# ---------------------------------------------------------------------------
# Legacy API: safe only for small skeletons
# ---------------------------------------------------------------------------

def skeleton_to_graph(
    skeleton: np.ndarray,
    max_pixels: int = 1_000_000,
) -> nx.Graph:
    """Convert a small binary skeleton to a NetworkX graph.

    This function exists for backwards compatibility and unit tests. The
    production pipeline must use :func:`skeleton_to_graph_and_branches`.
    """
    _validate_skeleton(skeleton)
    pixel_count = int(np.count_nonzero(skeleton))
    if pixel_count > max_pixels:
        raise RuntimeError(
            f"Refusing to create a NetworkX graph for {pixel_count:,} pixels. "
            f"The compatibility limit is {max_pixels:,}. Use "
            "skeleton_to_graph_and_branches(), which uses the NumPy backend."
        )

    y_indices, x_indices = np.where(skeleton > 0)
    points: Set[Pixel] = set(zip(y_indices.tolist(), x_indices.tolist()))
    graph = nx.Graph()
    graph.add_nodes_from(points)

    for y, x in points:
        # Forward canonical directions avoid adding every edge twice.
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            neighbour = (y + dy, x + dx)
            if neighbour in points:
                graph.add_edge((y, x), neighbour)

    return graph


def extract_branches(
    graph: nx.Graph,
) -> Tuple[List[Branch], List[Pixel], List[Pixel]]:
    """Extract branches from a small NetworkX graph without copying it."""
    degrees = dict(graph.degree())
    junctions = [node for node, degree in degrees.items() if degree > 2]
    endpoints = [node for node, degree in degrees.items() if degree == 1]
    interest_nodes = set(junctions + endpoints)

    visited_edges: Set[Tuple[Pixel, Pixel]] = set()
    branches: List[Branch] = []

    for start in interest_nodes:
        for neighbour in graph.neighbors(start):
            edge = _sorted_edge(start, neighbour)
            if edge in visited_edges:
                continue

            path: Branch = [start, neighbour]
            visited_edges.add(edge)
            previous = start
            current = neighbour

            while current not in interest_nodes:
                candidates = [node for node in graph.neighbors(current) if node != previous]
                if not candidates:
                    break
                next_node = candidates[0]
                visited_edges.add(_sorted_edge(current, next_node))
                path.append(next_node)
                previous, current = current, next_node

            branches.append(path)

    # Closed components have no endpoints or junctions.
    for component in nx.connected_components(graph):
        component_edges = [
            edge for edge in graph.subgraph(component).edges()
            if _sorted_edge(edge[0], edge[1]) not in visited_edges
        ]
        if not component_edges:
            continue

        subgraph = nx.Graph()
        subgraph.add_edges_from(component_edges)
        for cycle in nx.cycle_basis(subgraph):
            if len(cycle) >= 3:
                branches.append(cycle + [cycle[0]])

    return branches, junctions, endpoints


# ---------------------------------------------------------------------------
# Production NumPy backend
# ---------------------------------------------------------------------------

def skeleton_to_graph_and_branches(
    skeleton: np.ndarray,
    config: dict,
) -> Tuple[List[Branch], List[Pixel], List[Pixel]]:
    """Extract skeleton branches without a global NetworkX graph.

    Memory profile of the production backend:

    * original skeleton: supplied by caller;
    * interest-node scan: bounded row blocks;
    * visited edges: one uint8 value per raster pixel, optionally a memmap;
    * Python objects: only endpoints, junctions and final branch coordinates.

    Parameters read from ``postprocessing.graph``:

    ``backend``
         ``numpy`` (default) or ``networkx``. NetworkX is guarded by a strict
         pixel limit and should only be used for tiny debugging cases.
    ``max_networkx_pixels``
         Maximum skeleton pixels accepted by the legacy backend.
    ``max_total_skeleton_pixels``
         Safety ceiling for a single production run. The pipeline fails with a
         clear message instead of freezing the machine.
    ``scan_block_rows``
         Number of raster rows used in each interest-node scan block.
    ``visited_memmap_threshold_px``
         Raster size above which the visited-edge array is disk-backed.
    ``temp_dir``
         Optional directory for the visited-edge memmap.
    ``progress_interval_edges``
         Log progress after this many traced edges.
    ``max_stage_minutes``
         Cooperative timeout for this stage. Set to zero to disable.
    """
    _validate_skeleton(skeleton)

    graph_cfg = config.get("postprocessing", {}).get("graph", {})
    backend = str(graph_cfg.get("backend", "numpy")).lower()
    max_networkx_pixels = int(graph_cfg.get("max_networkx_pixels", 50_000))

    skeleton_pixels = int(np.count_nonzero(skeleton))
    logger.info(
        "Skeleton branch extraction: backend=%s, raster=%dx%d, skeleton_pixels=%s",
        backend,
        skeleton.shape[1],
        skeleton.shape[0],
        f"{skeleton_pixels:,}",
    )

    if skeleton_pixels == 0:
        return [], [], []

    if backend == "networkx":
        graph = skeleton_to_graph(skeleton, max_pixels=max_networkx_pixels)
        return extract_branches(graph)

    if backend != "numpy":
        raise ValueError(
            f"Unsupported postprocessing.graph.backend={backend!r}. "
            "Use 'numpy' or 'networkx'."
        )

    max_total_pixels = int(graph_cfg.get("max_total_skeleton_pixels", 20_000_000))
    skeleton_fraction = skeleton_pixels / float(skeleton.size)
    max_skeleton_fraction = float(graph_cfg.get("max_skeleton_fraction", 0.0))
    if max_skeleton_fraction > 0 and skeleton_fraction > max_skeleton_fraction:
        raise RuntimeError(
            f"Skeleton occupies {skeleton_fraction:.2%} of the raster, above the "
            f"adaptive safety limit of {max_skeleton_fraction:.2%}. The center "
            "probability is likely saturated or its threshold is not calibrated."
        )
    if max_total_pixels > 0 and skeleton_pixels > max_total_pixels:
        raise RuntimeError(
            f"Skeleton contains {skeleton_pixels:,} pixels, above the configured "
            f"safety limit of {max_total_pixels:,}. This usually indicates an "
            "overly permissive threshold or a failed mask cleanup. Reduce the ROI, "
            "raise center thresholds, or increase the limit only after measuring RAM."
        )

    scan_block_rows = max(64, int(graph_cfg.get("scan_block_rows", 2048)))
    progress_interval = max(1, int(graph_cfg.get("progress_interval_edges", 250_000)))
    timeout_minutes = float(graph_cfg.get("max_stage_minutes", 15.0))
    deadline = time.monotonic() + timeout_minutes * 60.0 if timeout_minutes > 0 else None

    endpoints, junctions = _find_interest_nodes_blocked(
        skeleton,
        block_rows=scan_block_rows,
        deadline=deadline,
    )
    interest_nodes: Set[Pixel] = set(endpoints)
    interest_nodes.update(junctions)

    logger.info(
        "Skeleton topology scan: endpoints=%s, junction_pixels=%s",
        f"{len(endpoints):,}",
        f"{len(junctions):,}",
    )

    visited, visited_path = _create_visited_edges(skeleton.shape, graph_cfg)
    branches: List[Branch] = []
    traced_edges = 0
    next_progress = progress_interval

    try:
        # Trace every branch that starts at an endpoint or junction.
        for node_index, start in enumerate(interest_nodes):
            _check_deadline(deadline, "interest-node branch tracing")

            for neighbour in _iter_neighbours(skeleton, start[0], start[1]):
                if _edge_is_visited(visited, start, neighbour):
                    continue

                path, edge_count = _trace_path(
                    skeleton=skeleton,
                    start=start,
                    first=neighbour,
                    interest_nodes=interest_nodes,
                    visited=visited,
                    deadline=deadline,
                )
                traced_edges += edge_count
                if len(path) >= 2:
                    branches.append(path)

                if traced_edges >= next_progress:
                    logger.info(
                        "Skeleton tracing progress: edges=%s, branches=%s, interest=%s/%s",
                        f"{traced_edges:,}",
                        f"{len(branches):,}",
                        f"{node_index + 1:,}",
                        f"{len(interest_nodes):,}",
                    )
                    next_progress += progress_interval

        # Remaining unvisited edges belong mainly to closed cycles.
        cycle_branches, cycle_edges = _trace_remaining_cycles(
            skeleton=skeleton,
            visited=visited,
            block_rows=scan_block_rows,
            deadline=deadline,
            progress_interval=progress_interval,
        )
        branches.extend(cycle_branches)
        traced_edges += cycle_edges

        logger.info(
            "Skeleton branch extraction completed: branches=%s, cycles=%s, traced_edges=%s",
            f"{len(branches):,}",
            f"{len(cycle_branches):,}",
            f"{traced_edges:,}",
        )
        return branches, junctions, endpoints
    finally:
        _close_visited_edges(visited, visited_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_skeleton(skeleton: np.ndarray) -> None:
    if not isinstance(skeleton, np.ndarray):
        raise TypeError("skeleton must be a NumPy array")
    if skeleton.ndim != 2:
        raise ValueError(f"skeleton must be 2-D, received shape={skeleton.shape}")


def _sorted_edge(a: Pixel, b: Pixel) -> Tuple[Pixel, Pixel]:
    return (a, b) if a <= b else (b, a)


def _check_deadline(deadline: Optional[float], stage: str) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError(
            f"Skeleton graph stage exceeded its configured timeout during {stage}."
        )


def _find_interest_nodes_blocked(
    skeleton: np.ndarray,
    block_rows: int,
    deadline: Optional[float],
) -> Tuple[List[Pixel], List[Pixel]]:
    """Find degree-1 and degree>2 pixels using bounded row blocks."""
    height, width = skeleton.shape
    endpoints: List[Pixel] = []
    junctions: List[Pixel] = []

    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0

    for row_start in range(0, height, block_rows):
        _check_deadline(deadline, "topology scan")
        row_end = min(height, row_start + block_rows)
        ext_start = max(0, row_start - 1)
        ext_end = min(height, row_end + 1)

        foreground = (skeleton[ext_start:ext_end, :] != 0).astype(np.uint8, copy=False)

        if cv2 is not None:
            degrees_ext = cv2.filter2D(
                foreground,
                ddepth=cv2.CV_8U,
                kernel=kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
        else:  # pragma: no cover - fallback for unusual environments
            degrees_ext = _neighbour_count_numpy(foreground)

        core_start = row_start - ext_start
        core_end = core_start + (row_end - row_start)
        core_fg = foreground[core_start:core_end, :]
        core_degree = degrees_ext[core_start:core_end, :]

        endpoint_rows, endpoint_cols = np.nonzero((core_fg != 0) & (core_degree == 1))
        junction_rows, junction_cols = np.nonzero((core_fg != 0) & (core_degree > 2))

        endpoints.extend(
            (int(row_start + row), int(col))
            for row, col in zip(endpoint_rows.tolist(), endpoint_cols.tolist())
        )
        junctions.extend(
            (int(row_start + row), int(col))
            for row, col in zip(junction_rows.tolist(), junction_cols.tolist())
        )

    return endpoints, junctions


def _neighbour_count_numpy(foreground: np.ndarray) -> np.ndarray:
    padded = np.pad(foreground, 1, mode="constant")
    height, width = foreground.shape
    result = np.zeros((height, width), dtype=np.uint8)
    for dy, dx in _NEIGHBOURS:
        result += padded[
            1 + dy:1 + dy + height,
            1 + dx:1 + dx + width,
        ]
    return result


def _create_visited_edges(
    shape: Tuple[int, int],
    graph_cfg: dict,
) -> Tuple[np.ndarray, Optional[str]]:
    pixel_count = int(shape[0] * shape[1])
    threshold = int(graph_cfg.get("visited_memmap_threshold_px", 50_000_000))

    if pixel_count <= threshold:
        return np.zeros(shape, dtype=np.uint8), None

    configured_dir = graph_cfg.get("temp_dir")
    temp_dir = Path(configured_dir) if configured_dir else Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        prefix="skeleton_visited_",
        suffix=".dat",
        dir=str(temp_dir),
        delete=False,
    )
    path = handle.name
    handle.close()

    visited = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
    # Newly extended files are zero-filled by the OS. Avoid a full-array write.
    logger.info(
        "Visited-edge map uses memmap: %s (%.2f MiB)",
        path,
        pixel_count / 1024**2,
    )
    return visited, path


def _close_visited_edges(visited: np.ndarray, path: Optional[str]) -> None:
    try:
        if isinstance(visited, np.memmap):
            visited.flush()
            mmap_obj = getattr(visited, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
    finally:
        if path:
            try:
                os.remove(path)
            except OSError as exc:
                logger.warning("Could not remove skeleton visited memmap %s: %s", path, exc)


def _iter_neighbours(skeleton: np.ndarray, row: int, col: int) -> Iterable[Pixel]:
    height, width = skeleton.shape
    for d_row, d_col in _NEIGHBOURS:
        n_row = row + d_row
        n_col = col + d_col
        if (
            0 <= n_row < height
            and 0 <= n_col < width
            and skeleton[n_row, n_col] != 0
        ):
            yield (n_row, n_col)


def _edge_owner_and_bit(a: Pixel, b: Pixel) -> Tuple[int, int, np.uint8]:
    row_a, col_a = a
    row_b, col_b = b
    d_row = row_b - row_a
    d_col = col_b - col_a

    if abs(d_row) > 1 or abs(d_col) > 1 or (d_row == 0 and d_col == 0):
        raise ValueError(f"Pixels are not direct neighbours: {a} -> {b}")

    if d_row < 0 or (d_row == 0 and d_col < 0):
        row_a, col_a = row_b, col_b
        d_row = -d_row
        d_col = -d_col

    try:
        bit = _EDGE_BITS[(d_row, d_col)]
    except KeyError as exc:  # pragma: no cover - protected by validation above
        raise ValueError(f"Unsupported neighbour direction: {(d_row, d_col)}") from exc

    return row_a, col_a, bit


def _edge_is_visited(visited: np.ndarray, a: Pixel, b: Pixel) -> bool:
    row, col, bit = _edge_owner_and_bit(a, b)
    return bool(visited[row, col] & bit)


def _mark_edge_visited(visited: np.ndarray, a: Pixel, b: Pixel) -> None:
    row, col, bit = _edge_owner_and_bit(a, b)
    visited[row, col] = np.uint8(visited[row, col] | bit)


def _trace_path(
    skeleton: np.ndarray,
    start: Pixel,
    first: Pixel,
    interest_nodes: Set[Pixel],
    visited: np.ndarray,
    deadline: Optional[float],
) -> Tuple[Branch, int]:
    path: Branch = [start]
    previous = start
    current = first
    _mark_edge_visited(visited, previous, current)
    path.append(current)
    edge_count = 1

    while current not in interest_nodes:
        if edge_count % 10_000 == 0:
            _check_deadline(deadline, "branch tracing")

        next_node: Optional[Pixel] = None
        for candidate in _iter_neighbours(skeleton, current[0], current[1]):
            if candidate == previous:
                continue
            if not _edge_is_visited(visited, current, candidate):
                next_node = candidate
                break

        if next_node is None:
            break

        _mark_edge_visited(visited, current, next_node)
        path.append(next_node)
        edge_count += 1
        previous, current = current, next_node

    return path, edge_count


def _trace_remaining_cycles(
    skeleton: np.ndarray,
    visited: np.ndarray,
    block_rows: int,
    deadline: Optional[float],
    progress_interval: int,
) -> Tuple[List[Branch], int]:
    """Trace unvisited edges, which are usually closed loops."""
    height, width = skeleton.shape
    branches: List[Branch] = []
    traced_edges = 0
    next_progress = progress_interval

    for row_start in range(0, height, block_rows):
        _check_deadline(deadline, "cycle scan")
        row_end = min(height, row_start + block_rows)
        rows, cols = np.nonzero(skeleton[row_start:row_end, :] != 0)

        for local_row, col in zip(rows.tolist(), cols.tolist()):
            row = int(row_start + local_row)
            col = int(col)
            start = (row, col)

            # Only canonical forward neighbours enumerate each edge once.
            for d_row, d_col in ((0, 1), (1, -1), (1, 0), (1, 1)):
                neighbour = (row + d_row, col + d_col)
                if not (
                    0 <= neighbour[0] < height
                    and 0 <= neighbour[1] < width
                    and skeleton[neighbour[0], neighbour[1]] != 0
                ):
                    continue
                if _edge_is_visited(visited, start, neighbour):
                    continue

                path, edge_count = _trace_cycle(
                    skeleton=skeleton,
                    start=start,
                    first=neighbour,
                    visited=visited,
                    deadline=deadline,
                )
                traced_edges += edge_count
                if len(path) >= 3:
                    branches.append(path)

                if traced_edges >= next_progress:
                    logger.info(
                        "Skeleton cycle scan progress: edges=%s, cycles=%s",
                        f"{traced_edges:,}",
                        f"{len(branches):,}",
                    )
                    next_progress += progress_interval

    return branches, traced_edges


def _trace_cycle(
    skeleton: np.ndarray,
    start: Pixel,
    first: Pixel,
    visited: np.ndarray,
    deadline: Optional[float],
) -> Tuple[Branch, int]:
    path: Branch = [start]
    previous = start
    current = first
    _mark_edge_visited(visited, previous, current)
    path.append(current)
    edge_count = 1

    while current != start:
        if edge_count % 10_000 == 0:
            _check_deadline(deadline, "cycle tracing")

        next_node: Optional[Pixel] = None
        for candidate in _iter_neighbours(skeleton, current[0], current[1]):
            if candidate == previous:
                continue
            if not _edge_is_visited(visited, current, candidate):
                next_node = candidate
                break

        if next_node is None:
            break

        _mark_edge_visited(visited, current, next_node)
        path.append(next_node)
        edge_count += 1
        previous, current = current, next_node

    return path, edge_count
