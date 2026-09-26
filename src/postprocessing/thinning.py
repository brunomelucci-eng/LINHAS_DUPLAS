"""Memory-conscious skeletonisation for large crop-row masks."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _find_endpoints_local(binary_strip: np.ndarray) -> np.ndarray:
    """Return endpoint coordinates (row, col) for a small binary strip."""
    kernel = np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        dtype=np.uint8,
    )
    foreground = (binary_strip != 0).astype(np.uint8, copy=False)
    neighbour_count = cv2.filter2D(
        foreground,
        ddepth=cv2.CV_8U,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    rows, cols = np.nonzero((foreground == 1) & (neighbour_count == 1))
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack((rows, cols)).astype(np.int32, copy=False)


def _connect_cross_boundary_pairs(
    skeleton: np.ndarray,
    endpoints_global: np.ndarray,
    boundary: int,
    vertical: bool,
    max_distance_px: float,
) -> int:
    """Connect only endpoint pairs that lie on opposite sides of one block seam."""
    if len(endpoints_global) < 2:
        return 0

    from scipy.spatial import cKDTree

    tree = cKDTree(endpoints_global.astype(np.float32, copy=False))
    pairs = tree.query_pairs(max_distance_px)
    connected = 0

    # Greedy one-to-one matching avoids joining one endpoint to several neighbours.
    used: set[int] = set()
    ranked_pairs = sorted(
        pairs,
        key=lambda pair: float(
            np.linalg.norm(endpoints_global[pair[0]] - endpoints_global[pair[1]])
        ),
    )

    for first_idx, second_idx in ranked_pairs:
        if first_idx in used or second_idx in used:
            continue

        first = endpoints_global[first_idx]
        second = endpoints_global[second_idx]

        if vertical:
            opposite_sides = (
                (first[1] < boundary <= second[1])
                or (second[1] < boundary <= first[1])
            )
        else:
            opposite_sides = (
                (first[0] < boundary <= second[0])
                or (second[0] < boundary <= first[0])
            )

        if not opposite_sides:
            continue

        cv2.line(
            skeleton,
            (int(first[1]), int(first[0])),
            (int(second[1]), int(second[0])),
            color=1,
            thickness=1,
            lineType=cv2.LINE_8,
        )
        used.add(first_idx)
        used.add(second_idx)
        connected += 1

    return connected


def _reconcile_thinning_seams(
    skeleton: np.ndarray,
    block_size: int,
    seam_half_width_px: int = 4,
    max_distance_px: float = 3.0,
) -> np.ndarray:
    """Heal block seams without scanning the complete raster.

    The previous implementation created multiple full-size arrays
    (foreground, neighbour count, endpoint mask and endpoint coordinates). For
    a 252 M-pixel raster this alone could exceed 1 GiB. This implementation
    inspects only narrow strips around each block boundary.
    """
    height, width = skeleton.shape
    if height < 3 or width < 3:
        return skeleton

    total_connected = 0

    # Vertical seams.
    for boundary_col in range(block_size, width, block_size):
        col_start = max(0, boundary_col - seam_half_width_px - 1)
        col_end = min(width, boundary_col + seam_half_width_px + 1)
        strip = skeleton[:, col_start:col_end]
        local_endpoints = _find_endpoints_local(strip)
        if local_endpoints.size == 0:
            continue
        local_endpoints[:, 1] += col_start
        total_connected += _connect_cross_boundary_pairs(
            skeleton,
            local_endpoints,
            boundary=boundary_col,
            vertical=True,
            max_distance_px=max_distance_px,
        )

    # Horizontal seams.
    for boundary_row in range(block_size, height, block_size):
        row_start = max(0, boundary_row - seam_half_width_px - 1)
        row_end = min(height, boundary_row + seam_half_width_px + 1)
        strip = skeleton[row_start:row_end, :]
        local_endpoints = _find_endpoints_local(strip)
        if local_endpoints.size == 0:
            continue
        local_endpoints[:, 0] += row_start
        total_connected += _connect_cross_boundary_pairs(
            skeleton,
            local_endpoints,
            boundary=boundary_row,
            vertical=False,
            max_distance_px=max_distance_px,
        )

    if total_connected:
        logger.info(
            "Thinning seam reconciliation connected %d endpoint pairs.",
            total_connected,
        )
    return skeleton


def perform_thinning_blocked(mask: np.ndarray, config: dict) -> np.ndarray:
    """Apply thinning block-by-block with overlap and bounded seam repair."""
    thin_cfg = config.get("postprocessing", {}).get("thinning", {})
    block_size = max(128, int(thin_cfg.get("block_size_px", 4096)))
    overlap = max(8, int(thin_cfg.get("block_overlap_px", 128)))

    height, width = mask.shape
    output = np.zeros((height, width), dtype=np.uint8)

    sub_config = dict(config)
    sub_post = dict(config.get("postprocessing", {}))
    sub_thin = dict(thin_cfg)
    sub_thin["block_processing_threshold_px"] = int(1e18)
    sub_post["thinning"] = sub_thin
    sub_config["postprocessing"] = sub_post

    logger.info(
        "Processing thinning in %dx%d blocks with %d px overlap.",
        block_size,
        block_size,
        overlap,
    )

    for row in range(0, height, block_size):
        row_start = max(0, row - overlap)
        row_end = min(height, row + block_size + overlap)
        core_row_start = row - row_start
        core_height = min(block_size, height - row)
        core_row_end = core_row_start + core_height

        for col in range(0, width, block_size):
            col_start = max(0, col - overlap)
            col_end = min(width, col + block_size + overlap)
            core_col_start = col - col_start
            core_width = min(block_size, width - col)
            core_col_end = core_col_start + core_width

            block = mask[row_start:row_end, col_start:col_end]
            thinned_block = perform_thinning(block, sub_config)
            output[row:row + core_height, col:col + core_width] = thinned_block[
                core_row_start:core_row_end,
                core_col_start:core_col_end,
            ]

    if thin_cfg.get("reconcile_seams", True):
        output = _reconcile_thinning_seams(
            output,
            block_size=block_size,
            seam_half_width_px=int(thin_cfg.get("seam_half_width_px", 4)),
            max_distance_px=float(thin_cfg.get("seam_max_distance_px", 3.0)),
        )

    return output


def perform_thinning(mask: np.ndarray, config: Optional[dict] = None) -> np.ndarray:
    """Skeletonise a binary mask using OpenCV, with a guarded fallback."""
    if mask.ndim != 2:
        raise ValueError(f"Thinning expects a 2-D mask, received shape={mask.shape}")

    backend = "opencv"
    allow_large_fallback = False
    block_threshold = 25_000_000

    if config is not None:
        thin_cfg = config.get("postprocessing", {}).get("thinning", {})
        backend = str(thin_cfg.get("backend", "opencv")).lower()
        allow_large_fallback = bool(
            thin_cfg.get("allow_large_skimage_fallback", False)
        )
        block_threshold = int(
            thin_cfg.get("block_processing_threshold_px", 25_000_000)
        )

        if mask.size > block_threshold:
            return perform_thinning_blocked(mask, config)

    if mask.dtype == np.uint8 and mask.max(initial=0) > 1:
        binary_mask = mask
    elif mask.dtype == np.uint8:
        binary_mask = mask * np.uint8(255)
    else:
        binary_mask = np.empty(mask.shape, dtype=np.uint8)
        np.greater(mask, 0, out=binary_mask)
        binary_mask *= np.uint8(255)

    if backend == "opencv":
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            thinned = cv2.ximgproc.thinning(
                binary_mask,
                thinningType=cv2.ximgproc.THINNING_GUOHALL,
            )
            return (thinned != 0).astype(np.uint8, copy=False)

        if mask.size > block_threshold and not allow_large_fallback:
            raise RuntimeError(
                "opencv-contrib-python with cv2.ximgproc.thinning is required "
                "for large rasters. Install opencv-contrib-python and remove "
                "the conflicting opencv-python package."
            )
        logger.warning("OpenCV ximgproc.thinning is unavailable; using scikit-image.")

    elif backend != "skimage":
        raise ValueError(
            f"Unsupported thinning backend={backend!r}. Use 'opencv' or 'skimage'."
        )

    if mask.size > block_threshold and not allow_large_fallback:
        raise RuntimeError(
            "scikit-image fallback is disabled for large rasters because it may "
            "exhaust RAM. Set allow_large_skimage_fallback=true only for testing."
        )

    try:
        from skimage.morphology import skeletonize
    except ImportError as exc:
        raise ImportError(
            "Neither OpenCV ximgproc thinning nor scikit-image skeletonize is available."
        ) from exc

    return skeletonize(binary_mask != 0).astype(np.uint8, copy=False)
