import cv2
import numpy as np
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def estimate_hysteresis_memory_bytes(pixel_count: int) -> int:
    """
    Conservative memory estimate for global hysteresis thresholding:
    - prob map (float32, 4 bytes)
    - strong mask (bool/uint8, 1 byte)
    - candidate mask (uint8, 1 byte)
    - labels array (int32, 4 bytes)
    - output mask (uint8, 1 byte)
    - internal workspace margin (~1 byte)
    Total: ~12 bytes per pixel.
    """
    return int(pixel_count * 12)


def _hysteresis_threshold(prob_map: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Hysteresis thresholding using a vectorised OpenCV lookup table.

    Strong seeds  : prob_map >= high  → always kept
    Candidates    : prob_map >= low   → kept only if connected to a seed
    Connectivity  : 8-connected (preserves diagonal runs)

    Complexity: O(pixels) regardless of the number of labels.
    The previous scipy.ndimage implementation iterated once per label over
    the full array, causing runtimes of up to 63 minutes on 252 M-pixel images.
    """
    strong    = (prob_map >= high).astype(np.uint8)
    candidate = (prob_map >= low).astype(np.uint8)

    # Label all candidate regions (8-connectivity, 32-bit labels for large images)
    num_labels, labels = cv2.connectedComponents(
        candidate, connectivity=8, ltype=cv2.CV_32S
    )

    if num_labels <= 1:          # only background
        return strong

    # Build a lookup table: label → keep (1) or discard (0)
    # A region is kept if ANY of its pixels is also a strong seed.
    seed_label_indices = np.unique(labels[strong > 0])
    keep_lut = np.zeros(num_labels, dtype=np.uint8)
    keep_lut[seed_label_indices] = 1
    keep_lut[0] = 0              # background always 0

    result = keep_lut[labels]    # single vectorised index operation
    return result


def _clean_mask_blocked(mask: np.ndarray, gsd: float, config: dict) -> np.ndarray:
    """
    Clean the mask block-by-block with overlap (halo) to prevent memory exhaustion (OOM)
    on large orthomosaics. (BUG P0-12)
    """
    post_cfg = config.get('postprocessing', {})
    block_size = post_cfg.get('block_size_px', 4096)
    overlap = post_cfg.get('block_overlap_px', 128)

    H, W = mask.shape
    output = np.zeros((H, W), dtype=np.uint8)

    # Create a config copy with block threshold disabled to prevent recursion.
    sub_config = config.copy()
    if 'postprocessing' in sub_config:
        sub_config['postprocessing'] = config['postprocessing'].copy()
        sub_config['postprocessing']['block_processing_threshold_px'] = 999999999999

    logger.info(
        f"Processing clean_mask in blocks of {block_size}x{block_size} (overlap={overlap}). "
        f"Global component labeling is disabled for memory efficiency (BUG P0-12)."
    )

    for r in range(0, H, block_size):
        r_start = max(0, r - overlap)
        r_end = min(H, r + block_size + overlap)
        r_core_start = r - r_start
        r_core_end = r_core_start + min(block_size, H - r)

        for c in range(0, W, block_size):
            c_start = max(0, c - overlap)
            c_end = min(W, c + block_size + overlap)
            c_core_start = c - c_start
            c_core_end = c_core_start + min(block_size, W - c)

            block_mask = mask[r_start:r_end, c_start:c_end]

            # Clean block (binarisation, morphology close/open, local area filter)
            cleaned_block = clean_mask(block_mask, gsd, sub_config)

            # Extract non-overlapping core and assign to global output
            output[r:r + min(block_size, H - r), c:c + min(block_size, W - c)] = \
                cleaned_block[r_core_start:r_core_end, c_core_start:c_core_end]

    logger.info("Block-wise clean_mask processing completed without global labeling.")
    return output


def _directional_close(mask: np.ndarray, gsd: float, settings: dict) -> np.ndarray:
    """Close only along the dominant row direction, never with a broad disk."""
    points = np.column_stack(np.nonzero(mask))
    if len(points) < 2:
        return mask
    # PCA supplies one global longitudinal direction; agricultural rows in one
    # ROI share it, and the narrow kernel prevents lateral row-to-row bridges.
    _, _, vectors = np.linalg.svd(points.astype(np.float32) - points.mean(0), full_matrices=False)
    direction = vectors[0]  # (row, col)
    max_gap_px = max(1, int(round(float(settings.get('max_gap_m', 1.2)) / gsd)))
    lateral_px = max(1, int(round(float(settings.get('lateral_width_m', 0.08)) / gsd)))
    size = max_gap_px * 2 + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = max_gap_px
    end = np.rint(direction * max_gap_px).astype(int)
    cv2.line(kernel, (center - end[1], center - end[0]), (center + end[1], center + end[0]), 1, thickness=lateral_px)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def clean_mask(
    mask: np.ndarray,
    gsd: float,
    config: dict,
) -> np.ndarray:
    """
    Clean a predicted binary (or probability) mask.

    Parameters
    ----------
    mask   : float32 probability map *or* uint8 binary mask
    gsd    : ground sampling distance in metres per pixel
    config : full pipeline configuration dict
    """
    post_cfg = config.get('postprocessing', {})

    # Check if block processing is needed based on size or available RAM
    block_threshold = post_cfg.get('block_processing_threshold_px', 25000000)
    use_blocking = False
    
    if mask.size > block_threshold:
        use_blocking = True
    else:
        try:
            import psutil
            required_mem = estimate_hysteresis_memory_bytes(mask.size)
            available_mem = psutil.virtual_memory().available
            if required_mem > available_mem * 0.55:
                logger.warning(
                    f"Estimated hysteresis memory ({required_mem / 1024**2:.1f} MB) "
                    f"exceeds 55% of available RAM ({available_mem / 1024**2:.1f} MB). "
                    "Enabling block processing."
                )
                use_blocking = True
        except Exception:
            pass

    if use_blocking and mask.ndim == 2:
        return _clean_mask_blocked(mask, gsd, config)

    # ------------------------------------------------------------------
    # 1. Binarise
    # ------------------------------------------------------------------
    if mask.dtype != np.uint8:
        threshold_mode = str(post_cfg.get('threshold_mode', 'hysteresis')).lower()
        threshold_high = post_cfg.get('center_threshold_high', 0.28)
        threshold_low  = post_cfg.get('center_threshold_low',  0.12)
        if threshold_mode == 'single':
            single_threshold = float(
                post_cfg.get('center_threshold', threshold_high)
            )
            bin_mask = (mask >= single_threshold).astype(np.uint8)
        elif threshold_mode == 'hysteresis' and threshold_low < threshold_high:
            bin_mask = _hysteresis_threshold(mask, low=threshold_low, high=threshold_high)
        elif threshold_mode not in {'single', 'hysteresis'}:
            raise ValueError(
                "postprocessing.threshold_mode must be 'single' or 'hysteresis'."
            )
        else:
            logger.warning(
                "center_threshold_low (%.2f) >= center_threshold_high (%.2f). "
                "Falling back to single threshold.",
                threshold_low, threshold_high,
            )
            bin_mask = (mask >= threshold_high).astype(np.uint8)
    else:
        bin_mask = mask.copy()

    # ------------------------------------------------------------------
    # 2. Pixel-space radii  (allow zero → skip operation)
    # ------------------------------------------------------------------
    # Conservative baseline: morphology is opt-in because a disk closing can
    # connect adjacent rows before skeletonisation.
    closing_radius_px = max(0, int(round(post_cfg.get('closing_radius_m', 0.00) / gsd)))
    opening_radius_px = max(0, int(round(post_cfg.get('opening_radius_m', 0.00) / gsd)))

    cleaned = bin_mask

    # ------------------------------------------------------------------
    # 3. Morphological closing
    # ------------------------------------------------------------------
    if closing_radius_px > 0:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * closing_radius_px + 1, 2 * closing_radius_px + 1),
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    directional_cfg = post_cfg.get('directional_closing', {})
    if directional_cfg.get('enabled', False):
        passes = directional_cfg.get('passes_m', [directional_cfg.get('max_gap_m', 1.2)])
        for max_gap_m in passes:
            settings = dict(directional_cfg, max_gap_m=max_gap_m)
            cleaned = _directional_close(cleaned, gsd, settings)

    # ------------------------------------------------------------------
    # 4. Morphological opening  (disabled when opening_radius_m == 0)
    # ------------------------------------------------------------------
    if opening_radius_px > 0:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * opening_radius_px + 1, 2 * opening_radius_px + 1),
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open)

    # ------------------------------------------------------------------
    # 5. Remove small connected components (BUG P0-02 fix)
    # ------------------------------------------------------------------
    if 'min_component_area_m2' in post_cfg:
        min_area_m2 = post_cfg['min_component_area_m2']
        min_area_px = max(1, int(round(min_area_m2 / (gsd * gsd))))
    elif 'min_component_length_m' in post_cfg:
        legacy_len_px = max(1, int(round(post_cfg['min_component_length_m'] / gsd)))
        min_area_px   = max(1, legacy_len_px)
        logger.warning(
            "DEPRECATED: 'min_component_length_m' used for area filtering is ambiguous "
            "(CC_STAT_AREA is in pixels², not pixels). "
            "Switch to 'min_component_area_m2' in postprocessing config. "
            "Current fallback: min_area_px=%d.", min_area_px,
        )
    else:
        min_area_px = max(1, int(round(0.02 / (gsd * gsd))))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    label_areas = stats[:, cv2.CC_STAT_AREA]
    keep_lut    = np.zeros(num_labels, dtype=np.uint8)
    keep_lut[label_areas >= min_area_px] = 1
    
    # Preserve border-touching components to prevent seam cuts during block processing (BUG P0-12)
    if num_labels > 1:
        border_pixels = np.concatenate([
            labels[0, :], labels[-1, :],
            labels[:, 0], labels[:, -1]
        ])
        border_labels = np.unique(border_pixels)
        for bl in border_labels:
            keep_lut[bl] = 1

    keep_lut[0] = 0   # background

    output = keep_lut[labels]
    logger.debug(
        "clean_mask: %d components, %d kept (min_area_px=%d).",
        num_labels - 1, int(keep_lut[1:].sum()), min_area_px,
    )
    return output
