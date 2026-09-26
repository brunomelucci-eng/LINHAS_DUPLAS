"""
deduplication.py — Remove redundant/duplicate crop-row lines.

Bug fixes vs previous version
------------------------------
* P0-07 : Previous implementation compared every line against every already-kept
           line (O(N²)). With 7 990 lines this took ~27 minutes.
           Now uses a Shapely STRtree spatial index so only geometrically close
           candidates are evaluated with the expensive distance/angle/overlap tests.

* Decoupled : The deduplication threshold is no longer read from
              `gap_bridging.max_lateral_offset_m`.  It uses its own independent
              section `deduplication:` in the config, so tuning gap tolerance
              no longer silently changes which lines are erased.
"""

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, Point
from typing import List
import logging

logger = logging.getLogger(__name__)


def _bearing_pca(line: LineString, n_samples: int = 20) -> float:
    """
    Estimate the principal direction of a (possibly curved) line using PCA of
    sampled points.  Returns angle in [0°, 180°).
    """
    coords = np.array(line.coords)
    if len(coords) < 2:
        return 0.0

    # Uniform sampling along the line
    n = min(n_samples, len(coords))
    idx = np.linspace(0, len(coords) - 1, n, dtype=int)
    pts = coords[idx]

    # PCA: first principal component = dominant direction
    pts_c = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    angle = float(np.degrees(np.arctan2(Vt[0, 1], Vt[0, 0]))) % 180.0
    return angle


def _longitudinal_overlap(short: LineString, long_: LineString) -> float:
    """
    Fraction of `short` that overlaps longitudinally with `long_`.
    Computed by projecting the start/end of `short` onto `long_` and
    measuring the projected span relative to `short.length`.
    """
    if short.length < 1e-6:
        return 0.0
    proj_s = long_.project(Point(list(short.coords)[0]))
    proj_e = long_.project(Point(list(short.coords)[-1]))
    overlap = abs(proj_e - proj_s)
    return overlap / short.length


def remove_duplicate_lines(
    lines: List[LineString],
    config: dict,
) -> List[LineString]:
    """
    Remove duplicate or parallel-redundant lines using an STRtree spatial index.

    A line is only considered a duplicate if ALL three conditions are met:
        1. Mean sampled distance to an already-kept line < max_mean_distance_m
        2. Angular difference (PCA-based)          < max_angle_difference_deg
        3. Longitudinal overlap ratio              >= min_overlap_ratio

    This is intentionally strict to avoid erasing one of the two lines in a
    double row (intra-pair distance 0.85–1.00 m; dedup threshold 0.20 m keeps
    them well separated).
    """
    if len(lines) < 2:
        return lines

    dup_cfg = config.get('deduplication', {})
    if not dup_cfg.get('enabled', True):
        return lines

    max_mean_dist        = dup_cfg.get('max_mean_distance_m',      0.20)
    max_angle_deg        = dup_cfg.get('max_angle_difference_deg',  5.0)
    min_overlap          = dup_cfg.get('min_overlap_ratio',         0.70)
    max_cands_per_line   = dup_cfg.get('max_candidates_per_line',   30)
    use_spatial_index    = dup_cfg.get('spatial_index',             True)

    # Sort descending by length — keep the longer line when duplicates exist
    order        = np.argsort([line.length for line in lines])[::-1]
    sorted_lines = [lines[i] for i in order]

    keep: List[LineString] = []

    if use_spatial_index and len(sorted_lines) > 10:
        # Build STRtree once; update it incrementally as lines are added to `keep`
        # Strategy: query the full list, but check `keep` membership via a set.
        # This avoids rebuilding the tree at each step while still limiting
        # the candidates to spatially nearby geometries.

        full_tree     = STRtree(sorted_lines)
        keep_set_idx  = set()    # indices into sorted_lines that are in `keep`

        for i, line in enumerate(sorted_lines):
            if i in keep_set_idx:
                # Already in keep (this shouldn't happen but guard anyway)
                continue

            # Query all geometries whose envelopes overlap the buffered line
            search_geom    = line.buffer(max_mean_dist)
            raw_candidates = full_tree.query(search_geom, predicate='intersects')
            # raw_candidates are indices into sorted_lines
            candidates_idx = [j for j in raw_candidates if j < i and j in keep_set_idx]
            candidates_idx.sort(key=lambda j: line.distance(sorted_lines[j]))
            candidates_idx = candidates_idx[:max_cands_per_line]

            is_duplicate = False
            ang_i = _bearing_pca(line)

            for j in candidates_idx:
                kept_line = sorted_lines[j]

                # 1. Mean distance check
                coords   = np.array(line.coords)
                n_pts    = min(20, len(coords))
                idx_pts  = np.linspace(0, len(coords) - 1, n_pts, dtype=int)
                dists    = [Point(coords[k]).distance(kept_line) for k in idx_pts]
                mean_d   = float(np.mean(dists))
                if mean_d >= max_mean_dist:
                    continue

                # 2. Angular check (PCA)
                ang_j    = _bearing_pca(kept_line)
                diff_ang = abs(ang_i - ang_j)
                diff_ang = min(diff_ang, 180.0 - diff_ang)
                if diff_ang > max_angle_deg:
                    continue

                # 3. Longitudinal overlap
                shorter, longer = (
                    (line, kept_line) if line.length <= kept_line.length
                    else (kept_line, line)
                )
                if _longitudinal_overlap(shorter, longer) < min_overlap:
                    continue

                is_duplicate = True
                break

            if not is_duplicate:
                keep.append(line)
                keep_set_idx.add(i)

    else:
        # Fallback: O(N²) for very small sets where tree overhead isn't worth it
        ang_cache: dict = {}
        for i, line in enumerate(sorted_lines):
            is_duplicate = False
            ang_i = ang_cache.setdefault(i, _bearing_pca(line))

            for j, kept_line in enumerate(keep):
                coords  = np.array(line.coords)
                n_pts   = min(20, len(coords))
                idx_pts = np.linspace(0, len(coords) - 1, n_pts, dtype=int)
                dists   = [Point(coords[k]).distance(kept_line) for k in idx_pts]
                if float(np.mean(dists)) >= max_mean_dist:
                    continue

                ang_j    = ang_cache.setdefault(-(j + 1), _bearing_pca(kept_line))
                diff_ang = abs(ang_i - ang_j)
                diff_ang = min(diff_ang, 180.0 - diff_ang)
                if diff_ang > max_angle_deg:
                    continue

                shorter, longer = (
                    (line, kept_line) if line.length <= kept_line.length
                    else (kept_line, line)
                )
                if _longitudinal_overlap(shorter, longer) < min_overlap:
                    continue

                is_duplicate = True
                break

            if not is_duplicate:
                keep.append(line)

    logger.info(
        "Deduplication: %d -> %d lines (removed %d). "
        "Thresholds: dist<%.2fm angle<%.1f° overlap>%.0f%%.",
        len(lines), len(keep), len(lines) - len(keep),
        max_mean_dist, max_angle_deg, min_overlap * 100,
    )
    return keep
