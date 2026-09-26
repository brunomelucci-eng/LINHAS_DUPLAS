import numpy as np
import geopandas as gpd
from shapely import STRtree, distance as shapely_distance, line_interpolate_point
from shapely.geometry import LineString, Point
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def _robust_two_cluster_ranges(values: np.ndarray, config: dict) -> Optional[Dict[str, float]]:
    """Split neighbour spacings into intra-pair and inter-pair robust ranges."""
    auto_cfg = config.get('auto_spacing', {})
    minimum_samples = int(auto_cfg.get('min_samples', 30))
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) < minimum_samples:
        return None
    lower_q = float(auto_cfg.get('trim_lower_quantile', 0.02))
    upper_q = float(auto_cfg.get('trim_upper_quantile', 0.98))
    low, high = np.quantile(values, [lower_q, upper_q])
    values = values[(values >= low) & (values <= high)]
    if len(values) < minimum_samples:
        return None

    centers = np.quantile(values, [0.30, 0.75]).astype(np.float64)
    for _ in range(30):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = np.asarray([
            np.median(values[labels == index]) if np.any(labels == index) else centers[index]
            for index in range(2)
        ])
        if np.allclose(updated, centers, atol=1e-5):
            centers = updated
            break
        centers = updated
    order = np.argsort(centers)
    centers = centers[order]
    labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
    clusters = [values[labels == index] for index in range(2)]
    if any(len(cluster) < max(5, minimum_samples // 5) for cluster in clusters):
        return None

    minimum_half_width = float(auto_cfg.get('minimum_half_width_m', 0.05))
    relative_half_width = float(auto_cfg.get('relative_half_width', 0.10))
    mad_multiplier = float(auto_cfg.get('mad_multiplier', 3.0))
    ranges = []
    for center, cluster in zip(centers, clusters):
        mad = float(np.median(np.abs(cluster - center)))
        half_width = max(
            minimum_half_width,
            relative_half_width * float(center),
            mad_multiplier * 1.4826 * mad,
        )
        ranges.append([float(center - half_width), float(center + half_width)])
    midpoint = float((centers[0] + centers[1]) / 2.0)
    separation_margin = float(auto_cfg.get('cluster_separation_margin_m', 0.02))
    ranges[0][1] = min(ranges[0][1], midpoint - separation_margin / 2.0)
    ranges[1][0] = max(ranges[1][0], midpoint + separation_margin / 2.0)
    if ranges[0][0] <= 0.0 or ranges[0][1] >= ranges[1][0]:
        return None
    return {
        'intra_pair_min_m': ranges[0][0],
        'intra_pair_max_m': ranges[0][1],
        'inter_pair_min_m': ranges[1][0],
        'inter_pair_max_m': ranges[1][1],
        'intra_pair_center_m': float(centers[0]),
        'inter_pair_center_m': float(centers[1]),
        'spacing_sample_count': int(len(values)),
    }


def estimate_double_row_spacing(lines: List[LineString], config: dict) -> Optional[Dict[str, float]]:
    """Estimate row-pair spacing from local parallel neighbours in one image."""
    double_cfg = config.get('postprocessing', {}).get('double_row_validation', {})
    auto_cfg = double_cfg.get('auto_spacing', {})
    if not auto_cfg.get('enabled', True) or len(lines) < 4:
        return None
    maximum_lines = int(auto_cfg.get('max_sample_lines', 2000))
    maximum_distance = float(auto_cfg.get('max_search_distance_m', 2.50))
    maximum_angle = float(auto_cfg.get('max_angle_difference_deg', 10.0))
    minimum_overlap = float(auto_cfg.get('min_longitudinal_overlap_ratio', 0.35))
    sampled_indices = np.linspace(
        0, len(lines) - 1, min(len(lines), maximum_lines), dtype=int
    )
    tree = STRtree(lines)
    bearings = np.asarray([_bearing_pca(line) for line in lines], dtype=np.float64)
    distances: List[float] = []
    for i in sampled_indices:
        line = lines[int(i)]
        local = []
        for raw_j in tree.query(line, predicate='dwithin', distance=maximum_distance):
            j = int(raw_j)
            if j == int(i) or _angle_diff(bearings[int(i)], bearings[j]) > maximum_angle:
                continue
            shorter, longer = (
                (line, lines[j]) if line.length <= lines[j].length else (lines[j], line)
            )
            if _longitudinal_overlap_ratio(shorter, longer) < minimum_overlap:
                continue
            distance = float(np.median(_sample_perpendicular_distances(line, lines[j], 12)))
            if 0.0 < distance <= maximum_distance:
                local.append(distance)
        distances.extend(sorted(local)[:2])
    return _robust_two_cluster_ranges(np.asarray(distances), auto_cfg)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _sample_perpendicular_distances(
    line_a: LineString,
    line_b: LineString,
    n_samples: int = 20,
) -> np.ndarray:
    """
    Sample `n_samples` points along `line_a` and measure their distance to
    `line_b`. Returns the distance array (metres).
    Uses the shorter line as the sampling source.
    """
    if line_a.length > line_b.length:
        line_a, line_b = line_b, line_a

    n = max(2, min(n_samples, max(2, int(line_a.length))))
    ts = np.linspace(0.0, 1.0, n)
    points = line_interpolate_point(line_a, ts, normalized=True)
    return np.asarray(shapely_distance(points, line_b), dtype=float)


def _bearing_pca(line: LineString, n_samples: int = 20) -> float:
    """Principal direction of a (possibly curved) line, in [0°, 180°)."""
    coords = np.array(line.coords)
    if len(coords) < 2:
        return 0.0
    n   = min(n_samples, len(coords))
    idx = np.linspace(0, len(coords) - 1, n, dtype=int)
    pts = coords[idx]
    pts_c = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)
    return float(np.degrees(np.arctan2(Vt[0, 1], Vt[0, 0]))) % 180.0


def _longitudinal_overlap_ratio(short: LineString, long_: LineString) -> float:
    """Fraction of `short` covered by the projected span onto `long_`."""
    if short.length < 1e-6:
        return 0.0
    p0 = long_.project(Point(list(short.coords)[0]))
    p1 = long_.project(Point(list(short.coords)[-1]))
    return abs(p1 - p0) / short.length


def _angle_diff(a: float, b: float) -> float:
    """Angle difference in [0°, 90°)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_double_rows(
    lines_gdf: gpd.GeoDataFrame,
    config: dict,
    *,
    preliminary: bool = False,
) -> gpd.GeoDataFrame:
    """
    Validate and annotate double crop-row pairs.

    Mode 'annotate' (default and recommended)
    -----------------------------------------
    Every line is preserved. Annotation columns are added:
      pair_id, pair_status, pair_distance_m, pair_angle_deg,
      pair_overlap_ratio, pair_confidence, inter_pair_status

    Mode 'filter'
    -------------
    Preserves 'unmatched' rows but filters out rows with definitive negative evidence
    such as 'invalid_distance' or 'invalid_angle' (BUG P1-04).
    """
    post_cfg        = config.get('postprocessing', {})
    double_row_cfg  = post_cfg.get('double_row_validation', {})

    if not double_row_cfg.get('enabled', False):
        logger.info("Double-row validation disabled.")
        return lines_gdf

    mode            = double_row_cfg.get('mode', 'annotate')
    runtime_spacing = config.get('_runtime_double_row_spacing')
    if runtime_spacing is None:
        runtime_spacing = estimate_double_row_spacing(
            lines_gdf.geometry.tolist(), config
        )
        if runtime_spacing is not None:
            config['_runtime_double_row_spacing'] = runtime_spacing
            logger.info(
                "Auto-calibrated double-row spacing from %d samples: "
                "intra=[%.3f, %.3f]m center=%.3fm; inter=[%.3f, %.3f]m center=%.3fm.",
                runtime_spacing['spacing_sample_count'],
                runtime_spacing['intra_pair_min_m'], runtime_spacing['intra_pair_max_m'],
                runtime_spacing['intra_pair_center_m'],
                runtime_spacing['inter_pair_min_m'], runtime_spacing['inter_pair_max_m'],
                runtime_spacing['inter_pair_center_m'],
            )
        else:
            logger.warning("Automatic double-row spacing calibration unavailable; using configured fallback ranges.")
    spacing_cfg = runtime_spacing or double_row_cfg
    intra_min       = spacing_cfg.get('intra_pair_min_m',              0.85)
    intra_max       = spacing_cfg.get('intra_pair_max_m',              1.00)
    inter_min       = spacing_cfg.get('inter_pair_min_m',              1.15)
    inter_max       = spacing_cfg.get('inter_pair_max_m',              1.40)
    tolerance       = double_row_cfg.get('tolerance_m',                   0.08)
    max_angle       = double_row_cfg.get('max_angle_difference_deg',       8.0)
    min_overlap     = double_row_cfg.get('min_longitudinal_overlap_ratio', 0.50)

    logger.info(
        "Double-row validation (mode=%s): intra=[%.2f–%.2f]m, "
        "inter=[%.2f–%.2f]m, angle<%.1f°, overlap>%.0f%%.",
        mode, intra_min, intra_max, inter_min, inter_max,
        max_angle, min_overlap * 100,
    )

    if lines_gdf.empty:
        return lines_gdf

    lines = lines_gdf.geometry.tolist()
    tree = STRtree(lines)
    bearings = np.asarray([_bearing_pca(line) for line in lines], dtype=np.float64)
    lengths = np.asarray([line.length for line in lines], dtype=np.float64)

    # --- 1. Safe Search Radius (BUG P1-01) ----------------------------------
    search_radius = (
        intra_max + tolerance
        if preliminary
        else max(intra_max, inter_max) + tolerance
    )
    
    # Collect candidate pairs and inter-pair contacts
    candidates = []
    inter_pair_found = np.zeros(len(lines), dtype=bool)

    # Only the highest-confidence failed candidate is consumed downstream.
    # Retaining a single record avoids quadratic diagnostic memory growth.
    best_failed_matches: List[Optional[Dict[str, Any]]] = [None] * len(lines)

    def update_best_failed(index: int, failed: Dict[str, Any]) -> None:
        current = best_failed_matches[index]
        # Strictly greater preserves the first candidate on confidence ties,
        # matching max(list, key=...) from the legacy implementation.
        if current is None or failed['confidence'] > current['confidence']:
            best_failed_matches[index] = failed

    for i, line in enumerate(lines):
        bear_i = bearings[i]
        neighbors = tree.query(line, predicate='dwithin', distance=search_radius)
        
        for j_raw in neighbors:
            j = int(j_raw)
            if j <= i:
                continue  # unique pairs only
            
            other = lines[j]
            angle_d = _angle_diff(bear_i, bearings[j])

            shorter = line if lengths[i] <= lengths[j] else other
            longer = other if lengths[i] <= lengths[j] else line
            ovlp = _longitudinal_overlap_ratio(shorter, longer)

            if preliminary:
                # These filters can only reject invalid intra-pair candidates;
                # no diagnostic fallback fields are consumed before refinement.
                if angle_d > max_angle or ovlp < min_overlap:
                    continue
                if line.distance(other) > (intra_max + tolerance):
                    continue

            dists = _sample_perpendicular_distances(line, other, n_samples=20)
            median_dist = float(np.median(dists))
            
            # Check inter-pair status (BUG P1-01 verification)
            if not preliminary and (inter_min - tolerance) <= median_dist <= (inter_max + tolerance):
                inter_pair_found[i] = True
                inter_pair_found[j] = True

            # Check intra-pair distance limits
            in_intra = (intra_min - tolerance) <= median_dist <= (intra_max + tolerance)
            
            if in_intra:
                # Calculate normalized scores (BUG P1-03)
                # Distance score: 1.0 at center, 0.0 at limits
                intra_mid = (intra_min + intra_max) / 2.0
                intra_half_width = (intra_max - intra_min) / 2.0 + tolerance
                dist_score = 1.0 - min(1.0, abs(median_dist - intra_mid) / max(intra_half_width, 1e-6))
                
                # Angle score
                angle_score = 1.0 - min(1.0, angle_d / max(max_angle, 1e-6))
                
                # Overlap score
                overlap_score = min(1.0, ovlp)
                
                # Weighted confidence in [0, 1]
                confidence = 0.40 * dist_score + 0.25 * angle_score + 0.35 * overlap_score
                
                # If it passes geometry checks, it's a valid candidate
                if angle_d <= max_angle and ovlp >= min_overlap:
                    candidates.append({
                        'i': i, 'j': j,
                        'distance': median_dist,
                        'angle_diff': angle_d,
                        'overlap': ovlp,
                        'confidence': float(confidence)
                    })
                else:
                    if preliminary:
                        continue
                    failed_info = {
                        'neighbor': j,
                        'distance': median_dist,
                        'angle_diff': angle_d,
                        'overlap': ovlp,
                        'confidence': float(confidence)
                    }
                    update_best_failed(i, failed_info)
                    update_best_failed(j, {
                        'neighbor': i,
                        'distance': median_dist,
                        'angle_diff': angle_d,
                        'overlap': ovlp,
                        'confidence': float(confidence)
                    })
            else:
                if preliminary:
                    continue
                # Distance is not in intra-pair range, record as fail with 0.0 confidence (BUG P1-04)
                failed_info = {
                    'neighbor': j,
                    'distance': median_dist,
                    'angle_diff': angle_d,
                    'overlap': ovlp,
                    'confidence': 0.0
                }
                update_best_failed(i, failed_info)
                update_best_failed(j, {
                    'neighbor': i,
                    'distance': median_dist,
                    'angle_diff': angle_d,
                    'overlap': ovlp,
                    'confidence': 0.0
                })

    # --- 2. Reciprocal Stable Matching (BUG P1-02) --------------------------
    # Sort candidates by confidence descending
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    
    matched_pair_id = {}
    pair_metadata = {}
    
    next_pair_id = 1
    for cand in candidates:
        i, j = cand['i'], cand['j']
        if i not in matched_pair_id and j not in matched_pair_id:
            # Assign unique pair id
            matched_pair_id[i] = next_pair_id
            matched_pair_id[j] = next_pair_id
            
            pair_metadata[i] = {
                'pair_id': next_pair_id,
                'pair_status': 'valid_pair',
                'pair_distance_m': cand['distance'],
                'pair_angle_deg': cand['angle_diff'],
                'pair_overlap_ratio': cand['overlap'],
                'pair_confidence': cand['confidence']
            }
            pair_metadata[j] = {
                'pair_id': next_pair_id,
                'pair_status': 'valid_pair',
                'pair_distance_m': cand['distance'],
                'pair_angle_deg': cand['angle_diff'],
                'pair_overlap_ratio': cand['overlap'],
                'pair_confidence': cand['confidence']
            }
            next_pair_id += 1

    # --- 3. Process Unmatched and Failed Lines ------------------------------
    annotations = []
    for i in range(len(lines)):
        inter_status = 'neighbor_found' if inter_pair_found[i] else 'no_neighbor'
        
        if i in pair_metadata:
            meta = pair_metadata[i].copy()
            meta['inter_pair_status'] = inter_status
            annotations.append(meta)
        else:
            # Categorize why this line is unmatched
            status = 'unmatched'
            pair_distance = float('nan')
            pair_angle = float('nan')
            pair_overlap = float('nan')
            best_conf = 0.0

            best_fail = best_failed_matches[i]
            if best_fail is not None:
                pair_distance = best_fail['distance']
                pair_angle = best_fail['angle_diff']
                pair_overlap = best_fail['overlap']
                best_conf = best_fail['confidence']
                
                if pair_distance < (intra_min - tolerance):
                    status = 'invalid_distance'
                elif pair_distance > (intra_max + tolerance):
                    status = 'unmatched'
                elif pair_overlap < min_overlap:
                    status = 'insufficient_overlap'
                elif pair_angle > max_angle:
                    status = 'invalid_angle'
                else:
                    status = 'possible_pair'
                    
            annotations.append({
                'pair_id': None,
                'pair_status': status,
                'pair_distance_m': pair_distance,
                'pair_angle_deg': pair_angle,
                'pair_overlap_ratio': pair_overlap,
                'pair_confidence': best_conf,
                'inter_pair_status': inter_status
            })

    # Attach annotations
    result_gdf = lines_gdf.copy()
    for col in ['pair_id', 'pair_status', 'pair_distance_m', 'pair_angle_deg',
                'pair_overlap_ratio', 'pair_confidence', 'inter_pair_status']:
        result_gdf[col] = [ann[col] for ann in annotations]
    result_gdf['spacing_calibration_source'] = (
        'automatic' if runtime_spacing is not None else 'configured_fallback'
    )
    result_gdf['calibrated_intra_center_m'] = float(
        spacing_cfg.get('intra_pair_center_m', (intra_min + intra_max) / 2.0)
    )
    result_gdf['calibrated_inter_center_m'] = float(
        spacing_cfg.get('inter_pair_center_m', (inter_min + inter_max) / 2.0)
    )

    status_counts = result_gdf['pair_status'].value_counts().to_dict()
    logger.info("Double-row status counts: %s", status_counts)

    if mode == 'filter':
        # Filter out rows with definitive negative evidence, preserving unmatched (BUG P1-04)
        n_before = len(result_gdf)
        invalid_statuses = {'invalid_distance', 'invalid_angle', 'insufficient_overlap', 'confirmed_duplicate'}
        keep_mask = ~result_gdf['pair_status'].isin(invalid_statuses)
        result_gdf = result_gdf[keep_mask].copy()
        logger.warning(
            "mode=filter: removed %d invalid lines (%d → %d). Preserved unmatched rows.",
            n_before - len(result_gdf), n_before, len(result_gdf),
        )
    else:
        logger.info(
            "mode=annotate: all %d lines preserved with double-row pair annotations.",
            len(result_gdf),
        )

    return result_gdf


def group_double_rows(
    lines_gdf: gpd.GeoDataFrame,
    intra_pair_min_m: float,
    intra_pair_max_m: float,
    inter_pair_min_m: float,
    inter_pair_max_m: float,
    tolerance_m: float = 0.10
) -> gpd.GeoDataFrame:
    """Legacy wrapper for backward compatibility."""
    config = {
        'postprocessing': {
            'double_row_validation': {
                'enabled': True,
                'mode': 'filter',
                'intra_pair_min_m': intra_pair_min_m,
                'intra_pair_max_m': intra_pair_max_m,
                'inter_pair_min_m': inter_pair_min_m,
                'inter_pair_max_m': inter_pair_max_m,
                'tolerance_m': tolerance_m
            }
        }
    }
    return validate_double_rows(lines_gdf, config)
