import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def extend_lines_to_roi(
    lines: List[LineString],
    roi_poly: Polygon,
    config: dict,
    probability_raster: Optional[np.ndarray] = None,
    transform=None,
) -> List[LineString]:
    """
    Extends endpoints of the lines towards the ROI boundary.
    Only extends if within max_extension_m, and if the extension does not cross other lines.
    """
    ext_cfg = config.get('line_extension', {})
    if not ext_cfg.get('enabled', True) or not lines:
        return lines
        
    max_ext = float(ext_cfg.get('max_extension_m', 0.30))
    if max_ext <= 0:
        return lines
    require_support = bool(ext_cfg.get('require_probability_support', True))
    if require_support and (probability_raster is None or transform is None):
        logger.warning("Line extension requires center-probability support; leaving lines unchanged.")
        return lines
    min_probability = float(ext_cfg.get('min_probability', 0.22))
    min_supported_fraction = float(ext_cfg.get('min_supported_fraction', 0.80))
    extended_lines = []
    tree = STRtree(lines)
    by_id = {id(geometry): index for index, geometry in enumerate(lines)}
    roi_boundary = roi_poly.boundary

    def supported(segment: LineString) -> bool:
        if not require_support:
            return True
        samples = max(3, int(np.ceil(segment.length / 0.05)) + 1)
        distances = np.linspace(0.0, segment.length, samples)
        values = []
        inverse = ~transform
        height, width = probability_raster.shape
        for distance in distances:
            point = segment.interpolate(float(distance))
            col_f, row_f = inverse * (point.x, point.y)
            row, col = int(np.floor(row_f)), int(np.floor(col_f))
            if 0 <= row < height and 0 <= col < width:
                values.append(float(probability_raster[row, col]))
            else:
                values.append(0.0)
        return float(np.mean(np.asarray(values) >= min_probability)) >= min_supported_fraction

    def crosses_other(segment: LineString, own_index: int) -> bool:
        for candidate in tree.query(segment):
            if hasattr(candidate, "geom_type"):
                other_index = by_id.get(id(candidate))
                other = candidate
            else:
                other_index = int(candidate)
                other = lines[other_index]
            if other_index != own_index and segment.intersects(other):
                return True
        return False
    
    for idx, line in enumerate(lines):
        coords = list(line.coords)
        if len(coords) < 2:
            extended_lines.append(line)
            continue
            
        # Start endpoint extension
        p0 = np.array(coords[0])
        p1 = np.array(coords[1])
        v_start = p0 - p1
        norm_start = np.linalg.norm(v_start)
        if norm_start > 1e-6:
            v_start /= norm_start
            ray_start_end = p0 + v_start * max_ext
            ext_segment_start = LineString([p0, ray_start_end])
            
            intersection = ext_segment_start.intersection(roi_boundary)
            if not intersection.is_empty:
                if isinstance(intersection, Point):
                    closest_pt = intersection
                else:
                    points = []
                    if hasattr(intersection, 'geoms'):
                        points = [g for g in intersection.geoms if isinstance(g, Point)]
                    else:
                        points = [Point(c) for c in intersection.coords]
                    if points:
                        closest_pt = min(points, key=lambda pt: pt.distance(Point(p0)))
                    else:
                        closest_pt = None
                        
                if closest_pt is not None:
                    new_seg = LineString([tuple(p0), (closest_pt.x, closest_pt.y)])
                    if supported(new_seg) and not crosses_other(new_seg, idx):
                        coords = [closest_pt.coords[0]] + coords
                        
        # End endpoint extension
        p_n = np.array(coords[-1])
        p_n_1 = np.array(coords[-2])
        v_end = p_n - p_n_1
        norm_end = np.linalg.norm(v_end)
        if norm_end > 1e-6:
            v_end /= norm_end
            ray_end_end = p_n + v_end * max_ext
            ext_segment_end = LineString([p_n, ray_end_end])
            
            intersection = ext_segment_end.intersection(roi_boundary)
            if not intersection.is_empty:
                if isinstance(intersection, Point):
                    closest_pt = intersection
                else:
                    points = []
                    if hasattr(intersection, 'geoms'):
                        points = [g for g in intersection.geoms if isinstance(g, Point)]
                    else:
                        points = [Point(c) for c in intersection.coords]
                    if points:
                        closest_pt = min(points, key=lambda pt: pt.distance(Point(p_n)))
                    else:
                        closest_pt = None
                        
                if closest_pt is not None:
                    new_seg = LineString([tuple(p_n), (closest_pt.x, closest_pt.y)])
                    if supported(new_seg) and not crosses_other(new_seg, idx):
                        coords = coords + [closest_pt.coords[0]]
                        
        extended_lines.append(LineString(coords))
        
    return extended_lines
