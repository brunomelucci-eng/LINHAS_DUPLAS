import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
from shapely.ops import clip_by_rect
import rasterio
from rasterio.features import rasterize
from scipy.spatial import KDTree
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

def get_line_tangents(line: LineString, step_m: float = 0.20) -> List[Tuple[float, float, float, float]]:
    """
    Sample points along a LineString and compute the tangent vector angles.
    Returns list of tuples: (x, y, sin_2theta, cos_2theta)
    """
    points = []
    length = line.length
    if length == 0:
        return points
        
    delta = 0.05  # small distance for numerical derivative
    # Determine distances to sample
    distances = np.arange(0, length, step_m)
    if len(distances) == 0 or distances[-1] < length:
        distances = np.append(distances, length)
        
    for d in distances:
        p_curr = line.interpolate(d)
        
        # Calculate numerical derivative for tangent
        d_prev = max(0.0, d - delta)
        d_next = min(length, d + delta)
        
        p_prev = line.interpolate(d_prev)
        p_next = line.interpolate(d_next)
        
        dx = p_next.x - p_prev.x
        dy = p_next.y - p_prev.y
        
        norm = np.hypot(dx, dy)
        if norm > 1e-6:
            dx /= norm
            dy /= norm
            theta = np.arctan2(dy, dx)
        else:
            theta = 0.0
            
        sin_2theta = np.sin(2.0 * theta)
        cos_2theta = np.cos(2.0 * theta)
        
        points.append((p_curr.x, p_curr.y, sin_2theta, cos_2theta))
        
    return points

def rasterize_targets(
    tile_window: Dict[str, Any],
    tile_transform: rasterio.Affine,
    tile_size_px: int,
    lines_gdf: gpd.GeoDataFrame,
    roi_gdf: gpd.GeoDataFrame,
    raster_nodata_mask: np.ndarray, # Shape: (H, W), True if nodata
    row_width_m: float = 1.20,
    center_width_m: float = 0.20,
    orientation_sample_step_m: float = 0.10,
    gsd: float = 0.10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterizes lines and ROI to produce the targets for training:
    1. row_mask: shape (H, W), binary
    2. center_mask: shape (H, W), binary
    3. orientation_sin: shape (H, W), float32
    4. orientation_cos: shape (H, W), float32
    5. valid_mask: shape (H, W), binary
    """
    H = W = tile_size_px
    
    # 1. Create Tile Bounding Box in World Coordinates
    # Corner offsets
    xs, ys = rasterio.transform.xy(
        tile_transform, 
        [0, H, H, 0], 
        [0, 0, W, W]
    )
    tile_poly = Polygon(zip(xs, ys))
    minx, miny, maxx, maxy = tile_poly.bounds
    
    # 2. Intersect/clip geometries by tile box (with 2m buffer to include nearby lines)
    buffered_tile_poly = tile_poly.buffer(2.0)
    
    # Filter geometries that intersect the buffered tile polygon
    sindex = lines_gdf.sindex
    possible_matches_index = list(sindex.intersection(buffered_tile_poly.bounds))
    lines_in_tile = lines_gdf.iloc[possible_matches_index]
    lines_in_tile = lines_in_tile[lines_in_tile.intersects(buffered_tile_poly)]
    
    # Clean/clip to buffered tile bounds to speed up rasterization
    clipped_lines = []
    for geom in lines_in_tile.geometry:
        intersection = geom.intersection(buffered_tile_poly)
        if intersection.is_empty:
            continue
        if isinstance(intersection, LineString):
            clipped_lines.append(intersection)
        elif isinstance(intersection, MultiLineString):
            for part in intersection.geoms:
                if not part.is_empty:
                    clipped_lines.append(part)
                    
    # 3. Row and Center Mask Rasterization
    row_mask = np.zeros((H, W), dtype=np.uint8)
    center_mask = np.zeros((H, W), dtype=np.uint8)
    
    # Ensure center mask has a minimum pixel width (between 1 and 3 pixels)
    min_center_width = max(center_width_m, 1.5 * gsd)
    
    if clipped_lines:
        row_shapes = [(line.buffer(row_width_m / 2.0), 1) for line in clipped_lines]
        center_shapes = [(line.buffer(min_center_width / 2.0), 1) for line in clipped_lines]
        
        row_mask = rasterize(
            row_shapes, out_shape=(H, W), transform=tile_transform, fill=0, dtype=np.uint8
        )
        center_mask = rasterize(
            center_shapes, out_shape=(H, W), transform=tile_transform, fill=0, dtype=np.uint8
        )
        
    # 4. Orientation Map Generation
    orientation_sin = np.zeros((H, W), dtype=np.float32)
    orientation_cos = np.zeros((H, W), dtype=np.float32)
    
    if clipped_lines and np.any(row_mask > 0):
        # Sample points with orientation tangents
        ref_points = []
        for line in clipped_lines:
            ref_points.extend(get_line_tangents(line, step_m=orientation_sample_step_m))
            
        if ref_points:
            ref_points_arr = np.array(ref_points) # shape: (N, 4) -> x, y, sin, cos
            kdtree = KDTree(ref_points_arr[:, :2])
            
            # Find coordinates of all positive pixels in row_mask
            row_y, row_x = np.where(row_mask > 0)
            
            # Convert pixel coords to world coords
            px_world_x, px_world_y = rasterio.transform.xy(tile_transform, row_y, row_x)
            pixels_world = np.stack([px_world_x, px_world_y], axis=1)
            
            # Query KDTree
            distances, indices = kdtree.query(pixels_world)
            
            # Populate orientation maps
            orientation_sin[row_y, row_x] = ref_points_arr[indices, 2]
            orientation_cos[row_y, row_x] = ref_points_arr[indices, 3]
            
    # 5. Valid Mask Rasterization (inside ROI and not nodata)
    roi_shapes = [(geom, 1) for geom in roi_gdf.geometry if geom.intersects(tile_poly)]
    if roi_shapes:
        roi_mask = rasterize(
            roi_shapes, out_shape=(H, W), transform=tile_transform, fill=0, dtype=np.uint8
        )
    else:
        roi_mask = np.zeros((H, W), dtype=np.uint8)
        
    valid_mask = roi_mask.copy()
    if raster_nodata_mask is not None:
        valid_mask[raster_nodata_mask] = 0
        
    return row_mask, center_mask, orientation_sin, orientation_cos, valid_mask
