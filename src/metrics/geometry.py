import geopandas as gpd
import numpy as np
from shapely.geometry import MultiLineString, LineString
from scipy.spatial import KDTree
from typing import Dict

def sample_points_from_gdf(gdf: gpd.GeoDataFrame, spacing_m: float = 0.5) -> np.ndarray:
    """
    Sample points along LineStrings in a GeoDataFrame at regular metric intervals.
    """
    points = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            geoms = [geom]
        elif isinstance(geom, MultiLineString):
            geoms = geom.geoms
        else:
            continue
            
        for g in geoms:
            length = g.length
            if length == 0:
                continue
            distances = np.arange(0, length, spacing_m)
            if len(distances) == 0 or distances[-1] < length:
                distances = np.append(distances, length)
            for d in distances:
                pt = g.interpolate(d)
                points.append([pt.x, pt.y])
                
    return np.array(points) if points else np.empty((0, 2))

def compute_geometric_metrics(pred_gdf: gpd.GeoDataFrame, ref_gdf: gpd.GeoDataFrame) -> Dict[str, float]:
    """
    Computes spatial error metrics between predicted lines and reference lines:
    - Mean Symmetric Distance (MSD)
    - Hausdorff Distance
    - 95th percentile distance (HD95)
    """
    if len(pred_gdf) == 0 or len(ref_gdf) == 0:
        return {
            'mean_symmetric_distance_m': float('nan'),
            'hausdorff_distance_m': float('nan'),
            'hd95_m': float('nan')
        }
        
    pts_pred = sample_points_from_gdf(pred_gdf, spacing_m=0.5)
    pts_ref = sample_points_from_gdf(ref_gdf, spacing_m=0.5)
    
    if len(pts_pred) == 0 or len(pts_ref) == 0:
        return {
            'mean_symmetric_distance_m': float('nan'),
            'hausdorff_distance_m': float('nan'),
            'hd95_m': float('nan')
        }
        
    tree_ref = KDTree(pts_ref)
    dists_pred_to_ref, _ = tree_ref.query(pts_pred)
    
    tree_pred = KDTree(pts_pred)
    dists_ref_to_pred, _ = tree_pred.query(pts_ref)
    
    all_dists = np.concatenate([dists_pred_to_ref, dists_ref_to_pred])
    
    mean_sym_dist = np.mean(all_dists)
    hausdorff = np.max(all_dists)
    hd95 = np.percentile(all_dists, 95)
    
    return {
        'mean_symmetric_distance_m': float(mean_sym_dist),
        'hausdorff_distance_m': float(hausdorff),
        'hd95_m': float(hd95)
    }
