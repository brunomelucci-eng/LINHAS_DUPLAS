import geopandas as gpd
from shapely.geometry import MultiPoint, Point
from typing import Dict

def compute_topological_metrics(pred_gdf: gpd.GeoDataFrame) -> Dict[str, int]:
    """
    Computes topological properties of predicted crop lines,
    counting self-intersections or cross-row intersections.
    """
    num_lines = len(pred_gdf)
    num_intersections = 0
    
    if num_lines >= 2:
        geoms = pred_gdf.geometry.values
        for i in range(num_lines):
            for j in range(i + 1, num_lines):
                if geoms[i].intersects(geoms[j]):
                    inter = geoms[i].intersection(geoms[j])
                    if not inter.is_empty:
                        if isinstance(inter, (Point, MultiPoint)):
                            if hasattr(inter, 'geoms'):
                                num_intersections += len(inter.geoms)
                            else:
                                num_intersections += 1
                        else:
                            num_intersections += 1
                            
    return {
        'num_reconstructed_lines': num_lines,
        'num_crossover_intersections': num_intersections
    }
