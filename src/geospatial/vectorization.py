from shapely.geometry import LineString
import geopandas as gpd
from typing import List, Tuple
import rasterio
from .pixel_world import pixel_to_world

def vectorize_branches(
    branches: List[List[Tuple[int, int]]],
    transform: rasterio.Affine,
    crs: str
) -> gpd.GeoDataFrame:
    """
    Converts lists of pixel coordinates (row, col) into a GeoDataFrame containing LineString geometries.
    """
    geoms = []
    for branch in branches:
        if len(branch) < 2:
            continue
        world_pts = pixel_to_world(branch, transform)
        geoms.append(LineString(world_pts))
        
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    return gdf
