import rasterio
from typing import List, Tuple

def pixel_to_world(coords: List[Tuple[int, int]], transform: rasterio.Affine) -> List[Tuple[float, float]]:
    """
    Convert a list of pixel coordinates (row, col) to world coordinates (x, y) in the raster's CRS.
    """
    world_coords = []
    for pt in coords:
        r, c = pt[0], pt[1]
        x, y = rasterio.transform.xy(transform, r, c, offset="center")
        world_coords.append((x, y))
    return world_coords

def world_to_pixel(coords: List[Tuple[float, float]], transform: rasterio.Affine) -> List[Tuple[int, int]]:
    """
    Convert a list of world coordinates (x, y) to pixel coordinates (row, col).
    """
    pixel_coords = []
    for pt in coords:
        x, y = pt[0], pt[1]
        r, c = rasterio.transform.rowcol(transform, x, y)
        pixel_coords.append((r, c))
    return pixel_coords
