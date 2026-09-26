import pytest
import rasterio
from src.geospatial.pixel_world import pixel_to_world, world_to_pixel
from src.geospatial.vectorization import vectorize_branches

def test_pixel_to_world_conversions():
    # Affine transform: GSD=0.5m, Top-Left coordinate: (1000.0, 5000.0)
    transform = rasterio.Affine(0.5, 0.0, 1000.0, 0.0, -0.5, 5000.0)
    
    # (0, 0) maps to center of top-left pixel: (1000.25, 4999.75)
    w_pts = pixel_to_world([(0, 0)], transform)
    assert w_pts[0] == (1000.25, 4999.75)
    
    # Test round trip conversion
    p_pts = world_to_pixel(w_pts, transform)
    assert p_pts[0] == (0, 0)

def test_vectorize_branches():
    transform = rasterio.Affine(1.0, 0.0, 10.0, 0.0, -1.0, 100.0)
    branches = [
        [(0, 0), (1, 1), (2, 2)]
    ]
    gdf = vectorize_branches(branches, transform, "EPSG:32630")
    assert len(gdf) == 1
    assert gdf.crs == "EPSG:32630"
    geom = gdf.geometry.iloc[0]
    assert len(geom.coords) == 3
