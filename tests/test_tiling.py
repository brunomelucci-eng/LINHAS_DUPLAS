import pytest
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
from src.data.tiling import TileGenerator

def test_tile_generator_basic():
    # Mock a 100x100 pixel raster with simple identity transform
    transform = rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
    
    # ROI covering the entire area
    roi_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    roi_gdf = gpd.GeoDataFrame(geometry=[roi_poly], crs="EPSG:32632")
    
    # Use tile size of 50 with 10 pixel overlap
    generator = TileGenerator(tile_size_px=50, overlap_px=10, min_valid_fraction=0.5)
    tiles = generator.generate_tiles(100, 100, transform, roi_gdf)
    
    assert len(tiles) > 0
    for tile in tiles:
        assert tile['width'] == 50
        assert tile['height'] == 50
        assert tile['valid_fraction'] >= 0.5
