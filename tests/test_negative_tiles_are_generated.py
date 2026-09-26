import geopandas as gpd
import rasterio
from shapely.geometry import LineString, box

from src.data.tiling import TileGenerator


def test_only_explicit_zones_generate_negative_tiles():
    transform = rasterio.Affine(1, 0, 0, 0, -1, 100)
    roi = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs="EPSG:32722")
    lines = gpd.GeoDataFrame(geometry=[LineString([(5, 90), (35, 90)])], crs=roi.crs)
    # Buffer absorbs the half-pixel centre convention used to build tile polygons.
    negative_zones = gpd.GeoDataFrame(geometry=[box(49, 49, 101, 101)], crs=roi.crs)
    generator = TileGenerator(
        tile_size_px=50,
        overlap_px=0,
        min_valid_fraction=0.95,
        include_empty_tiles_ratio=1.0,
        negative_sampling_mode="explicit_zones",
        negative_zone_min_fraction=0.8,
    )
    tiles = generator.generate_tiles(100, 100, transform, roi)
    without_zones = generator.filter_tiles(tiles, lines, seed=1)
    assert all(not tile["is_empty"] for tile in without_zones)

    selected = generator.filter_tiles(tiles, lines, negative_zones, seed=1)
    negatives = [tile for tile in selected if tile["is_empty"]]
    assert len(negatives) == 1
    assert negatives[0]["sampling_class"] == "hard_negative"
