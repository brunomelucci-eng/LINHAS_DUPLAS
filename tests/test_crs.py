import pytest
import geopandas as gpd
from shapely.geometry import Point
from src.data.crs_utils import estimate_utm_epsg, get_centroid_lon_lat, reproject_gdf

def test_estimate_utm_epsg():
    # Test standard UTM zones
    epsg_london = estimate_utm_epsg(-0.1, 51.5)
    assert epsg_london == 32630  # UTM zone 30N
    
    epsg_sp = estimate_utm_epsg(-46.6, -23.5)
    assert epsg_sp == 32723  # UTM zone 23S

def test_get_centroid_lon_lat():
    gdf = gpd.GeoDataFrame(geometry=[Point(-46.6333, -23.5505)], crs="EPSG:4326")
    lon, lat = get_centroid_lon_lat(gdf)
    assert pytest.approx(lon, abs=1e-4) == -46.6333
    assert pytest.approx(lat, abs=1e-4) == -23.5505
