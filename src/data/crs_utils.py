import pyproj
from pyproj import CRS
from shapely.geometry import Point
from shapely.ops import transform
import geopandas as gpd

def estimate_utm_epsg(lon: float, lat: float) -> int:
    """
    Estimate the UTM EPSG code for a given longitude and latitude.
    """
    # Zone number: 1 to 60
    zone_num = int((lon + 180) / 6) + 1
    # Check hemisphere
    if lat >= 0:
        epsg = 32600 + zone_num
    else:
        epsg = 32700 + zone_num
    return epsg

def get_centroid_lon_lat(gdf: gpd.GeoDataFrame) -> tuple[float, float]:
    """
    Compute centroid of GeoDataFrame in geographic coordinates (EPSG:4326).
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    
    if not gdf.crs.is_geographic:
        gdf_geom = gdf.to_crs(epsg=4326)
    else:
        gdf_geom = gdf
    
    union_geom = gdf_geom.unary_union
    centroid = union_geom.centroid
    return centroid.x, centroid.y

def reproject_gdf(gdf: gpd.GeoDataFrame, target_epsg: int) -> gpd.GeoDataFrame:
    """
    Reproject GeoDataFrame to a target EPSG code.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    return gdf.to_crs(epsg=target_epsg)

def reproject_geometry(geom, source_epsg: int, target_epsg: int):
    """
    Reproject a Shapely geometry from source_epsg to target_epsg.
    """
    project = pyproj.Transformer.from_crs(
        CRS.from_epsg(source_epsg),
        CRS.from_epsg(target_epsg),
        always_xy=True
    ).transform
    return transform(project, geom)
