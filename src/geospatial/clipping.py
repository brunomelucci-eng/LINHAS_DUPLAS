import geopandas as gpd
from ..data.geometry_validation import clip_lines_to_roi

def clip_predictions_to_roi(lines_gdf: gpd.GeoDataFrame, roi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clips final predicted LineStrings to the boundary of the ROI polygon.
    """
    return clip_lines_to_roi(lines_gdf, roi_gdf)
