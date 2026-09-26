import geopandas as gpd
from shapely.geometry import box, LineString, MultiLineString
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def validate_spatial_alignment(
    raster_bounds,
    raster_crs,
    roi_gdf: gpd.GeoDataFrame,
    lines_gdf: gpd.GeoDataFrame,
    strict_check: bool = False
) -> Tuple[bool, bool]:
    """
    Validate that the raster, ROI, and lines are spatially aligned.
    Returns:
        (roi_intersects_raster, lines_intersect_roi)
    """
    # Create shapely box for raster bounds
    raster_box = box(*raster_bounds)
    
    # Check raster and ROI intersection
    roi_in_raster_crs = roi_gdf.to_crs(raster_crs)
    roi_union = roi_in_raster_crs.unary_union
    
    roi_intersects_raster = roi_union.intersects(raster_box)
    
    if not roi_intersects_raster:
        msg = (
            f"CRITICAL: The ROI does not intersect the raster bounds! "
            f"Raster Bounds: {raster_bounds}, "
            f"ROI Bounds: {roi_gdf.total_bounds}"
        )
        if strict_check:
            raise ValueError(msg)
        else:
            logger.warning(msg)
    else:
        # Check percentage of ROI overlap with raster
        overlap_area = roi_union.intersection(raster_box).area
        roi_area = roi_union.area
        if roi_area > 0:
            fraction = overlap_area / roi_area
            if fraction < 0.99:
                logger.warning(f"Warning: Only {fraction:.1%} of the ROI intersects the raster area.")
                
    # Check lines and ROI intersection (both should be in the working metric CRS)
    lines_union = lines_gdf.unary_union
    roi_working_crs = roi_gdf.to_crs(lines_gdf.crs)
    roi_working_union = roi_working_crs.unary_union
    
    lines_intersect_roi = lines_union.intersects(roi_working_union)
    if not lines_intersect_roi:
        msg = "CRITICAL: The reference lines do not intersect the ROI!"
        if strict_check:
            raise ValueError(msg)
        else:
            logger.warning(msg)
        
    return roi_intersects_raster, lines_intersect_roi

def clip_lines_to_roi(lines_gdf: gpd.GeoDataFrame, roi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clip lines to the Region of Interest (ROI) polygon.
    Ensure both are in the same CRS.
    """
    if lines_gdf.crs != roi_gdf.crs:
        roi_gdf = roi_gdf.to_crs(lines_gdf.crs)
        
    clipped_gdf = gpd.clip(lines_gdf, roi_gdf)
    
    # Filter and unpack geometries to LineStrings
    valid_rows = []
    for idx, row in clipped_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            valid_rows.append(row.to_dict())
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                if isinstance(part, LineString) and not part.is_empty:
                    row_dict = row.to_dict()
                    row_dict['geometry'] = part
                    valid_rows.append(row_dict)
                    
    if not valid_rows:
        logger.warning("No line geometries remain after clipping to ROI!")
        return gpd.GeoDataFrame(columns=lines_gdf.columns, crs=lines_gdf.crs)
        
    return gpd.GeoDataFrame(valid_rows, crs=lines_gdf.crs)
