import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def read_roi(filepath: str, layer: Optional[str] = None, target_epsg: Optional[int] = None) -> gpd.GeoDataFrame:
    """
    Read the Region of Interest (ROI) polygon and reproject if target_epsg is provided.
    """
    logger.info(f"Reading ROI from {filepath} (layer: {layer})")
    gdf = gpd.read_file(filepath, layer=layer)
    
    if gdf.crs is None:
        raise ValueError(f"Arquivo ROI '{filepath}' não possui CRS definido. Por favor, forneça um arquivo com CRS válido.")
    
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    valid_rows = []
    
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if not geom.is_valid:
            logger.warning(f"Invalid ROI geometry detected at index {idx}, fixing with buffer(0)")
            geom = geom.buffer(0)
        
        if isinstance(geom, (Polygon, MultiPolygon)):
            row_dict = row.to_dict()
            row_dict['geometry'] = geom
            valid_rows.append(row_dict)
        else:
            logger.warning(f"Ignored non-polygon geometry type at index {idx}: {type(geom)}")
            
    if not valid_rows:
        raise ValueError(f"No valid Polygon or MultiPolygon found in ROI file: {filepath}")
    
    gdf = gpd.GeoDataFrame(valid_rows, crs=gdf.crs)
    
    if target_epsg is not None:
        gdf = gdf.to_crs(epsg=target_epsg)
        
    return gdf

def read_reference_lines(filepath: str, layer: Optional[str] = None, id_field: Optional[str] = None, target_epsg: Optional[int] = None) -> gpd.GeoDataFrame:
    """
    Read the reference lines, explodes MultiLineStrings into LineStrings, cleans and reprojects.
    """
    logger.info(f"Reading reference lines from {filepath} (layer: {layer})")
    gdf = gpd.read_file(filepath, layer=layer)
    
    if gdf.crs is None:
        raise ValueError(f"Arquivo de linhas de referência '{filepath}' não possui CRS definido. Por favor, forneça um arquivo com CRS válido.")
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    
    exploded_rows = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if not geom.is_valid:
            logger.warning(f"Invalid reference line geometry at index {idx}, fixing with buffer(0)")
            geom = geom.buffer(0)
            
        if isinstance(geom, LineString):
            row_dict = row.to_dict()
            row_dict['geometry'] = geom
            exploded_rows.append(row_dict)
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                if isinstance(part, LineString) and not part.is_empty:
                    row_dict = row.to_dict()
                    row_dict['geometry'] = part
                    exploded_rows.append(row_dict)
        else:
            logger.warning(f"Ignored unsupported geometry type at index {idx}: {type(geom)}")
            
    if not exploded_rows:
        raise ValueError(f"No valid LineString or MultiLineString found in reference lines: {filepath}")
        
    gdf_exploded = gpd.GeoDataFrame(exploded_rows, crs=gdf.crs)
    
    # Remove duplicates based on WKT representation to be safe
    wkt_geom = gdf_exploded.geometry.to_wkt()
    gdf_exploded = gdf_exploded.loc[wkt_geom.drop_duplicates().index]
    
    if target_epsg is not None:
        gdf_exploded = gdf_exploded.to_crs(epsg=target_epsg)
        
    # Generate unique ID for each line
    if id_field and id_field in gdf_exploded.columns:
        gdf_exploded[id_field] = gdf_exploded[id_field].fillna(-1)
        unique_ids = []
        counts = {}
        for val in gdf_exploded[id_field]:
            if val == -1:
                val = "generated"
            counts[val] = counts.get(val, 0) + 1
            if counts[val] > 1:
                unique_ids.append(f"{val}_{counts[val]-1}")
            else:
                unique_ids.append(str(val))
        gdf_exploded['unique_row_id'] = unique_ids
    else:
        gdf_exploded['unique_row_id'] = [f"row_{i}" for i in range(len(gdf_exploded))]
        
    return gdf_exploded
