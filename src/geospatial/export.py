import geopandas as gpd
import os
import datetime
import numpy as np
import pandas as pd
from typing import Optional


def sample_line_probabilities(
    gdf: gpd.GeoDataFrame,
    probability_raster: np.ndarray,
    transform,
    spacing_m: float = 0.10,
) -> gpd.GeoDataFrame:
    """Sample real center probabilities along each line in raster coordinates."""
    df = gdf.copy()
    inverse = ~transform
    height, width = probability_raster.shape
    means, medians, p20s = [], [], []
    for geometry in df.geometry:
        count = max(2, int(np.ceil(geometry.length / max(spacing_m, 1e-3))) + 1)
        values = []
        for distance in np.linspace(0.0, geometry.length, count):
            point = geometry.interpolate(float(distance))
            col_f, row_f = inverse * (point.x, point.y)
            row, col = int(np.floor(row_f)), int(np.floor(col_f))
            if 0 <= row < height and 0 <= col < width:
                value = float(probability_raster[row, col])
                if np.isfinite(value):
                    values.append(value)
        if values:
            array = np.asarray(values, dtype=np.float64)
            means.append(float(array.mean()))
            medians.append(float(np.median(array)))
            p20s.append(float(np.percentile(array, 20)))
        else:
            means.append(np.nan)
            medians.append(np.nan)
            p20s.append(np.nan)
    df['mean_probability'] = means
    df['median_probability'] = medians
    df['p20_probability'] = p20s
    return df

def populate_metadata(
    gdf: gpd.GeoDataFrame,
    config: dict,
    model_name: str = "unet",
    checkpoint_name: str = "best.pt",
    source_id: str = "sugarcane_field"
) -> gpd.GeoDataFrame:
    """
    Populate columns into the predictions GeoDataFrame to align with the database specification.
    """
    df = gdf.copy()
    if len(df) == 0:
        return df
        
    current_time = datetime.datetime.now().isoformat()
    epsg = gdf.crs.to_epsg() if gdf.crs else None
    
    row_ids = []
    lengths = []
    num_verts = []
    geom_types = []
    
    for i, geom in enumerate(df.geometry):
        row_ids.append(f"row_{i}")
        lengths.append(geom.length)
        num_verts.append(len(geom.coords))
        geom_types.append(geom.geom_type)
        
    df['row_id'] = row_ids
    df['source_id'] = source_id
    
    df['length_m'] = lengths
    df['geometry_type'] = geom_types
    
    if 'fit_type' not in df.columns:
        df['fit_type'] = 'spline'
    if 'curvature_score' not in df.columns:
        df['curvature_score'] = 0.0
    if 'num_gaps_bridged' not in df.columns:
        df['num_gaps_bridged'] = pd.NA
    if 'max_gap_bridged_m' not in df.columns:
        df['max_gap_bridged_m'] = np.nan
        
    df['num_vertices'] = num_verts
    df['crs_epsg'] = str(epsg)
    df['model_name'] = model_name
    df['source_model'] = checkpoint_name
    df['checkpoint_name'] = checkpoint_name
    df['processing_date'] = current_time
    
    q_flags = []
    probabilities = df['mean_probability'] if 'mean_probability' in df.columns else [np.nan] * len(df)
    for p in probabilities:
        if pd.isna(p):
            q_flags.append('unknown')
        elif p >= 0.7:
            q_flags.append('high')
        elif p >= 0.5:
            q_flags.append('medium')
        else:
            q_flags.append('low')
    df['quality_flag'] = q_flags
    
    # Preserve topology, pairing, fitting, and future audit columns. Geometry stays last.
    columns = [column for column in df.columns if column != df.geometry.name]
    df = df[columns + [df.geometry.name]]
    return df

def export_geopackage(
    gdf: gpd.GeoDataFrame,
    filepath: str,
    layer_name: str = 'predicted_rows',
    export_epsg: Optional[int] = None
):
    """
    Save GeoDataFrame to a layer inside a GeoPackage.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if export_epsg is not None and gdf.crs is not None:
        gdf = gdf.to_crs(epsg=export_epsg)
    gdf.to_file(filepath, layer=layer_name, driver="GPKG")
    
def export_geojson(
    gdf: gpd.GeoDataFrame,
    filepath: str,
    export_epsg: Optional[int] = None
):
    """
    Save GeoDataFrame to a GeoJSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if export_epsg is not None and gdf.crs is not None:
        gdf = gdf.to_crs(epsg=export_epsg)
    gdf.to_file(filepath, driver="GeoJSON")
