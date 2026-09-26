import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio
import os
from typing import Optional

def save_prediction_overlay(
    image: np.ndarray, 
    pred_gdf: gpd.GeoDataFrame,
    ref_gdf: Optional[gpd.GeoDataFrame],
    transform: rasterio.Affine,
    output_path: str
):
    """
    Saves a visualization showing predicted crop lines (and reference lines, if provided)
    overlaid on the input RGB imagery (pixels).
    """
    # Handle band structure
    if image.shape[0] in [1, 3]:
        img = np.transpose(image[:3], (1, 2, 0))
    else:
        img = image
        
    if img.dtype != np.uint8:
        # Scale if float in [0, 1]
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)
    
    # Draw predicted lines
    has_legend = False
    for geom in pred_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        coords = np.array(geom.coords)
        cols, rows = [], []
        for pt in coords:
            x, y = float(pt[0]), float(pt[1])
            r, c = rasterio.transform.rowcol(transform, x, y)
            cols.append(c)
            rows.append(r)
        
        lbl = 'Predicted' if not has_legend else ""
        ax.plot(cols, rows, color='cyan', linewidth=2.0, label=lbl)
        has_legend = True
        
    # Draw reference lines
    if ref_gdf is not None:
        has_ref_legend = False
        for geom in ref_gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            coords = np.array(geom.coords)
            cols, rows = [], []
            for pt in coords:
                x, y = float(pt[0]), float(pt[1])
                r, c = rasterio.transform.rowcol(transform, x, y)
                cols.append(c)
                rows.append(r)
                
            lbl = 'Reference' if not has_ref_legend else ""
            ax.plot(cols, rows, color='red', linestyle='--', linewidth=1.5, label=lbl)
            has_ref_legend = True
            
    if has_legend or (ref_gdf is not None):
        ax.legend(loc='upper right')
        
    ax.axis('off')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
