import os
import json
import numpy as np
import pytest
import rasterio
import geopandas as gpd
from shapely.geometry import box
from src.data.raster_reprojection import (
    prepare_working_raster,
    prepare_working_raster_cached,
)
from rasterio.crs import CRS

def test_prepare_working_raster_bounds(tmp_path):
    # 1. Create a dummy geographic raster (EPSG:4326), 200x200 pixels
    raster_path = os.path.join(tmp_path, "dummy_geographic.tif")
    
    # 0.0001 degrees resolution (~11m)
    transform = rasterio.transform.from_origin(-48.0, -22.0, 0.0001, 0.0001)
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint16',
        'nodata': 0,
        'width': 200,
        'height': 200,
        'count': 3,
        'crs': 'EPSG:4326',
        'transform': transform,
        'tiled': False
    }
    
    data = np.ones((3, 200, 200), dtype=np.uint16) * 1000
    with rasterio.open(raster_path, 'w', **profile) as dst:
        dst.write(data)
        
    # 2. Create a dummy ROI at the center of the raster, covering ~40x40 pixels
    roi_box = box(-47.992, -22.012, -47.988, -22.008)
    roi_gdf = gpd.GeoDataFrame(geometry=[roi_box], crs='EPSG:4326')
    
    # Target CRS: UTM Zone 22S (EPSG:32722)
    target_crs = CRS.from_epsg(32722)
    
    out_path = os.path.join(tmp_path, "working_raster_cropped.tif")
    
    # Run crop and reproject
    prepare_working_raster(
        src_path=raster_path,
        dst_path=out_path,
        target_crs=target_crs,
        roi_gdf=roi_gdf,
        roi_buffer_m=10.0,
        rgb_bands=[1, 2, 3],
        output_resolution_m=1.0 # 1 meter resolution
    )
    
    # Verify file is written
    assert os.path.exists(out_path)
    
    # Read output and verify metadata
    with rasterio.open(out_path) as src_out:
        assert src_out.crs.to_epsg() == 32722
        assert src_out.dtypes[0] == 'uint16' # Dtype preserved! (BUG P0-02)
        assert src_out.count == 3
        
        # Verify output dimensions are small (proportional to ROI + buffer, NOT the full raster size)
        # Full raster reprojected would be ~2000x2000 meters. Crop is ~44x44 meters + buffer.
        # Dimensions should be around 60x60 to 100x100, far below the full raster.
        assert src_out.width < 150
        assert src_out.height < 150


def test_working_raster_cache_is_reused_only_after_validation(tmp_path):
    raster_path = os.path.join(tmp_path, "cache_source.tif")
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': 40,
        'height': 40,
        'count': 3,
        'crs': 'EPSG:4326',
        'transform': rasterio.transform.from_origin(-48.0, -22.0, 0.0001, 0.0001),
    }
    with rasterio.open(raster_path, 'w', **profile) as dst:
        dst.write(np.full((3, 40, 40), 50, dtype=np.uint8))
    roi = gpd.GeoDataFrame(
        geometry=[box(-47.998, -22.003, -47.997, -22.002)],
        crs='EPSG:4326',
    )
    cache_dir = os.path.join(tmp_path, 'cache')
    kwargs = dict(
        src_path=raster_path,
        cache_dir=cache_dir,
        target_crs=CRS.from_epsg(32722),
        roi_gdf=roi,
        roi_buffer_m=5.0,
        rgb_bands=[1, 2, 3],
        output_resolution_m=1.0,
    )

    first_path, first_hit = prepare_working_raster_cached(**kwargs)
    second_path, second_hit = prepare_working_raster_cached(**kwargs)

    assert first_hit is False
    assert second_hit is True
    assert first_path == second_path

    manifest_path = os.path.splitext(first_path)[0] + '.json'
    with open(manifest_path, 'r', encoding='utf-8') as stream:
        manifest = json.load(stream)
    manifest['width'] += 1
    with open(manifest_path, 'w', encoding='utf-8') as stream:
        json.dump(manifest, stream)

    rebuilt_path, rebuilt_hit = prepare_working_raster_cached(**kwargs)
    assert rebuilt_path == first_path
    assert rebuilt_hit is False
