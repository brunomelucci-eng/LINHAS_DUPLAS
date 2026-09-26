import pytest
import json
import os
import tempfile
import sys
import yaml
import shutil
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, Polygon
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_synthetic_data(temp_dir: str) -> tuple:
    """
    Generate synthetic datasets for the E2E integration test:
    - 3-band raster of 512x512 pixels with GSD of 0.1m.
    - ROI polygon covering the raster bounds.
    - 2 parallel reference LineStrings.
    """
    H = W = 512
    gsd = 0.1
    # Top-Left coordinate: (1000.0, 5000.0)
    transform = rasterio.Affine(gsd, 0.0, 1000.0, 0.0, -gsd, 5000.0)
    
    # Green background with normal noise
    img = np.random.normal(110, 10, (3, H, W)).astype(np.uint8)
    
    # Draw two light crop rows on the imagery
    # Row index 200: coordinate y = 5000 - 200 * 0.1 = 4980.0
    img[:, 195:205, 50:462] = 220
    # Row index 300: coordinate y = 5000 - 300 * 0.1 = 4970.0
    img[:, 295:305, 50:462] = 220
    
    raster_path = os.path.join(temp_dir, 'synthetic_ortho.tif')
    with rasterio.open(
        raster_path, 'w',
        driver='GTiff',
        width=W, height=H,
        count=3,
        dtype='uint8',
        crs='EPSG:32630',
        transform=transform
    ) as dst:
        dst.write(img)
        
    # ROI
    min_x, max_y = 1000.0, 5000.0
    max_x = min_x + W * gsd
    min_y = max_y - H * gsd
    roi_poly = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
    roi_gdf = gpd.GeoDataFrame(geometry=[roi_poly], crs='EPSG:32630')
    roi_path = os.path.join(temp_dir, 'synthetic_roi.gpkg')
    roi_gdf.to_file(roi_path, driver='GPKG')
    
    # Lines
    # Col index 50 to 462 -> x = 1000 + 50*0.1 = 1005.0 to 1046.2
    line1 = LineString([(1005.0, 4980.0), (1046.2, 4980.0)])
    line2 = LineString([(1005.0, 4970.0), (1046.2, 4970.0)])
    
    lines_gdf = gpd.GeoDataFrame(geometry=[line1, line2], crs='EPSG:32630')
    lines_gdf['talhao_id'] = [1, 1]
    lines_gdf['row_id'] = [101, 102]
    
    lines_path = os.path.join(temp_dir, 'synthetic_lines.gpkg')
    lines_gdf.to_file(lines_path, driver='GPKG')
    
    return raster_path, roi_path, lines_path

def test_end_to_end_pipeline():
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Generate synthetic files
        raster_path, roi_path, lines_path = create_synthetic_data(temp_dir)
        
        # 2. Write temp config YAML file
        config_data = {
            'project': {
                'name': 'synthetic_project',
                'seed': 42,
                'output_dir': os.path.join(temp_dir, 'outputs')
            },
            'input': {
                'orthomosaic': raster_path,
                'roi': roi_path,
                'reference_lines': lines_path,
                'id_field': 'row_id',
                'group_field': 'talhao_id',
                'rgb_bands': [1, 2, 3]
            },
            'crs': {
                'auto_utm': True,
                'working_epsg': None,
                'export_epsg': None
            },
            'tiling': {
                'tile_size_px': 256,
                'overlap_px': 64,
                'min_valid_fraction': 0.50,
                'include_empty_tiles_ratio': 0.10
            },
            'targets': {
                'row_width_m': 1.20,
                'center_width_m': 0.20,
                'orientation_sample_step_m': 0.20
            },
            'split': {
                'strategy': 'spatial_group',
                'group_field': 'talhao_id',
                'train_fraction': 0.60,
                'val_fraction': 0.20,
                'test_fraction': 0.20
            },
            'model': {
                'architecture': 'unet',
                'encoder': 'resnet18',
                'pretrained': False,
                'input_channels': 3,
                'output_channels': 4
            },
            'loss': {
                'row_bce_weight': 0.20,
                'row_dice_weight': 0.20,
                'center_focal_weight': 0.15,
                'center_dice_weight': 0.20,
                'cldice_weight': 0.15,
                'orientation_weight': 0.10
            },
            'training': {
                'epochs': 1,
                'batch_size': 2,
                'num_workers': 0,
                'optimizer': 'adamw',
                'learning_rate': 0.001,
                'weight_decay': 0.0,
                'mixed_precision': False,
                'gradient_clip_norm': 1.0,
                'early_stopping_patience': 2,
                'monitor': 'val_loss',
                'monitor_mode': 'min'
            },
            'augmentation': {
                'enabled': False
            },
            'inference': {
                'batch_size': 1,
                'use_tta': False,
                'blending': 'hann'
            },
            'postprocessing': {
                'row_threshold': 0.50,
                'center_threshold_high': 0.50,
                'center_threshold_low': 0.30,
                'closing_radius_m': 0.10,
                'opening_radius_m': 0.05,
                'min_component_length_m': 2.0,
                'max_hole_area_m2': 0.50,
                'double_row_validation': {
                    'enabled': False
                }
            },
            'gap_bridging': {
                'enabled': True,
                'max_gap_m': 2.0,
                'max_angle_difference_deg': 20.0,
                'max_lateral_offset_m': 0.40,
                'min_corridor_probability': 0.10
            },
            'line_extension': {
                'enabled': True,
                'max_extension_m': 1.0,
                'max_angle_change_deg': 10.0
            },
            'vector': {
                'simplify_tolerance_m': 0.05,
                'vertex_spacing_m': 2.0,
                'min_line_length_m': 2.0
            }
        }
        # Override the multi-input list inherited from configs/base.yaml. The
        # integration test must remain hermetic and use only its synthetic data.
        config_data['inputs'] = [dict(config_data['input'], name='synthetic')]
        
        config_path = os.path.join(temp_dir, 'synthetic_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
            
        # 3. Run dataset prep
        from scripts.prepare_dataset import main as prep_main
        with patch('sys.argv', ['prepare_dataset.py', '--config', config_path]):
            prep_main()
            
        # Assert database partitions exist
        train_path = os.path.join(temp_dir, 'outputs', 'dataset', 'train')
        val_path = os.path.join(temp_dir, 'outputs', 'dataset', 'val')
        assert os.path.exists(train_path)
        assert os.path.exists(val_path)
        assert len(os.listdir(train_path)) > 0

        # 4. Run the same mandatory dataset inspection required in production.
        output_dir = config_data['project']['output_dir']
        inspection_path = os.path.join(output_dir, 'dataset_inspection_report.json')
        from scripts.inspect_data import main as inspect_main
        with patch('sys.argv', [
            'inspect_data.py',
            '--dataset-dir', os.path.join(output_dir, 'dataset'),
            '--output', inspection_path,
        ]):
            inspect_main()
        with open(inspection_path, encoding='utf-8') as handle:
            assert json.load(handle)['status'] == 'approved'

        # 5. Run mock training for 1 epoch
        from scripts.train import main as train_main
        with patch('sys.argv', ['train.py', '--config', config_path]):
            train_main()
            
        checkpoint_dir = os.path.join(temp_dir, 'outputs', 'checkpoints')
        best_checkpoints = [
            os.path.join(checkpoint_dir, name)
            for name in os.listdir(checkpoint_dir)
            if name.startswith('best_') and name.endswith('.pt')
        ]
        assert len(best_checkpoints) == 1
        checkpoint_file = best_checkpoints[0]
        
        # 6. Run predict / inference and postprocessing
        output_vector_path = os.path.join(temp_dir, 'outputs', 'predicted_rows.gpkg')
        from scripts.predict import main as predict_main

        def deterministic_prediction(_predictor, reader, _config, roi_window, **_kwargs):
            height, width = int(roi_window.height), int(roi_window.width)
            center = np.zeros((height, width), dtype=np.float32)
            row = np.zeros((height, width), dtype=np.float32)
            for row_index in (int(height * 200 / 512), int(height * 300 / 512)):
                start, end = int(width * 50 / 512), int(width * 462 / 512)
                center[max(0, row_index - 1):row_index + 2, start:end] = 0.95
                row[max(0, row_index - 5):row_index + 6, start:end] = 0.95
            mosaic = SimpleNamespace(close=lambda: None)
            return SimpleNamespace(
                center=center,
                row=row,
                orientation_sin=None,
                orientation_cos=None,
                mosaic=mosaic,
                tile_windows=[],
            )

        with patch('scripts.predict.predict_full_raster', side_effect=deterministic_prediction), patch(
            'sys.argv', [
                'predict.py',
                '--config', config_path,
                '--checkpoint', checkpoint_file,
                '--orthomosaic', raster_path,
                '--roi', roi_path,
                '--output', output_vector_path,
            ],
        ):
            predict_main()
            
        # Assert final output exists and contains LineString data
        assert os.path.exists(output_vector_path)
        gdf_out = gpd.read_file(output_vector_path, layer='predicted_rows')
        assert len(gdf_out) > 0
        assert gdf_out.geometry.iloc[0].geom_type == 'LineString'
        
        # Check standard fields populated
        assert 'row_id' in gdf_out.columns
        assert 'length_m' in gdf_out.columns
        assert 'quality_flag' in gdf_out.columns
        
    finally:
        import logging
        logger = logging.getLogger()
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        shutil.rmtree(temp_dir)
