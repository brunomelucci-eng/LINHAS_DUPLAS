import argparse

import sys
import os
import uuid
import numpy as np
import geopandas as gpd
import rasterio
import shutil
import logging
import csv
import json
import hashlib

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.seed import set_seed
from src.logging_utils import setup_logging
from src.data import (
    RasterReader, read_roi, read_reference_lines, reproject_raster,
    estimate_utm_epsg, get_centroid_lon_lat, reproject_gdf,
    validate_spatial_alignment, clip_lines_to_roi, TileGenerator,
    rasterize_targets, split_dataset_groups
)
from src.data.raster_reprojection import estimate_native_gsd_m, resolve_raster_source

logger = logging.getLogger(__name__)


def _remove_existing_dataset(dataset_dir: str, policy: str) -> None:
    if policy not in {"replace", "append", "fail"}:
        raise ValueError("dataset.existing_output_policy must be replace, append, or fail.")
    if not os.path.exists(dataset_dir) or policy == "append":
        return
    if policy == "fail":
        raise FileExistsError(f"Dataset output already exists: {dataset_dir}")
    logger.info("Replacing generated dataset only: %s", dataset_dir)
    shutil.rmtree(dataset_dir)


def _count_tile_types(tiles):
    return sum(not tile.get("is_empty", False) for tile in tiles), sum(tile.get("is_empty", False) for tile in tiles)

def main():
    parser = argparse.ArgumentParser(description="Prepare the Sugarcane Row training dataset.")
    parser.add_argument('--config', type=str, default='configs/unet_resnet34.yaml', help="Path to config YAML file.")
    args = parser.parse_args()
    
    # 1. Load config
    config = load_config(args.config)
    output_dir = config.get('project', {}).get('output_dir', 'outputs')
    setup_logging(output_dir)
    
    set_seed(config.get('project', {}).get('seed', 42))
    
    logger.info("Starting dataset preparation...")
    
    # 2. Extract inputs list and deduplicate identical orthomosaics
    raw_inputs = config.get('inputs')
    if not raw_inputs:
        in_cfg = config.get('input', {})
        if not in_cfg:
            raise ValueError("Config must define 'input' or 'inputs'.")
        raw_inputs = [in_cfg]
        
    # Deduplicate: if two inputs use identical rasters (same bounds & dimensions),
    # keep the one with more annotated reference lines and discard the duplicate.
    seen_rasters = {}
    inputs_list = []
    for item in raw_inputs:
        ortho_path = item.get('orthomosaic')
        item_name = item.get('name', 'unnamed')
        if not ortho_path or not os.path.exists(ortho_path):
            inputs_list.append(item)
            continue
        try:
            with rasterio.open(ortho_path) as r:
                r_key = (
                    str(r.crs),
                    round(float(r.bounds.left), 1),
                    round(float(r.bounds.bottom), 1),
                    round(float(r.bounds.right), 1),
                    round(float(r.bounds.top), 1),
                    r.width,
                    r.height,
                )
            lines_path = item.get('reference_lines')
            line_count = 0
            if lines_path and os.path.exists(lines_path):
                try:
                    line_count = len(gpd.read_file(lines_path))
                except Exception:
                    line_count = 0

            if r_key in seen_rasters:
                prev_idx, prev_name, prev_count = seen_rasters[r_key]
                if line_count > prev_count:
                    logger.warning(
                        f"DUPLICATE DETECTED: '{item_name}' ({line_count} lines) and '{prev_name}' ({prev_count} lines) "
                        f"share the exact same raster ({ortho_path}). Discarding '{prev_name}' and keeping '{item_name}'."
                    )
                    inputs_list[prev_idx] = item
                    seen_rasters[r_key] = (prev_idx, item_name, line_count)
                else:
                    logger.warning(
                        f"DUPLICATE DETECTED: '{item_name}' ({line_count} lines) shares the exact same raster as "
                        f"'{prev_name}' ({prev_count} lines). Discarding duplicate '{item_name}'."
                    )
            else:
                idx = len(inputs_list)
                inputs_list.append(item)
                seen_rasters[r_key] = (idx, item_name, line_count)
        except Exception as exc:
            logger.warning(f"Could not check raster bounds for {ortho_path}: {exc}")
            inputs_list.append(item)

    logger.info(f"Loaded {len(inputs_list)} unique dataset inputs after duplicate check.")
        
    output_dir = config.get('project', {}).get('output_dir', 'outputs')
    dataset_dir = os.path.join(output_dir, 'dataset')
    dataset_cfg = config.get('dataset', {})
    _remove_existing_dataset(dataset_dir, dataset_cfg.get('existing_output_policy', 'replace'))
    gsd_report_rows = []
    split_report_rows = []
        
    for item_idx, in_cfg in enumerate(inputs_list):
        input_name = in_cfg.get('name', f"input_{item_idx}")
        logger.info(f"\n=== [{input_name.upper()}] Starting dataset preparation ({item_idx + 1}/{len(inputs_list)}) ===")
        
        ortho_path = in_cfg.get('orthomosaic')
        roi_path = in_cfg.get('roi')
        lines_path = in_cfg.get('reference_lines')
        
        if not (ortho_path and roi_path and lines_path):
            raise ValueError(f"[{input_name}] Config must define 'orthomosaic', 'roi', and 'reference_lines'.")
            
        # 3. Read ROI and lines
        roi_gdf = read_roi(roi_path, layer=in_cfg.get('roi_layer'))
        lines_gdf = read_reference_lines(lines_path, layer=in_cfg.get('lines_layer'), id_field=in_cfg.get('id_field'))
        hard_negative_zones = None
        hard_negative_path = in_cfg.get('hard_negative_zones')
        if hard_negative_path:
            hard_negative_zones = read_roi(
                hard_negative_path,
                layer=in_cfg.get('hard_negative_zones_layer', 'hard_negative_zones'),
            )
        
        # 4. Check CRS and Reprojection needs
        crs_cfg = config.get('crs', {})
        auto_utm = crs_cfg.get('auto_utm', True)
        
        working_epsg = crs_cfg.get('working_epsg')
        if working_epsg is None:
            if auto_utm:
                logger.info(f"[{input_name}] Automatically estimating local UTM projection...")
                lon, lat = get_centroid_lon_lat(roi_gdf)
                working_epsg = estimate_utm_epsg(lon, lat)
                logger.info(f"[{input_name}] Estimated local metric UTM EPSG: {working_epsg}")
            else:
                # Fallback to raster CRS
                with rasterio.open(ortho_path) as tmp_r:
                    if tmp_r.crs.is_projected:
                        working_epsg = tmp_r.crs.to_epsg()
                        logger.info(f"[{input_name}] Using projected raster CRS EPSG: {working_epsg}")
                    else:
                        raise ValueError(f"[{input_name}] Working CRS is geographic and auto_utm is False. Please provide a metric working_epsg.")
                        
        # Reproject ROI and Lines
        roi_working = reproject_gdf(roi_gdf, working_epsg)
        lines_working = reproject_gdf(lines_gdf, working_epsg)
        if hard_negative_zones is not None:
            hard_negative_zones = reproject_gdf(hard_negative_zones, working_epsg)
        
        # Reproject raster if needed — nome exclusivo por PID + UUID para evitar WinError 32
        _unique_suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        reprojected_raster_path = os.path.join(
            output_dir, f'temp_raster_working_{input_name}_{_unique_suffix}.tif'
        )
        with rasterio.open(ortho_path) as tmp_r:
            raster_epsg = tmp_r.crs.to_epsg()

        rgb_bands = in_cfg.get('rgb_bands', [1, 2, 3])
        output_res_m = crs_cfg.get('output_resolution_m')
        preserve_gsd = crs_cfg.get('preserve_source_gsd', True)
        source_path = resolve_raster_source(ortho_path, in_cfg.get('orthomosaic_layer'))
        with rasterio.open(source_path) as source_raster:
            source_crs = source_raster.crs
            source_pixel_x = abs(source_raster.transform.a)
            source_pixel_y = abs(source_raster.transform.e)
            native_gsd_x, native_gsd_y = estimate_native_gsd_m(
                source_raster, roi_gdf, rasterio.crs.CRS.from_epsg(working_epsg),
            )
        native_gsd_mean = (native_gsd_x + native_gsd_y) / 2.0
        logger.info(
            "[%s] Native source GSD at ROI: x=%.4f m/pixel y=%.4f m/pixel mean=%.4f m/pixel",
            input_name, native_gsd_x, native_gsd_y, native_gsd_mean,
        )

        # Reprojection required if EPSG differs OR if we want to ensure specific resolution/alignment
        if raster_epsg != working_epsg:
            logger.info(f"[{input_name}] Reprojecting orthomosaic (ROI only) EPSG:{raster_epsg} -> EPSG:{working_epsg}...")
            reproject_raster(
                ortho_path,
                reprojected_raster_path,
                working_epsg,
                layer_name=in_cfg.get('orthomosaic_layer'),
                roi_gdf=roi_gdf,
                rgb_bands=rgb_bands,
                output_resolution_m=output_res_m if output_res_m else None,
                preserve_source_gsd=preserve_gsd,
                # Note: if output_res_m is None and preserve_gsd is True, 
                # reproject_raster (via prepare_working_raster) will estimate native GSD.
            )
            raster_input_path = reprojected_raster_path
        else:
            raster_input_path = ortho_path
            
        # 5. Clip lines to ROI
        logger.info(f"[{input_name}] Clipping reference lines to ROI boundary...")
        lines_working = clip_lines_to_roi(lines_working, roi_working)
        
        # 6. Read Raster properties
        with RasterReader(raster_input_path) as reader:
            raster_bounds = reader.bounds
            raster_crs = reader.crs
            raster_w = reader.width
            raster_h = reader.height
            gsd = reader.get_gsd()
            if raster_crs is None or not raster_crs.is_projected:
                raise ValueError(f"[{input_name}] Raster de trabalho sem CRS projetado.")
            if raster_w <= 0 or raster_h <= 0 or abs(reader.transform.a) <= 0 or abs(reader.transform.e) <= 0:
                raise ValueError(f"[{input_name}] Raster de trabalho com dimensões ou GSD inválidos.")
            logger.info(
                "[%s] Working raster: width=%d height=%d GSD X=%.6f GSD Y=%.6f CRS=%s",
                input_name, raster_w, raster_h, abs(reader.transform.a), abs(reader.transform.e), raster_crs,
            )
            
            strict_check = crs_cfg.get('strict_alignment_check', False)
            validate_spatial_alignment(raster_bounds, raster_crs, roi_working, lines_working, strict_check=strict_check)
            
            # 7. Generate tiles
            tile_cfg = config.get('tiling', {})
            tile_size = tile_cfg.get('tile_size_px', 512)
            overlap = tile_cfg.get('overlap_px', 128)
            min_valid = tile_cfg.get('min_valid_fraction', 0.60)
            empty_ratio = tile_cfg.get('include_empty_tiles_ratio', 0.15)
            negative_cfg = config.get('negative_sampling', {})
            
            generator = TileGenerator(
                tile_size_px=tile_size,
                overlap_px=overlap,
                min_valid_fraction=min_valid,
                include_empty_tiles_ratio=empty_ratio,
                negative_sampling_mode=negative_cfg.get('mode', 'explicit_zones'),
                negative_zone_min_fraction=negative_cfg.get('min_zone_fraction', 0.80),
                min_negative_line_distance_m=negative_cfg.get('min_line_distance_m', 0.0),
            )
            
            # Assign group field for spatial split if configured
            group_field = in_cfg.get('group_field') or config.get('split', {}).get('group_field')
            
            if group_field and group_field in roi_working.columns:
                # We will generate tiles separately per ROI polygon to map group_ids
                all_tiles = []
                for idx, roi_row in roi_working.iterrows():
                    sub_roi = gpd.GeoDataFrame([roi_row], crs=roi_working.crs)
                    gid = roi_row[group_field]
                    sub_tiles = generator.generate_tiles(raster_w, raster_h, reader.transform, sub_roi, group_id=gid)
                    all_tiles.extend(sub_tiles)
            else:
                all_tiles = generator.generate_tiles(raster_w, raster_h, reader.transform, roi_working)
                
            logger.info(f"[{input_name}] Generated {len(all_tiles)} candidate tiles within valid ROI.")
            
            # Filter tiles based on line intersection and empty ratio
            filtered_tiles = generator.filter_tiles(
                all_tiles,
                lines_working,
                hard_negative_zones_gdf=hard_negative_zones,
                seed=config.get('project', {}).get('seed', 42),
            )
            
            # 8. Spatial Split
            split_cfg = config.get('split', {})
            train_tiles, val_tiles, test_tiles = split_dataset_groups(
                filtered_tiles,
                strategy=split_cfg.get('strategy', 'spatial_group'),
                group_field=group_field,
                train_fraction=split_cfg.get('train_fraction', 0.70),
                val_fraction=split_cfg.get('val_fraction', 0.15),
                test_fraction=split_cfg.get('test_fraction', 0.15),
                seed=config.get('project', {}).get('seed', 42)
            )
            split_sets = {'train': train_tiles, 'val': val_tiles, 'test': test_tiles}
            split_report_rows.append({
                'input_name': input_name,
                'total_tiles': len(filtered_tiles),
                'total_groups': len({tile['group_id'] for tile in filtered_tiles}),
                **{name + '_groups': len({tile['group_id'] for tile in tiles}) for name, tiles in split_sets.items()},
                **{f'{name}_tiles': len(tiles) for name, tiles in split_sets.items()},
                **{f'positive_{name}': _count_tile_types(tiles)[0] for name, tiles in split_sets.items()},
                **{f'empty_{name}': _count_tile_types(tiles)[1] for name, tiles in split_sets.items()},
            })
            gsd_report_rows.append({
                'input_name': input_name, 'raster_path': ortho_path,
                'source_crs': str(source_crs), 'target_crs': str(raster_crs),
                'source_pixel_size_x': source_pixel_x, 'source_pixel_size_y': source_pixel_y,
                'native_gsd_x_m': native_gsd_x, 'native_gsd_y_m': native_gsd_y,
                'native_gsd_mean_m': native_gsd_mean, 'output_gsd_m': gsd,
                'output_width': raster_w, 'output_height': raster_h,
                'roi_area_m2': float(roi_working.geometry.area.sum()),
            })
            
            # 9. Target parameters
            target_cfg = config.get('targets', {})
            row_w = target_cfg.get('row_width_m', 1.20)
            center_w = target_cfg.get('center_width_m', 0.20)
            sample_step = target_cfg.get('orientation_sample_step_m', 0.10)
            # rgb_bands already defined above
            
            # 10. Generate and save patches
            for split_name, split_tiles in [('train', train_tiles), ('val', val_tiles), ('test', test_tiles)]:
                split_path = os.path.join(dataset_dir, split_name)
                os.makedirs(split_path, exist_ok=True)
                
                logger.info(f"[{input_name}] Writing {split_name} split NPZ files ({len(split_tiles)} tiles)...")
                
                for idx, tile in enumerate(split_tiles):
                    col = tile['col_off']
                    row = tile['row_off']
                    w = tile['width']
                    h = tile['height']
                    
                    # Window reading
                    win = rasterio.windows.Window(col, row, w, h)
                    img_patch = reader.read_normalized(rgb_bands, window=win)
                    
                    # Check nodata mask
                    valid_pixels = reader.src.dataset_mask(window=win) > 0
                    raster_nodata_mask = ~valid_pixels
                        
                    # Rasterize targets
                    tile_transform = rasterio.windows.transform(win, reader.transform)
                    r_mask, c_mask, o_sin, o_cos, v_mask = rasterize_targets(
                        tile_window=tile,
                        tile_transform=tile_transform,
                        tile_size_px=tile_size,
                        lines_gdf=lines_working,
                        roi_gdf=roi_working,
                        raster_nodata_mask=raster_nodata_mask,
                        row_width_m=row_w,
                        center_width_m=center_w,
                        orientation_sample_step_m=sample_step,
                        gsd=gsd
                    )
                    
                    # Save as NPZ with dataset-specific name prefix
                    npz_name = f"{input_name}_tile_{idx:05d}_r{row:05d}_c{col:05d}.npz"
                    npz_path = os.path.join(split_path, npz_name)
                    np.savez_compressed(
                        npz_path,
                        image=img_patch,
                        row_mask=r_mask,
                        center_mask=c_mask,
                        orientation_sin=o_sin,
                        orientation_cos=o_cos,
                        valid_mask=v_mask,
                        group_id=str(tile['group_id']),
                        input_name=input_name,
                        sampling_class=tile.get('sampling_class', 'positive'),
                        gsd_m=np.float32(gsd),
                    )
                    
        # Clean up temp working raster if created
        if raster_input_path == reprojected_raster_path and os.path.exists(reprojected_raster_path):
            try:
                os.remove(reprojected_raster_path)
                logger.info(f"[{input_name}] Cleaned up temp raster: {reprojected_raster_path}")
            except OSError as e:
                logger.warning(f"[{input_name}] Could not remove temp raster: {e}")
            
    def write_csv(path, rows):
        if not rows:
            return
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(os.path.join(output_dir, 'dataset_gsd_report.csv'), gsd_report_rows)
    write_csv(os.path.join(output_dir, 'dataset_split_report.csv'), split_report_rows)
    gsds = [row['output_gsd_m'] for row in gsd_report_rows]
    if gsds:
        gsd_min, gsd_max, gsd_median = min(gsds), max(gsds), float(np.median(gsds))
        variation = (gsd_max - gsd_min) / gsd_median if gsd_median else 0.0
        logger.info("GSD report: min=%.6f max=%.6f median=%.6f variation=%.3f", gsd_min, gsd_max, gsd_median, variation)
        if variation > crs_cfg.get('max_gsd_variation_ratio', 0.25):
            message = f"Critical GSD variation: {variation:.3f}"
            if crs_cfg.get('fail_on_excessive_gsd_variation', False):
                raise ValueError(message)
            logger.warning(message)
    manifest = {
        'created_at': __import__('datetime').datetime.now().isoformat(),
        'inputs': gsd_report_rows,
        'splits': split_report_rows,
        'config': args.config,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, default=str).encode('utf-8')
    manifest['sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    with open(os.path.join(dataset_dir, 'manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, default=str)
    logger.info("Dataset preparation successfully completed!")

if __name__ == '__main__':
    main()
