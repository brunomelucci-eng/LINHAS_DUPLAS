import argparse
import sys
import os
import uuid
import json
import time
from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.windows
from rasterio.features import rasterize
import logging
from typing import Optional

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.seed import set_seed
from src.logging_utils import setup_logging, configure_runtime_threads, StageTimer
from src.data import (
    RasterReader, read_roi, read_reference_lines, reproject_raster,
    estimate_utm_epsg, get_centroid_lon_lat, reproject_gdf,
    clip_lines_to_roi
)
from src.inference.predictor import RowPredictor
from src.inference.sliding_window import predict_full_raster
from src.data.raster_reprojection import (
    resolve_raster_source,
    prepare_working_raster_cached,
)
from src.postprocessing import (
    clean_mask, perform_thinning, skeleton_to_graph_and_branches
)
from src.postprocessing.postprocess_pipeline import run_postprocessing
from src.postprocessing.output_products import (
    build_comparison_report,
    config_hash,
    create_output_layout,
    persist_two_stage_vector_outputs,
    select_production_lines,
    utc_now_iso,
    write_compatibility_vector_product,
    write_manifest,
    write_reports,
)
from src.geospatial import pixel_to_world, populate_metadata
from src.metrics import (
    compute_binary_metrics, compute_cldice, compute_geometric_metrics,
    compute_topological_metrics, compile_metrics_report
)
from src.visualization import (
    save_prediction_overlay, save_probability_maps, save_orientation_quiver,
    generate_html_report
)

logger = logging.getLogger(__name__)


def _write_debug_raster(path, array, transform, crs, dtype=None, threshold=None):
    """Persist a 2-D diagnostic array without materialising memmaps in RAM."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.stem}.partial-{uuid.uuid4().hex}{target.suffix}"
    )
    target_dtype = dtype or ('uint8' if np.asarray(array).dtype == np.uint8 else 'float32')
    height, width = array.shape
    dtype_kind = np.dtype(target_dtype).kind
    profile = {
        'driver': 'GTiff', 'height': height, 'width': width, 'count': 1,
        'dtype': target_dtype, 'crs': crs, 'transform': transform,
        'compress': 'DEFLATE', 'tiled': True, 'blockxsize': 512, 'blockysize': 512,
        'predictor': 3 if dtype_kind == 'f' else 2,
        'num_threads': os.environ.get('GDAL_NUM_THREADS', '2'),
        'BIGTIFF': 'IF_SAFER', 'nodata': 0,
    }
    try:
        with rasterio.open(temporary, 'w', **profile) as dst:
            for row in range(0, height, 1024):
                rows = min(1024, height - row)
                window = rasterio.windows.Window(0, row, width, rows)
                stripe = np.asarray(array[row:row + rows])
                if threshold is not None:
                    stripe = stripe >= threshold
                dst.write(np.asarray(stripe, dtype=target_dtype), 1, window=window)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_tile_grid(path, windows, transform, crs):
    """Export every inference tile so seams can be inspected in QGIS."""
    from shapely.geometry import box
    geometries = []
    for index, (col, row, width, height) in enumerate(windows):
        bounds = rasterio.windows.bounds(rasterio.windows.Window(col, row, width, height), transform)
        geometries.append({'tile_id': index, 'geometry': box(*bounds)})
    gpd.GeoDataFrame(geometries, crs=crs).to_file(path, layer='tile_grid', driver='GPKG')


_REFINEMENT_DEBUG_LAYERS = (
    'linhas_originais',
    'linhas_suavizadas',
    'trechos_substituidos_pela_irma',
    'pares_reconstruidos',
    'extensoes_terminais',
    'linhas_rejeitadas',
    'linhas_para_revisao',
    'linhas_finais',
)


def _write_refinement_debug_layers(
    path,
    debug_gdf,
    *,
    originals_gdf=None,
    finals_gdf=None,
):
    """Write each refinement action category as a dedicated GeoPackage layer."""
    if debug_gdf.empty and originals_gdf is None and finals_gdf is None:
        return
    wrote_layer = False
    for layer_name in _REFINEMENT_DEBUG_LAYERS:
        if layer_name == 'linhas_originais' and originals_gdf is not None:
            layer_gdf = gpd.GeoDataFrame(
                {
                    'line_index': np.arange(len(originals_gdf)),
                    'action': 'original',
                },
                geometry=originals_gdf.geometry.to_numpy(copy=False),
                crs=originals_gdf.crs,
            )
        elif layer_name == 'linhas_finais' and finals_gdf is not None:
            layer_gdf = gpd.GeoDataFrame(
                {
                    'line_index': np.arange(len(finals_gdf)),
                    'action': 'final',
                    'quality_after': finals_gdf['quality_after'].to_numpy(),
                    'review_required': finals_gdf['review_required'].to_numpy(),
                },
                geometry=finals_gdf.geometry.to_numpy(copy=False),
                crs=finals_gdf.crs,
            )
        elif layer_name == 'linhas_para_revisao' and finals_gdf is not None:
            review = finals_gdf.loc[finals_gdf['review_required'].fillna(False)].copy()
            layer_gdf = gpd.GeoDataFrame(
                {
                    'line_index': review.index.to_numpy(),
                    'action': 'review_required',
                    'quality_after': review['quality_after'].to_numpy(),
                },
                geometry=review.geometry.to_numpy(copy=False),
                crs=review.crs,
            )
        elif not debug_gdf.empty and 'debug_layer' in debug_gdf.columns:
            layer_gdf = debug_gdf.loc[debug_gdf['debug_layer'].eq(layer_name)].copy()
            layer_gdf = layer_gdf.drop(columns=['debug_layer'])
        else:
            continue
        if layer_gdf.empty:
            continue
        layer_gdf = layer_gdf.dropna(axis=1, how='all')
        layer_gdf.to_file(
            path,
            layer=layer_name,
            driver='GPKG',
            mode='a' if wrote_layer else 'w',
        )
        wrote_layer = True

def main():
    parser = argparse.ArgumentParser(description="Predict crop rows on a new orthomosaic.")
    parser.add_argument('--config', type=str, default='configs/unet_resnet34.yaml', help="Path to config YAML.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best model checkpoint.")
    parser.add_argument('--orthomosaic', type=str, required=True, help="Path to orthomosaic TIFF.")
    parser.add_argument('--roi', type=str, required=True, help="Path to ROI GPKG.")
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help=(
            "Legacy single-layer compatibility output. The two canonical "
            "products are always written under outputs/runs/<run_id>/vectors."
        ),
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    from src.training.threshold_calibration import apply_checkpoint_threshold
    apply_checkpoint_threshold(config, args.checkpoint)
    output_dir = config.get('project', {}).get('output_dir', 'outputs')

    # --- Run isolation (P0-04) -------------------------------------------
    run_id = f'{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}'
    pid    = os.getpid()
    run_dir = os.path.join(output_dir, 'runs', run_id)
    os.makedirs(run_dir, exist_ok=True)
    output_layout = create_output_layout(
        run_dir,
        os.path.splitext(os.path.basename(args.roi))[0],
        config,
    )
    requested_output = os.path.abspath(args.output)
    canonical_raw_output = os.path.abspath(output_layout.post_inference_path)
    canonical_output = os.path.abspath(output_layout.postprocessing_path)
    compatibility_alias_required = requested_output != canonical_output
    if requested_output == canonical_raw_output:
        raise ValueError("The legacy --output path cannot be the canonical raw product.")
    outputs_overwrite = bool(
        config.get('outputs', {}).get('overwrite_existing', False)
    )
    if (
        compatibility_alias_required
        and os.path.exists(requested_output)
        and not outputs_overwrite
    ):
        raise FileExistsError(
            f"Refusing to reuse an existing legacy --output product: {requested_output}"
        )

    setup_logging(run_dir, run_id=run_id, pid=pid)

    # Lock file to prevent two simultaneous runs on the same output
    runtime_cfg = config.get('runtime', {})
    lock_path = os.path.join(output_dir, '.prediction.lock')
    if runtime_cfg.get('prevent_concurrent_runs', True):
        if os.path.exists(lock_path):
            try:
                with open(lock_path) as lf:
                    lock_info = json.load(lf)
                other_pid = lock_info.get('pid')
                # Check if the locking process is still alive
                import psutil
                if other_pid and psutil.pid_exists(other_pid):
                    raise SystemExit(
                        f"Another prediction is already running (PID {other_pid}, "
                        f"run_id={lock_info.get('run_id')}). "
                        f"Delete {lock_path!r} manually if that process is no longer running."
                    )
                else:
                    logger.warning("Stale lock found (PID %s no longer exists). Removing.", other_pid)
                    os.remove(lock_path)
            except SystemExit:
                raise
            except Exception as exc:
                policy = runtime_cfg.get('lock_validation_policy', 'fail')
                if policy == 'fail':
                    raise RuntimeError(
                        f"Lock validation failed (lock_validation_policy='fail'): "
                        f"Could not read/validate the existing lock file {lock_path!r}. "
                        f"Error: {exc}"
                    )
                else:
                    logger.warning("Could not validate existing lock: %s. Proceeding.", exc)

        lock_info = {
            'pid': pid, 'run_id': run_id,
            'checkpoint': args.checkpoint,
            'orthomosaic': args.orthomosaic,
            'roi': args.roi,
            'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        with open(lock_path, 'w') as lf:
            json.dump(lock_info, lf, indent=2)

    # Apply CPU/GDAL thread limits (P1-05)
    configure_runtime_threads(config)

    # Configure PyTorch (kept for backward compat)
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        num_threads = config.get('runtime', {}).get('max_cpu_threads',
                      config.get('inference', {}).get('num_threads', 4))
        torch.set_num_threads(num_threads)
        logger.info("Using CPU for inference. PyTorch CPU threads: %d", num_threads)

    set_seed(config.get('project', {}).get('seed', 42))
    logger.info("[run=%s pid=%d] Prediction pipeline initialised.", run_id, pid)

    # --- Main pipeline wrapped in try/finally for guaranteed cleanup -----------
    temp_raster_path = None
    raster_was_warped = False
    row_prob = None
    center_prob = None
    orientation_sin = None
    orientation_cos = None
    clean_center_mask = None
    skeleton_raster = None
    row_prob_down = None
    center_prob_down = None
    pipeline_started = time.perf_counter()
    try:
        # 1. Load ROI and estimate metric CRS
        in_cfg = config.get('input', {})
        orthomosaic_layer = in_cfg.get('orthomosaic_layer')
        roi_gdf = read_roi(args.roi, layer=in_cfg.get('roi_layer'))
        roi_gdf['geometry'] = roi_gdf.geometry.make_valid()
        resolved_orthomosaic = resolve_raster_source(
            args.orthomosaic,
            layer_name=orthomosaic_layer,
        )

        crs_cfg = config.get('crs', {})
        working_epsg = crs_cfg.get('working_epsg')
        if working_epsg is None:
            if crs_cfg.get('auto_utm', True):
                lon, lat = get_centroid_lon_lat(roi_gdf)
                working_epsg = estimate_utm_epsg(lon, lat)
                logger.info("Estimated local UTM EPSG: %d", working_epsg)
            else:
                with rasterio.open(resolved_orthomosaic) as tmp_r:
                    if tmp_r.crs.is_projected:
                        working_epsg = tmp_r.crs.to_epsg()
                    else:
                        raise ValueError(
                            "Automatic UTM estimation is disabled and orthomosaic "
                            "has geographic coordinates."
                        )

        roi_working = reproject_gdf(roi_gdf, working_epsg)

        # 2. Warp Orthomosaic if CRS differs — only ROI window (P0-03)
        with rasterio.open(resolved_orthomosaic) as tmp_r:
            raster_epsg = tmp_r.crs.to_epsg()

        if raster_epsg != working_epsg:
            from rasterio.crs import CRS as _CRS
            logger.info(
                "Warping orthomosaic EPSG:%d → EPSG:%d (ROI window only)...",
                raster_epsg, working_epsg,
            )
            with StageTimer('warp_orthomosaic', disk_path=run_dir):
                raster_working_path, cache_hit = prepare_working_raster_cached(
                    args.orthomosaic,
                    os.path.join(output_dir, 'cache', 'working_rasters'),
                    target_crs=_CRS.from_epsg(working_epsg),
                    roi_gdf=roi_gdf,
                    roi_buffer_m=crs_cfg.get('roi_warp_buffer_m', 10.0),
                    rgb_bands=config.get('input', {}).get('rgb_bands', [1, 2, 3]),
                    layer_name=config.get('input', {}).get('orthomosaic_layer'),
                    compression=crs_cfg.get('compression'),
                    warp_mem_limit_mb=crs_cfg.get('warp_mem_limit_mb', 256),
                )
            raster_was_warped = True
            logger.info("Working raster cache_hit=%s path=%s", cache_hit, raster_working_path)
        else:
            raster_working_path = args.orthomosaic
        
        # 3. Read image and run predictor
        predictor = RowPredictor(config, args.checkpoint)

        rgb_bands = in_cfg.get('rgb_bands', [1, 2, 3])

        reader_layer = None if raster_was_warped else orthomosaic_layer
        with RasterReader(raster_working_path, layer=reader_layer) as reader:
            transform = reader.transform
            gsd = reader.get_gsd()
            raster_crs = reader.crs

            if reader.crs is None or reader.crs.is_geographic:
                raise ValueError(
                    "The working raster must use a projected metric CRS before inference. "
                    f"Received CRS={reader.crs}."
                )

            minx, miny, maxx, maxy = roi_working.total_bounds
            raw_window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform)

            col_off = max(0, min(reader.width - 1, int(np.floor(raw_window.col_off))))
            row_off = max(0, min(reader.height - 1, int(np.floor(raw_window.row_off))))
            w = max(1, min(reader.width - col_off, int(np.ceil(raw_window.width))))
            h = max(1, min(reader.height - row_off, int(np.ceil(raw_window.height))))
            roi_window = rasterio.windows.Window(col_off, row_off, w, h)
            transform_roi = rasterio.windows.transform(roi_window, transform)
            H, W = int(roi_window.height), int(roi_window.width)

            logger.info(
                "[run=%s] ROI window: %dx%d px (%.1f Mpx)",
                run_id,
                W,
                H,
                W * H / 1e6,
            )

            inference_started = time.perf_counter()
            with StageTimer('sliding_window_inference', disk_path=run_dir):
                probs = predict_full_raster(
                    predictor,
                    reader,
                    config,
                    roi_window=roi_window,
                    roi_geometry=roi_working.geometry.union_all(),
                    temp_root=run_dir,
                )
                row_prob = probs.row
                center_prob = probs.center
                orientation_sin = probs.orientation_sin
                orientation_cos = probs.orientation_cos
            inference_seconds = time.perf_counter() - inference_started

        debug_cfg = config.get('debug', {})
        debug_enabled = bool(debug_cfg.get('enabled', False))
        if debug_enabled:
            roi_working.to_file(os.path.join(run_dir, 'roi_effective.gpkg'), layer='roi_effective', driver='GPKG')
        # Required reproducibility product; it is not a debug-only artifact.
        with StageTimer('probability_center_export', disk_path=run_dir):
            _write_debug_raster(
                output_layout.probability_center_path,
                center_prob,
                transform_roi,
                raster_crs,
            )
        if debug_enabled and debug_cfg.get('save_weights', False):
            probs.mosaic.export_weights_geotiff(os.path.join(run_dir, 'mosaic_weights.tif'), transform_roi, raster_crs)
        if debug_enabled and debug_cfg.get('save_tile_grid', False):
            _write_tile_grid(os.path.join(run_dir, 'tile_grid.gpkg'), probs.tile_windows, transform_roi, raster_crs)

        # Accumulators and blending weights are no longer needed after the final
        # probability/debug exports. Releasing them here is essential for very
        # large ROIs before allocating masks and Python graph coordinates.
        if probs.mosaic is not None and hasattr(probs.mosaic, 'release_accumulators'):
            probs.mosaic.release_accumulators()

        # Post-processing must only start after inference completed successfully.
        if center_prob is None:
            raise RuntimeError("Inference completed without a center probability raster.")

        with StageTimer('mask_cleaning', disk_path=run_dir):
            clean_center_mask = clean_mask(center_prob, gsd, config)
        if debug_enabled and debug_cfg.get('save_masks', False):
            with StageTimer('debug_mask_export', disk_path=run_dir):
                _write_debug_raster(
                    os.path.join(run_dir, 'binary_mask_raw.tif'),
                    center_prob,
                    transform_roi,
                    raster_crs,
                    'uint8',
                    threshold=config.get('postprocessing', {}).get('center_threshold_high', 0.28),
                )
                _write_debug_raster(os.path.join(run_dir, 'binary_mask_cleaned.tif'), clean_center_mask, transform_roi, raster_crs, 'uint8')

        with StageTimer('thinning', disk_path=run_dir):
            skeleton_raster = perform_thinning(clean_center_mask, config)
        if debug_enabled and debug_cfg.get('save_skeleton', False):
            with StageTimer('debug_skeleton_export', disk_path=run_dir):
                _write_debug_raster(os.path.join(run_dir, 'skeleton.tif'), skeleton_raster, transform_roi, raster_crs, 'uint8')

        # The cleaned mask is only required later when reference-line metrics
        # are enabled. Avoid retaining another full-size uint8 raster otherwise.
        if not in_cfg.get('reference_lines'):
            clean_center_mask = None

        with StageTimer('skeleton_to_graph', disk_path=run_dir):
            branches, _junctions, _endpoints = skeleton_to_graph_and_branches(
                skeleton_raster,
                config,
            )

        # 9. Convert the unmodified skeleton branches to metric LineStrings.
        # Spur removal, gap bridging, loop cleanup and fitting now live inside
        # run_postprocessing so the saved raw product can reproduce the final.
        logger.info("Transforming raw skeleton branches to metric coordinates...")
        branches_world = []
        with StageTimer('pixel_to_world', disk_path=run_dir):
            for path in branches:
                if len(path) >= 2:
                    world_pts = pixel_to_world(path, transform_roi)
                    branches_world.append(np.array(world_pts))

        # 10. Build the strict post-inference GeoDataFrame without corrections.
        from shapely.geometry import LineString
        raw_vector_gdf = gpd.GeoDataFrame(
            geometry=[LineString(points) for points in branches_world if len(points) >= 2],
            crs=working_epsg,
        )

        # 11. Persist raw first; only then run and persist the robust engine.
        export_epsg = config.get('crs', {}).get('export_epsg') or raster_epsg
        output_crs = (
            f"EPSG:{int(export_epsg)}" if export_epsg is not None else str(raw_vector_gdf.crs)
        )
        two_stage_started = time.perf_counter()
        stage_timings = {}
        try:
            with StageTimer('two_stage_vector_outputs', disk_path=run_dir):
                persisted = persist_two_stage_vector_outputs(
                    raw_vector_gdf,
                    center_prob,
                    transform_roi,
                    roi_working,
                    config,
                    layout=output_layout,
                    run_id=run_id,
                    checkpoint=os.path.basename(args.checkpoint),
                    debug=debug_enabled,
                    export_epsg=export_epsg,
                    postprocess_fn=run_postprocessing,
                    finalizer=lambda frame: populate_metadata(
                        frame,
                        config,
                        model_name=config.get('model', {}).get('architecture', 'unet'),
                        checkpoint_name=os.path.basename(args.checkpoint),
                        source_id=os.path.basename(args.orthomosaic),
                    ),
                    timings=stage_timings,
                )
        except Exception as error:
            # persist_two_stage_vector_outputs records failures that happen
            # after the raw product is durable.  Keep a fallback manifest for
            # validation/I/O errors that occur before that boundary.
            if not output_layout.manifest_path.exists():
                failure_manifest = {
                    'run_id': run_id,
                    'status': 'postprocessing_failed',
                    'post_inference': {
                        'complete': output_layout.post_inference_path.exists(),
                        'path': os.path.relpath(output_layout.post_inference_path, run_dir),
                        'layer': 'post_inference_lines',
                    },
                    'postprocessing': {
                        'complete': False,
                        'path': os.path.relpath(output_layout.postprocessing_path, run_dir),
                        'layer': 'final_lines',
                    },
                    'error_type': type(error).__name__,
                    'error': str(error),
                    'checkpoint': os.path.abspath(args.checkpoint),
                    'config': os.path.abspath(args.config),
                    'config_hash': config_hash(config),
                    'timing_seconds': {
                        'inference': inference_seconds,
                        'postprocessing': stage_timings.get('postprocessing'),
                        'vector_products_total': time.perf_counter() - two_stage_started,
                    },
                    'created_at': utc_now_iso(),
                }
                write_manifest(
                    output_layout,
                    failure_manifest,
                    overwrite=bool(
                        config.get('outputs', {}).get('overwrite_existing', False)
                    ),
                )
            raise
        vector_products_seconds = time.perf_counter() - two_stage_started
        postprocess_seconds = stage_timings['postprocessing']
        raw_gdf = persisted.raw_lines
        postprocess_result = persisted.postprocess_result
        final_gdf = persisted.final_lines
        pred_gdf = postprocess_result.final_lines
        logger.info("Postprocess metrics: %s", dict(postprocess_result.metrics))

        if debug_enabled:
            debug_frames = []
            for frame, layer_name in (
                (postprocess_result.repaired_segments, 'trechos_substituidos_pela_irma'),
                (postprocess_result.reconstructed_pairs, 'pares_reconstruidos'),
                (postprocess_result.terminal_extensions, 'extensoes_terminais'),
                (postprocess_result.rejected_lines, 'linhas_rejeitadas'),
            ):
                if frame.empty:
                    continue
                layer_frame = frame.copy()
                layer_frame['debug_layer'] = layer_name
                debug_frames.append(layer_frame)
            postprocess_debug = (
                gpd.GeoDataFrame(
                    pd.concat(debug_frames, ignore_index=True, sort=False),
                    geometry='geometry',
                    crs=pred_gdf.crs,
                )
                if debug_frames
                else gpd.GeoDataFrame(geometry=[], crs=pred_gdf.crs)
            )
            _write_refinement_debug_layers(
                os.path.join(run_dir, 'double_row_refinement_debug.gpkg'),
                postprocess_debug,
                originals_gdf=postprocess_result.raw_lines,
                finals_gdf=pred_gdf,
            )

        # Compatibility alias for existing commands. Canonical products remain
        # under run_dir/vectors and are never overwritten.
        compatibility_output = {
            'path': requested_output,
            'layer': 'predicted_rows',
            'status': 'canonical' if not compatibility_alias_required else 'pending',
        }
        if compatibility_alias_required:
            production_gdf = select_production_lines(final_gdf)
            with StageTimer('export_compatibility_alias', disk_path=run_dir):
                write_compatibility_vector_product(
                    production_gdf,
                    requested_output,
                    layer_name='predicted_rows',
                    overwrite=outputs_overwrite,
                    export_epsg=export_epsg,
                )
            compatibility_output['status'] = 'created'
            compatibility_output['line_count'] = len(production_gdf)

        comparison_report = build_comparison_report(
            raw_gdf,
            postprocess_result,
            config,
            run_id=run_id,
            inference_seconds=inference_seconds,
            postprocess_seconds=postprocess_seconds,
        )
        manifest = {
            'run_id': run_id,
            'status': str(postprocess_result.metrics.get('status', 'completed')),
            'post_inference': {
                'complete': True,
                'path': os.path.relpath(output_layout.post_inference_path, run_dir),
                'layer': 'post_inference_lines',
                'line_count': len(raw_gdf),
            },
            'postprocessing': {
                'complete': True,
                'path': os.path.relpath(output_layout.postprocessing_path, run_dir),
                'layer': 'final_lines',
                'line_count': len(final_gdf),
                'manual_review_count': len(postprocess_result.manual_review),
                'rejected_count': len(postprocess_result.rejected_lines),
            },
            'probability_center': os.path.relpath(
                output_layout.probability_center_path, run_dir
            ),
            'reports': {
                'postprocess': os.path.relpath(
                    output_layout.postprocess_report_path, run_dir
                ),
                'comparison': os.path.relpath(
                    output_layout.comparison_report_path, run_dir
                ),
            },
            'crs': output_crs,
            'checkpoint': os.path.abspath(args.checkpoint),
            'config': os.path.abspath(args.config),
            'config_hash': config_hash(config),
            'timing_seconds': {
                'inference': inference_seconds,
                'postprocessing': postprocess_seconds,
                'vector_products_total': vector_products_seconds,
            },
            'compatibility_output': compatibility_output,
            'created_at': utc_now_iso(),
        }
        with StageTimer('two_stage_reports', disk_path=run_dir):
            write_reports(
                output_layout,
                comparison_report,
                manifest=manifest,
                overwrite=bool(
                    config.get('outputs', {}).get('overwrite_existing', False)
                ),
            )
        
        # 20. Compile evaluation report if reference lines are available
        ref_gdf = None
        ref_path = in_cfg.get('reference_lines')
        if ref_path and os.path.exists(ref_path):
            try:
                ref_gdf = read_reference_lines(ref_path, layer=in_cfg.get('lines_layer'), id_field=in_cfg.get('id_field'))
                ref_gdf = reproject_gdf(ref_gdf, working_epsg)
                ref_gdf = clip_lines_to_roi(ref_gdf, roi_working)
            except Exception as e:
                logger.warning(f"Could not load reference lines for metric comparison: {e}")
                
        with StageTimer('metrics', disk_path=run_dir):
            if ref_gdf is not None:
                logger.info("Computing evaluation metrics against reference rows...")
                target_cfg = config.get('targets', {})
                row_w = target_cfg.get('row_width_m', 1.20)
                center_w = target_cfg.get('center_width_m', 0.20)

                # Rasterize GT geometries
                row_shapes_gt = [(line.buffer(row_w / 2.0), 1) for line in ref_gdf.geometry]
                center_shapes_gt = [(line.buffer(center_w / 2.0), 1) for line in ref_gdf.geometry]

                row_mask_gt = rasterize(row_shapes_gt, out_shape=(H, W), transform=transform_roi, fill=0, dtype=np.uint8)
                center_mask_gt = rasterize(center_shapes_gt, out_shape=(H, W), transform=transform_roi, fill=0, dtype=np.uint8)

                if row_prob is not None:
                    row_pred_bin = (row_prob >= config.get('postprocessing', {}).get('row_threshold', 0.5)).astype(np.uint8)
                    seg_metrics = compute_binary_metrics(row_pred_bin, row_mask_gt)
                else:
                    seg_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                    seg_metrics['row_metrics_status'] = 'not_computed'
                seg_metrics['cldice'] = compute_cldice(clean_center_mask, center_mask_gt)

                geom_metrics = compute_geometric_metrics(final_gdf, ref_gdf)
            else:
                seg_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'cldice': 0.0}
                geom_metrics = {'mean_symmetric_distance_m': float('nan'), 'hausdorff_distance_m': float('nan'), 'hd95_m': float('nan')}

            topo_metrics = compute_topological_metrics(final_gdf)

            # Compile JSON validation report inside run_dir
            compile_metrics_report(seg_metrics, geom_metrics, topo_metrics, run_dir)
        
        # 21. Generate diagnostic visual plots
        logger.info("Saving diagnostic visual overlays and heatmaps...")
        
        with StageTimer('visualization', disk_path=run_dir):
            # Calculate dynamic downsample factor to prevent MemoryError in matplotlib
            downsample_factor = max(1, max(H, W) // 2048)

            # Reopen reader briefly just for reading the diagnostic image (BUG P0-21)
            with RasterReader(raster_working_path, layer=reader_layer) as vis_reader:
                if downsample_factor > 1:
                    logger.info(f"Rasters are large ({W}x{H}). Downsampling by {downsample_factor}x for visual diagnostics...")
                    vis_h = max(1, H // downsample_factor)
                    vis_w = max(1, W // downsample_factor)
                    vis_img = vis_reader.read_normalized(rgb_bands, window=roi_window, out_shape=(vis_h, vis_w))

                    from rasterio.transform import Affine
                    scaled_transform = transform_roi * Affine.scale(downsample_factor, downsample_factor)

                    save_prediction_overlay(vis_img, final_gdf, ref_gdf, scaled_transform, os.path.join(run_dir, 'overlays.png'))

                    row_prob_down = row_prob[::downsample_factor, ::downsample_factor] if row_prob is not None else None
                    center_prob_down = center_prob[::downsample_factor, ::downsample_factor]
                    save_probability_maps(row_prob_down, center_prob_down, os.path.join(run_dir, 'probabilities.png'))
                else:
                    vis_img = vis_reader.read_normalized(rgb_bands, window=roi_window)
                    save_prediction_overlay(vis_img, final_gdf, ref_gdf, transform_roi, os.path.join(run_dir, 'overlays.png'))
                    save_probability_maps(row_prob, center_prob, os.path.join(run_dir, 'probabilities.png'))

            generate_html_report(run_dir)

    except Exception as pipeline_error:
        raw_complete = output_layout.post_inference_path.exists()
        final_complete = output_layout.postprocessing_path.exists()
        if not raw_complete:
            failure_status = 'inference_failed'
        elif not final_complete:
            failure_status = 'postprocessing_failed'
        else:
            failure_status = 'pipeline_failed_after_vector_products'
        failure_manifest = {}
        if output_layout.manifest_path.exists():
            try:
                failure_manifest = json.loads(
                    output_layout.manifest_path.read_text(encoding='utf-8')
                )
            except (OSError, ValueError, json.JSONDecodeError):
                failure_manifest = {}
        failure_manifest.update({
            'run_id': run_id,
            'status': failure_status,
            'post_inference': {
                'complete': raw_complete,
                'path': os.path.relpath(output_layout.post_inference_path, run_dir),
                'layer': 'post_inference_lines',
            },
            'postprocessing': {
                'complete': final_complete,
                'path': os.path.relpath(output_layout.postprocessing_path, run_dir),
                'layer': 'final_lines',
            },
            'error_type': type(pipeline_error).__name__,
            'error': str(pipeline_error),
            'checkpoint': os.path.abspath(args.checkpoint),
            'config': os.path.abspath(args.config),
            'config_hash': config_hash(config),
            'crs': locals().get('output_crs'),
            'timing_seconds': {
                'inference': locals().get('inference_seconds'),
                'postprocessing': locals().get(
                    'postprocess_seconds',
                    locals().get('stage_timings', {}).get('postprocessing'),
                ),
                'pipeline_until_failure': time.perf_counter() - pipeline_started,
            },
            'failed_at': utc_now_iso(),
        })
        try:
            write_manifest(output_layout, failure_manifest, overwrite=True)
        except Exception as manifest_error:
            logger.error(
                "Could not update failure manifest %s: %s",
                output_layout.manifest_path,
                manifest_error,
            )
        raise

    finally:
        # --- Guaranteed cleanup (P0-04) ------------------------------------
        # Remove lock file
        if runtime_cfg.get('prevent_concurrent_runs', True) and os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except OSError:
                pass

        # Remove temporary raster
        if temp_raster_path and os.path.exists(temp_raster_path):
            try:
                os.remove(temp_raster_path)
                logger.info("Cleaned up temporary raster: %s", temp_raster_path)
            except OSError as e:
                logger.warning("Could not remove temp raster %s: %s", temp_raster_path, e)

        # Release memmaps
        for arr in [row_prob, center_prob, orientation_sin, orientation_cos]:
            mmap_obj = getattr(arr, '_mmap', None)
            if mmap_obj is not None and not mmap_obj.closed:
                mmap_obj.close()
        if 'probs' in locals() and probs.mosaic is not None:
            probs.mosaic.close()

        row_prob_down = None
        center_prob_down = None
        row_prob = None
        center_prob = None
        orientation_sin = None
        orientation_cos = None
        clean_center_mask = None
        skeleton_raster = None
        import gc
        gc.collect()

    logger.info("[run=%s] Prediction pipeline completed.", run_id)

if __name__ == '__main__':
    main()
