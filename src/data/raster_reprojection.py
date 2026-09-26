import math
import os
import shutil
import logging
import hashlib
import json
from typing import Optional, List, Tuple
import numpy as np
import rasterio
import rasterio.windows
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds, Window
from rasterio.warp import Resampling, transform_bounds, calculate_default_transform, reproject
from rasterio.enums import ColorInterp
import geopandas as gpd
from rasterio.crs import CRS

logger = logging.getLogger(__name__)


def resolve_raster_source(path: str, layer_name: Optional[str] = None) -> str:
    """Resolve the actual raster dataset, including GeoPackage subdatasets.

    Ambiguous multi-layer GeoPackages fail explicitly instead of silently
    falling back to the container path.
    """
    if path.startswith(("GPKG:", "NETCDF:", "HDF5:")):
        return path

    if not os.path.exists(path):
        raise FileNotFoundError(f"Raster source not found: {path}")

    extension = os.path.splitext(path)[1].lower()
    if extension != ".gpkg":
        return path

    try:
        with rasterio.open(path) as container:
            subdatasets = list(container.subdatasets)
    except rasterio.errors.RasterioIOError as exc:
        raise ValueError(f"Could not open GeoPackage raster container {path!r}: {exc}") from exc

    if not subdatasets:
        return path

    if layer_name:
        exact_matches = [
            dataset for dataset in subdatasets
            if dataset.rsplit(":", 1)[-1] == layer_name
        ]
        matches = exact_matches or [
            dataset for dataset in subdatasets if layer_name in dataset
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"Raster layer {layer_name!r} was not found in {path!r}. "
                f"Available subdatasets: {subdatasets}"
            )
        raise ValueError(
            f"Raster layer name {layer_name!r} is ambiguous in {path!r}. "
            f"Matches: {matches}"
        )

    if len(subdatasets) == 1:
        return subdatasets[0]

    raise ValueError(
        f"GeoPackage {path!r} contains multiple raster layers. Set "
        f"input.orthomosaic_layer. Available subdatasets: {subdatasets}"
    )


def _check_free_disk(path: str, needed_bytes: int, label: str = '') -> None:
    total, used, free = shutil.disk_usage(path)
    if free < needed_bytes:
        raise OSError(
            f"Not enough disk space for {label}. "
            f"Need {needed_bytes / 1024**3:.2f} GiB, "
            f"only {free / 1024**3:.2f} GiB available at {path!r}."
        )


def _write_raster(part_path: str, profile: dict, vrt: WarpedVRT, rgb_bands: List[int], dst_window: Window) -> None:
    """Write VRT data to a target GeoTIFF chunk-by-chunk."""
    with rasterio.open(part_path, "w", **profile) as dst:
        for _, window in dst.block_windows(1):
            data = vrt.read(
                rgb_bands,
                window=Window(
                    dst_window.col_off + window.col_off,
                    dst_window.row_off + window.row_off,
                    window.width,
                    window.height,
                ),
                out_shape=(
                    len(rgb_bands),
                    int(window.height),
                    int(window.width),
                ),
            )
            dst.write(data, window=window)

def resolve_output_nodata(
    src,
    bands: List[int],
) -> Optional[float]:
    """
    Preserve a valid source NoData value only when it is consistently defined
    for all selected bands.

    For RGB imagery, zero must not be forced as NoData because it can be a
    valid pixel value (black pixels).
    """
    nodata_values = []
    for band_index in bands:
        value = src.nodatavals[band_index - 1]
        if value is None:
            return None
        nodata_values.append(value)
    if not nodata_values:
        return None
    if len(set(nodata_values)) != 1:
        return None
    nodata_value = nodata_values[0]
    if nodata_value == 0:
        return None
    return nodata_value


def estimate_native_gsd_m(
    src,
    roi_gdf,
    target_crs,
) -> Tuple[float, float]:
    """
    Estimate the source pixel size in meters around the ROI centroid.
    Returns (gsd_x_m, gsd_y_m).
    """
    from rasterio.warp import transform as warp_transform
    if src.crs is None:
        raise ValueError("O ortomosaico nao possui CRS definido.")
    roi_in_source = roi_gdf.to_crs(src.crs)
    try:
        centroid = roi_in_source.geometry.union_all().centroid
    except AttributeError:
        centroid = roi_in_source.geometry.unary_union.centroid
    x0, y0 = centroid.x, centroid.y
    pixel_x = abs(src.transform.a)
    pixel_y = abs(src.transform.e)
    # GSD X
    tx, ty = warp_transform(src.crs, target_crs, [x0, x0 + pixel_x], [y0, y0])
    gsd_x_m = math.hypot(tx[1] - tx[0], ty[1] - ty[0])
    # GSD Y
    tx, ty = warp_transform(src.crs, target_crs, [x0, x0], [y0, y0 + pixel_y])
    gsd_y_m = math.hypot(tx[1] - tx[0], ty[1] - ty[0])
    if gsd_x_m <= 0 or gsd_y_m <= 0:
        raise ValueError("Nao foi possivel estimar um GSD metrico valido.")
    return gsd_x_m, gsd_y_m



def prepare_working_raster(
    src_path: str,
    dst_path: str,
    target_crs: CRS,
    roi_gdf: gpd.GeoDataFrame,
    roi_buffer_m: float = 10.0,
    rgb_bands: Optional[List[int]] = None,
    output_resolution_m: Optional[float] = None,
    preserve_source_gsd: bool = True,
    layer_name: Optional[str] = None,
    compression: Optional[str] = None,
    warp_mem_limit_mb: int = 256,
) -> str:
    """
    Reproject and crop a raster to the ROI + buffer window, writing only selected bands.
    Uses WarpedVRT to avoid reprojecting the entire file (BUG P0-01).
    """
    if rgb_bands is None:
        rgb_bands = [1, 2, 3]

    compress_algo = compression or 'ZSTD'
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    part_path = dst_path + '.part.tif'

    # 1. Resolve subdataset path (GPKG support)
    open_path = resolve_raster_source(src_path, layer_name)

    try:
        with rasterio.open(open_path) as src:
            src_crs = src.crs
            if src_crs is None:
                raise ValueError("Source raster does not have a defined CRS.")

            # --- 2. Safe Buffer in Metric Workspace (BUG P0-04) -----------------
            roi_metric = roi_gdf.to_crs(target_crs)
            roi_metric["geometry"] = roi_metric.geometry.buffer(roi_buffer_m)
            roi_src_crs = roi_metric.to_crs(src_crs)
            clip_bounds = roi_src_crs.total_bounds

            # Clip bounds to the raster extent in source CRS
            rast_bounds = src.bounds
            clip_bounds = (
                max(clip_bounds[0], rast_bounds.left),
                max(clip_bounds[1], rast_bounds.bottom),
                min(clip_bounds[2], rast_bounds.right),
                min(clip_bounds[3], rast_bounds.top),
            )
            if clip_bounds[0] >= clip_bounds[2] or clip_bounds[1] >= clip_bounds[3]:
                raise ValueError(
                    f"ROI + buffer does not intersect the raster extent. "
                    f"ROI bounds: {clip_bounds}, Raster extent: {rast_bounds}."
                )

            # --- 3. Read original dtype and safe nodata --------------------------
            dtype = src.dtypes[rgb_bands[0] - 1]
            output_nodata = resolve_output_nodata(src, rgb_bands)

            # --- 3b. Estimate native GSD in meters --------------------------------
            crs_cfg_gsd = {}
            try:
                gsd_x_m, gsd_y_m = estimate_native_gsd_m(src, roi_gdf, target_crs)
                native_gsd_m = (gsd_x_m + gsd_y_m) / 2.0
                logger.info(
                    "Native source GSD at ROI: x=%.4f m/px  y=%.4f m/px  mean=%.4f m/px",
                    gsd_x_m, gsd_y_m, native_gsd_m,
                )
                crs_cfg_gsd['native_gsd_m'] = native_gsd_m
            except Exception as _gsd_err:
                logger.warning("Could not estimate native GSD: %s", _gsd_err)
                native_gsd_m = None

            # Resolution to use: explicit override > preserve native > let VRT decide
            target_resolution = output_resolution_m if output_resolution_m is not None else (
                native_gsd_m if preserve_source_gsd else None
            )

            # --- 4. Reproject limited window using WarpedVRT --------------------
            vrt_kwargs = dict(
                crs=target_crs,
                resampling=Resampling.bilinear,
                warp_mem_limit=warp_mem_limit_mb,
            )
            if output_nodata is not None:
                vrt_kwargs['src_nodata'] = output_nodata
                vrt_kwargs['nodata'] = output_nodata
            else:
                # A source tag of nodata=0 must not make black RGB pixels
                # transparent or cause GDAL to rewrite them to 1.  An alpha
                # mask preserves validity without reserving an RGB value.
                # Rasterio otherwise inherits the source nodata tag; use a
                # value outside the source integer range only for the warp.
                # It is never written to the output profile.
                dtype_info = np.dtype(dtype)
                if dtype_info.kind == 'u':
                    vrt_kwargs['src_nodata'] = -1
                elif dtype_info.kind == 'i':
                    vrt_kwargs['src_nodata'] = np.iinfo(dtype_info).min - 1
                if ColorInterp.alpha not in src.colorinterp:
                    vrt_kwargs['add_alpha'] = True
            if target_resolution is not None:
                vrt_kwargs['resolution'] = target_resolution
                logger.info(
                    "Reprojecting ROI preserving source GSD: %.4f m/pixel",
                    target_resolution,
                )
            with WarpedVRT(src, **vrt_kwargs) as vrt:
                # Target bounds of the clipped window in target CRS
                target_bounds = transform_bounds(
                    src_crs,
                    target_crs,
                    *clip_bounds,
                    densify_pts=21,
                )

                # Compute bounds in target pixel coordinates
                dst_window = from_bounds(
                    *target_bounds,
                    transform=vrt.transform,
                )
                dst_window = dst_window.round_offsets().round_lengths()
                dst_window = dst_window.intersection(
                    Window(0, 0, vrt.width, vrt.height)
                )

                width_dst = int(dst_window.width)
                height_dst = int(dst_window.height)
                if width_dst <= 0 or height_dst <= 0:
                    raise ValueError("Calculated reprojected raster dimension is empty.")

                transform_dst = vrt.window_transform(dst_window)

                # --- 5. Disk Space Verification ---------------------------------
                # uint8 = 1 byte, uint16 = 2 bytes, float32 = 4 bytes
                item_size = np.dtype(dtype).itemsize
                est_bytes = width_dst * height_dst * len(rgb_bands) * item_size
                _check_free_disk(
                    os.path.dirname(os.path.abspath(dst_path)) or '.',
                    est_bytes * 3,
                    label=f'reprojected raster ({width_dst}x{height_dst})',
                )

                profile = src.profile.copy()
                profile.update(
                    driver='GTiff',
                    crs=target_crs,
                    transform=transform_dst,
                    width=width_dst,
                    height=height_dst,
                    count=len(rgb_bands),
                    dtype=dtype,
                    nodata=output_nodata,
                    tiled=True,
                    blockxsize=512,
                    blockysize=512,
                    compress=compress_algo,
                    predictor=2 if np.dtype(dtype).kind in ('i', 'u') else 1,
                    BIGTIFF='IF_SAFER',
                )

                logger.info(
                    f"prepare_working_raster: WarpedVRT window {width_dst}x{height_dst} px "
                    f"(target GSD={target_resolution}m). Writing with compress={compress_algo}."
                )

                # --- 6. Write with fallback compression check (BUG P0-05) --------
                try:
                    _write_raster(part_path, profile, vrt, rgb_bands, dst_window)
                except Exception as e:
                    if compress_algo == 'ZSTD':
                        logger.warning(
                            f"Writing with ZSTD failed (codec probably unavailable): {e}. "
                            "Retrying with DEFLATE compression."
                        )
                        if os.path.exists(part_path):
                            try:
                                os.remove(part_path)
                            except OSError:
                                pass
                        profile['compress'] = 'DEFLATE'
                        _write_raster(part_path, profile, vrt, rgb_bands, dst_window)
                    else:
                        raise

        # Atomic swap
        os.replace(part_path, dst_path)
        logger.info(f"prepare_working_raster: written successfully to {dst_path}")
        return dst_path

    except Exception:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
        raise


def _working_raster_cache_key(
    src_path: str,
    target_crs: CRS,
    roi_gdf: gpd.GeoDataFrame,
    roi_buffer_m: float,
    rgb_bands: List[int],
    output_resolution_m: Optional[float],
    preserve_source_gsd: bool,
    layer_name: Optional[str],
    compression: Optional[str],
) -> Tuple[str, dict]:
    """Build a content key from every input that can affect warped pixels."""
    resolved = resolve_raster_source(src_path, layer_name)
    physical_source = os.path.abspath(src_path) if os.path.exists(src_path) else None
    with rasterio.open(resolved) as src:
        if physical_source is None:
            physical_source = next(
                (os.path.abspath(path) for path in src.files if os.path.exists(path)),
                resolved,
            )
        source_driver = src.driver
        source_subdataset = resolved
    stat = os.stat(physical_source)

    roi_metric = roi_gdf.to_crs(target_crs)
    try:
        roi_geometry = roi_metric.geometry.union_all()
    except AttributeError:
        roi_geometry = roi_metric.geometry.unary_union
    roi_hash = hashlib.sha256(roi_geometry.wkb).hexdigest()
    payload = {
        'format_version': 1,
        'source_path': os.path.normcase(os.path.abspath(physical_source)),
        'source_size': int(stat.st_size),
        'source_mtime_ns': int(stat.st_mtime_ns),
        'source_driver': source_driver,
        'source_subdataset': source_subdataset,
        'layer_name': layer_name,
        'roi_sha256': roi_hash,
        'target_crs_wkt': target_crs.to_wkt(),
        'roi_buffer_m': float(roi_buffer_m),
        'rgb_bands': [int(index) for index in rgb_bands],
        'output_resolution_m': (
            None if output_resolution_m is None else float(output_resolution_m)
        ),
        'preserve_source_gsd': bool(preserve_source_gsd),
        'compression': compression or 'ZSTD',
        'resampling': 'bilinear',
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest(), payload


def _working_raster_manifest(path: str, cache_key: str, key_payload: dict) -> dict:
    with rasterio.open(path) as src:
        return {
            'cache_key': cache_key,
            'key_payload': key_payload,
            'file_size': int(os.path.getsize(path)),
            'crs_wkt': src.crs.to_wkt() if src.crs is not None else None,
            'transform': list(src.transform),
            'width': int(src.width),
            'height': int(src.height),
            'count': int(src.count),
            'dtypes': list(src.dtypes),
            'bounds': list(src.bounds),
            'gsd': [abs(float(src.transform.a)), abs(float(src.transform.e))],
        }


def validate_cached_working_raster(
    raster_path: str,
    manifest_path: str,
    cache_key: str,
    target_crs: CRS,
    expected_band_count: int,
) -> bool:
    """Validate cached raster structure and georeferencing before reuse."""
    try:
        if not os.path.exists(raster_path) or not os.path.exists(manifest_path):
            return False
        with open(manifest_path, 'r', encoding='utf-8') as stream:
            manifest = json.load(stream)
        if manifest.get('cache_key') != cache_key:
            return False
        if int(manifest.get('file_size', -1)) != os.path.getsize(raster_path):
            return False
        with rasterio.open(raster_path) as src:
            if src.crs != target_crs or src.count != expected_band_count:
                return False
            if src.width != int(manifest['width']) or src.height != int(manifest['height']):
                return False
            if list(src.dtypes) != list(manifest['dtypes']):
                return False
            if not np.allclose(tuple(src.transform), manifest['transform'], rtol=0.0, atol=1e-12):
                return False
            if not np.allclose(tuple(src.bounds), manifest['bounds'], rtol=0.0, atol=1e-8):
                return False
            gsd = [abs(float(src.transform.a)), abs(float(src.transform.e))]
            if not np.allclose(gsd, manifest['gsd'], rtol=0.0, atol=1e-12):
                return False
            if src.width <= 0 or src.height <= 0 or min(gsd) <= 0:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, rasterio.errors.RasterioError) as exc:
        logger.warning("Working raster cache validation failed for %s: %s", raster_path, exc)
        return False


def prepare_working_raster_cached(
    src_path: str,
    cache_dir: str,
    target_crs: CRS,
    roi_gdf: gpd.GeoDataFrame,
    roi_buffer_m: float = 10.0,
    rgb_bands: Optional[List[int]] = None,
    output_resolution_m: Optional[float] = None,
    preserve_source_gsd: bool = True,
    layer_name: Optional[str] = None,
    compression: Optional[str] = None,
    warp_mem_limit_mb: int = 256,
) -> Tuple[str, bool]:
    """Reuse a structurally validated deterministic working-raster cache."""
    bands = list(rgb_bands or [1, 2, 3])
    os.makedirs(cache_dir, exist_ok=True)
    cache_key, key_payload = _working_raster_cache_key(
        src_path,
        target_crs,
        roi_gdf,
        roi_buffer_m,
        bands,
        output_resolution_m,
        preserve_source_gsd,
        layer_name,
        compression,
    )
    raster_path = os.path.join(cache_dir, f'{cache_key}.tif')
    manifest_path = os.path.join(cache_dir, f'{cache_key}.json')
    if validate_cached_working_raster(
        raster_path,
        manifest_path,
        cache_key,
        target_crs,
        len(bands),
    ):
        logger.info("Working raster cache HIT: %s", raster_path)
        return raster_path, True

    logger.info("Working raster cache MISS: %s", raster_path)
    prepare_working_raster(
        src_path,
        raster_path,
        target_crs=target_crs,
        roi_gdf=roi_gdf,
        roi_buffer_m=roi_buffer_m,
        rgb_bands=bands,
        output_resolution_m=output_resolution_m,
        preserve_source_gsd=preserve_source_gsd,
        layer_name=layer_name,
        compression=compression,
        warp_mem_limit_mb=warp_mem_limit_mb,
    )
    manifest = _working_raster_manifest(raster_path, cache_key, key_payload)
    manifest_part = manifest_path + '.part'
    with open(manifest_part, 'w', encoding='utf-8') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
    os.replace(manifest_part, manifest_path)
    if not validate_cached_working_raster(
        raster_path,
        manifest_path,
        cache_key,
        target_crs,
        len(bands),
    ):
        raise RuntimeError(f"New working raster cache entry failed validation: {raster_path}")
    return raster_path, False


def reproject_raster(
    src_path: str,
    dst_path: str,
    target_epsg: int,
    layer_name: Optional[str] = None,
    roi_gdf: Optional[gpd.GeoDataFrame] = None,
    roi_buffer_m: float = 10.0,
    rgb_bands: Optional[List[int]] = None,
    output_resolution_m: Optional[float] = None,
    preserve_source_gsd: bool = True,
) -> str:
    """Legacy backward-compatible wrapper. Delegates to prepare_working_raster if roi_gdf exists."""
    target_crs = CRS.from_epsg(target_epsg)
    if roi_gdf is not None:
        return prepare_working_raster(
            src_path, dst_path, target_crs, roi_gdf,
            roi_buffer_m=roi_buffer_m,
            rgb_bands=rgb_bands,
            layer_name=layer_name,
            output_resolution_m=output_resolution_m,
            preserve_source_gsd=preserve_source_gsd,
        )

    logger.warning("reproject_raster called without roi_gdf - fallback to full raster reprojection.")
    open_path = resolve_raster_source(src_path, layer_name)
    import uuid as _uuid
    _suffix = f"{os.getpid()}_{_uuid.uuid4().hex[:8]}"
    part_path = dst_path + f'.{_suffix}.part.tif'
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)

    try:
        with rasterio.open(open_path) as src:
            dst_crs = f'EPSG:{target_epsg}'
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update(
                crs=dst_crs,
                transform=transform,
                width=width,
                height=height,
                driver='GTiff',
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress='ZSTD',
                BIGTIFF='IF_SAFER',
            )
            try:
                with rasterio.open(part_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear,
                        )
            except Exception as e:
                logger.warning(f"ZSTD full reprojection failed: {e}. Trying DEFLATE.")
                if os.path.exists(part_path):
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass
                kwargs['compress'] = 'DEFLATE'
                with rasterio.open(part_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear,
                        )
        # All rasterio/GDAL contexts are now closed. Force GC before rename.
        import gc
        gc.collect()
        # Retry-safe rename for Windows (transient lock may take ~500ms to release)
        import time as _time
        for _attempt in range(5):
            try:
                os.replace(part_path, dst_path)
                break
            except PermissionError:
                if _attempt == 4:
                    raise PermissionError(
                        f"[WinError 32] Could not rename {part_path!r} -> {dst_path!r} "
                        "after 5 attempts. Another process may have the file open."
                    )
                logger.warning(
                    "WinError 32: file still locked, retrying in 0.5s (attempt %d/5)...", _attempt + 1
                )
                _time.sleep(0.5)
        return dst_path
    except Exception:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
        raise
