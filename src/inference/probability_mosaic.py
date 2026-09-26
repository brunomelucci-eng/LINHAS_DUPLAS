import numpy as np
from typing import Tuple, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field
import tempfile
import os
import shutil
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import rasterio


@dataclass
class ProbabilityOutputs:
    """
    Container for sliding window probability results.
    Channels that were not selected for storage are set to None,
    preventing massive allocations of zero-filled arrays (BUG P0-06).
    """
    center: np.ndarray
    row: Optional[np.ndarray] = None
    orientation_sin: Optional[np.ndarray] = None
    orientation_cos: Optional[np.ndarray] = None
    mosaic: Optional['ProbabilityMosaic'] = None
    tile_windows: List[Tuple[int, int, int, int]] = field(default_factory=list)


class ProbabilityMosaic:
    def __init__(
        self,
        width: int,
        height: int,
        tile_size: int = 512,
        blend_type: str = 'hann',
        memmap_threshold_px: int = 25000000,
        config: Optional[dict] = None,
        temp_root: Optional[str] = None
    ):
        self.width = width
        self.height = height
        
        # Channels selection (BUG P0-06)
        self.selected_indices = [1] # default to center channel only (index 1)
        self.final_dtype = np.float32
        
        # Parse configurations
        max_disk_gb = 4.0
        inf_cfg = {}
        debug_cfg = {}
        if config is not None:
            inf_cfg = config.get('inference', {})
            debug_cfg = config.get('debug', {})
            tile_size = config.get('tiling', {}).get('tile_size_px', tile_size)
            blend_type = inf_cfg.get('blend_mode', inf_cfg.get('blending', blend_type))
            memmap_threshold_px = inf_cfg.get('memmap_threshold_px', memmap_threshold_px)
            max_disk_gb = float(inf_cfg.get('max_temp_disk_gb', 4.0))
            
            channels_cfg = inf_cfg.get('output_channels_to_store', ['center'])
            
            # Support explicit boolean requests or metrics requests
            # If reference lines exist, row is automatically added to make sure metrics compute correctly
            in_cfg = config.get('input', {})
            ref_path = in_cfg.get('reference_lines')
            if ref_path and os.path.exists(ref_path):
                if 'row' not in channels_cfg:
                    channels_cfg = list(channels_cfg) + ['row']

            if inf_cfg.get('retain_row_probability', False) and 'row' not in channels_cfg:
                channels_cfg = list(channels_cfg) + ['row']
            if inf_cfg.get('retain_orientation_probability', False) and 'orientation_sin' not in channels_cfg:
                channels_cfg = list(channels_cfg) + ['orientation_sin', 'orientation_cos']
                
            selected = []
            if 'row' in channels_cfg:
                selected.append(0)
            if 'center' in channels_cfg:
                selected.append(1)
            if 'orientation_sin' in channels_cfg or 'orientation' in channels_cfg:
                selected.append(2)
            if 'orientation_cos' in channels_cfg or 'orientation' in channels_cfg:
                selected.append(3)
                
            if selected:
                self.selected_indices = sorted(list(set(selected)))
                
            dtype_cfg = inf_cfg.get('final_probability_dtype', 'float32')
            if dtype_cfg == 'float16':
                self.final_dtype = np.float16

        self.tile_size = tile_size
        self.blend_type = blend_type
        self.num_channels = len(self.selected_indices)
        
        # Switch to memory-mapped files on disk if the image is large
        self.use_memmap = (width * height > memmap_threshold_px)
        # Large mosaics are accumulated as a rolling tile-height stripe. This
        # keeps the exact row-major summation order while avoiding full-size
        # float32 accumulator and weight maps.
        self.streaming = bool(inf_cfg.get('streaming_mosaic', self.use_memmap)) and self.use_memmap
        self.active_height = min(height, tile_size) if self.streaming else height
        self.store_full_weights = bool(
            self.streaming
            and debug_cfg.get('enabled', False)
            and debug_cfg.get('save_weights', False)
        )
        self.temp_dir = None
        
        # --- Disk Space and Size Enforcement (BUG P0-10) --------------------
        # Estimate size of temp files (float32 accumulators + float16/32 final outputs)
        item_size = np.dtype(self.final_dtype).itemsize
        accumulator_bytes = self.active_height * width * 4 * self.num_channels
        weights_bytes = self.active_height * width * 4
        final_channels_bytes = height * width * item_size * self.num_channels
        retained_weights_bytes = height * width * 4 if self.store_full_weights else 0
        estimated_total_bytes = (
            accumulator_bytes
            + weights_bytes
            + final_channels_bytes
            + retained_weights_bytes
        )
        
        limit_bytes = int(max_disk_gb * 1024**3)
        if estimated_total_bytes > limit_bytes:
            raise RuntimeError(
                f"Inference requires approximately {estimated_total_bytes / 1024**3:.2f} GiB "
                f"of temporary files, which exceeds the configured max_temp_disk_gb limit "
                f"of {max_disk_gb:.1f} GiB. Aborting execution."
            )
        
        if self.use_memmap:
            # Use run_dir if provided, otherwise default to outputs/ (BUG P0-09)
            root_dir = temp_root if temp_root else 'outputs'
            os.makedirs(root_dir, exist_ok=True)
            
            # --- Check Free Disk Space (BUG P0-07) ---------------------------
            try:
                _, _, free = shutil.disk_usage(root_dir)
                if free < estimated_total_bytes * 1.3:  # 1.3x safety margin
                    raise OSError(
                        f"Insufficient disk space at {root_dir!r} for temporary files. "
                        f"Estimated: {estimated_total_bytes / 1024**3:.2f} GiB; "
                        f"Available: {free / 1024**3:.2f} GiB."
                    )
            except OSError as e:
                if "Insufficient disk space" in str(e):
                    raise
                logger.warning(f"Could not verify free disk space: {e}")

            self.temp_dir = tempfile.TemporaryDirectory(dir=os.path.abspath(root_dir), prefix="probability_mosaic_")
            self.accum_path = os.path.join(self.temp_dir.name, 'accum.dat')
            self.weights_path = os.path.join(self.temp_dir.name, 'weights.dat')
            
            self.accum = np.memmap(
                self.accum_path,
                dtype=np.float32,
                mode='w+',
                shape=(self.num_channels, self.active_height, width),
            )
            self.weights = np.memmap(
                self.weights_path,
                dtype=np.float32,
                mode='w+',
                shape=(self.active_height, width),
            )
            
            # Initialize memmaps with zeros
            self.accum[:] = 0.0
            self.weights[:] = 0.0

            self._stream_outputs = {}
            self._stream_row0 = 0
            self._last_update_row = -1
            self._stream_weights = None
            if self.streaming:
                output_filenames = {
                    0: 'row_prob.dat',
                    1: 'center_prob.dat',
                    2: 'sin.dat',
                    3: 'cos.dat',
                }
                for idx in self.selected_indices:
                    path = os.path.join(self.temp_dir.name, output_filenames[idx])
                    self._stream_outputs[idx] = np.memmap(
                        path,
                        dtype=self.final_dtype,
                        mode='w+',
                        shape=(height, width),
                    )
                if self.store_full_weights:
                    self.stream_weights_path = os.path.join(self.temp_dir.name, 'weights_final.dat')
                    self._stream_weights = np.memmap(
                        self.stream_weights_path,
                        dtype=np.float32,
                        mode='w+',
                        shape=(height, width),
                    )
        else:
            self.accum = np.zeros((self.num_channels, height, width), dtype=np.float32)
            self.weights = np.zeros((height, width), dtype=np.float32)
            self._stream_outputs = {}
            self._stream_weights = None
        
        self.window = self._create_window(tile_size, blend_type)
        logger.info(
            f"Initialized ProbabilityMosaic (size {width}x{height} px, "
            f"use_memmap={self.use_memmap}, streaming={self.streaming}, "
            f"active_rows={self.active_height}). "
            f"Active channels: {self.selected_indices} ({self.num_channels} channels). "
            f"Temp disk limit: {max_disk_gb:.1f} GiB (estimated: {estimated_total_bytes / 1024**3:.2f} GiB)."
        )

    def _create_window(self, size: int, blend_type: str) -> np.ndarray:
        if blend_type == 'hann':
            w = np.hanning(size)
            window_2d = np.outer(w, w)
        elif blend_type == 'gaussian':
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            xx, yy = np.meshgrid(x, y)
            window_2d = np.exp(-0.5 * (xx**2 + yy**2) / 0.25)
        else:
            w = np.minimum(np.arange(size), np.arange(size)[::-1])
            w = w / (w.max() + 1e-8)
            window_2d = np.outer(w, w)
            
        window_2d = np.maximum(window_2d, 1e-4)
        return window_2d.astype(np.float32)

    def update(self, col_off: int, row_off: int, patch_pred: np.ndarray):
        """
        Accumulate patch predictions at specified grid offsets.
        patch_pred shape: (4, h, w)
        """
        _, h, w = patch_pred.shape
        win_sec = self.window[:h, :w]

        if patch_pred.shape[0] == self.num_channels:
            selected_patch = patch_pred
        else:
            selected_patch = patch_pred[self.selected_indices]

        if self.streaming:
            if row_off < self._last_update_row:
                raise ValueError("Streaming mosaic updates must be ordered by non-decreasing row offset.")
            if row_off > self._last_update_row and self._last_update_row >= 0:
                self._finalize_stream_rows(row_off)
            self._last_update_row = row_off
            relative_row = row_off - self._stream_row0
            if relative_row < 0 or relative_row + h > self.active_height:
                raise ValueError(
                    f"Tile rows [{row_off}, {row_off + h}) do not fit active streaming stripe "
                    f"[{self._stream_row0}, {self._stream_row0 + self.active_height})."
                )
            self.accum[:, relative_row:relative_row+h, col_off:col_off+w] += selected_patch * win_sec
            self.weights[relative_row:relative_row+h, col_off:col_off+w] += win_sec
        else:
            self.accum[:, row_off:row_off+h, col_off:col_off+w] += selected_patch * win_sec
            self.weights[row_off:row_off+h, col_off:col_off+w] += win_sec

    def _finalize_stream_rows(self, row_limit: int) -> None:
        """Normalise completed rows, write them once, and roll the stripe."""
        row_limit = min(int(row_limit), self.height)
        eps = 1e-8
        clip_ranges = {
            0: (0.0, 1.0),
            1: (0.0, 1.0),
            2: (-1.0, 1.0),
            3: (-1.0, 1.0),
        }
        while self._stream_row0 < row_limit:
            count = min(row_limit - self._stream_row0, self.active_height)
            block_end = self._stream_row0 + count
            norm_weights = np.maximum(self.weights[:count], eps)
            for pos, idx in enumerate(self.selected_indices):
                final = self.accum[pos, :count] / norm_weights
                low, high = clip_ranges[idx]
                self._stream_outputs[idx][self._stream_row0:block_end] = np.clip(
                    final, low, high
                ).astype(self.final_dtype)
            if self._stream_weights is not None:
                self._stream_weights[self._stream_row0:block_end] = self.weights[:count]

            remaining = self.active_height - count
            if remaining > 0:
                self.accum[:, :remaining] = self.accum[:, count:].copy()
                self.weights[:remaining] = self.weights[count:].copy()
            self.accum[:, remaining:] = 0.0
            self.weights[remaining:] = 0.0
            self._stream_row0 = block_end

    def get_final(self) -> ProbabilityOutputs:
        """
        Normalise accumulated patches and returns final predictions wrapped in a ProbabilityOutputs class.
        If a channel was not selected in configurations, its field is set to None (BUG P0-06).
        """
        eps = 1e-8
        
        name_map = {
            0: ('row_prob', 'row_prob.dat', (0.0, 1.0)),
            1: ('center_prob', 'center_prob.dat', (0.0, 1.0)),
            2: ('orientation_sin', 'sin.dat', (-1.0, 1.0)),
            3: ('orientation_cos', 'cos.dat', (-1.0, 1.0))
        }
        
        outputs = {}

        if self.streaming:
            self._finalize_stream_rows(self.height)
            for idx in self.selected_indices:
                self._stream_outputs[idx].flush()
                outputs[idx] = self._stream_outputs[idx]
            if self._stream_weights is not None:
                self._stream_weights.flush()
            for idx in [0, 1, 2, 3]:
                outputs.setdefault(idx, None)
            return ProbabilityOutputs(
                row=outputs.get(0),
                center=outputs.get(1),
                orientation_sin=outputs.get(2),
                orientation_cos=outputs.get(3),
                mosaic=self,
            )
        
        if self.use_memmap:
            # 1. Allocate final memmap files directly using the target final_dtype
            # to avoid large full-matrix copies in RAM (BUG P0-08)
            for idx in [0, 1, 2, 3]:
                if idx in self.selected_indices:
                    name, filename, _ = name_map[idx]
                    path = os.path.join(self.temp_dir.name, filename)
                    outputs[idx] = np.memmap(path, dtype=self.final_dtype, mode='w+', shape=(self.height, self.width))
                else:
                    outputs[idx] = None
                    
            # 2. Compute chunk-by-chunk to keep memory usage low
            chunk_size = 1000
            for i in range(0, self.height, chunk_size):
                end_i = min(i + chunk_size, self.height)
                weights_chunk = self.weights[i:end_i, :]
                norm_weights = np.maximum(weights_chunk, eps)
                
                for idx in self.selected_indices:
                    pos = self.selected_indices.index(idx)
                    accum_chunk = self.accum[pos, i:end_i, :]
                    final_chunk = accum_chunk / norm_weights
                    
                    _, _, clip_range = name_map[idx]
                    clipped_chunk = np.clip(final_chunk, clip_range[0], clip_range[1])
                    
                    # Convert to final_dtype during write to disk (BUG P0-08)
                    outputs[idx][i:end_i, :] = clipped_chunk.astype(self.final_dtype)
                    
            # Flush memmaps
            for idx in self.selected_indices:
                outputs[idx].flush()
                
            # Attach the temp directory to the arrays to prevent them from being garbage-collected early
            for idx in self.selected_indices:
                arr = outputs[idx]
                if hasattr(arr, 'base') and isinstance(arr.base, np.memmap):
                    arr.base._temp_dir_ref = self.temp_dir
                elif hasattr(arr, '_temp_dir_ref'):
                    arr._temp_dir_ref = self.temp_dir
                else:
                    try:
                        arr._temp_dir_ref = self.temp_dir
                    except AttributeError:
                        pass
        else:
            norm_weights = np.maximum(self.weights, eps)
            
            for idx in [0, 1, 2, 3]:
                if idx in self.selected_indices:
                    pos = self.selected_indices.index(idx)
                    final = self.accum[pos] / norm_weights
                    _, _, clip_range = name_map[idx]
                    outputs[idx] = np.clip(final, clip_range[0], clip_range[1]).astype(self.final_dtype)
                else:
                    outputs[idx] = None
                    
        return ProbabilityOutputs(
            row=outputs.get(0),
            center=outputs.get(1),
            orientation_sin=outputs.get(2),
            orientation_cos=outputs.get(3),
            mosaic=self,
        )

    def export_weights_geotiff(self, output_path: str, transform, crs) -> None:
        """Write unnormalised mosaic weights in bounded row chunks."""
        import rasterio

        profile = {
            'driver': 'GTiff', 'height': self.height, 'width': self.width,
            'count': 1, 'dtype': 'float32', 'crs': crs, 'transform': transform,
            'compress': 'DEFLATE', 'predictor': 3, 'tiled': True,
            'blockxsize': 512, 'blockysize': 512, 'nodata': 0.0,
        }
        weights_source = self._stream_weights if self.streaming else self.weights
        if weights_source is None:
            raise RuntimeError(
                "Streaming mosaic weights were not retained; enable debug.save_weights before inference."
            )
        with rasterio.open(output_path, 'w', **profile) as dst:
            for row in range(0, self.height, 1024):
                height = min(1024, self.height - row)
                window = rasterio.windows.Window(0, row, self.width, height)
                dst.write(np.asarray(weights_source[row:row + height], dtype=np.float32), 1, window=window)

    def release_accumulators(self) -> None:
        """Release accumulation maps after final probabilities are materialised.

        The final channel memmaps live in the same temporary directory but are
        independent files, so inference consumers can keep using them while the
        much larger float32 accumulator and weight maps are closed and removed.
        """
        for attr_name, path_name in (("accum", "accum_path"), ("weights", "weights_path")):
            arr = getattr(self, attr_name, None)
            mmap_obj = getattr(arr, "_mmap", None)
            if mmap_obj is not None and not mmap_obj.closed:
                mmap_obj.close()
            setattr(self, attr_name, None)

            path = getattr(self, path_name, None)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    logger.warning("Could not remove released mosaic file %s: %s", path, exc)

        stream_weights = getattr(self, '_stream_weights', None)
        if stream_weights is not None:
            mmap_obj = getattr(stream_weights, '_mmap', None)
            if mmap_obj is not None and not mmap_obj.closed:
                mmap_obj.close()
            self._stream_weights = None
            path = getattr(self, 'stream_weights_path', None)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    logger.warning("Could not remove released mosaic weights %s: %s", path, exc)

        logger.info("Released probability mosaic accumulator and weight maps.")

    def close(self) -> None:
        """Release temporary memmaps and their directory after debug exports."""
        arrays = [getattr(self, 'accum', None), getattr(self, 'weights', None)]
        arrays.extend(getattr(self, '_stream_outputs', {}).values())
        arrays.append(getattr(self, '_stream_weights', None))
        for arr in arrays:
            mmap_obj = getattr(arr, '_mmap', None)
            if mmap_obj is not None and not mmap_obj.closed:
                mmap_obj.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None
