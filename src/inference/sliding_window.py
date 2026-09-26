import numpy as np
from tqdm import tqdm
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, List
import rasterio
from shapely.geometry import box
from shapely.prepared import prep
from .predictor import RowPredictor
from .probability_mosaic import ProbabilityMosaic, ProbabilityOutputs
from .tta import predict_with_tta
from src.logging_utils import StageTimer

logger = logging.getLogger(__name__)


def generate_offsets(size: int, tile_size: int, step: int) -> List[int]:
    """
    Generate coordinate offsets along one dimension of the raster,
    ensuring that the final tile aligns with the boundary and no coordinates
    are duplicated (BUG P0-24).
    """
    if size <= tile_size:
        return [0]

    offsets = list(range(0, size - tile_size + 1, step))
    last = size - tile_size
    if not offsets or offsets[-1] != last:
        offsets.append(last)

    return sorted(list(set(offsets)))


def predict_full_raster(
    predictor: RowPredictor,
    raster_reader, # RasterReader instance
    config: dict,
    roi_window: Optional[rasterio.windows.Window] = None,
    roi_geometry = None,
    temp_root: Optional[str] = None
) -> ProbabilityOutputs:
    """
    Run sliding window prediction on a full raster image by reading patches directly from disk.
    If roi_window and roi_geometry are provided, limits computation to the window and skips
    tiles that do not intersect with the ROI.
    """
    inf_cfg = config.get('inference', {})
    tile_size = config.get('tiling', {}).get('tile_size_px', 512)
    overlap = config.get('tiling', {}).get('overlap_px', 128)
    configured_overlap = inf_cfg.get('overlap')
    if configured_overlap is not None:
        overlap = int(round(tile_size * float(configured_overlap))) if float(configured_overlap) < 1 else int(configured_overlap)
    use_tta = inf_cfg.get('use_tta', False)
    
    in_cfg = config.get('input', {})
    rgb_bands = in_cfg.get('rgb_bands', [1, 2, 3])
    c = len(rgb_bands)
    
    if roi_window is not None:
        col_min, row_min = int(roi_window.col_off), int(roi_window.row_off)
        H = roi_window.height
        W = roi_window.width
    else:
        col_min, row_min = 0, 0
        H = raster_reader.height
        W = raster_reader.width
        
    mosaic = ProbabilityMosaic(
        width=W, 
        height=H, 
        config=config,
        temp_root=temp_root
    )
    
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("Overlap must be strictly smaller than tile size.")
        
    y_offsets = generate_offsets(H, tile_size, step)
    x_offsets = generate_offsets(W, tile_size, step)
    
    total_tiles = len(y_offsets) * len(x_offsets)
    logger.info(f"Running inference on {total_tiles} sliding window tiles...")
    
    # Prepare ROI geometry for fast intersection queries
    prep_roi = None
    if roi_geometry is not None:
        prep_roi = prep(roi_geometry)
        
    batch_size = int(inf_cfg.get('batch_size', 2))
    batch_buffer = np.empty((batch_size, c, tile_size, tile_size), dtype=np.float32)
    batch_count = 0
    batch_offsets = []  # list of (col_off, row_off, h, w)
    used_tile_windows: List[Tuple[int, int, int, int]] = []

    def _flush_batch(batch_arr: np.ndarray, offsets_list: List[Tuple[int, int, int, int]]):
        """Send a batch to the model, with retry/reduction on CUDA OOM (BUG P0-22)."""
        nonlocal batch_size
        if len(batch_arr) == 0:
            return
        
        try:
            if use_tta:
                # predict_with_tta expects (C, H, W). We run TTA per item in the batch.
                pred_list = []
                for patch_in in batch_arr:
                    pred_list.append(predict_with_tta(predictor, patch_in))
                pred_batch = np.stack(pred_list, axis=0)
            else:
                pred_batch = predictor.predict_batch(
                    batch_arr,
                    output_indices=mosaic.selected_indices,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > 1:
                logger.warning(
                    f"CUDA Out of Memory. Reducing batch_size from {batch_size} to {batch_size // 2}."
                )
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                
                batch_size = max(1, batch_size // 2)
                mid = len(batch_arr) // 2
                _flush_batch(batch_arr[:mid], offsets_list[:mid])
                _flush_batch(batch_arr[mid:], offsets_list[mid:])
                return
            else:
                raise

        # Accumulate predictions into ProbabilityMosaic
        for idx, (col_off, row_off, h, w) in enumerate(offsets_list):
            pred = pred_batch[idx]
            # Crop padding if necessary
            if pred.shape[1] > h or pred.shape[2] > w:
                pred = pred[:, :h, :w]
            mosaic.update(col_off, row_off, pred)

    tile_specs = [
        (x, y, min(tile_size, W - x), min(tile_size, H - y))
        for y in y_offsets
        for x in x_offsets
    ]

    def _read_tile(spec: Tuple[int, int, int, int]):
        col_off, row_off, w, h = spec
        col_off_full = col_off + col_min
        row_off_full = row_off + row_min
        window_full = rasterio.windows.Window(col_off_full, row_off_full, w, h)

        if prep_roi is not None:
            xs, ys = rasterio.transform.xy(
                raster_reader.transform,
                [row_off_full, row_off_full + h],
                [col_off_full, col_off_full + w],
            )
            tile_box = box(min(xs), min(ys), max(xs), max(ys))
            if not prep_roi.intersects(tile_box):
                return spec, None

        patch = raster_reader.read_normalized(rgb_bands, window=window_full)
        if np.all(patch == 0):
            return spec, None
        if h < tile_size or w < tile_size:
            padded_patch = np.zeros((c, tile_size, tile_size), dtype=np.float32)
            padded_patch[:, :h, :w] = patch
            patch = padded_patch
        return spec, patch

    prefetch_depth = max(2, batch_size * 2)
    with tqdm(total=total_tiles, desc="Sliding Window Inference") as pbar:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='raster-prefetch') as executor:
            pending = deque()
            spec_iter = iter(tile_specs)
            for _ in range(min(prefetch_depth, total_tiles)):
                pending.append(executor.submit(_read_tile, next(spec_iter)))

            while pending:
                future = pending.popleft()
                try:
                    next_spec = next(spec_iter)
                except StopIteration:
                    next_spec = None
                if next_spec is not None:
                    pending.append(executor.submit(_read_tile, next_spec))

                (col_off, row_off, w, h), patch_in = future.result()
                pbar.update(1)
                if patch_in is None:
                    continue

                batch_buffer[batch_count] = patch_in
                batch_count += 1
                batch_offsets.append((col_off, row_off, h, w))
                used_tile_windows.append((col_off, row_off, w, h))

                if batch_count >= batch_size:
                    _flush_batch(batch_buffer[:batch_count], batch_offsets)
                    batch_count = 0
                    batch_offsets.clear()

        if batch_count:
            _flush_batch(batch_buffer[:batch_count], batch_offsets)
            
    with StageTimer('probability_mosaic_finalize', disk_path=temp_root or '.'):
        outputs = mosaic.get_final()
    outputs.tile_windows = used_tile_windows
    return outputs
