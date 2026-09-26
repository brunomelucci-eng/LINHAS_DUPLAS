import numpy as np
import pytest
from src.postprocessing.thinning import perform_thinning_blocked, perform_thinning

def test_thinning_block_seams_continuity():
    # 1. Create a binary mask (200x200) with a continuous horizontal crop-row line
    # crossing vertical block boundaries at col 64 and col 128.
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[95:105, :] = 255  # a horizontal stripe of width 10
    
    # Configuration with small block size (64) to force boundary crossings
    config = {
        'postprocessing': {
            'thinning': {
                'block_size_px': 64,
                'block_overlap_px': 8,
                'reconcile_seams': True,
                'seam_half_width_px': 4,
                'seam_max_distance_px': 3.0,
                'block_processing_threshold_px': 0, # force blocked processing
                'backend': 'opencv'
            }
        }
    }
    
    # 2. Perform thinned skeletonisation
    thinned = perform_thinning_blocked(mask, config)
    
    # 3. Check that the thinned line is continuous (no gap at col 64 or 128)
    # The line was drawn at y=100. Let's trace it from left to right.
    # In each column, there should be at least one skeleton pixel, and they should form a path.
    cols_with_skeleton = [int(np.any(thinned[:, col])) for col in range(200)]
    
    # There should be no gap (i.e. every column must have a skeleton pixel)
    assert all(cols_with_skeleton), "Thinning created a gap across block boundaries!"
    
    # The number of skeleton pixels should be small (close to 200, i.e. 1 pixel per col)
    assert np.count_nonzero(thinned) < 400, f"Thinning output has too many pixels: {np.count_nonzero(thinned)}"


def test_thinning_block_seams_disabled():
    # Verify that block processing runs and reconstructs the image when reconcile_seams is False
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[45:55, :] = 255
    
    config = {
        'postprocessing': {
            'thinning': {
                'block_size_px': 32,
                'block_overlap_px': 4,
                'reconcile_seams': False,
                'block_processing_threshold_px': 0,
                'backend': 'opencv'
            }
        }
    }
    thinned = perform_thinning_blocked(mask, config)
    assert np.any(thinned > 0)
