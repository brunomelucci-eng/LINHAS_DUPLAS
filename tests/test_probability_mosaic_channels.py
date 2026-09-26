import numpy as np
import os
from src.inference.probability_mosaic import ProbabilityMosaic

def test_probability_mosaic_channels_selection():
    # Configure ProbabilityMosaic to only store the 'center' channel
    config = {
        'inference': {
            'output_channels_to_store': ['center'],
            'memmap_threshold_px': 999999999999, # force in-memory
            'final_probability_dtype': 'float32'
        },
        'tiling': {
            'tile_size_px': 64
        }
    }
    
    mosaic = ProbabilityMosaic(width=100, height=100, config=config)
    
    # Check shape of internal accumulator (should be 1 channel, not 4)
    assert mosaic.accum.shape == (1, 100, 100)
    assert mosaic.num_channels == 1
    assert mosaic.selected_indices == [1]
    
    # Create dummy patch prediction (4, 64, 64)
    patch = np.ones((4, 64, 64), dtype=np.float32)
    mosaic.update(0, 0, patch)
    
    # Retrieve final predictions
    outputs = mosaic.get_final()
    row_prob = outputs.row
    center_prob = outputs.center
    sin = outputs.orientation_sin
    cos = outputs.orientation_cos
    
    # Center prob should contain accumulated values, others should be None (not selected)
    assert center_prob.shape == (100, 100)
    assert np.all(center_prob[:64, :64] > 0)
    
    assert row_prob is None
    assert sin is None
    assert cos is None


def test_release_accumulators_keeps_final_memmap_available(tmp_path):
    config = {
        'inference': {
            'output_channels_to_store': ['center'],
            'memmap_threshold_px': 0,
            'final_probability_dtype': 'float16',
            'max_temp_disk_gb': 1.0,
        },
        'tiling': {'tile_size_px': 8},
    }
    mosaic = ProbabilityMosaic(
        width=16,
        height=16,
        config=config,
        temp_root=str(tmp_path),
    )
    mosaic.update(0, 0, np.ones((4, 8, 8), dtype=np.float32))
    outputs = mosaic.get_final()
    accum_path = mosaic.accum_path
    weights_path = mosaic.weights_path

    mosaic.release_accumulators()

    assert mosaic.accum is None
    assert mosaic.weights is None
    assert not os.path.exists(accum_path)
    assert not os.path.exists(weights_path)
    assert np.all(outputs.center[:8, :8] > 0)

    outputs.center._mmap.close()
    mosaic.close()


def test_streaming_mosaic_matches_full_accumulator(tmp_path):
    base_inference = {
        'output_channels_to_store': ['center'],
        'final_probability_dtype': 'float32',
        'max_temp_disk_gb': 1.0,
    }
    legacy = ProbabilityMosaic(
        width=13,
        height=11,
        config={
            'inference': {**base_inference, 'memmap_threshold_px': 10**9},
            'tiling': {'tile_size_px': 8},
        },
    )
    streaming = ProbabilityMosaic(
        width=13,
        height=11,
        config={
            'inference': {
                **base_inference,
                'memmap_threshold_px': 0,
                'streaming_mosaic': True,
            },
            'tiling': {'tile_size_px': 8},
        },
        temp_root=str(tmp_path),
    )

    rng = np.random.default_rng(20260719)
    for row_off in (0, 3):
        for col_off in (0, 5):
            patch = rng.random((4, 8, 8), dtype=np.float32)
            legacy.update(col_off, row_off, patch)
            streaming.update(col_off, row_off, patch)

    expected = legacy.get_final().center
    streamed = streaming.get_final().center
    assert np.array_equal(streamed, expected)
    assert np.array_equal(streamed >= 0.42, expected >= 0.42)
    assert streaming.accum.shape == (1, 8, 13)

    streamed._mmap.close()
    streaming.close()
