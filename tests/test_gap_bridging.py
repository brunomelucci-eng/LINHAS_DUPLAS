import pytest
import numpy as np
import rasterio
from src.postprocessing.gap_bridging import bridge_gaps

def test_gap_bridging_aligned():
    # Two horizontal rows separated by a 2-pixel gap
    # GSD = 0.1 -> distance = 0.2m
    transform = rasterio.Affine(0.1, 0.0, 0.0, 0.0, -0.1, 10.0)
    
    branches = [
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(2, 6), (2, 7), (2, 8), (2, 9)]
    ]
    
    center_prob = np.ones((20, 20), dtype=np.float32)
    
    config = {
        'gap_bridging': {
            'enabled': True,
            'max_gap_m': 1.0,
            'max_angle_difference_deg': 20.0,
            'max_lateral_offset_m': 0.5,
            'min_corridor_probability': 0.10
        }
    }
    
    merged = bridge_gaps(branches, center_prob, transform, gsd=0.1, config=config)
    
    # Check they reassembled into 1 branch
    assert len(merged) == 1
    assert len(merged[0]) == 8


def _safe_config(**overrides):
    settings = {
        'enabled': True,
        'max_gap_m': 0.6,
        'max_angle_deg': 5.0,
        'max_lateral_offset_m': 0.05,
        'min_corridor_probability': 0.0,
        'require_facing_endpoints': True,
        'reject_crossing_connections': True,
        'reject_neighbor_row_connections': True,
        'max_connections_per_endpoint': 1,
    }
    settings.update(overrides)
    return {'gap_bridging': settings}


def test_rejects_connection_crossing_another_row():
    transform = rasterio.Affine(0.1, 0.0, 0.0, 0.0, -0.1, 10.0)
    branches = [
        [(5, 0), (5, 1), (5, 2), (5, 3)],
        [(5, 7), (5, 8), (5, 9), (5, 10)],
        [(2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5)],
    ]
    result = bridge_gaps(
        branches, np.ones((20, 20), dtype=np.float32), transform, 0.1, _safe_config()
    )
    assert len(result) == 3


def test_rejects_lateral_neighbor_row_connection():
    transform = rasterio.Affine(0.1, 0.0, 0.0, 0.0, -0.1, 10.0)
    branches = [
        [(5, 0), (5, 1), (5, 2), (5, 3)],
        [(6, 5), (6, 6), (6, 7), (6, 8)],
    ]
    result = bridge_gaps(
        branches, np.ones((20, 20), dtype=np.float32), transform, 0.1, _safe_config()
    )
    assert len(result) == 2


def test_uses_each_endpoint_at_most_once():
    transform = rasterio.Affine(0.1, 0.0, 0.0, 0.0, -0.1, 10.0)
    branches = [
        [(5, 0), (5, 1), (5, 2), (5, 3)],
        [(5, 6), (5, 7), (5, 8)],
        [(5, 7), (5, 8), (5, 9)],
    ]
    result = bridge_gaps(
        branches, np.ones((20, 20), dtype=np.float32), transform, 0.1, _safe_config()
    )
    assert len(result) == 2
