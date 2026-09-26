import numpy as np
from shapely.geometry import LineString

from src.postprocessing.curve_fitting import fit_curve_to_points


def test_curve_fitting_respects_max_deviation():
    points = np.array([(0, 0), (1, 0.6), (2, -0.5), (3, 0.5), (4, 0)], dtype=float)
    config = {
        "curve_fitting": {
            "enabled": True,
            "smoothing_tolerance_m": 2.0,
            "max_deviation_m": 0.01,
            "fallback_to_simplified": False,
            "reject_self_intersection": True,
        },
        "vector": {"vertex_spacing_m": 0.2, "simplify_tolerance_m": 0.01},
    }
    fitted = fit_curve_to_points(points, config)
    assert LineString(points).hausdorff_distance(LineString(fitted)) <= 0.01
    assert np.array_equal(fitted[0], points[0])
    assert np.array_equal(fitted[-1], points[-1])
