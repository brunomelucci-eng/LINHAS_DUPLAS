import numpy as np

import src.postprocessing.curve_fitting as curve_fitting
from src.postprocessing.curve_fitting import fit_curve_to_points


def test_curve_fitting_preserves_good_line():
    points = np.array([(0, 0), (1, 0), (2, 0), (3, 0)], dtype=float)
    config = {
        "curve_fitting": {
            "enabled": True,
            "smoothing_tolerance_m": 0.05,
            "max_deviation_m": 0.10,
        }
    }
    fitted, metadata = fit_curve_to_points(points, config, return_metadata=True)
    assert np.array_equal(fitted, points)
    assert metadata == {"fit_quality_class": "GOOD", "fit_type": "preserved"}


def test_disabled_fitting_bypasses_geometry_operations_and_preserves_metadata(monkeypatch):
    branches = [
        np.array([(0, 0), (1, 0), (2, 0), (3, 0)], dtype=float),
        np.array([(0, 0), (1, 0.6), (2, -0.5), (3, 0.5), (4, 0)], dtype=float),
    ]
    config = {
        "curve_fitting": {
            "enabled": False,
            "smoothing_tolerance_m": 0.08,
            "max_deviation_m": 0.20,
        }
    }
    expected_metadata = [
        fit_curve_to_points(branch, config, return_metadata=True)[1]
        for branch in branches
    ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("smooth_and_fit_branches must not run when disabled")

    monkeypatch.setattr(curve_fitting, "smooth_and_fit_branches", fail_if_called)
    fitted, metadata = curve_fitting.fit_branches_if_enabled(
        branches, config, return_metadata=True
    )

    assert fitted is branches
    assert metadata == expected_metadata
    assert all(output is source for output, source in zip(fitted, branches))
