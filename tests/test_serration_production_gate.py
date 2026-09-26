import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.affinity import rotate
from shapely.geometry import LineString

from src.postprocessing.output_products import select_production_lines
from src.postprocessing.regularized_centerline_fairing import (
    annotate_serration_status,
    compute_serration_diagnostics,
)


def _config(gsd):
    return {
        "final_centerline_fairing": {
            "enabled": True,
            "micro_serration": {
                "enabled": True,
                "default_gsd_m": gsd,
                "max_sample_spacing_m": 0.05,
                "window_pixels": [3, 5, 9],
                "residual_window_pixels": 9,
                "min_line_length_m": 1.50,
                "min_meaningful_vertex_turn_deg": 0.50,
                "max_micro_p95_turn_deg": 2.50,
                "max_vertex_p95_turn_deg": 3.00,
                "max_lateral_inversions_per_m": 0.75,
                "max_vertex_density_per_m": 8.0,
                "max_length_excess_ratio": 0.0025,
                "max_residual_p95_gsd_ratio": 0.25,
                "max_score": 1.0,
            },
        }
    }


def _pixel_staircase(gsd, angle_deg):
    index = np.arange(420, dtype=float)
    # Digital line: each vertical pixel step is interleaved with horizontal
    # steps, reproducing the repeated horizontal/diagonal skeleton geometry.
    coordinates = np.column_stack((index * gsd, np.floor(index * 0.37) * gsd))
    return rotate(LineString(coordinates), angle_deg, origin=(0.0, 0.0))


@pytest.mark.parametrize("gsd", [0.032, 0.035, 0.045])
@pytest.mark.parametrize("angle_deg", [0.0, 35.0, 90.0, 145.0])
def test_gsd_aware_detector_rejects_pixel_staircases(gsd, angle_deg):
    transform = Affine.rotation(7.0) * Affine.scale(gsd, -gsd)
    diagnostics = compute_serration_diagnostics(
        _pixel_staircase(gsd, angle_deg), _config(gsd), transform
    )

    assert diagnostics.failed
    assert diagnostics.score >= 1.0
    assert diagnostics.vertex_density_per_m >= 8.0


@pytest.mark.parametrize("gsd", [0.032, 0.035, 0.045])
def test_gsd_aware_detector_preserves_dense_smooth_curves(gsd):
    x = np.arange(0.0, 18.0, gsd)
    smooth_wave = LineString(np.column_stack((x, 0.35 * np.sin(x / 7.0))))
    theta = np.linspace(0.0, np.pi / 2.0, int(np.ceil(12.0 * np.pi / (2 * gsd))))
    smooth_arc = LineString(
        np.column_stack((12.0 * np.sin(theta), 12.0 * (1.0 - np.cos(theta))))
    )
    transform = Affine.scale(gsd, -gsd)

    assert not compute_serration_diagnostics(
        smooth_wave, _config(gsd), transform
    ).failed
    assert not compute_serration_diagnostics(
        smooth_arc, _config(gsd), transform
    ).failed


def test_final_audit_is_fail_closed_for_production_lines():
    gsd = 0.035
    clean = LineString([(0.0, 2.0), (15.0, 2.0)])
    serrated = _pixel_staircase(gsd, 0.0)
    lines = gpd.GeoDataFrame(
        {
            "postprocess_status": ["approved", "approved"],
            "review_required": [False, False],
        },
        geometry=[clean, serrated],
        crs="EPSG:3857",
    )

    audited = annotate_serration_status(
        lines, _config(gsd), Affine.scale(gsd, -gsd)
    )
    production = select_production_lines(audited)

    assert audited["serration_status"].tolist() == ["clean", "failed"]
    assert audited["review_required"].tolist() == [False, True]
    assert len(production) == 1
    assert production.iloc[0].geometry.equals(clean)
    assert production["serration_status"].eq("clean").all()
