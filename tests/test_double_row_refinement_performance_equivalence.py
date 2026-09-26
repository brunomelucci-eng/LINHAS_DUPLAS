import numpy as np
from affine import Affine
from shapely.geometry import LineString

from src.postprocessing.double_row_refinement import (
    _resample,
    _sample_probability_xy,
)


def test_segment_resampling_matches_legacy_shapely_interpolation():
    line = LineString([(0, 0), (1.1, 0.2), (1.4, 1.3), (3.8, 1.7)])
    spacing = 0.20
    count = max(2, int(np.ceil(line.length / spacing)) + 1)
    expected_distances = np.linspace(0.0, line.length, count)
    expected_coords = np.asarray([
        [line.interpolate(float(distance)).x, line.interpolate(float(distance)).y]
        for distance in expected_distances
    ])

    distances, coords = _resample(line, spacing)

    assert np.array_equal(distances, expected_distances)
    assert np.max(np.abs(coords - expected_coords)) <= 1e-12


def test_vectorized_bilinear_probability_sampling_matches_legacy_loop():
    raster = np.arange(48, dtype=np.float32).reshape(6, 8) / 47.0
    transform = Affine.translation(100.0, 220.0) * Affine.scale(0.25, -0.25)
    points = np.asarray([
        (100.125, 219.875),
        (100.61, 219.42),
        (101.37, 218.93),
        (99.0, 219.0),
        (102.0, 218.0),
    ])
    inv = ~transform
    expected = np.zeros(len(points), dtype=np.float64)
    height, width = raster.shape
    for index, (x, y) in enumerate(points):
        col_f, row_f = inv * (float(x), float(y))
        if not (0.0 <= row_f < height - 1 and 0.0 <= col_f < width - 1):
            continue
        r0, c0 = int(np.floor(row_f)), int(np.floor(col_f))
        dr, dc = row_f - r0, col_f - c0
        v00 = float(raster[r0, c0])
        v01 = float(raster[r0, c0 + 1])
        v10 = float(raster[r0 + 1, c0])
        v11 = float(raster[r0 + 1, c0 + 1])
        expected[index] = (
            v00 * (1 - dr) * (1 - dc)
            + v01 * (1 - dr) * dc
            + v10 * dr * (1 - dc)
            + v11 * dr * dc
        )

    actual = _sample_probability_xy(points, raster, transform)

    assert np.array_equal(actual, expected)
