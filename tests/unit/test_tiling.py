from __future__ import annotations

import numpy as np
import pytest

from rowgraphnet.data import HannBlender, blend_tiles, generate_tiles, hann_blend_weights


def test_tiles_are_in_bounds_deterministic_and_cover_every_pixel():
    shape = (77, 103)
    windows = generate_tiles(shape, tile_size=(32, 40), overlap=(8, 10))
    repeated = generate_tiles(shape, tile_size=(32, 40), overlap=(8, 10))

    assert windows == repeated
    assert [window.tile_id for window in windows] == list(range(len(windows)))
    coverage = np.zeros(shape, dtype=np.uint16)
    for window in windows:
        assert window.row_off >= 0 and window.col_off >= 0
        assert window.row_off + window.height <= shape[0]
        assert window.col_off + window.width <= shape[1]
        coverage[window.slices] += 1
    assert coverage.min() == 1
    assert coverage.max() > 1
    assert windows[-1].row_off + windows[-1].height == shape[0]
    assert windows[-1].col_off + windows[-1].width == shape[1]


def test_image_smaller_than_tile_produces_one_clipped_window():
    windows = generate_tiles((11, 17), tile_size=32, overlap=8)

    assert len(windows) == 1
    assert windows[0].row_off == windows[0].col_off == 0
    assert (windows[0].height, windows[0].width) == (11, 17)


@pytest.mark.parametrize(
    ("tile_size", "overlap"),
    [((32, 32), 32), ((32, 32), (31, 32)), (0, 0), (32, -1)],
)
def test_invalid_tile_parameters_are_rejected(tile_size, overlap):
    with pytest.raises(ValueError):
        generate_tiles((100, 100), tile_size=tile_size, overlap=overlap)


def test_hann_weights_are_positive_symmetric_and_normalized():
    weights = hann_blend_weights((17, 23), dtype=np.float64)

    assert weights.shape == (17, 23)
    assert np.all(weights > 0)
    assert weights.max() == pytest.approx(1.0)
    np.testing.assert_allclose(weights, weights[::-1, :])
    np.testing.assert_allclose(weights, weights[:, ::-1])


@pytest.mark.parametrize("channels", [None, 3])
def test_hann_blending_exactly_reconstructs_consistent_overlapping_tiles(channels):
    height, width = 57, 83
    yy, xx = np.mgrid[:height, :width]
    base = (yy * 0.1 + xx * 0.01).astype(np.float32)
    source = base if channels is None else np.stack([base, base + 1, base * 2])
    windows = generate_tiles((height, width), tile_size=(24, 31), overlap=(7, 9))
    tiled = [(window, source[..., window.row_slice, window.col_slice]) for window in windows]

    result = blend_tiles(tiled, source.shape)

    np.testing.assert_allclose(result, source, rtol=1e-6, atol=1e-6)


def test_blender_refuses_to_finalize_uncovered_output():
    blender = HannBlender((20, 20))
    window = generate_tiles((10, 10), tile_size=10)[0]
    blender.add(np.ones((10, 10), dtype=np.float32), window)

    with pytest.raises(ValueError, match="uncovered"):
        blender.finalize()
