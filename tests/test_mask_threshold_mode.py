import numpy as np
import pytest

from src.postprocessing.mask_cleaning import clean_mask


def _config(mode, **values):
    return {
        "postprocessing": {
            "threshold_mode": mode,
            "center_threshold": 0.50,
            "center_threshold_low": 0.20,
            "center_threshold_high": 0.40,
            "min_component_area_m2": 0.0,
            "block_processing_threshold_px": 10_000,
            **values,
        }
    }


def test_single_threshold_does_not_keep_low_probability_bridge():
    probability = np.zeros((7, 11), dtype=np.float32)
    probability[3, 1:4] = 0.80
    probability[3, 4:8] = 0.30
    probability[3, 8:10] = 0.80

    single = clean_mask(probability, 0.10, _config("single"))
    hysteresis = clean_mask(probability, 0.10, _config("hysteresis"))

    assert single[3, 5] == 0
    assert hysteresis[3, 5] == 1


def test_invalid_threshold_mode_is_rejected():
    with pytest.raises(ValueError, match="threshold_mode"):
        clean_mask(np.ones((3, 3), dtype=np.float32), 0.10, _config("unknown"))
