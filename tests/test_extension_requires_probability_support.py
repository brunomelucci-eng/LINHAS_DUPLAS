import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString, box

from src.postprocessing.line_extension import extend_lines_to_roi


CONFIG = {
    "line_extension": {
        "enabled": True,
        "max_extension_m": 0.5,
        "require_probability_support": True,
        "min_probability": 0.22,
        "min_supported_fraction": 0.8,
    }
}


def test_extension_requires_probability_support():
    line = LineString([(0.5, 0.5), (0.8, 0.5)])
    roi = box(0, 0, 1, 1)
    transform = from_origin(0, 1, 0.1, 0.1)
    unsupported = np.zeros((10, 10), dtype=np.float32)
    assert extend_lines_to_roi([line], roi, CONFIG, unsupported, transform)[0].equals(line)

    supported = np.ones((10, 10), dtype=np.float32)
    extended = extend_lines_to_roi([line], roi, CONFIG, supported, transform)[0]
    assert extended.length > line.length
