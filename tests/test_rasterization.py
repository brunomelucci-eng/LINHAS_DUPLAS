import pytest
import numpy as np
from shapely.geometry import LineString
from src.data.rasterization import get_line_tangents

def test_get_line_tangents_horizontal():
    # Horizontal line pointing right: theta = 0
    # sin(2*0) = 0, cos(2*0) = 1
    line = LineString([(0, 0), (10, 0)])
    tangents = get_line_tangents(line, step_m=2.0)
    assert len(tangents) > 0
    for pt in tangents:
        x, y, s, c = pt
        assert pytest.approx(s, abs=1e-5) == 0.0
        assert pytest.approx(c, abs=1e-5) == 1.0

def test_get_line_tangents_vertical():
    # Vertical line pointing up: theta = pi/2
    # sin(2 * pi/2) = sin(pi) = 0, cos(2 * pi/2) = cos(pi) = -1
    line = LineString([(0, 0), (0, 10)])
    tangents = get_line_tangents(line, step_m=2.0)
    assert len(tangents) > 0
    for pt in tangents:
        x, y, s, c = pt
        assert pytest.approx(s, abs=1e-5) == 0.0
        assert pytest.approx(c, abs=1e-5) == -1.0
