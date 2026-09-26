from shapely.geometry import LineString

from src.postprocessing.topology_validation import analyze_topology


CONFIG = {"topology_validation": {"node_snap_tolerance_m": 0.001, "max_cycle_perimeter_m": 5.0}}


def test_detects_crossing_lines():
    records = analyze_topology(
        [LineString([(0, 0), (2, 0)]), LineString([(1, -1), (1, 1)])], CONFIG
    )
    assert all(record["has_crossing"] for record in records)
    assert all(record["node_degree_max"] == 4 for record in records)


def test_detects_y_branch():
    lines = [
        LineString([(0, 0), (1, 0)]),
        LineString([(1, 0), (2, 1)]),
        LineString([(1, 0), (2, -1)]),
    ]
    records = analyze_topology(lines, CONFIG)
    assert all(record["has_branch"] for record in records)
    assert max(record["node_degree_max"] for record in records) == 3


def test_detects_short_diamond_cycle():
    lines = [
        LineString([(0, 0), (1, 0.5)]),
        LineString([(1, 0.5), (2, 0)]),
        LineString([(2, 0), (1, -0.5)]),
        LineString([(1, -0.5), (0, 0)]),
    ]
    records = analyze_topology(lines, CONFIG)
    assert all(record["has_short_cycle"] for record in records)


def test_parallel_rows_are_valid():
    records = analyze_topology(
        [LineString([(0, 0), (10, 0)]), LineString([(0, 1), (10, 1)])], CONFIG
    )
    assert all(record["topology_status"] == "valid" for record in records)


def test_curve_is_valid():
    records = analyze_topology([LineString([(0, 0), (1, 0.2), (2, 0.7), (3, 1.5)])], CONFIG)
    assert records[0]["topology_status"] == "valid"
