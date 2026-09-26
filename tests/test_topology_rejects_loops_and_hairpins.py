from shapely.geometry import LineString

from src.postprocessing.topology_validation import analyze_topology


def _config():
    return {
        "topology_validation": {
            "node_snap_tolerance_m": 0.01,
            "reject_closed_loops": True,
            "reject_hairpins": True,
            "hairpin_min_length_m": 1.0,
            "hairpin_max_endpoint_ratio": 0.35,
            "hairpin_min_total_turn_deg": 120.0,
            "hairpin_sample_spacing_m": 0.10,
        }
    }


def test_closed_loop_is_rejected_regardless_of_perimeter():
    loop = LineString([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])

    record = analyze_topology([loop], _config())[0]

    assert record["has_closed_loop"] is True
    assert "closed_loop" in record["rejection_reason"]


def test_open_teardrop_hairpin_is_rejected():
    hairpin = LineString(
        [(0, 0), (2, 0), (3, 1), (2, 2), (0.25, 2), (0.10, 0.15)]
    )

    record = analyze_topology([hairpin], _config())[0]

    assert record["has_hairpin"] is True
    assert "hairpin" in record["rejection_reason"]


def test_gentle_crop_row_is_not_classified_as_hairpin():
    row = LineString([(0, 0), (2, 0.05), (4, 0.15), (6, 0.30), (8, 0.50)])

    record = analyze_topology([row], _config())[0]

    assert record["has_closed_loop"] is False
    assert record["has_hairpin"] is False
    assert record["topology_status"] == "valid"
