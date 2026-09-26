from src.postprocessing.network_cleanup import cleanup_network


CONFIG = {
    "network_cleanup": {
        "enabled": True,
        "max_branch_angle_deg": 8,
        "max_short_branch_m": 1.5,
        "reject_degree_gt_2": True,
        "reject_short_cycles": True,
        "max_cycle_perimeter_m": 5.0,
    }
}


def test_keeps_straight_path_and_removes_short_lateral_branch():
    branches = [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 2), (0, 3), (0, 4)],
        [(0, 2), (1, 2)],
    ]
    result = cleanup_network(branches, CONFIG, gsd=1.0)
    assert result == branches[:2]


def test_preserves_legitimate_curve_without_junction():
    curve = [(0, 0), (0, 1), (1, 2), (2, 3)]
    assert cleanup_network([curve], CONFIG, gsd=1.0) == [curve]


def test_breaks_short_diamond_cycle():
    diamond = [
        [(0, 0), (1, 1)],
        [(1, 1), (0, 2)],
        [(0, 2), (-1, 1)],
        [(-1, 1), (0, 0)],
    ]
    result = cleanup_network(diamond, CONFIG, gsd=0.5)
    assert len(result) == 3
