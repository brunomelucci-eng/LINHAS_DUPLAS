from shapely.geometry import LineString

from src.postprocessing.double_row_refinement import Track, _select_reference_track


def _track(track_id, index, quality="GOOD"):
    return Track(track_id, (index,), 0.0, 5.0, quality)


def test_neighbour_is_not_used_when_no_confirmed_sister_exists():
    lines = [
        LineString([(0, 0), (5, 0)]),
        LineString([(0, 0.75), (5, 0.75)]),
    ]
    target = _track(1, 0, "SUSPECT")
    neighbour = _track(2, 1, "GOOD")
    config = {
        "double_row_refinement": {
            "reference_guided_repair": {
                "require_same_pair_id": True,
                "allow_neighbor_reference": False,
            }
        }
    }

    selected = _select_reference_track(
        target, [neighbour], lines, [None, None], config
    )

    assert selected is None


def test_confirmed_sister_with_same_pair_id_is_selected():
    lines = [
        LineString([(0, 0), (5, 0)]),
        LineString([(0, 0.75), (5, 0.75)]),
    ]
    target = _track(1, 0, "SUSPECT")
    sister = _track(2, 1, "GOOD")
    config = {
        "double_row_refinement": {
            "reference_guided_repair": {
                "require_same_pair_id": True,
                "allow_neighbor_reference": False,
            }
        }
    }

    selected = _select_reference_track(target, [sister], lines, [7, 7], config)

    assert selected == sister
