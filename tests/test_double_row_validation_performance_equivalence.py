import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from src.postprocessing.double_row_validation import (
    _sample_perpendicular_distances,
    validate_double_rows,
)


def _config():
    return {
        "postprocessing": {
            "double_row_validation": {
                "enabled": True,
                "mode": "annotate",
                "intra_pair_min_m": 0.85,
                "intra_pair_max_m": 1.00,
                "inter_pair_min_m": 1.15,
                "inter_pair_max_m": 1.40,
                "tolerance_m": 0.08,
                "max_angle_difference_deg": 8.0,
                "min_longitudinal_overlap_ratio": 0.50,
            }
        }
    }


def test_vectorized_perpendicular_sampling_matches_legacy_loop():
    line_a = LineString([(0, 0), (2, 0.1), (4, -0.1), (8, 0)])
    line_b = LineString([(0, 0.92), (3, 1.02), (8, 0.88)])
    source, target = (line_a, line_b) if line_a.length <= line_b.length else (line_b, line_a)
    n = max(2, min(20, max(2, int(source.length))))
    legacy = np.asarray([
        source.interpolate(t, normalized=True).distance(target)
        for t in np.linspace(0.0, 1.0, n)
    ])

    vectorized = _sample_perpendicular_distances(line_a, line_b, n_samples=20)

    assert np.array_equal(vectorized, legacy)


def test_preliminary_mode_preserves_every_valid_pair_used_by_refinement():
    lines = [
        LineString([(0, 0), (10, 0)]),
        LineString([(0, 0.92), (10, 0.92)]),
        LineString([(0, 2.20), (10, 2.20)]),
        LineString([(0, 4.0), (3, 5.0)]),
    ]
    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:31982")

    complete = validate_double_rows(gdf, _config())
    preliminary = validate_double_rows(gdf, _config(), preliminary=True)

    complete_valid = complete["pair_status"].eq("valid_pair")
    preliminary_valid = preliminary["pair_status"].eq("valid_pair")
    assert preliminary_valid.equals(complete_valid)
    assert preliminary.loc[complete_valid, "pair_id"].tolist() == complete.loc[complete_valid, "pair_id"].tolist()
    assert np.array_equal(
        preliminary.loc[complete_valid, "pair_confidence"].to_numpy(),
        complete.loc[complete_valid, "pair_confidence"].to_numpy(),
    )
