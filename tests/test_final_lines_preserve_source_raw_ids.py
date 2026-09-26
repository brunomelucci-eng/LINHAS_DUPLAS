import json

import geopandas as gpd
from shapely.geometry import LineString, Polygon

from src.postprocessing.output_products import prepare_final_lines
from src.postprocessing.postprocess_pipeline import run_postprocessing


def _config():
    return {
        "postprocess_pipeline": {
            "enabled": True,
            "version": "test-v1",
            "prevector_cleanup": {"enabled": False},
            "track_merge": {"enabled": True},
            "fail_on_unmatched": True,
            "fail_on_crossing": True,
        },
        "postprocessing": {"double_row_validation": {"enabled": False}},
        "deduplication": {"enabled": False},
        "double_row_refinement": {"enabled": False},
        "final_centerline_fairing": {"enabled": False},
        "line_extension": {"enabled": False},
        "vector": {"min_line_length_m": 0.1},
    }


def test_final_lines_preserve_source_raw_ids(tmp_path):
    del tmp_path  # The lineage contract is exercised in memory by the real engine.
    raw = gpd.GeoDataFrame(
        {
            "raw_line_id": ["raw_001", "raw_002", "raw_003"],
            "source_raw_ids": ["raw_001", "raw_002", "raw_003"],
            "track_id": [7, 7, 8],
            "pair_id": [1, 1, 1],
            "pair_status": ["valid_pair", "valid_pair", "valid_pair"],
        },
        geometry=[
            LineString([(1.0, 0.0), (5.0, 0.0)]),
            LineString([(5.0, 0.0), (9.0, 0.0)]),
            LineString([(1.0, 0.9), (9.0, 0.9)]),
        ],
        crs="EPSG:3857",
    )
    roi = gpd.GeoDataFrame(
        geometry=[Polygon([(0, -1), (10, -1), (10, 2), (0, 2)])],
        crs=raw.crs,
    )

    result = run_postprocessing(raw, None, None, roi, _config())
    final = prepare_final_lines(
        result.final_lines,
        run_id="run_lineage",
        created_at="2026-07-20T12:00:00+00:00",
        valid_raw_line_ids=raw["raw_line_id"].tolist(),
    )

    assert len(final) == 2
    assert final["final_line_id"].is_unique
    merged = final.loc[final["track_id"].eq(7)].iloc[0]
    untouched = final.loc[final["track_id"].eq(8)].iloc[0]
    assert json.loads(merged["source_raw_ids"]) == ["raw_001", "raw_002"]
    assert json.loads(untouched["source_raw_ids"]) == ["raw_003"]
    assert bool(merged["was_merged"]) is True
    assert bool(untouched["was_merged"]) is False
