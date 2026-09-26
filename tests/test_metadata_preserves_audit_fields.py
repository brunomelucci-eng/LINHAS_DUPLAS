import geopandas as gpd
from shapely.geometry import LineString

from src.geospatial.export import populate_metadata


def test_metadata_preserves_topology_and_pair_fields_without_fake_confidence():
    gdf = gpd.GeoDataFrame(
        {
            "pair_id": ["pair_1"],
            "pair_status": ["valid_pair"],
            "topology_status": ["valid"],
            "has_crossing": [False],
            "mean_probability": [0.63],
            "median_probability": [0.65],
            "p20_probability": [0.44],
        },
        geometry=[LineString([(0, 0), (2, 0)])],
        crs="EPSG:32722",
    )
    result = populate_metadata(gdf, {}, checkpoint_name="best.pt")
    assert result.loc[0, "pair_id"] == "pair_1"
    assert result.loc[0, "topology_status"] == "valid"
    assert result.loc[0, "mean_probability"] == 0.63
    assert "confidence" not in result.columns
    assert result.loc[0, "source_model"] == "best.pt"
