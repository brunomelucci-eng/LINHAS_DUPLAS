import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString

from src.geospatial.export import sample_line_probabilities


def test_export_uses_real_probability_samples():
    raster = np.zeros((5, 5), dtype=np.float32)
    raster[2, :] = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    transform = from_origin(0, 5, 1, 1)
    line = LineString([(0.5, 2.5), (4.5, 2.5)])
    gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:32722")
    sampled = sample_line_probabilities(gdf, raster, transform, spacing_m=1.0)
    assert np.isclose(sampled.loc[0, "mean_probability"], 0.5)
    assert np.isclose(sampled.loc[0, "median_probability"], 0.5)
    assert sampled.loc[0, "p20_probability"] < 0.5
