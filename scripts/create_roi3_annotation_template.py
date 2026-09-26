"""Create an empty QGIS layer for explicit ROI 3 hard-negative annotation."""

from pathlib import Path

import geopandas as gpd
import pandas as pd


def main() -> None:
    source = Path("outputs/diagnostico_tres_rois/rois/roi_3.geojson")
    output = Path("data/ANNOTATIONS/72696_roi3_sampling_zones.gpkg")
    output.parent.mkdir(parents=True, exist_ok=True)
    roi = gpd.read_file(source).to_crs("EPSG:32722")
    context = roi.copy()
    context["quality_class"] = "LOW_CONFIDENCE"
    context["instruction"] = "Annotate only verified no-row areas in hard_negative_zones"
    context.to_file(output, layer="roi3_context", driver="GPKG")
    empty = gpd.GeoDataFrame(
        {
            "zone_id": pd.Series(dtype="str"),
            "review_status": pd.Series(dtype="str"),
            "notes": pd.Series(dtype="str"),
        },
        geometry=gpd.GeoSeries([], crs=roi.crs),
    )
    empty.to_file(output, layer="hard_negative_zones", driver="GPKG", mode="a")
    empty_lines = gpd.GeoDataFrame(
        {
            "row_id": pd.Series(dtype="str"),
            "review_status": pd.Series(dtype="str"),
            "notes": pd.Series(dtype="str"),
        },
        geometry=gpd.GeoSeries([], crs=roi.crs),
    )
    empty_lines.to_file(output, layer="additional_reference_lines", driver="GPKG", mode="a")
    print(output)


if __name__ == "__main__":
    main()
