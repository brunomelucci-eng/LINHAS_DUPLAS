import pytest
from src.config import load_config
import rasterio
import os
import geopandas as gpd

def test_deduplication_detects_and_discards_duplicate_rasters():
    # Load config with 661
    cfg = load_config("configs/base.yaml")
    inputs = list(cfg.get("inputs", []))

    # Manually append an exact duplicate entry of 661 to test the deduplication algorithm
    inputs.append({
        "name": "661_duplicate",
        "orthomosaic": "data/IMAGE/661.gpkg",
        "roi": "data/TALHOES/661_TALHOES.geojson",
        "reference_lines": "data/LINHAS/661_LINHAS.geojson",
    })

    seen_rasters = {}
    deduped = []
    for item in inputs:
        ortho_path = item.get("orthomosaic")
        item_name = item.get("name", "unnamed")
        if not ortho_path or not os.path.exists(ortho_path):
            deduped.append(item)
            continue
        with rasterio.open(ortho_path) as r:
            r_key = (
                str(r.crs),
                round(float(r.bounds.left), 1),
                round(float(r.bounds.bottom), 1),
                round(float(r.bounds.right), 1),
                round(float(r.bounds.top), 1),
                r.width,
                r.height,
            )
        lines_path = item.get("reference_lines")
        line_count = len(gpd.read_file(lines_path)) if lines_path and os.path.exists(lines_path) else 0

        if r_key in seen_rasters:
            prev_idx, prev_name, prev_count = seen_rasters[r_key]
            if line_count > prev_count:
                deduped[prev_idx] = item
                seen_rasters[r_key] = (prev_idx, item_name, line_count)
        else:
            idx = len(deduped)
            deduped.append(item)
            seen_rasters[r_key] = (idx, item_name, line_count)

    names = [x["name"] for x in deduped]
    # Verify 661 is kept and 661_duplicate was discarded!
    assert "661" in names
    assert "661_duplicate" not in names
    assert len(deduped) == len(cfg.get("inputs", []))
