import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.diagnostics import build_roi_diagnostic_report, write_diagnostic_reports


def _write_raster(path, array, transform):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        transform=transform,
        crs="EPSG:31982",
    ) as dst:
        dst.write(array, 1)


def test_roi_diagnostic_reports_required_metrics(tmp_path):
    transform = from_origin(0, 2, 0.25, 0.25)
    probability = np.full((8, 12), 0.1, dtype=np.float32)
    probability[3, 1:11] = 0.8
    mask = (probability >= 0.42).astype(np.uint8)
    skeleton = mask.copy()
    probability_path = tmp_path / "probability.tif"
    mask_path = tmp_path / "mask.tif"
    skeleton_path = tmp_path / "skeleton.tif"
    weights_path = tmp_path / "weights.tif"
    _write_raster(probability_path, probability, transform)
    _write_raster(mask_path, mask, transform)
    _write_raster(skeleton_path, skeleton, transform)
    _write_raster(weights_path, np.ones_like(probability), transform)

    config = {
        "postprocessing": {
            "center_threshold_low": 0.22,
            "center_threshold_high": 0.42,
            "graph": {"backend": "numpy"},
        }
    }
    report = build_roi_diagnostic_report(
        probability_path,
        mask_path,
        skeleton_path,
        config,
        mosaic_weights_path=weights_path,
    )
    topology = report["topology_before_vectorization"]
    assert topology["connected_components"] == 1
    assert topology["skeleton_pixels"] == 10
    assert topology["endpoints"] == 2
    assert topology["branches"] == 1
    assert topology["median_branch_length_m"] == 2.25
    assert report["probability"]["fraction_above_high_threshold"] == 10 / 96

    write_diagnostic_reports(report, tmp_path)
    loaded = json.loads((tmp_path / "diagnostic_report.json").read_text("utf-8"))
    assert loaded == report
    assert "endpoints_per_100m" in (tmp_path / "diagnostic_report.html").read_text("utf-8")
