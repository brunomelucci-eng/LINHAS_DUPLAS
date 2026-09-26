from pathlib import Path

import numpy as np
import rasterio

from src.config import load_config
from scripts.predict import _write_debug_raster


def test_debug_outputs_are_disabled_by_default_and_guarded():
    config = load_config("configs/base.yaml")
    assert config["debug"] == {
        "enabled": False,
        "save_probability": False,
        "save_weights": False,
        "save_masks": False,
        "save_skeleton": False,
        "save_tile_grid": False,
    }
    source = Path("scripts/predict.py").read_text(encoding="utf-8")
    engine_source = Path(
        "src/postprocessing/postprocess_pipeline.py"
    ).read_text(encoding="utf-8")
    # probability_center.tif is a mandatory reproducibility product, while all
    # additional diagnostic rasters remain opt-in.
    assert "output_layout.probability_center_path" in source
    assert "with StageTimer('probability_center_export'" in source
    assert "if debug_enabled and debug_cfg.get('save_skeleton', False):" in source
    assert "debug=debug_enabled" in source
    assert "collect_debug=debug" in engine_source
    assert "if debug_enabled:" in source
    for layer_name in (
        "linhas_originais",
        "linhas_suavizadas",
        "trechos_substituidos_pela_irma",
        "pares_reconstruidos",
        "extensoes_terminais",
        "linhas_rejeitadas",
        "linhas_para_revisao",
        "linhas_finais",
    ):
        assert layer_name in source


def test_roi_refinement_debug_overlay_enables_required_artifacts():
    config = load_config("configs/roi_refinement_debug.yaml")
    assert config["project"]["output_dir"] == "outputs/roi_refinement_debug"
    assert config["debug"] == {
        "enabled": True,
        "save_probability": True,
        "save_weights": False,
        "save_masks": True,
        "save_skeleton": True,
        "save_tile_grid": True,
    }
    assert config["double_row_refinement"]["enabled"] is True


def test_debug_threshold_raster_is_written_by_stripes_without_value_changes(tmp_path):
    values = np.linspace(0.0, 1.0, 2050 * 17, dtype=np.float32).reshape(2050, 17)
    output = tmp_path / 'threshold_stripes.tif'
    transform = rasterio.transform.from_origin(0.0, 2050.0, 1.0, 1.0)

    _write_debug_raster(
        output,
        values,
        transform,
        'EPSG:3857',
        dtype='uint8',
        threshold=0.42,
    )

    with rasterio.open(output) as src:
        written = src.read(1)
    assert np.array_equal(written, (values >= 0.42).astype(np.uint8))
