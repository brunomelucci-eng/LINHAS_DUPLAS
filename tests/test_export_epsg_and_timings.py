"""Testes explícitos de CRS persistido e dicionário timings.

Auditoria pendente (seção "Pendência arquitetural / últimas alterações ainda
sem revalidação local" do documento de continuação):

1. ``export_epsg`` diferente do CRS de trabalho: verificar que o GPKG persiste
   no CRS exportado, não no CRS de trabalho.
2. ``timings``: verificar que ``persist_two_stage_vector_outputs`` preenche a
   chave ``postprocessing`` no dicionário fornecido e que o valor é um número
   positivo.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from src.postprocessing.output_products import (
    create_output_layout,
    persist_two_stage_vector_outputs,
)
from src.postprocessing.postprocess_pipeline import run_postprocessing

# ---------------------------------------------------------------------------
# Fixtures compartilhadas
# ---------------------------------------------------------------------------

WORKING_EPSG = 3857  # CRS de trabalho (métrico)
EXPORT_EPSG = 4326   # CRS de exportação (geográfico — diferente do trabalho)


def _two_line_gdf():
    """Par de linhas válido no CRS de trabalho."""
    return gpd.GeoDataFrame(
        {
            "track_id": [10, 11],
            "pair_id": [1, 1],
            "pair_status": ["valid_pair", "valid_pair"],
        },
        geometry=[
            LineString([(1.0, 0.0), (9.0, 0.0)]),
            LineString([(1.0, 0.9), (9.0, 0.9)]),
        ],
        crs=f"EPSG:{WORKING_EPSG}",
    )


def _roi_gdf():
    return gpd.GeoDataFrame(
        geometry=[Polygon([(0, -1), (10, -1), (10, 2), (0, 2)])],
        crs=f"EPSG:{WORKING_EPSG}",
    )


def _probability():
    transform = Affine.translation(0.0, 3.0) * Affine.scale(0.10, -0.10)
    prob = np.linspace(0.55, 0.95, 50 * 120, dtype=np.float32).reshape(50, 120)
    return prob, transform


def _config(export_epsg: int):
    return {
        "outputs": {
            "always_save_post_inference": True,
            "always_save_postprocessing": True,
            "overwrite_existing": False,
            "save_lineage": True,
            "save_comparison_report": True,
        },
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
        "metadata": {"probability_sample_spacing_m": 0.10},
        "crs": {"export_epsg": export_epsg},
    }


# ---------------------------------------------------------------------------
# Teste 1 — CRS persistido é o export_epsg, não o CRS de trabalho
# ---------------------------------------------------------------------------

def test_persisted_crs_matches_export_epsg(tmp_path):
    """O GPKG gravado deve ter CRS igual a export_epsg, não ao CRS de trabalho.

    O CRS de trabalho é EPSG:3857; o export_epsg é EPSG:4326. Após a
    persistência os dois GPKGs devem estar em EPSG:4326.
    """
    config = _config(export_epsg=EXPORT_EPSG)
    prob, transform = _probability()
    layout = create_output_layout(tmp_path / "run", "talhao.gpkg", config)

    persist_two_stage_vector_outputs(
        _two_line_gdf(),
        prob,
        transform,
        _roi_gdf(),
        config,
        layout=layout,
        run_id="run_crs_test",
        checkpoint="best.pt",
        postprocess_fn=run_postprocessing,
        export_epsg=EXPORT_EPSG,
    )

    # Produto bruto
    raw = gpd.read_file(layout.post_inference_path, layer="post_inference_lines")
    assert raw.crs is not None, "post_inference_lines sem CRS"
    assert raw.crs.to_epsg() == EXPORT_EPSG, (
        f"post_inference_lines em EPSG:{raw.crs.to_epsg()}, esperado EPSG:{EXPORT_EPSG}"
    )

    # Produto final
    final = gpd.read_file(layout.postprocessing_path, layer="final_lines")
    assert final.crs is not None, "final_lines sem CRS"
    assert final.crs.to_epsg() == EXPORT_EPSG, (
        f"final_lines em EPSG:{final.crs.to_epsg()}, esperado EPSG:{EXPORT_EPSG}"
    )

    # CRS de trabalho não deve aparecer no GPKG
    assert raw.crs.to_epsg() != WORKING_EPSG, (
        "post_inference_lines foi gravado no CRS de trabalho em vez do export_epsg"
    )
    assert final.crs.to_epsg() != WORKING_EPSG, (
        "final_lines foi gravado no CRS de trabalho em vez do export_epsg"
    )


# ---------------------------------------------------------------------------
# Teste 2 — dicionário timings recebe chave 'postprocessing' com valor positivo
# ---------------------------------------------------------------------------

def test_timings_dict_receives_postprocessing_key(tmp_path):
    """persist_two_stage_vector_outputs deve preencher timings['postprocessing'].

    O valor deve ser um float positivo que reflete apenas o tempo de execução
    de run_postprocessing(), separado de preparação e I/O.
    """
    config = _config(export_epsg=WORKING_EPSG)
    prob, transform = _probability()
    layout = create_output_layout(tmp_path / "run", "talhao.gpkg", config)
    timings: dict = {}

    persist_two_stage_vector_outputs(
        _two_line_gdf(),
        prob,
        transform,
        _roi_gdf(),
        config,
        layout=layout,
        run_id="run_timings_test",
        checkpoint="best.pt",
        postprocess_fn=run_postprocessing,
        export_epsg=WORKING_EPSG,
        timings=timings,
    )

    assert "postprocessing" in timings, (
        f"Chave 'postprocessing' ausente em timings. Chaves encontradas: {list(timings)}"
    )
    value = timings["postprocessing"]
    assert isinstance(value, float), (
        f"timings['postprocessing'] deve ser float, obtido {type(value)}"
    )
    assert value >= 0.0, (
        f"timings['postprocessing'] deve ser não-negativo, obtido {value}"
    )
    # Sanidade: o tempo de pós-processamento não deve ser absurdamente longo em
    # um teste sintético (< 30 s é um limite muito generoso).
    assert value < 30.0, (
        f"timings['postprocessing'] suspeito: {value:.3f} s em teste sintético"
    )
