import pytest
from unittest.mock import MagicMock
from src.data.raster_reprojection import resolve_output_nodata

def test_resolve_nodata_rgb_no_nodata():
    # Caso 1 — RGB sem NoData
    src = MagicMock()
    src.nodatavals = [None, None, None]
    bands = [1, 2, 3]
    assert resolve_output_nodata(src, bands) is None

def test_resolve_nodata_rgb_zero_as_nodata():
    # Caso 2 — RGB com pixel preto válido (0 não deve ser NoData)
    src = MagicMock()
    src.nodatavals = [0, 0, 0]
    bands = [1, 2, 3]
    assert resolve_output_nodata(src, bands) is None

def test_resolve_nodata_explicit_valid():
    # Caso 3 — NoData explícito diferente de zero
    src = MagicMock()
    src.nodatavals = [255, 255, 255]
    bands = [1, 2, 3]
    assert resolve_output_nodata(src, bands) == 255

def test_resolve_nodata_inconsistent():
    # Caso 4 — bandas com NoData diferentes
    src = MagicMock()
    src.nodatavals = [255, 0, 255]
    bands = [1, 2, 3]
    assert resolve_output_nodata(src, bands) is None

def test_resolve_nodata_partial_none():
    src = MagicMock()
    src.nodatavals = [255, None, 255]
    bands = [1, 2, 3]
    assert resolve_output_nodata(src, bands) is None
