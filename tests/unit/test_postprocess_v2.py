"""Testes sintéticos do pós-processamento v2 (um caso por problema relatado)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon

from rowgraphnet.geometry.postprocess_v2 import (
    LineV2,
    PostprocessV2Config,
    _reversal_cuts,
    clean_loops_and_hooks,
    export_complete_v2,
    export_uncertain_v2,
    extend_to_field_boundary,
    join_broken_lines,
    run_postprocess_v2,
)

SP = 1.5


def _ln(coords, source: str = "simples") -> LineV2:
    g = LineString(coords)
    return LineV2(geom=g, source=source, confidence=0.9, observed_len=g.length, spacing=SP, orig_ids=[str(id(g))])


def _curve(x0: float, y0: float, y1: float, amp: float = 2.0, step: float = 0.25) -> list[tuple[float, float]]:
    ys = np.arange(y0, y1 + 1e-9, step)
    return [(x0 + amp * np.sin(y / 15.0), float(y)) for y in ys]


def test_hairpin_is_removed() -> None:
    # Linha vai até y=30 e volta paralela 0,1 m ao lado (print 2)
    coords = [(0.0, 0.0), (0.0, 30.0), (0.1, 30.0), (0.1, 25.0)]
    cfg = PostprocessV2Config()
    out, _unc, stats = clean_loops_and_hooks([_ln(coords)], SP, cfg)
    assert stats["hairpins_cortados"] >= 1
    assert len(out) == 1
    assert not _reversal_cuts(out[0].geom, SP, cfg)
    assert out[0].geom.length > 28.0


def test_l_hook_at_tip_is_trimmed() -> None:
    # Gancho em L na ponta subindo em direção à linha de cima (print 5)
    coords = [(0.0, 0.0), (20.0, 0.0), (20.0, 0.9)]
    out, _unc, stats = clean_loops_and_hooks([_ln(coords)], SP, PostprocessV2Config())
    assert stats["ganchos_cortados"] >= 1
    end = out[0].geom.coords[-1]
    assert abs(end[1]) < 0.15, end


def test_slight_wobble_tip_is_kept() -> None:
    coords = [(0.0, 0.0), (19.5, 0.0), (20.0, 0.05)]
    out, _unc, stats = clean_loops_and_hooks([_ln(coords)], SP, PostprocessV2Config())
    assert stats["ganchos_cortados"] == 0
    assert out[0].geom.length > 19.9


def test_broken_line_joined_copying_curved_neighbour() -> None:
    # Vizinha curva contínua; linha quebrada com falha de 8 m (print 1)
    neighbour = _ln(_curve(SP, 0.0, 60.0))
    full = _curve(0.0, 0.0, 60.0)
    part_a = _ln([p for p in full if p[1] <= 25.0])
    part_b = _ln([p for p in full if p[1] >= 33.0])
    out, unc, stats = join_broken_lines([part_a, part_b, neighbour], SP, PostprocessV2Config())
    assert stats["unioes_copia_vizinha"] == 1
    assert len(out) == 2 and not unc
    merged = max(out, key=lambda r: r.repaired_len)
    truth = LineString(full)
    # o conector deve seguir a curva (copiado da vizinha), não uma reta
    assert merged.geom.hausdorff_distance(truth) < 0.10
    assert merged.repaired_len > 7.5


def test_long_gap_without_neighbour_not_joined() -> None:
    a = _ln([(0.0, 0.0), (0.0, 10.0)])
    b = _ln([(0.0, 15.0), (0.0, 25.0)])
    out, _unc, stats = join_broken_lines([a, b], SP, PostprocessV2Config())
    assert sum(stats[k] for k in ("unioes_copia_vizinha", "unioes_reta_curta")) == 0
    assert len(out) == 2


def test_join_never_creates_loop() -> None:
    # Dois pedaços cujas duas pontas se aproximam -> união única, sem fechar ciclo
    neighbour = _ln([(SP, -5.0), (SP, 40.0)])
    a = _ln([(0.0, 0.0), (0.0, 10.0)])
    b = _ln([(0.0, 12.0), (0.0, 20.0)])
    out, _unc, _stats = join_broken_lines([a, b, neighbour], SP, PostprocessV2Config())
    for r in out:
        assert r.geom.is_simple
        assert not _reversal_cuts(r.geom, SP, PostprocessV2Config())


def test_extension_uses_neighbour_direction_and_touches_boundary() -> None:
    # Talhão com borda sul em y=0 e borda norte em y=40; vizinhas chegam de 0 até 40; alvo para em 30
    boundary = Polygon([(-10, 0), (20, 0), (20, 40), (-10, 40)])
    rows = [_ln([(x, 0.0), (x, 40.0)]) for x in (0.0, SP, 3 * SP)]
    target = _ln([(2 * SP, 0.0), (2 * SP, 29.5), (2 * SP + 0.4, 30.0)])  # ponta torta
    cfg = PostprocessV2Config()
    cleaned, _, _ = clean_loops_and_hooks([*rows, target], SP, cfg)
    out, unc, stats = extend_to_field_boundary(cleaned, boundary, SP, cfg)
    ext = [r for r in out if any(h.startswith("extensao") for h in r.history)]
    assert len(ext) == 1, (stats, [u.motivo for u in unc])
    end = ext[0].geom.coords[-1]
    assert abs(end[1] - 40.0) < 0.02
    assert abs(end[0] - 2 * SP) < 0.15  # reta, paralela à vizinha


def test_extension_refused_into_matacao_without_neighbour_support() -> None:
    # Todas as linhas param em y=30 (início de matação); borda do talhão só em y=40.
    boundary = Polygon([(-10, 0), (20, 0), (20, 40), (-10, 40)])
    rows = [_ln([(x, 0.0), (x, 30.0)]) for x in (0.0, SP, 2 * SP, 3 * SP)]
    out, unc, stats = extend_to_field_boundary(rows, boundary, SP, PostprocessV2Config())
    assert stats["extensoes_copia_vizinha"] + stats["extensoes_consenso"] == 0
    assert all(max(c[1] for c in r.geom.coords) <= 30.0 + 1e-6 for r in out)
    assert any(u.motivo == "extensao_sem_suporte" for u in unc)


def test_short_headland_extension_without_support_is_allowed() -> None:
    # Todas param 2,5 m antes da estrada (cabeceira): margem livre cobre.
    boundary = Polygon([(-10, 0), (20, 0), (20, 32.5), (-10, 32.5)])
    rows = [_ln([(x, 0.0), (x, 30.0)]) for x in (0.0, SP, 2 * SP, 3 * SP)]
    out, _unc, stats = extend_to_field_boundary(rows, boundary, SP, PostprocessV2Config())
    assert stats["extensoes_copia_vizinha"] + stats["extensoes_consenso"] == 4
    assert all(abs(max(c[1] for c in r.geom.coords) - 32.5) < 0.02 for r in out)


def test_seam_break_across_all_rows_is_joined() -> None:
    # Costura diagonal: todas as linhas curvas quebram com falha ~2,5 m, sem vizinha contínua.
    lines = []
    truths = []
    for k in range(5):
        x0 = k * SP
        full = _curve(x0, 0.0, 60.0)
        cut = 25.0 + k * 1.5  # costura inclinada
        lines.append(_ln([p for p in full if p[1] <= cut]))
        lines.append(_ln([p for p in full if p[1] >= cut + 2.5]))
        truths.append(LineString(full))
    out, unc, stats = join_broken_lines(lines, SP, PostprocessV2Config())
    assert len(out) == 5, (stats, [u.motivo for u in unc])
    for r in out:
        assert min(r.geom.hausdorff_distance(t) for t in truths) < 0.15


def test_extension_disabled_without_boundary() -> None:
    rows = [_ln([(x, 0.0), (x, 30.0)]) for x in (0.0, SP, 2 * SP)]
    res = run_postprocess_v2(rows, None, spacing_m=SP)
    assert isinstance(res.diagnostics["etapa3_extensao"], str)


def test_export_v2_files(tmp_path: Path) -> None:
    rows = [_ln([(500000.0 + x, 7000000.0), (500000.0 + x, 7000040.0)]) for x in (0.0, SP, 2 * SP)]
    res = run_postprocess_v2(rows, None, spacing_m=SP)
    p1 = export_complete_v2(tmp_path / "linhas_completas_v2.gpkg", res.lines, "EPSG:32722", SP)
    p2 = export_uncertain_v2(tmp_path / "linhas_incertas_v2.gpkg", res.uncertain, "EPSG:32722")
    g1 = gpd.read_file(p1, layer="linhas_completas_v2")
    g2 = gpd.read_file(p2, layer="linhas_incertas_v2")
    assert len(g1) == 3 and g1.crs.to_epsg() == 32722
    assert "motivo_revisao" in g2.columns
