"""Testes sintéticos de Fase 2 para o pós-processamento v2.

Valida:
1. Emendas C1 com recolhimento de pontas tortas (max_turn <= 6°).
2. Preferência estrita de mesma fileira sobre distância euclidiana mais curta (caso dos prints 2 e 3).
3. Linhas que matam (mata): classificadas, isoladas, sem extensão para o talhão e sem união com fileira completa.
4. União longa guiada (gap até 25 m com suporte bilateral, caso da linha vermelha dos prints 4 e 5).
5. Detecção e corte de trocas de fileira espúrias em linhas de entrada.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

from rowgraphnet.geometry.postprocess_v2 import (
    LineV2,
    PostprocessV2Config,
    run_postprocess_v2,
)
from rowgraphnet.geometry.postprocess_v2_rows import (
    build_row_context,
    detect_row_switches,
    measure_max_turn_deg,
)

SP = 1.5


def _ln(coords, source: str = "simples") -> LineV2:
    g = LineString(coords)
    return LineV2(
        geom=g,
        source=source,
        confidence=0.9,
        observed_len=float(g.length),
        spacing=SP,
        orig_ids=[f"id_{id(g)}"],
    )


def _curve(x0: float, y0: float, y1: float, amp: float = 2.0, step: float = 0.25) -> list[tuple[float, float]]:
    ys = np.arange(y0, y1 + 1e-9, step)
    return [(float(x0 + amp * np.sin(y / 15.0)), float(y)) for y in ys]


def test_seam_is_c1_when_tips_drift() -> None:
    """Seam entre fragmentos cujas pontas cruas derivam em x até 0,15 m."""
    guide_l = _ln([(-SP, 0.0), (-SP, 60.0)])
    guide_r = _ln([(SP, 0.0), (SP, 60.0)])
    # Fileira x=0 quebrada: A em y in [0, 25], B em y in [31, 60] (gap 6 m).
    # Último 1.0 m de A desvia linearmente até +0.15 m em x.
    # Primeiro 1.0 m de B desvia linearmente de +0.15 m para 0.0 em x.
    coords_a = [(0.0, float(y)) for y in np.arange(0.0, 24.0, 0.5)]
    coords_a += [(0.05, 24.5), (0.15, 25.0)]

    coords_b = [(0.15, 31.0), (0.05, 31.5)]
    coords_b += [(0.0, float(y)) for y in np.arange(32.0, 60.1, 0.5)]

    part_a = _ln(coords_a)
    part_b = _ln(coords_b)

    cfg = PostprocessV2Config(seam_max_turn_deg=6.0)
    res = run_postprocess_v2([part_a, part_b, guide_l, guide_r], None, spacing_m=SP, config=cfg)

    # Deve unir part_a e part_b em 1 linha
    complete_target = [r for r in res.lines if abs(r.geom.coords[0][0]) < 0.5]
    assert len(complete_target) == 1
    merged = complete_target[0]
    assert merged.geom.length > 58.0
    turn = measure_max_turn_deg(merged.geom, window_m=0.6)
    assert turn <= 6.5, f"Turn={turn} > 6.5 deg"
    truth = LineString([(0.0, 0.0), (0.0, 60.0)])
    assert merged.geom.hausdorff_distance(truth) < 0.12


def test_seam_c1_on_curve() -> None:
    """Emenda C1 sobre traçado curvo com deriva nas pontas."""
    g_left = _ln(_curve(-SP, 0.0, 60.0, amp=2.0))
    g_right = _ln(_curve(SP, 0.0, 60.0, amp=2.0))

    full = _curve(0.0, 0.0, 60.0, amp=2.0)
    pts_a = [p for p in full if p[1] <= 24.0]
    # deriva de 0.12 m nos últimos pontos de A
    pts_a.append((pts_a[-1][0] + 0.12, 25.0))

    pts_b = [(p[0] + 0.12, 33.0) if p[1] == 33.0 else p for p in full if p[1] >= 33.0]

    part_a = _ln(pts_a)
    part_b = _ln(pts_b)

    cfg = PostprocessV2Config(seam_max_turn_deg=6.0)
    res = run_postprocess_v2([part_a, part_b, g_left, g_right], None, spacing_m=SP, config=cfg)

    complete_target = [r for r in res.lines if abs(r.geom.coords[0][0]) < 0.5]
    assert len(complete_target) == 1
    merged = complete_target[0]
    assert merged.geom.length > 58.0
    truth = LineString(full)
    assert merged.geom.hausdorff_distance(truth) < 0.15


def test_join_prefers_same_row_over_closer_wrong_row() -> None:
    """Print 2 e 3: A deve unir com B na mesma fileira (y=0), e não com C na fileira convergente mais próxima."""
    # Guias estáveis acima: y = k*SP (k=1..4), x in [0, 70]
    guides = [_ln([(float(x), k * SP) for x in np.arange(0, 71, 2.0)]) for k in (1, 2, 3)]

    # Fileira alvo em y=0: A x in [0, 25], B x in [30.5, 65] (gap 5.5 m)
    a_coords = [(float(x), 0.0) for x in np.arange(0, 25.1, 1.0)]
    b_coords = [(float(x), 0.0) for x in np.arange(30.5, 65.1, 1.0)]
    row_a = _ln(a_coords)
    row_b = _ln(b_coords)

    # Fileira convergente C abaixo: começa em x=28.0, y=-0.64*SP (mais perto euclidiano de A do que B!)
    # e segue descendo até y = -SP em x=45, continuando até x=65
    c_coords = [(28.0, -0.64 * SP)]
    for x in np.arange(30.0, 65.1, 2.0):
        w = min((x - 28.0) / 17.0, 1.0)
        y_val = -0.64 * SP + (-SP - (-0.64 * SP)) * w
        c_coords.append((float(x), float(y_val)))
    row_c = _ln(c_coords)

    # Fileira y=-SP existe para x in [0, 24]
    row_d = _ln([(float(x), -SP) for x in np.arange(0, 24.1, 1.0)])

    cfg = PostprocessV2Config(row_index_tolerance=0.25)
    res = run_postprocess_v2([row_a, row_b, row_c, row_d, *guides], None, spacing_m=SP, config=cfg)

    # A deve ter sido unida com B
    merged_ab = [r for r in res.lines if any("uniao" in h for h in r.history)]
    assert len(merged_ab) >= 1
    best_ab = max(merged_ab, key=lambda r: r.geom.length)
    # A união deve cobrir o vão de x=25 a x=30.5 com y ~= 0.0
    p_mid = best_ab.geom.interpolate(best_ab.geom.project(shapely.geometry.Point(27.5, 0.0)))
    assert abs(p_mid.y) < 0.15, f"União derivou para y={p_mid.y} em vez de y=0.0"


def test_dying_row_not_joined_nor_extended() -> None:
    """Linha que mata por convergência: classificada como mata, não é estendida nem unida."""
    # Guias estáveis y = k*SP, k=0..3, x in [0, 80] (y=0 é a completa)
    guides = [_ln([(float(x), k * SP) for x in np.arange(0, 81, 2.0)]) for k in (0, 1, 2)]

    # Linha M sai de (0, -SP) e converge linearmente até (40, -0.35*SP), onde encerra
    m_coords = []
    for x in np.arange(0, 40.1, 1.0):
        w = x / 40.0
        y_val = -SP + ((-0.35 * SP) - (-SP)) * w
        m_coords.append((float(x), float(y_val)))
    row_m = _ln(m_coords)

    # Linha inferior completa em y = -2*SP
    row_inf = _ln([(float(x), -2 * SP) for x in np.arange(0, 81, 2.0)])

    boundary = Polygon([(-5, -5 * SP), (85, -5 * SP), (85, 5 * SP), (-5, 5 * SP)])
    cfg = PostprocessV2Config(detect_mata=True)
    res = run_postprocess_v2([row_m, row_inf, *guides], boundary, spacing_m=SP, config=cfg)

    # M deve permanecer encerrada perto de x=40 sem avançar até x=85 (limite do talhão)
    m_out = [r for r in res.lines if abs(r.geom.coords[0][1] - (-SP)) < 0.1]
    assert len(m_out) == 1
    m_geom = m_out[0].geom
    max_x = max(c[0] for c in m_geom.coords)
    assert max_x <= 42.0, f"Linha que mata foi estendida indevidamente para x={max_x}"


def test_red_line_long_guided_join() -> None:
    """Prints 4 e 5: lacuna longa de 18 m com suporte bilateral estável é unida."""
    # Guias estáveis em y = -SP, y = +SP e y = +2*SP (x in [0, 80])
    g1 = _ln([(float(x), -SP) for x in np.arange(0, 81, 2.0)])
    g2 = _ln([(float(x), SP) for x in np.arange(0, 81, 2.0)])
    g3 = _ln([(float(x), 2 * SP) for x in np.arange(0, 81, 2.0)])

    # Alvo y=0: A com x in [0, 30] e B com x in [48, 80] (gap de 18 m)
    row_a = _ln([(float(x), 0.0) for x in np.arange(0, 30.1, 1.0)])
    row_b = _ln([(float(x), 0.0) for x in np.arange(48.0, 80.1, 1.0)])

    cfg = PostprocessV2Config(max_gap_guiado_m=25.0)
    res = run_postprocess_v2([row_a, row_b, g1, g2, g3], None, spacing_m=SP, config=cfg)

    # Deve haver 1 união longa guiada completando a fileira y=0
    target = [r for r in res.lines if abs(r.geom.coords[0][1]) < 0.2 and r.geom.length > 70.0]
    assert len(target) == 1, f"União longa não realizada: {[r.geom.length for r in res.lines]}"
    truth = LineString([(0.0, 0.0), (80.0, 0.0)])
    assert target[0].geom.hausdorff_distance(truth) < 0.10


def test_row_switch_in_input_is_cut() -> None:
    """Linha que salta de fileira na inferência é cortada pelo RowContext."""
    # Guias estáveis acima: y=SP e y=2*SP (x in [0, 60])
    g1 = _ln([(float(x), SP) for x in np.arange(0, 61, 2.0)])
    g2 = _ln([(float(x), 2 * SP) for x in np.arange(0, 61, 2.0)])

    # Linha L vai por y=0 de x=0 a 25, faz diagonal abrupta para y=-SP em x=28, e segue em y=-SP até x=60
    coords_bad = [(float(x), 0.0) for x in np.arange(0, 25.1, 1.0)]
    coords_bad += [(26.5, -0.5 * SP), (28.0, -SP)]
    coords_bad += [(float(x), -SP) for x in np.arange(29.0, 60.1, 1.0)]
    bad_line = _ln(coords_bad)

    cfg = PostprocessV2Config(row_switch_delta=0.5)
    ctx = build_row_context([bad_line, g1, g2], SP, cfg)
    lines_cut, n_cut = detect_row_switches([bad_line, g1, g2], ctx, cfg)

    assert n_cut >= 1, "Troca de fileira não foi detectada/cortada"
    # A linha errada foi cortada em pelo menos 2 pedaços
    parts = [ln for ln in lines_cut if ln.geom.distance(g1.geom) > 0.5]
    assert len(parts) >= 2
    for p in parts:
        assert not ((p.geom.coords[0][1] == 0.0) and (p.geom.coords[-1][1] == -SP))


def test_long_guided_join_refused_without_bilateral_support() -> None:
    """Gap longo de 18 m é recusado se não houver pelo menos 2 guias estáveis."""
    # Apenas 1 guia em y = SP
    g1 = _ln([(float(x), SP) for x in np.arange(0, 81, 2.0)])

    row_a = _ln([(float(x), 0.0) for x in np.arange(0, 30.1, 1.0)])
    row_b = _ln([(float(x), 0.0) for x in np.arange(48.0, 80.1, 1.0)])

    cfg = PostprocessV2Config(max_gap_guiado_m=25.0)
    res = run_postprocess_v2([row_a, row_b, g1], None, spacing_m=SP, config=cfg)

    # Gap longo > 12m sem suporte bilateral não é unido
    target = [r for r in res.lines if abs(r.geom.coords[0][1]) < 0.2 and r.geom.length > 70.0]
    assert len(target) == 0


def test_extension_seam_is_c1() -> None:
    """Extensão até a borda possui transição suave C1 sem quina angular na emenda."""
    boundary = Polygon([(-10, 0), (20, 0), (20, 40), (-10, 40)])
    rows = [_ln([(x, 0.0), (x, 40.0)]) for x in (0.0, SP, 3 * SP)]
    # Target com leve deriva de 0.10 m no final
    coords_target = [(2 * SP, float(y)) for y in np.arange(0, 29.0, 1.0)]
    coords_target.append((2 * SP + 0.10, 30.0))
    target = _ln(coords_target)

    cfg = PostprocessV2Config()
    res = run_postprocess_v2([*rows, target], boundary, spacing_m=SP, config=cfg)

    ext = [r for r in res.lines if abs(r.geom.coords[0][0] - 2 * SP) < 0.5]
    assert len(ext) == 1
    assert abs(ext[0].geom.coords[-1][1] - 40.0) < 0.05
    turn = measure_max_turn_deg(ext[0].geom, window_m=0.6)
    assert turn <= 6.5


def test_row_context_stability_flags() -> None:
    """Verifica que linhas paralelas têm estações estáveis e linha convergente perde estabilidade."""
    parallels = [_ln([(float(x), k * SP) for x in np.arange(0, 61, 2.0)]) for k in range(4)]
    # Convergente C
    c_pts = [(float(x), -SP + 0.65 * SP * (x / 60.0)) for x in np.arange(0, 61, 2.0)]
    row_c = _ln(c_pts)

    cfg = PostprocessV2Config()
    ctx = build_row_context([*parallels, row_c], SP, cfg)

    # Paralelas interiores (k=1 e k=2) devem ser estáveis em quase 100% das estações
    assert float(np.mean(ctx.stable[1])) >= 0.85
    assert float(np.mean(ctx.stable[2])) >= 0.85
    # Convergente tem variação contínua de espaçamento
    assert float(np.mean(ctx.stable[4])) < 0.70
