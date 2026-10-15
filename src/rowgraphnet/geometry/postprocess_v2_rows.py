"""Módulo de contexto de fileiras, detecção de linhas que matam e emendas C1 suaves.

Complementa postprocess_v2.py com:
1. RowContext: estações longitudinais, identificação de guias estáveis e índice de fileira local.
2. classify_endpoints: classificação das pontas em 'borda', 'mata' (linha convergente que encerra) e 'quebra'.
3. detect_row_switches: detecção e corte de trocas de fileira espúrias em linhas de entrada.
4. build_c1_seam: emendas com recolhimento de pontas tortas, offset Hermite suave e transição C1 sem quinas.
5. join_same_row: união estrita na mesma fileira usando índice contínuo de fileira em guias estáveis.
6. smooth_residual_kinks: suavização de quinas locais remanescentes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import substring
from shapely.strtree import STRtree

if TYPE_CHECKING:
    from rowgraphnet.geometry.postprocess_v2 import LineV2, PostprocessV2Config

EndpointType = Literal["borda", "mata", "quebra", "aberta"]


@dataclass(slots=True)
class RowContext:
    """Contexto topológico e de espaçamento local ao longo de fileiras agrícolas."""

    spacing_global: float
    geoms: list[LineString]
    tree: STRtree
    stations: list[np.ndarray]  # s em metros por linha (passo regular)
    nb_left: list[np.ndarray]  # índice da vizinha à esquerda por estação (-1 se ausente)
    nb_right: list[np.ndarray]  # índice da vizinha à direita por estação (-1 se ausente)
    d_left: list[np.ndarray]  # distância à vizinha esquerda (nan se ausente)
    d_right: list[np.ndarray]  # distância à vizinha direita (nan se ausente)
    spacing_local: list[np.ndarray]  # espaçamento local estimado (mediana móvel)
    stable: list[np.ndarray]  # bool indicando se a estação atua como guia estável


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-12 else np.zeros(2, dtype=float)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = _unit(a), _unit(b)
    if not ua.any() or not ub.any():
        return 0.0
    return float(math.degrees(math.acos(float(np.clip(np.dot(ua, ub), -1.0, 1.0)))))


def _tangent_at_s(geom: LineString, s: float, window: float = 1.0) -> np.ndarray:
    L = float(geom.length)
    if L <= 1e-6:
        return np.array([1.0, 0.0])
    s0 = max(0.0, s - window)
    s1 = min(L, s + window)
    if s1 - s0 < 1e-6:
        s0, s1 = max(0.0, L - 0.1), L
    p0 = geom.interpolate(s0)
    p1 = geom.interpolate(s1)
    return _unit(np.array([p1.x - p0.x, p1.y - p0.y]))


def _tangents_at_s_vec(geom: LineString, s_arr: np.ndarray, window: float = 1.0) -> np.ndarray:
    """Calcula vetores tangentes unitários em lote para um array de s."""
    L = float(geom.length)
    n = len(s_arr)
    if L <= 1e-6 or n == 0:
        return np.tile(np.array([1.0, 0.0]), (n, 1))
    s0 = np.clip(s_arr - window, 0.0, L)
    s1 = np.clip(s_arr + window, 0.0, L)
    deg = (s1 - s0) < 1e-6
    if np.any(deg):
        s0[deg] = np.maximum(0.0, L - 0.1)
        s1[deg] = L
    p0 = shapely.get_coordinates(shapely.line_interpolate_point(geom, s0))
    p1 = shapely.get_coordinates(shapely.line_interpolate_point(geom, s1))
    diff = p1 - p0
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return diff / norms


def measure_max_turn_deg(geom: LineString, window_m: float = 0.6) -> float:
    """Calcula a máxima deflexão angular local (graus) em janela móvel de +/- window_m."""
    L = float(geom.length)
    if 3.0 * window_m > L:
        return 0.0
    s_arr = np.arange(window_m, L - window_m, 0.2)
    if len(s_arr) == 0:
        return 0.0
    pts_a = shapely.get_coordinates(shapely.line_interpolate_point(geom, s_arr - window_m))
    pts_b = shapely.get_coordinates(shapely.line_interpolate_point(geom, s_arr))
    pts_c = shapely.get_coordinates(shapely.line_interpolate_point(geom, s_arr + window_m))
    v1 = pts_b - pts_a
    v2 = pts_c - pts_b
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    valid = (norm1 > 1e-9) & (norm2 > 1e-9)
    if not np.any(valid):
        return 0.0
    cos = np.sum(v1[valid] * v2[valid], axis=1) / (norm1[valid] * norm2[valid])
    angles = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return float(np.max(angles)) if len(angles) else 0.0


def build_row_context(
    lines: list[LineV2], spacing_global: float, cfg: PostprocessV2Config
) -> RowContext:
    """Gera o contexto de fileiras com vizinhas e classificação de estabilidade."""
    geoms = [ln.geom for ln in lines]
    tree = STRtree(geoms)
    step = getattr(cfg, "station_step_m", 1.0)
    stable_win = getattr(cfg, "stable_window_m", 10.0)
    stable_tol_fac = getattr(cfg, "stable_tolerance_factor", 0.12)
    guide_min_len = getattr(cfg, "guide_min_length_m", 20.0)

    stations_list: list[np.ndarray] = []
    nb_left_list: list[np.ndarray] = []
    nb_right_list: list[np.ndarray] = []
    d_left_list: list[np.ndarray] = []
    d_right_list: list[np.ndarray] = []
    spacing_local_list: list[np.ndarray] = []
    stable_list: list[np.ndarray] = []

    for i, g in enumerate(geoms):
        L = float(g.length)
        if step > L:
            s_arr = np.array([0.0, L])
        else:
            s_arr = np.linspace(0.0, L, max(math.ceil(L / step) + 1, 2))
        n_pts = len(s_arr)

        nb_l = np.full(n_pts, -1, dtype=int)
        nb_r = np.full(n_pts, -1, dtype=int)
        d_l = np.full(n_pts, np.nan, dtype=float)
        d_r = np.full(n_pts, np.nan, dtype=float)

        pts = shapely.line_interpolate_point(g, s_arr)
        max_search = 2.5 * spacing_global
        nearby_candidates = [int(j) for j in tree.query(g.envelope.buffer(max_search)) if int(j) != i]

        t_curr_all = _tangents_at_s_vec(g, s_arr)
        n_curr_all = np.column_stack([-t_curr_all[:, 1], t_curr_all[:, 0]])
        coords_p = shapely.get_coordinates(pts)

        for j in nearby_candidates:
            gj = geoms[j]
            Lj = float(gj.length)
            if Lj <= 0.6:
                continue
            sj_arr = shapely.line_locate_point(gj, pts)
            valid_mask = (sj_arr >= 0.3) & (sj_arr <= Lj - 0.3)
            if not np.any(valid_mask):
                continue

            valid_idx = np.where(valid_mask)[0]
            sj_sub = sj_arr[valid_idx]
            q_pts = shapely.line_interpolate_point(gj, sj_sub)
            coords_q = shapely.get_coordinates(q_pts)

            vec = coords_q - coords_p[valid_idx]
            dist = np.linalg.norm(vec, axis=1)
            dist_mask = (dist >= 0.2) & (dist <= max_search)
            if not np.any(dist_mask):
                continue

            final_idx = valid_idx[dist_mask]
            vec = vec[dist_mask]
            dist = dist[dist_mask]
            sj_sub = sj_sub[dist_mask]

            tj = _tangents_at_s_vec(gj, sj_sub)
            ti = t_curr_all[final_idx]
            cos_sim = np.abs(np.sum(ti * tj, axis=1))
            # cos(35 deg) ~= 0.819152
            cos_mask = cos_sim >= 0.819152
            if not np.any(cos_mask):
                continue

            final_idx = final_idx[cos_mask]
            vec = vec[cos_mask]
            dist = dist[cos_mask]

            side_proj = np.sum(vec * n_curr_all[final_idx], axis=1)

            for k_idx, d_val, s_proj in zip(final_idx, dist, side_proj, strict=False):
                if s_proj > 0:  # esquerda
                    if np.isnan(d_l[k_idx]) or d_val < d_l[k_idx]:
                        d_l[k_idx] = float(d_val)
                        nb_l[k_idx] = j
                else:  # direita
                    if np.isnan(d_r[k_idx]) or d_val < d_r[k_idx]:
                        d_r[k_idx] = float(d_val)
                        nb_r[k_idx] = j

        # Espaçamento local móvel
        sp_loc = np.full(n_pts, spacing_global, dtype=float)
        w_half = max(round((stable_win / 2.0) / step), 1)
        for k in range(n_pts):
            k0 = max(0, k - w_half)
            k1 = min(n_pts, k + w_half + 1)
            vals = []
            for d in [d_l[k0:k1], d_r[k0:k1]]:
                v = d[np.isfinite(d)]
                v = v[(v >= 0.55 * spacing_global) & (v <= 1.65 * spacing_global)]
                vals.extend(v.tolist())
            if len(vals) >= 2:
                sp_loc[k] = float(np.median(vals))

        # Estabilidade
        is_stable = np.zeros(n_pts, dtype=bool)
        is_long = guide_min_len <= L
        repaired_frac = (
            lines[i].repaired_len / (lines[i].observed_len + lines[i].repaired_len)
            if (lines[i].observed_len + lines[i].repaired_len) > 0
            else 0.0
        )
        is_reliable = repaired_frac <= 0.5 and is_long

        if is_reliable:
            for k in range(n_pts):
                k0 = max(0, k - w_half)
                k1 = min(n_pts, k + w_half + 1)
                s_lo = s_arr[k0]
                s_hi = s_arr[min(k1 - 1, n_pts - 1)]
                sub_g = substring(g, s_lo, s_hi)
                local_turn = measure_max_turn_deg(sub_g) if sub_g.length >= 1.8 else 0.0

                has_imm_neighbor = False
                stable_on_side = False
                for d_arr, _nb_arr in [(d_l[k0:k1], nb_l[k0:k1]), (d_r[k0:k1], nb_r[k0:k1])]:
                    imm_mask = (d_arr >= 0.55 * sp_loc[k]) & (d_arr <= 1.45 * sp_loc[k])
                    valid_d = d_arr[imm_mask]
                    if len(valid_d) >= 2:
                        has_imm_neighbor = True
                        spread = float(np.max(valid_d) - np.min(valid_d))
                        if spread <= stable_tol_fac * sp_loc[k]:
                            stable_on_side = True
                            break

                if has_imm_neighbor:
                    is_stable[k] = (local_turn <= 6.0) and stable_on_side
                else:
                    is_stable[k] = (local_turn <= 3.0)

        stations_list.append(s_arr)
        nb_left_list.append(nb_l)
        nb_right_list.append(nb_r)
        d_left_list.append(d_l)
        d_right_list.append(d_r)
        spacing_local_list.append(sp_loc)
        stable_list.append(is_stable)

    return RowContext(
        spacing_global=spacing_global,
        geoms=geoms,
        tree=tree,
        stations=stations_list,
        nb_left=nb_left_list,
        nb_right=nb_right_list,
        d_left=d_left_list,
        d_right=d_right_list,
        spacing_local=spacing_local_list,
        stable=stable_list,
    )


def guide_is_stable_over(ctx: RowContext, guide_idx: int, s0: float, s1: float) -> bool:
    """Verifica se a guia indicada possui estabilidade de fileira sobre o intervalo [s0, s1]."""
    if guide_idx < 0 or guide_idx >= len(ctx.geoms):
        return False
    st_arr = ctx.stations[guide_idx]
    sb_arr = ctx.stable[guide_idx]
    lo = max(0.0, min(s0, s1) - 4.0)
    hi = min(float(ctx.geoms[guide_idx].length), max(s0, s1) + 4.0)
    mask = (st_arr >= lo) & (st_arr <= hi)
    if not np.any(mask):
        return False
    return float(np.mean(sb_arr[mask])) >= 0.70


def row_offset_from_guide(
    ctx: RowContext, guide_idx: int, pt: tuple[float, float] | Point
) -> tuple[float, float, float]:
    """Retorna (s_proj, offset_metros_com_sinal, indice_fracionario_n)."""
    p = Point(pt) if not isinstance(pt, Point) else pt
    g = ctx.geoms[guide_idx]
    s = float(g.project(p))
    q = g.interpolate(s)
    t = _tangent_at_s(g, s)
    n = np.array([-t[1], t[0]])
    off_m = float(np.dot(np.array([p.x - q.x, p.y - q.y]), n))

    st_arr = ctx.stations[guide_idx]
    sp_arr = ctx.spacing_local[guide_idx]
    idx = int(np.clip(np.searchsorted(st_arr, s), 0, len(sp_arr) - 1))
    sp_local = sp_arr[idx] if len(sp_arr) else ctx.spacing_global
    n_idx = off_m / max(sp_local, 0.5)
    return s, off_m, n_idx


def classify_endpoints(
    lines: list[LineV2],
    ctx: RowContext,
    boundary: shapely.geometry.base.BaseGeometry | None,
    cfg: PostprocessV2Config,
) -> dict[tuple[int, int], EndpointType]:
    """Classifica cada ponta (linha_i, lado) em 'borda', 'mata', 'quebra' ou 'aberta'."""
    res: dict[tuple[int, int], EndpointType] = {}
    touch_tol = getattr(cfg, "touch_tolerance_m", 0.05)
    mata_tip_fac = getattr(cfg, "mata_tip_spacing_factor", 0.75)
    mata_drop_fac = getattr(cfg, "mata_drop_factor", 0.20)
    mata_cont_m = getattr(cfg, "mata_neighbor_continue_m", 3.0)
    detect_mata_enabled = getattr(cfg, "detect_mata", True)

    polygons = []
    if boundary is not None and not boundary.is_empty:
        if isinstance(boundary, shapely.geometry.Polygon):
            polygons = [boundary]
        elif hasattr(boundary, "geoms"):
            polygons = [p for p in boundary.geoms if not p.is_empty]
    buffered_polys = [p.buffer(touch_tol) for p in polygons]

    if len(lines) != len(ctx.geoms):
        ctx = build_row_context(lines, ctx.spacing_global, cfg)

    for i, ln in enumerate(lines):
        g = ln.geom
        L = float(g.length)
        st_arr = ctx.stations[i]
        d_l = ctx.d_left[i]
        d_r = ctx.d_right[i]
        nb_l = ctx.nb_left[i]
        nb_r = ctx.nb_right[i]
        sp_loc = ctx.spacing_local[i]

        for side in (0, 1):
            pt = Point(g.coords[0] if side == 0 else g.coords[-1])

            # 1. Borda do talhão
            near_border = False
            for poly, poly_buf in zip(polygons, buffered_polys, strict=False):
                if poly_buf.covers(pt) and poly.boundary.distance(pt) <= 1.0:
                    near_border = True
                    break
            if near_border:
                res[(i, side)] = "borda"
                continue

            # 2. Linha que mata (convergência agronômica)
            is_mata = False
            if detect_mata_enabled and L >= 5.0:
                s_tip = 0.0 if side == 0 else L
                # Amostra distâncias para trás a partir da ponta
                dists_back = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
                s_samples = s_tip + (dists_back if side == 0 else -dists_back)
                valid_mask = (s_samples >= 0.0) & (s_samples <= L)
                s_samples = s_samples[valid_mask]

                if len(s_samples) >= 3:
                    indices = [int(np.clip(np.searchsorted(st_arr, sv), 0, len(st_arr) - 1)) for sv in s_samples]
                    for d_arr, nb_arr in [(d_l, nb_l), (d_r, nb_r)]:
                        vals = d_arr[indices]
                        nbs = nb_arr[indices]
                        sp0 = sp_loc[indices[0]]
                        if np.isfinite(vals[0]) and vals[0] <= mata_tip_fac * sp0:
                            finite_vals = vals[np.isfinite(vals)]
                            if len(finite_vals) >= 2:
                                delta_d = finite_vals[-1] - finite_vals[0]
                                if delta_d >= mata_drop_fac * sp0:
                                    # Vizinha absorvente continua além da ponta
                                    absorb_nb = int(nbs[0])
                                    if absorb_nb >= 0:
                                        g_nb = ctx.geoms[absorb_nb]
                                        s_proj = float(g_nb.project(pt))
                                        rem = min(s_proj, g_nb.length - s_proj)
                                        if rem >= mata_cont_m:
                                            is_mata = True
                                            break

            if is_mata:
                res[(i, side)] = "mata"
            else:
                res[(i, side)] = "quebra"

    return res


def _offset_copy_smooth(
    guide: LineString,
    s0: float,
    s1: float,
    off0: float,
    off1: float,
    step: float = 0.20,
) -> np.ndarray:
    """Copia o traçado da guia com variação C1 Hermite suave no offset (derivadas zero nas pontas)."""
    dist = abs(s1 - s0)
    n = max(math.ceil(dist / step), 2)
    ss = np.linspace(s0, s1, n + 1)
    pts = shapely.get_coordinates(shapely.line_interpolate_point(guide, ss))
    out = np.empty_like(pts)
    for k, s in enumerate(ss):
        w = k / n
        # Polinômio Hermite smoothstep com derivada nula em w=0 e w=1
        smooth_w = 3.0 * w**2 - 2.0 * w**3
        off_k = off0 + (off1 - off0) * smooth_w
        t = _tangent_at_s(guide, float(s))
        norm = np.array([-t[1], t[0]])
        out[k] = pts[k] + norm * off_k
    return out


def build_c1_seam(
    geom_a: LineString,
    side_a: int,
    geom_b: LineString,
    side_b: int,
    guide: LineString,
    s_a: float,
    s_b: float,
    off_a: float,
    off_b: float,
    spacing: float,
    cfg: PostprocessV2Config,
) -> tuple[np.ndarray | None, float]:
    """Constrói conector suave C1 com recolhimento de pontas tortas e mesclagem C1."""
    step = getattr(cfg, "resample_step_m", 0.20)
    max_turn_deg = getattr(cfg, "seam_max_turn_deg", 6.0)
    turn_margin_deg = getattr(cfg, "seam_turn_margin_deg", 3.0)
    retract_max = getattr(cfg, "seam_retract_max_factor", 1.0) * spacing
    tip_residual_m = getattr(cfg, "seam_tip_residual_m", 0.05)

    # 1. Avalia se ponta A possui deriva espúria no raster e recolhe
    def _clean_tip_geom(geom: LineString, side: int, off_target: float) -> tuple[LineString, float]:
        coords = np.asarray(geom.coords, dtype=float)
        L = float(geom.length)
        if L < 2.0:
            return geom, 0.0
        line_oriented = geom if side == 1 else LineString(coords[::-1])
        s_eval = np.linspace(max(0.0, L - retract_max), L, max(int(retract_max / 0.2), 3))
        pts_eval = shapely.get_coordinates(shapely.line_interpolate_point(line_oriented, s_eval))

        retract_s = L
        for k in range(len(s_eval) - 1, -1, -1):
            p = Point(pts_eval[k])
            sp = float(guide.project(p))
            t = _tangent_at_s(guide, sp)
            norm = np.array([-t[1], t[0]])
            q = guide.interpolate(sp)
            cur_off = float(np.dot(np.array([p.x - q.x, p.y - q.y]), norm))
            if abs(cur_off - off_target) <= tip_residual_m:
                retract_s = s_eval[k]
                break
        cut_len = L - retract_s
        if cut_len > 0.15:
            trimmed = substring(line_oriented, 0.0, retract_s)
            res = trimmed if side == 1 else LineString(np.asarray(trimmed.coords)[::-1])
            return res, cut_len
        return geom, 0.0

    g_a_clean, _ = _clean_tip_geom(geom_a, side_a, off_a)
    g_b_clean, _ = _clean_tip_geom(geom_b, side_b, off_b)

    pt_a = tuple(g_a_clean.coords[-1] if side_a == 1 else g_a_clean.coords[0])
    pt_b = tuple(g_b_clean.coords[0] if side_b == 0 else g_b_clean.coords[-1])

    s_proj_a = float(guide.project(Point(pt_a)))
    s_proj_b = float(guide.project(Point(pt_b)))

    connector = _offset_copy_smooth(guide, s_proj_a, s_proj_b, off_a, off_b, step=step)
    connector[0] = pt_a
    connector[-1] = pt_b

    conn_line = LineString(connector)
    if not conn_line.is_valid or not conn_line.is_simple:
        return None, 0.0

    # Validação de giro
    guide_turn = measure_max_turn_deg(substring(guide, min(s_proj_a, s_proj_b), max(s_proj_a, s_proj_b)))
    allowed_turn = max(max_turn_deg, guide_turn + turn_margin_deg)
    conn_turn = measure_max_turn_deg(conn_line)
    if conn_turn > allowed_turn:
        return None, 0.0

    return connector, float(conn_line.length)


def detect_row_switches(
    lines: list[LineV2], ctx: RowContext, cfg: PostprocessV2Config
) -> tuple[list[LineV2], int]:
    """Detecta e corta trechos de troca de fileira abrupta em linhas de entrada."""
    out: list[LineV2] = []
    switches_cut = 0
    delta_thr = getattr(cfg, "row_switch_delta", 0.5)
    win_len = getattr(cfg, "row_switch_window_factor", 4.0) * ctx.spacing_global

    for i, ln in enumerate(lines):
        g = ln.geom
        L = float(g.length)
        if L < 12.0:
            out.append(ln)
            continue

        st_arr = ctx.stations[i]
        # Procura guia estável contínua que cubra a linha g
        best_guide = -1
        best_cov = 0.0
        best_in_bounds: np.ndarray | None = None
        nearby_cand = [int(j) for j in ctx.tree.query(g.envelope.buffer(3.2 * ctx.spacing_global)) if int(j) != i]
        for cand_j in nearby_cand:
            gj = ctx.geoms[cand_j]
            if gj.length < 25.0:
                continue
            d = gj.distance(g)
            if not (0.5 * ctx.spacing_global <= d <= 3.2 * ctx.spacing_global):
                continue
            # Verifica cobertura dos pontos de g sobre gj
            s_projs = np.array([float(gj.project(g.interpolate(s))) for s in st_arr])
            in_bounds = (s_projs >= 1.0) & (s_projs <= gj.length - 1.0)
            cov = float(np.mean(in_bounds))
            if (
                cov >= 0.85
                and cov > best_cov
                and guide_is_stable_over(ctx, cand_j, float(s_projs[in_bounds].min()), float(s_projs[in_bounds].max()))
            ):
                best_guide = cand_j
                best_cov = cov
                best_in_bounds = in_bounds

        if best_guide < 0 or best_in_bounds is None:
            out.append(ln)
            continue

        n_indices = np.array([row_offset_from_guide(ctx, best_guide, g.interpolate(s))[2] for s in st_arr])
        # Detecta salto rápido em janela de win_len
        cut_windows: list[tuple[float, float]] = []
        w_steps = max(round(win_len / (st_arr[1] - st_arr[0])), 2) if len(st_arr) > 1 else 2

        for k in range(len(st_arr) - w_steps):
            if not (best_in_bounds[k] and best_in_bounds[k + w_steps]):
                continue
            diff = abs(n_indices[k + w_steps] - n_indices[k])
            rate = diff / win_len
            if diff >= delta_thr and rate >= 0.08:
                cut_windows.append((st_arr[k], st_arr[k + w_steps]))

        if not cut_windows:
            out.append(ln)
            continue

        # Realiza corte e preserva fragmentos limpos
        switches_cut += 1
        clean_ranges: list[tuple[float, float]] = []
        cur_s = 0.0
        for w_lo, w_hi in cut_windows:
            if w_lo > cur_s + 2.0:
                clean_ranges.append((cur_s, w_lo))
            cur_s = max(cur_s, w_hi)
        if cur_s < L - 2.0:
            clean_ranges.append((cur_s, L))

        for idx_r, (r0, r1) in enumerate(clean_ranges):
            sub = substring(g, r0, r1)
            if isinstance(sub, LineString) and sub.length >= 2.0:
                out.append(
                    type(ln)(
                        geom=sub,
                        source=ln.source,
                        confidence=ln.confidence,
                        observed_len=sub.length,
                        repaired_len=0.0,
                        spacing=ln.spacing,
                        model_version=ln.model_version,
                        orig_ids=list(ln.orig_ids),
                        history=[*ln.history, f"troca_fileira_cortada_{idx_r}"],
                        original_reason=ln.original_reason,
                    )
                )

    return out, switches_cut


def join_same_row(
    lines: list[LineV2],
    ctx: RowContext,
    endpoints_class: dict[tuple[int, int], EndpointType],
    cfg: PostprocessV2Config,
) -> tuple[list[LineV2], list[dict], dict[str, int]]:
    """Une linhas quebradas restritas estritamente à mesma fileira através de guias estáveis."""
    from rowgraphnet.geometry.postprocess_v2 import (
        UncertainV2,
        _as_line,
        _assemble_chains,
        _segment_clear,
    )

    stats = {
        "unioes_copia_vizinha": 0,
        "unioes_longas_guiadas": 0,
        "unioes_costura": 0,
        "unioes_ambiguas": 0,
        "rejeitadas_indice": 0,
    }
    uncertain: list[UncertainV2] = []
    max_gap_guided = getattr(cfg, "max_gap_guiado_m", 25.0)
    max_gap_std = getattr(cfg, "max_gap_m", 12.0)
    row_idx_tol = getattr(cfg, "row_index_tolerance", 0.25)
    corridor = cfg.safe_corridor_factor * ctx.spacing_global

    if len(lines) != len(ctx.geoms):
        ctx = build_row_context(lines, ctx.spacing_global, cfg)

    # Coleta todas as pontas 'quebra'
    ends: list[tuple[int, int, tuple[float, float], np.ndarray]] = []
    for i, ln in enumerate(lines):
        g = ln.geom
        c = list(g.coords)
        for side in (0, 1):
            if endpoints_class.get((i, side)) == "quebra":
                pt = (c[0][0], c[0][1]) if side == 0 else (c[-1][0], c[-1][1])
                t_out = _tangent_at_s(g, 0.0 if side == 0 else g.length) * (-1.0 if side == 0 else 1.0)
                ends.append((i, side, pt, t_out))

    if len(ends) < 2:
        return lines, uncertain, stats

    end_pts = [Point(e[2]) for e in ends]
    end_tree = STRtree(end_pts)
    candidates = []

    for ei, (a, sa, pa, ta) in enumerate(ends):
        for ej in end_tree.query(end_pts[ei], predicate="dwithin", distance=max_gap_guided):
            ej = int(ej)
            if ej <= ei:
                continue
            b, sb, pb, tb = ends[ej]
            if a == b:
                continue

            delta = np.array([pb[0] - pa[0], pb[1] - pa[1]])
            dist = float(np.linalg.norm(delta))
            if dist < 0.10 or dist > max_gap_guided:
                continue

            u = delta / dist
            if np.dot(ta, u) < math.cos(math.radians(35.0)) or np.dot(tb, -u) < math.cos(math.radians(35.0)):
                continue

            # Busca guias estáveis abrangendo o vão
            stable_guides = []
            nearby_guides = [
                int(idx_j)
                for idx_j in ctx.tree.query(Point(pa).buffer(3.2 * ctx.spacing_global))
                if int(idx_j) not in (a, b)
            ]
            for j in nearby_guides:
                gj = ctx.geoms[j]
                sa_j = float(gj.project(Point(pa)))
                sb_j = float(gj.project(Point(pb)))
                if sa_j <= 0.5 or sa_j >= gj.length - 0.5 or sb_j <= 0.5 or sb_j >= gj.length - 0.5:
                    continue
                d_cand = gj.distance(Point(pa))
                if d_cand <= 3.2 * ctx.spacing_global and guide_is_stable_over(ctx, j, sa_j, sb_j):
                    stable_guides.append((d_cand, j, sa_j, sb_j))

            stable_guides.sort(key=lambda item: item[0])

            connector: np.ndarray | None = None
            delta_n = 0.0
            method = ""

            if stable_guides:
                _, best_g, s_ga, s_gb = stable_guides[0]
                _, off_ga, na = row_offset_from_guide(ctx, best_g, pa)
                _, off_gb, nb = row_offset_from_guide(ctx, best_g, pb)
                delta_n = abs(na - nb)

                if delta_n > row_idx_tol:
                    stats["rejeitadas_indice"] += 1
                    continue

                if dist > max_gap_std:
                    # Gap longo: requer suporte de guias dos dois lados ou 2 guias
                    if len(stable_guides) < 2:
                        continue
                    method = "uniao_longa_guiada"
                else:
                    method = "copia_vizinha"

                g_guide = ctx.geoms[best_g]
                conn_arr, _ = build_c1_seam(
                    lines[a].geom, sa, lines[b].geom, sb, g_guide, s_ga, s_gb, off_ga, off_gb, ctx.spacing_global, cfg
                )
                connector = conn_arr
            else:
                if dist > getattr(cfg, "straight_gap_max_m", 2.0):
                    continue
                # Costura reta curta se perfeitamente alinhada
                connector = np.array([pa, pb])
                method = "costura_tangente"

            if connector is None:
                continue

            conn_line = _as_line(connector)
            if conn_line is None or not _segment_clear(conn_line, {a, b}, ctx.geoms, ctx.tree, corridor):
                continue

            cost = 10.0 * delta_n + 1.0 * (dist / max_gap_guided) + 2.0 * (_angle_deg(ta, -tb) / 35.0)
            candidates.append((cost, delta_n, a, sa, b, sb, connector, method))

    # Regra de preferência estrita de mesma fileira:
    # Se uma ponta tem candidato com delta_n <= 0.15, descarta os com delta_n > 0.20
    cand_by_end: dict[tuple[int, int], list] = {}
    for c in candidates:
        ka, kb = (c[2], c[3]), (c[4], c[5])
        cand_by_end.setdefault(ka, []).append(c)
        cand_by_end.setdefault(kb, []).append(c)

    filtered_candidates = []
    for c in candidates:
        cost, dn, a, sa, b, sb, conn, method = c
        ka, kb = (a, sa), (b, sb)
        best_dn_a = min((item[1] for item in cand_by_end[ka]), default=1.0)
        best_dn_b = min((item[1] for item in cand_by_end[kb]), default=1.0)
        if (best_dn_a <= 0.15 and dn > 0.20) or (best_dn_b <= 0.15 and dn > 0.20):
            continue
        filtered_candidates.append(c)

    filtered_candidates.sort(key=lambda item: item[0])

    # Ambiguidade
    ambiguous_ends = set()
    for key, c_list in cand_by_end.items():
        c_list_sorted = sorted(c_list, key=lambda item: item[0])
        if len(c_list_sorted) >= 2:
            best, sec = c_list_sorted[0], c_list_sorted[1]
            if sec[0] <= 1.15 * best[0] + 0.05 and abs(sec[1] - best[1]) <= 0.10:
                ambiguous_ends.add(key)

    from rowgraphnet.geometry.postprocess_v2 import _UnionFind

    uf = _UnionFind(len(lines))
    used_ends = set()
    links: dict[tuple[int, int], tuple[int, int, np.ndarray, str]] = {}

    for _cost, dn, a, sa, b, sb, conn, method in filtered_candidates:
        ka, kb = (a, sa), (b, sb)
        if ka in used_ends or kb in used_ends:
            continue
        if ka in ambiguous_ends or kb in ambiguous_ends:
            c_line = _as_line(conn)
            if c_line is not None:
                uncertain.append(
                    UncertainV2(
                        geom=c_line,
                        motivo="uniao_ambigua",
                        origem="conector",
                        orig_ids=lines[a].orig_ids + lines[b].orig_ids,
                        detalhe=f"{method}; delta_n={dn:.2f}",
                    )
                )
                stats["unioes_ambiguas"] += 1
            continue
        if not uf.union(a, b):
            continue
        used_ends.add(ka)
        used_ends.add(kb)
        links[ka] = (b, sb, conn, method)
        links[kb] = (a, sa, conn[::-1], method)
        if method == "uniao_longa_guiada":
            stats["unioes_longas_guiadas"] += 1
        elif method == "copia_vizinha":
            stats["unioes_copia_vizinha"] += 1
        else:
            stats["unioes_costura"] += 1

    joined_lines = _assemble_chains(lines, links)
    return joined_lines, uncertain, stats


def smooth_residual_kinks(lines: list[LineV2], cfg: PostprocessV2Config) -> tuple[list[LineV2], int]:
    """Suaviza quinas residuais com ângulo > seam_max_turn_deg em emendas de pós-processamento."""
    out: list[LineV2] = []
    smoothed_count = 0
    max_turn = getattr(cfg, "seam_max_turn_deg", 6.0)

    for ln in lines:
        g = ln.geom
        if not any(h.startswith("uniao") for h in ln.history):
            out.append(ln)
            continue
        if measure_max_turn_deg(g) <= max_turn:
            out.append(ln)
            continue

        # Suavização local via spline preservando desvio <= seam_max_dev_m
        from rowgraphnet.geometry.vector_refinement import _smooth_safely

        max_dev = getattr(cfg, "seam_max_dev_m", 0.10)
        best_sm = None
        curr_turn = measure_max_turn_deg(g)
        dev_candidates = (max_dev, 0.15) if max_dev < 0.15 else (max_dev,)
        for dev in dev_candidates:
            sm = _smooth_safely(g, dev)
            if sm.is_valid and sm.is_simple:
                sm_turn = measure_max_turn_deg(sm)
                if sm_turn < curr_turn:
                    best_sm = sm
                    curr_turn = sm_turn
                    if curr_turn <= max_turn:
                        break

        if best_sm is not None:
            smoothed_count += 1
            out.append(
                type(ln)(
                    geom=best_sm,
                    source=ln.source,
                    confidence=ln.confidence,
                    observed_len=ln.observed_len,
                    repaired_len=ln.repaired_len,
                    spacing=ln.spacing,
                    model_version=ln.model_version,
                    orig_ids=list(ln.orig_ids),
                    history=[*ln.history, "quina_suavizada"],
                    original_reason=ln.original_reason,
                )
            )
        else:
            out.append(ln)

    return out, smoothed_count
