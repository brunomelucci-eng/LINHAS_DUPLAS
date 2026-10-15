"""Pós-processamento v2 separado (gera ``linhas_completas_v2`` e ``linhas_incertas_v2``).

Este módulo NÃO altera o pipeline principal nem ``enhanced_refinement``. Ele trabalha apenas sobre
geometrias vetoriais já exportadas e executa, em ordem:

1. Remoção de loops/hairpins (linha que "volta" sobre si) e ganchos tortos nas pontas.
2. União de linhas quebradas copiando o trecho da vizinha válida (conector com o mesmo formato
   da fileira adjacente, deslocado lateralmente) — falhas sem guia só são unidas se curtas.
3. Extensão das pontas até o contorno do talhão, guiada pela vizinha (cópia do trecho terminal
   da vizinha ou raio reto na direção de consenso), ignorando a ponta torta da própria linha.
4. Validação final: geometria simples, sem cruzamento, sem retorno; o que falhar ou for ambíguo
   vai para ``linhas_incertas_v2`` com ``motivo_revisao``.

Todas as distâncias são em metros e exigem CRS métrico. Limites são relativos ao espaçamento
estimado dos próprios dados (nenhuma regra agronômica universal).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring, unary_union
from shapely.strtree import STRtree

from rowgraphnet.geometry.postprocess_v2_rows import (
    EndpointType,
    _offset_copy_smooth,
    smooth_residual_kinks,
)

COMPLETE_V2_LAYER = "linhas_completas_v2"
UNCERTAIN_V2_LAYER = "linhas_incertas_v2"


# --------------------------------------------------------------------------------------------
# Configuração e estruturas
# --------------------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class PostprocessV2Config:
    """Parâmetros do pós-processamento v2 (metros ou frações do espaçamento local)."""

    # Geral
    resample_step_m: float = 0.20
    safe_corridor_factor: float = 0.30
    min_line_length_factor: float = 1.0
    duplicate_overlap_fraction: float = 0.60
    duplicate_distance_factor: float = 0.25
    # Etapa 1: loops / hairpins / ganchos
    hairpin_turn_deg: float = 120.0
    hairpin_window_factor: float = 1.0
    tip_window_factor: float = 1.5
    tip_body_factor: float = 3.0
    tip_max_offset_factor: float = 0.08
    tip_min_offset_m: float = 0.12
    tip_min_bend_deg: float = 25.0
    # Etapa 2: união de quebradas
    max_gap_m: float = 12.0
    straight_gap_max_m: float = 2.0
    seam_gap_max_factor: float = 2.5
    max_lateral_factor: float = 0.28
    min_facing_deg: float = 35.0
    ambiguity_ratio: float = 0.15
    guide_min_factor: float = 0.45
    guide_max_factor: float = 1.70
    join_passes: int = 2
    # Etapa 3: extensão até o talhão
    max_extension_m: float = 15.0
    neighbors_k: int = 5
    neighbor_search_factor: float = 3.0
    own_tip_tolerance_deg: float = 10.0
    touch_tolerance_m: float = 0.05
    # Extensão só avança onde há vizinhas observadas ao lado (+ margem livre curta de cabeceira).
    extension_free_m: float = 4.0
    support_sample_m: float = 0.5
    support_hole_m: float = 2.0
    support_max_angle_deg: float = 30.0
    # Etapa Contexto de Fileiras, Matação e Emendas C1 (Fase 2)
    station_step_m: float = 1.0
    stable_window_m: float = 10.0
    stable_tolerance_factor: float = 0.12
    guide_min_length_m: float = 20.0
    mata_tip_spacing_factor: float = 0.75
    mata_drop_factor: float = 0.20
    mata_neighbor_continue_m: float = 3.0
    detect_mata: bool = True
    max_gap_guiado_m: float = 25.0
    row_index_tolerance: float = 0.25
    guide_agreement_factor: float = 0.15
    guided_support_min_fraction: float = 0.90
    seam_max_turn_deg: float = 6.0
    seam_turn_margin_deg: float = 3.0
    seam_retract_max_factor: float = 1.0
    seam_tip_residual_m: float = 0.05
    seam_blend_factor: float = 1.0
    seam_max_dev_m: float = 0.10
    row_switch_delta: float = 0.5
    row_switch_window_factor: float = 4.0


@dataclass(slots=True)
class LineV2:
    """Linha em processamento com proveniência."""

    geom: LineString
    source: str
    confidence: float
    observed_len: float
    repaired_len: float = 0.0
    spacing: float | None = None
    model_version: str = "pos-v2"
    orig_ids: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    original_reason: str = ""


@dataclass(slots=True)
class UncertainV2:
    geom: LineString
    motivo: str
    origem: str
    orig_ids: list[str] = field(default_factory=list)
    detalhe: str = ""


@dataclass(slots=True)
class PostprocessV2Result:
    lines: list[LineV2]
    uncertain: list[UncertainV2]
    diagnostics: dict[str, Any]


# --------------------------------------------------------------------------------------------
# Utilidades geométricas
# --------------------------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(2, dtype=float)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = _unit(a), _unit(b)
    if not ua.any() or not ub.any():
        return 0.0
    return float(math.degrees(math.acos(float(np.clip(np.dot(ua, ub), -1.0, 1.0)))))


def _stable_id(geom: LineString) -> str:
    data = np.round(np.asarray(geom.coords, dtype=float), 3).tobytes()
    return "rowv2_" + hashlib.sha256(data).hexdigest()[:16]


def _axial_orientation_deg(geom: LineString) -> float:
    c = np.asarray(geom.coords, dtype=float)
    d = np.diff(c, axis=0)
    lengths = np.linalg.norm(d, axis=1)
    ok = lengths > 1e-12
    if not ok.any():
        return 0.0
    az = np.arctan2(d[ok, 0], d[ok, 1])
    s = np.average(np.sin(2 * az), weights=lengths[ok])
    co = np.average(np.cos(2 * az), weights=lengths[ok])
    deg = float(np.degrees(0.5 * np.arctan2(s, co)) % 180.0)
    if deg >= 180.0 or deg < 0.0 or abs(deg - 180.0) < 1e-7:
        deg = 0.0
    return float(np.clip(deg, 0.0, 179.999999))


def _resample(geom: LineString, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Return resampled points and their arclength positions (always includes both ends)."""
    length = float(geom.length)
    if length <= 0:
        c = np.asarray(geom.coords, dtype=float)
        return c[:1], np.array([0.0])
    n = max(math.ceil(length / step), 1)
    dists = np.linspace(0.0, length, n + 1)
    pts = shapely.get_coordinates(shapely.line_interpolate_point(geom, dists))
    return pts, dists


def _clean_coords(coords: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return coords
    keep = np.r_[True, np.linalg.norm(np.diff(coords, axis=0), axis=1) > 1e-9]
    return coords[keep]


def _as_line(coords: np.ndarray | list) -> LineString | None:
    c = _clean_coords(np.asarray(coords, dtype=float))
    if len(c) < 2:
        return None
    line = LineString(c)
    if line.is_empty or not np.isfinite(line.length) or line.length <= 0:
        return None
    return line


def _line_parts(geom: BaseGeometry | None) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if hasattr(geom, "geoms"):
        out: list[LineString] = []
        for g in geom.geoms:
            out.extend(_line_parts(g))
        return out
    return []


def _outward_tangent(geom: LineString, side: int, length_m: float) -> np.ndarray:
    """Unit vector pointing outward at ``side`` (0=start, 1=end) fitted over ``length_m``."""
    c = np.asarray(geom.coords, dtype=float)
    if side == 1:
        c = c[::-1]
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    take = int(np.searchsorted(d, min(length_m, d[-1]), side="right"))
    local = c[: max(take, 2)]
    if len(local) >= 3:
        centered = local - local.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        inward = vh[0]
        if np.dot(inward, local[-1] - local[0]) < 0:
            inward = -inward
    else:
        inward = local[-1] - local[0]
    return -_unit(inward)


def _tangent_at(geom: LineString, s: float, half_window: float = 0.75) -> np.ndarray:
    """Forward unit tangent (start→end) at arclength ``s``."""
    length = float(geom.length)
    a = max(0.0, s - half_window)
    b = min(length, s + half_window)
    if b - a < 1e-6:
        a, b = max(0.0, length - 1e-3), length
    pa = geom.interpolate(a)
    pb = geom.interpolate(b)
    return _unit(np.array([pb.x - pa.x, pb.y - pa.y]))


def _signed_offset(geom: LineString, pt: tuple[float, float]) -> tuple[float, float]:
    """Return (arclength projection, signed lateral offset, left positive) of ``pt`` on ``geom``."""
    p = Point(pt)
    s = float(geom.project(p))
    q = geom.interpolate(s)
    t = _tangent_at(geom, s)
    normal = np.array([-t[1], t[0]])
    off = float(np.dot(np.array([p.x - q.x, p.y - q.y]), normal))
    return s, off


def _offset_copy(guide: LineString, s0: float, s1: float, off0: float, off1: float, step: float) -> np.ndarray:
    """Copy guide between arclengths s0→s1 applying a C1 smooth Hermite varying lateral offset."""
    n = max(math.ceil(abs(s1 - s0) / step), 2)
    ss = np.linspace(s0, s1, n + 1)
    pts = shapely.get_coordinates(shapely.line_interpolate_point(guide, ss))
    out = np.empty_like(pts)
    for i, s in enumerate(ss):
        t = _tangent_at(guide, float(s))
        normal = np.array([-t[1], t[0]])
        w = i / n
        smooth_w = 3.0 * (w**2) - 2.0 * (w**3)
        out[i] = pts[i] + normal * (off0 * (1.0 - smooth_w) + off1 * smooth_w)
    return out


def _bezier_connector(pa: np.ndarray, ta: np.ndarray, pb: np.ndarray, tb: np.ndarray, step: float) -> np.ndarray:
    """Cubic Bézier from ``pa`` leaving along ``ta`` to ``pb`` arriving against ``tb`` (outward tangents)."""
    dist = float(np.linalg.norm(pb - pa))
    h = dist / 3.0
    c1 = pa + ta * h
    c2 = pb + tb * h
    n = max(math.ceil(dist / step), 2)
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    pts = (1 - t) ** 3 * pa + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * pb
    pts[0], pts[-1] = pa, pb
    return pts


def _field_polygons(boundary: BaseGeometry | None) -> list[Polygon]:
    if boundary is None or boundary.is_empty:
        return []
    if isinstance(boundary, Polygon):
        return [boundary]
    if isinstance(boundary, MultiPolygon):
        return [p for p in boundary.geoms if not p.is_empty]
    return []


def estimate_spacing(lines: list[LineString], fallback: float = 1.5) -> float:
    """Median nearest-neighbour distance between lines sampled at mid-points."""
    if len(lines) < 2:
        return fallback
    tree = STRtree(lines)
    samples: list[float] = []
    for i, ln in enumerate(lines):
        if ln.length < 2.0:
            continue
        for frac in (0.25, 0.5, 0.75):
            p = ln.interpolate(frac, normalized=True)
            cand = tree.query(p.buffer(6.0 * fallback))
            ds = [lines[j].distance(p) for j in cand if j != i]
            ds = [d for d in ds if d > 0.2]
            if ds:
                samples.append(min(ds))
    if not samples:
        return fallback
    return float(np.median(samples))


# --------------------------------------------------------------------------------------------
# Etapa 1: loops, hairpins e ganchos
# --------------------------------------------------------------------------------------------


def _reversal_cuts(geom: LineString, spacing: float, cfg: PostprocessV2Config) -> list[float]:
    """Arclength positions where the line turns back (hairpin/loop) within a short window."""
    step = cfg.resample_step_m
    pts, dists = _resample(geom, step)
    if len(pts) < 5:
        return []
    w = max(round(cfg.hairpin_window_factor * spacing / step), 2)
    if len(pts) <= 2 * w:
        w = max((len(pts) - 1) // 2, 1)
    angles = np.zeros(len(pts))
    for i in range(w, len(pts) - w):
        angles[i] = _angle_deg(pts[i] - pts[i - w], pts[i + w] - pts[i])
    hit = angles > cfg.hairpin_turn_deg
    cuts: list[float] = []
    i = 0
    while i < len(hit):
        if hit[i]:
            j = i
            while j + 1 < len(hit) and hit[j + 1]:
                j += 1
            k = i + int(np.argmax(angles[i : j + 1]))
            cuts.append(float(dists[k]))
            i = j + 1
        else:
            i += 1
    return cuts


def _split_self_intersections(geom: LineString) -> list[LineString]:
    if geom.is_simple:
        return [geom]
    noded = unary_union(geom)
    return [p for p in _line_parts(noded) if p.length > 0]


def _trim_tip(geom: LineString, side: int, spacing: float, cfg: PostprocessV2Config) -> tuple[LineString, float]:
    """Trim a crooked tip (hook) whose offset from the body axis exceeds tolerance."""
    length = float(geom.length)
    tip_len = min(cfg.tip_window_factor * spacing, 0.35 * length)
    body_len = min(cfg.tip_body_factor * spacing, length - tip_len)
    if tip_len < 2 * cfg.resample_step_m or body_len < spacing:
        return geom, 0.0
    work = geom if side == 0 else LineString(list(geom.coords)[::-1])
    body = substring(work, tip_len, tip_len + body_len)
    bc = np.asarray(body.coords, dtype=float)
    if len(bc) < 2:
        return geom, 0.0
    origin = bc.mean(axis=0)
    if len(bc) >= 3:
        _, _, vh = np.linalg.svd(bc - origin, full_matrices=False)
        axis = vh[0]
    else:
        axis = _unit(bc[-1] - bc[0])
    normal = np.array([-axis[1], axis[0]])
    tol = max(cfg.tip_min_offset_m, cfg.tip_max_offset_factor * spacing)
    pts, dists = _resample(substring(work, 0.0, tip_len), cfg.resample_step_m)
    offsets = np.abs((pts - origin) @ normal)
    # body residual: offsets of the body itself (curved rows) raise the tolerance
    body_res = float(np.max(np.abs((bc - origin) @ normal))) if len(bc) else 0.0
    tol = max(tol, 1.5 * body_res)
    bad = np.flatnonzero(offsets > tol)
    if not len(bad):
        return geom, 0.0
    innermost_bad = int(bad.max())
    cut_at = float(dists[min(innermost_bad + 1, len(dists) - 1)])
    # walk inward until the line is back on axis
    good = np.flatnonzero((offsets <= 0.5 * tol) & (np.arange(len(offsets)) > innermost_bad))
    if len(good):
        cut_at = float(dists[int(good.min())])
    cut_at = min(cut_at, tip_len)
    if cut_at <= 0:
        return geom, 0.0
    knee = work.interpolate(cut_at)
    chord = np.array([pts[0][0] - knee.x, pts[0][1] - knee.y])
    bend = _angle_deg(chord, axis)
    bend = min(bend, 180.0 - bend)
    if bend < cfg.tip_min_bend_deg and float(offsets.max()) < cfg.safe_corridor_factor * spacing:
        return geom, 0.0
    trimmed = substring(work, cut_at, length)
    if side == 1:
        trimmed = LineString(list(trimmed.coords)[::-1])
    return trimmed, cut_at


def clean_loops_and_hooks(
    lines: list[LineV2], spacing: float, cfg: PostprocessV2Config
) -> tuple[list[LineV2], list[UncertainV2], dict[str, int]]:
    out: list[LineV2] = []
    uncertain: list[UncertainV2] = []
    stats = {"hairpins_cortados": 0, "autointersecoes_divididas": 0, "ganchos_cortados": 0, "trechos_retorno_descartados": 0}
    min_len = cfg.min_line_length_factor * spacing

    for ln in lines:
        sp = ln.spacing or spacing
        pieces = _split_self_intersections(ln.geom)
        if len(pieces) > 1:
            stats["autointersecoes_divididas"] += 1
        segs: list[LineString] = []
        for piece in pieces:
            cuts = _reversal_cuts(piece, sp, cfg)
            if not cuts:
                segs.append(piece)
                continue
            stats["hairpins_cortados"] += len(cuts)
            bounds = [0.0, *cuts, float(piece.length)]
            for a, b in itertools.pairwise(bounds):
                if b - a > 1e-3:
                    sub = substring(piece, a, b)
                    if isinstance(sub, LineString) and sub.length > 0:
                        segs.append(sub)
        if not segs:
            continue
        segs.sort(key=lambda g: g.length, reverse=True)
        kept: list[LineString] = []
        for seg in segs:
            # Return leg of a hairpin runs alongside an already-kept piece → discard.
            if kept and any(seg.hausdorff_distance(k) < 0.5 * sp or _covered_fraction(seg, k, 0.5 * sp) > 0.5 for k in kept):
                stats["trechos_retorno_descartados"] += 1
                continue
            kept.append(seg)
        for idx, seg in enumerate(kept):
            geom = seg
            trimmed_total = 0.0
            for side in (0, 1):
                geom2, cut = _trim_tip(geom, side, sp, cfg)
                if cut > 0:
                    stats["ganchos_cortados"] += 1
                    trimmed_total += cut
                    geom = geom2
            if geom.length < min_len:
                if geom.length > 0.5:
                    uncertain.append(UncertainV2(geom, "loop_nao_resolvido" if len(kept) > 1 or len(segs) > 1 else "fragmento_curto", ln.source, list(ln.orig_ids)))
                continue
            hist = list(ln.history)
            if len(segs) > 1 or len(pieces) > 1:
                hist.append("loop_removido")
            if trimmed_total > 0:
                hist.append(f"gancho_cortado_{trimmed_total:.2f}m")
            out.append(
                LineV2(
                    geom=geom,
                    source=ln.source,
                    confidence=ln.confidence,
                    observed_len=min(ln.observed_len, geom.length) if idx == 0 else geom.length,
                    repaired_len=0.0,
                    spacing=ln.spacing,
                    model_version=ln.model_version,
                    orig_ids=list(ln.orig_ids),
                    history=hist,
                    original_reason=ln.original_reason,
                )
            )
    return out, uncertain, stats


def _covered_fraction(a: LineString, b: LineString, tol: float) -> float:
    if a.length <= 0:
        return 0.0
    inter = a.intersection(b.buffer(tol, cap_style="flat"))
    return float(inter.length / a.length) if not inter.is_empty else 0.0


def remove_duplicates(lines: list[LineV2], spacing: float, cfg: PostprocessV2Config) -> tuple[list[LineV2], int]:
    """Drop lines mostly covered by a longer line (incertas duplicating simples, etc.)."""
    if not lines:
        return lines, 0
    geoms = [ln.geom for ln in lines]
    tree = STRtree(geoms)
    order = sorted(range(len(lines)), key=lambda i: geoms[i].length, reverse=True)
    kept: set[int] = set()
    tol = cfg.duplicate_distance_factor * spacing
    removed = 0
    for i in order:
        g = geoms[i]
        dup = False
        for j in tree.query(g, predicate="dwithin", distance=tol):
            j = int(j)
            if j == i or j not in kept:
                continue
            if _covered_fraction(g, geoms[j], tol) >= cfg.duplicate_overlap_fraction:
                dup = True
                break
        if dup:
            removed += 1
            continue
        kept.add(i)
    return [lines[i] for i in sorted(kept)], removed


# --------------------------------------------------------------------------------------------
# Etapa 2: união de linhas quebradas copiando a vizinha
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _JoinCandidate:
    score: float
    a: int
    side_a: int
    b: int
    side_b: int
    connector: np.ndarray
    method: str


def _guides_for_gap(
    pa: tuple[float, float],
    pb: tuple[float, float],
    exclude: set[int],
    geoms: list[LineString],
    tree: STRtree,
    spacing: float,
    cfg: PostprocessV2Config,
) -> list[tuple[int, float, float, float, float]]:
    """Neighbour rows covering the whole gap: (idx, sA, offA, sB, offB), nearest per side."""
    gap = LineString([pa, pb])
    window = gap.buffer(cfg.guide_max_factor * spacing)
    lo, hi = cfg.guide_min_factor * spacing, cfg.guide_max_factor * spacing
    best: dict[int, tuple[float, tuple[int, float, float, float, float]]] = {}
    for j in tree.query(window):
        j = int(j)
        if j in exclude:
            continue
        g = geoms[j]
        s_a, off_a = _signed_offset(g, pa)
        s_b, off_b = _signed_offset(g, pb)
        if not (lo <= abs(off_a) <= hi and lo <= abs(off_b) <= hi):
            continue
        if np.sign(off_a) != np.sign(off_b):
            continue
        # projections must be interior (guide actually spans the gap)
        if min(s_a, s_b) <= 1e-3 or max(s_a, s_b) >= g.length - 1e-3:
            continue
        if abs(s_b - s_a) < 0.5 * gap.length:
            continue
        side = int(np.sign(off_a))
        key = 0.5 * (abs(off_a) + abs(off_b))
        if side not in best or key < best[side][0]:
            best[side] = (key, (j, s_a, off_a, s_b, off_b))
    return [v[1] for v in best.values()]


def _segment_clear(
    seg: LineString, exclude: set[int], geoms: list[LineString], tree: STRtree, corridor: float
) -> bool:
    for j in tree.query(seg, predicate="dwithin", distance=corridor):
        j = int(j)
        if j in exclude:
            continue
        return False
    return True


def _build_join_candidates(
    lines: list[LineV2], spacing: float, cfg: PostprocessV2Config
) -> list[_JoinCandidate]:
    geoms = [ln.geom for ln in lines]
    tree = STRtree(geoms)
    tlen = max(1.5 * spacing, 2.0)
    ends: list[tuple[int, int, tuple[float, float], np.ndarray]] = []
    for i, g in enumerate(geoms):
        c = list(g.coords)
        ends.append((i, 0, (c[0][0], c[0][1]), _outward_tangent(g, 0, tlen)))
        ends.append((i, 1, (c[-1][0], c[-1][1]), _outward_tangent(g, 1, tlen)))
    end_pts = [Point(e[2]) for e in ends]
    end_tree = STRtree(end_pts)
    corridor = cfg.safe_corridor_factor * spacing
    max_lat = cfg.max_lateral_factor * spacing
    min_facing_cos = math.cos(math.radians(cfg.min_facing_deg))
    out: list[_JoinCandidate] = []
    seen: set[tuple[int, int]] = set()

    search_dist = max(cfg.max_gap_m, cfg.max_gap_guiado_m)
    for ei, (a, sa, pa, ta) in enumerate(ends):
        for ej in end_tree.query(end_pts[ei].buffer(search_dist)):
            ej = int(ej)
            if ej == ei or (min(ei, ej), max(ei, ej)) in seen:
                continue
            seen.add((min(ei, ej), max(ei, ej)))
            b, sb, pb, tb = ends[ej]
            if a == b:
                continue
            delta = np.array([pb[0] - pa[0], pb[1] - pa[1]])
            dist = float(np.linalg.norm(delta))
            if dist < 0.05 or dist > search_dist:
                continue
            u = delta / dist
            guides = _guides_for_gap(pa, pb, {a, b}, geoms, tree, spacing, cfg)
            connector: np.ndarray | None = None
            lateral = 0.0
            method = ""
            copies = []
            for j, s_a, off_a, s_b, off_b in guides:
                g = geoms[j]
                lat_diff = abs(off_a - off_b)
                if lat_diff > max_lat:
                    continue
                # guide direction from A towards B must agree with tangents of both tips
                gdir_a = _tangent_at(g, s_a) * (1 if s_b > s_a else -1)
                gdir_b = _tangent_at(g, s_b) * (1 if s_b > s_a else -1)
                if np.dot(ta, gdir_a) < min_facing_cos or np.dot(tb, -gdir_b) < min_facing_cos:
                    continue
                lateral = max(lateral, lat_diff)
                copies.append(_offset_copy(g, s_a, s_b, off_a, off_b, cfg.resample_step_m))

            if dist > cfg.max_gap_m and len(copies) < 2:
                continue

            if copies:
                n = min(len(c) for c in copies)
                if len(copies) > 1:
                    resampled = []
                    for c in copies:
                        ln_c = LineString(c)
                        resampled.append(
                            shapely.get_coordinates(
                                shapely.line_interpolate_point(ln_c, np.linspace(0, 1, n), normalized=True)
                            )
                        )
                    connector = np.mean(resampled, axis=0)
                else:
                    connector = copies[0]
                connector[0] = pa
                connector[-1] = pb
                method = "copia_vizinha" + ("_dupla" if len(copies) > 1 else "")
            else:
                # Se não há cópia válida por guia, tenta costura ou reta curta em gaps pequenos
                seam_max = max(cfg.straight_gap_max_m, cfg.seam_gap_max_factor * spacing)
                if (
                    dist <= seam_max
                    and np.dot(ta, u) >= min_facing_cos
                    and np.dot(tb, -u) >= min_facing_cos
                    and np.dot(ta, -tb) >= min_facing_cos
                ):
                    bis = _unit(ta - tb)
                    normal_b = np.array([-bis[1], bis[0]])
                    lateral = abs(float(np.dot(delta, normal_b)))
                    if lateral <= max_lat:
                        connector = _bezier_connector(np.asarray(pa), ta, np.asarray(pb), tb, cfg.resample_step_m)
                        method = "reta_curta" if dist <= cfg.straight_gap_max_m else "costura_tangente"
            if connector is None:
                continue
            conn_line = _as_line(connector)
            if conn_line is None:
                continue
            if not _segment_clear(conn_line, {a, b}, geoms, tree, corridor):
                continue
            # connector must not cross its own parent lines
            if conn_line.crosses(geoms[a]) or conn_line.crosses(geoms[b]):
                continue
            score = dist + 5.0 * lateral + (0.0 if guides else 0.5)
            out.append(_JoinCandidate(score, a, sa, b, sb, np.asarray(conn_line.coords), method))
    return out


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def join_broken_lines(
    lines: list[LineV2], spacing: float, cfg: PostprocessV2Config
) -> tuple[list[LineV2], list[UncertainV2], dict[str, int]]:
    stats = {"unioes_copia_vizinha": 0, "unioes_costura": 0, "unioes_reta_curta": 0, "unioes_ambiguas": 0}
    uncertain: list[UncertainV2] = []
    for _ in range(max(cfg.join_passes, 1)):
        if len(lines) < 2:
            break
        cands = _build_join_candidates(lines, spacing, cfg)
        if not cands:
            break
        # Ambiguity: per endpoint, best vs second best towards different lines
        by_end: dict[tuple[int, int], list[_JoinCandidate]] = {}
        for c in cands:
            by_end.setdefault((c.a, c.side_a), []).append(c)
            by_end.setdefault((c.b, c.side_b), []).append(c)
        ambiguous_ends: set[tuple[int, int]] = set()
        for key, lst in by_end.items():
            lst.sort(key=lambda c: c.score)
            if len(lst) >= 2:
                best, second = lst[0], lst[1]
                other_best = best.b if (best.a, best.side_a) == key else best.a
                other_second = second.b if (second.a, second.side_a) == key else second.a
                if other_best != other_second and second.score <= best.score * (1.0 + cfg.ambiguity_ratio) + 0.05:
                    ambiguous_ends.add(key)
        cands.sort(key=lambda c: c.score)
        used: set[tuple[int, int]] = set()
        uf = _UnionFind(len(lines))
        links: dict[tuple[int, int], tuple[int, int, np.ndarray, str]] = {}
        reported: set[tuple[int, int, int, int]] = set()
        for c in cands:
            ka, kb = (c.a, c.side_a), (c.b, c.side_b)
            if ka in used or kb in used:
                continue
            if ka in ambiguous_ends or kb in ambiguous_ends:
                sig = (c.a, c.side_a, c.b, c.side_b)
                if sig not in reported:
                    reported.add(sig)
                    conn = _as_line(c.connector)
                    if conn is not None:
                        uncertain.append(
                            UncertainV2(conn, "uniao_ambigua", "conector",
                                        lines[c.a].orig_ids + lines[c.b].orig_ids, c.method)
                        )
                        stats["unioes_ambiguas"] += 1
                continue
            if not uf.union(c.a, c.b):
                continue  # would close a cycle → loop
            used.add(ka)
            used.add(kb)
            links[ka] = (c.b, c.side_b, c.connector, c.method)
            links[kb] = (c.a, c.side_a, c.connector[::-1], c.method)
            key = {"copia_vizinha": "unioes_copia_vizinha", "copia_vizinha_dupla": "unioes_copia_vizinha",
                   "costura_tangente": "unioes_costura"}.get(c.method, "unioes_reta_curta")
            stats[key] = stats.get(key, 0) + 1
        if not links:
            break
        lines = _assemble_chains(lines, links)
    return lines, uncertain, stats


def _assemble_chains(lines: list[LineV2], links: dict[tuple[int, int], tuple[int, int, np.ndarray, str]]) -> list[LineV2]:
    visited: set[int] = set()
    out: list[LineV2] = []
    for start in range(len(lines)):
        if start in visited:
            continue
        # find chain head: walk from start backwards via side 0 links
        head, head_entry_side = start, 0
        seen_walk = {start}
        while (head, head_entry_side) in links:
            nb, nb_side, _, _ = links[(head, head_entry_side)]
            if nb in seen_walk:
                break
            seen_walk.add(nb)
            head, head_entry_side = nb, 1 - nb_side
        # walk forward from head, leaving through the side opposite to entry
        coords: list[np.ndarray] = []
        members: list[LineV2] = []
        repaired = 0.0
        methods: list[str] = []
        cur, entry_side = head, head_entry_side
        while True:
            visited.add(cur)
            members.append(lines[cur])
            c = np.asarray(lines[cur].geom.coords, dtype=float)
            if entry_side == 1:
                c = c[::-1]
            coords.append(c)
            exit_side = 1 - entry_side
            link = links.get((cur, exit_side))
            if link is None:
                break
            nb, nb_side, conn, method = link
            if nb in visited:
                break
            coords.append(np.asarray(conn, dtype=float)[1:-1])
            repaired += float(LineString(conn).length)
            methods.append(method)
            cur, entry_side = nb, nb_side
        if len(members) == 1:
            out.append(members[0])
            continue
        merged = _as_line(np.vstack([c for c in coords if len(c)]))
        if merged is None:
            out.extend(members)
            continue
        hist: list[str] = []
        for m in members:
            hist.extend(m.history)
        hist.append("uniao:" + ",".join(methods))
        out.append(
            LineV2(
                geom=merged,
                source="+".join(sorted({m.source for m in members})),
                confidence=float(np.average([m.confidence for m in members], weights=[m.geom.length for m in members])),
                observed_len=sum(m.observed_len for m in members),
                repaired_len=sum(m.repaired_len for m in members) + repaired,
                spacing=members[0].spacing,
                model_version=members[0].model_version,
                orig_ids=[i for m in members for i in m.orig_ids],
                history=hist,
                original_reason=members[0].original_reason,
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# Etapa 3: extensão até o talhão guiada pela vizinha
# --------------------------------------------------------------------------------------------


def _first_boundary_hit(path: LineString, exterior: BaseGeometry, start: tuple[float, float]) -> tuple[float, float] | None:
    inter = path.intersection(exterior)
    pts = [p for p in _points_of(inter)]
    if not pts:
        return None
    best = min(pts, key=lambda p: path.project(p))
    if path.project(best) < 1e-3 and Point(start).distance(best) < 1e-3:
        others = [p for p in pts if path.project(p) > 1e-3]
        if not others:
            return None
        best = min(others, key=lambda p: path.project(p))
    return (best.x, best.y)


def _points_of(geom: BaseGeometry) -> list[Point]:
    if geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [geom]
    if isinstance(geom, LineString):
        return [Point(geom.coords[0])]
    if hasattr(geom, "geoms"):
        out: list[Point] = []
        for g in geom.geoms:
            out.extend(_points_of(g))
        return out
    return []


def _neighbour_guides_at_tip(
    idx: int,
    pt: tuple[float, float],
    out_dir: np.ndarray,
    geoms: list[LineString],
    tree: STRtree,
    spacing: float,
    cfg: PostprocessV2Config,
) -> list[tuple[float, int, float, float, int]]:
    """Return neighbours near the tip: (distance, idx, s, signed_offset, outward_sign)."""
    p = Point(pt)
    res = []
    for j in tree.query(p.buffer(cfg.neighbor_search_factor * spacing)):
        j = int(j)
        if j == idx:
            continue
        g = geoms[j]
        s, off = _signed_offset(g, pt)
        q = g.interpolate(s)
        perp = math.hypot(q.x - p.x, q.y - p.y)
        if perp < cfg.guide_min_factor * spacing or perp > cfg.neighbor_search_factor * spacing:
            continue
        t = _tangent_at(g, s, half_window=max(1.0, spacing))
        sign = 1 if np.dot(t, out_dir) >= 0 else -1
        res.append((perp, j, s, off, sign))
    res.sort(key=lambda r: r[0])
    return res[: cfg.neighbors_k]


def _supported_length(
    path: LineString, idx: int, geoms: list[LineString], tree: STRtree, spacing: float, cfg: PostprocessV2Config
) -> float:
    """Length from the tip along ``path`` that has an observed parallel neighbour beside it.

    Small holes (neighbour gaps up to ``support_hole_m``) are tolerated; support ends at the first
    longer unsupported stretch. Only ORIGINAL geometries are used, so extensions never chain.
    """
    lo, hi = cfg.guide_min_factor * spacing, cfg.guide_max_factor * spacing
    max_ang = cfg.support_max_angle_deg
    n = max(math.ceil(path.length / cfg.support_sample_m), 1)
    last_supported = 0.0
    for k in range(1, n + 1):
        s = min(k * cfg.support_sample_m, path.length)
        p = path.interpolate(s)
        t = _tangent_at(path, s, half_window=max(0.5, 0.5 * spacing))
        ok = False
        for j in tree.query(p, predicate="dwithin", distance=hi):
            j = int(j)
            if j == idx:
                continue
            g = geoms[j]
            sj = float(g.project(p))
            q = g.interpolate(sj)
            d = math.hypot(q.x - p.x, q.y - p.y)
            # ponto deve estar ao lado da vizinha (não além da ponta dela)
            if not (lo <= d <= hi) or sj <= 1e-3 or sj >= g.length - 1e-3:
                continue
            a = _angle_deg(_tangent_at(g, sj), t)
            if min(a, 180.0 - a) <= max_ang:
                ok = True
                break
        if ok:
            last_supported = s
        elif s - last_supported > cfg.support_hole_m:
            break
    return last_supported


def extend_to_field_boundary(
    lines: list[LineV2],
    boundary: BaseGeometry,
    spacing: float,
    cfg: PostprocessV2Config,
    endpoints_class: dict[tuple[int, int], EndpointType] | None = None,
) -> tuple[list[LineV2], list[UncertainV2], dict[str, int]]:
    stats = {
        "extensoes_copia_vizinha": 0,
        "extensoes_consenso": 0,
        "extensoes_vetadas": 0,
        "pontas_ja_na_borda": 0,
        "pontas_mata_ignoradas": 0,
    }
    uncertain: list[UncertainV2] = []
    polygons = _field_polygons(boundary)
    if not polygons or not lines:
        return lines, uncertain, stats
    buffered_polygons = [pg.buffer(cfg.touch_tolerance_m) for pg in polygons]
    geoms = [ln.geom for ln in lines]
    tree = STRtree(geoms)
    added: list[LineString] = []
    corridor = cfg.safe_corridor_factor * spacing
    tlen = max(1.5 * spacing, 2.0)
    step = cfg.resample_step_m
    new_lines: list[LineV2] = []

    for idx, ln in enumerate(lines):
        coords = np.asarray(ln.geom.coords, dtype=float)
        extra = 0.0
        hist = list(ln.history)
        for side in (0, 1):
            if endpoints_class is not None and endpoints_class.get((idx, side)) == "mata":
                stats["pontas_mata_ignoradas"] += 1
                continue
            pt = tuple(coords[0] if side == 0 else coords[-1])
            p = Point(pt)
            poly_match = next(((pg, pg_buf) for pg, pg_buf in zip(polygons, buffered_polygons, strict=False) if pg_buf.covers(p)), None)
            if poly_match is None:
                continue
            poly, poly_buf = poly_match
            exterior = poly.boundary
            if exterior.distance(p) <= cfg.touch_tolerance_m:
                stats["pontas_ja_na_borda"] += 1
                continue
            if exterior.distance(p) > cfg.max_extension_m:
                continue
            own_line = _as_line(coords)
            if own_line is None:
                continue
            own_dir = _outward_tangent(own_line, side, tlen)
            neigh = _neighbour_guides_at_tip(idx, pt, own_dir, geoms, tree, spacing, cfg)
            # consensus direction from neighbours (outward oriented)
            dirs = []
            for _, j, s, _, sign in neigh:
                dirs.append(_tangent_at(geoms[j], s, half_window=max(1.0, spacing)) * sign)
            if dirs:
                angs = [math.atan2(d[0], d[1]) for d in dirs]
                mean = math.atan2(sum(math.sin(a) for a in angs), sum(math.cos(a) for a in angs))
                diffs = [math.atan2(math.sin(a - mean), math.cos(a - mean)) for a in angs]
                med = mean + float(np.median(diffs))
                consensus = np.array([math.sin(med), math.cos(med)])
                if _angle_deg(own_dir, consensus) <= cfg.own_tip_tolerance_deg:
                    m2 = math.atan2(own_dir[0] + 2 * consensus[0], own_dir[1] + 2 * consensus[1])
                    direction = np.array([math.sin(m2), math.cos(m2)])
                else:
                    direction = consensus
            else:
                direction = own_dir

            path_coords: np.ndarray | None = None
            method = ""
            # Option A: copy the nearest neighbour that continues outward towards the border.
            for _, j, s, off, sign in neigh:
                g = geoms[j]
                remaining = (g.length - s) if sign > 0 else s
                if remaining < 0.5:
                    continue
                s_end = s + sign * min(remaining, cfg.max_extension_m * 1.2)
                _, off_tip = _signed_offset(g, pt)
                copy = _offset_copy_smooth(g, s, s_end, off_tip, off, step)
                copy[0] = pt
                tail_dir = _unit(copy[-1] - copy[-2]) if len(copy) >= 2 else direction
                if _angle_deg(tail_dir, direction) > 30.0:
                    continue
                tail = copy[-1] + tail_dir * (cfg.max_extension_m * 1.5)
                path_coords = np.vstack([copy, tail])
                method = "copia_vizinha"
                break
            if path_coords is None:
                far = np.asarray(pt) + direction * (cfg.max_extension_m * 1.5)
                path_coords = np.array([pt, far])
                method = "consenso_reto"
            path = _as_line(path_coords)
            if path is None:
                continue
            hit = _first_boundary_hit(path, exterior, pt)
            if hit is None:
                continue
            s_hit = float(path.project(Point(hit)))
            if s_hit > cfg.max_extension_m or s_hit < cfg.touch_tolerance_m:
                continue
            ext = substring(path, 0.0, s_hit)
            if not isinstance(ext, LineString) or ext.length <= 0:
                continue
            ext_c = np.asarray(ext.coords, dtype=float)
            ext_c[-1] = hit
            ext = _as_line(ext_c)
            if ext is None:
                continue
            # Suporte: só avança onde há vizinhas OBSERVADAS ao lado (evita invadir matação/mata).
            supported = _supported_length(ext, idx, geoms, tree, spacing, cfg)
            allowed = supported + cfg.extension_free_m
            veto = None
            if ext.length > allowed + 1e-6:
                veto = "extensao_sem_suporte"
            elif not poly_buf.covers(ext):
                veto = "extensao_fora_talhao"
            elif not _segment_clear(ext, {idx}, geoms, tree, corridor):
                veto = "extensao_conflito_vizinha"
            elif any(ext.distance(a) < corridor for a in added if a.envelope.intersects(ext.envelope.buffer(corridor))):
                veto = "extensao_conflito_extensao"
            else:
                trial = _as_line(np.vstack([ext_c[::-1][:-1], coords]) if side == 0 else np.vstack([coords, ext_c[1:]]))
                if trial is None or not trial.is_simple:
                    veto = "extensao_autointersecao"
            if veto:
                stats["extensoes_vetadas"] += 1
                stats[veto] = stats.get(veto, 0) + 1
                uncertain.append(
                    UncertainV2(ext, veto, "extensao", list(ln.orig_ids),
                                f"{method}; suporte={supported:.1f}m; proposta={ext.length:.1f}m")
                )
                continue
            if side == 0:
                coords = np.vstack([ext_c[::-1][:-1], coords])
            else:
                coords = np.vstack([coords, ext_c[1:]])
            extra += float(ext.length)
            added.append(ext)
            hist.append(f"extensao_{method}_{ext.length:.2f}m")
            stats["extensoes_copia_vizinha" if method == "copia_vizinha" else "extensoes_consenso"] += 1
        if extra > 0:
            geom = _as_line(coords)
            if geom is not None and geom.is_simple:
                new_lines.append(
                    LineV2(
                        geom=geom, source=ln.source, confidence=ln.confidence, observed_len=ln.observed_len,
                        repaired_len=ln.repaired_len + extra, spacing=ln.spacing, model_version=ln.model_version,
                        orig_ids=list(ln.orig_ids), history=hist, original_reason=ln.original_reason,
                    )
                )
                continue
        new_lines.append(ln)
    return new_lines, uncertain, stats


# --------------------------------------------------------------------------------------------
# Etapa 4: validação final
# --------------------------------------------------------------------------------------------


def final_validation(
    lines: list[LineV2], spacing: float, cfg: PostprocessV2Config
) -> tuple[list[LineV2], list[UncertainV2], dict[str, int]]:
    stats = {"invalidas": 0, "com_retorno": 0, "cruzamentos": 0, "curtas": 0}
    uncertain: list[UncertainV2] = []
    candidates: list[LineV2] = []
    min_len = cfg.min_line_length_factor * spacing
    for ln in lines:
        g = ln.geom
        if not (isinstance(g, LineString) and g.is_valid and g.is_simple and g.length > 0):
            stats["invalidas"] += 1
            uncertain.append(UncertainV2(g, "geometria_invalida", ln.source, ln.orig_ids))
            continue
        if _reversal_cuts(g, ln.spacing or spacing, cfg):
            stats["com_retorno"] += 1
            uncertain.append(UncertainV2(g, "loop_nao_resolvido", ln.source, ln.orig_ids))
            continue
        if g.length < min_len:
            stats["curtas"] += 1
            uncertain.append(UncertainV2(g, "fragmento_curto", ln.source, ln.orig_ids))
            continue
        candidates.append(ln)

    # crossings: drop the line with more repaired fraction / shorter
    geoms = [c.geom for c in candidates]
    tree = STRtree(geoms) if geoms else None
    rejected: set[int] = set()
    if tree is not None:
        def _weakness(k: int) -> tuple[float, float]:
            c = candidates[k]
            total = c.observed_len + c.repaired_len
            return (c.repaired_len / total if total > 0 else 0.0, -c.geom.length)

        for i, g in enumerate(geoms):
            if i in rejected:
                continue
            for j in tree.query(g):
                j = int(j)
                if j <= i or j in rejected:
                    continue
                if g.crosses(geoms[j]) or (g.intersects(geoms[j]) and not g.touches(geoms[j])):
                    loser = i if _weakness(i) > _weakness(j) else j
                    rejected.add(loser)
                    stats["cruzamentos"] += 1
                    if loser == i:
                        break
    final: list[LineV2] = []
    for k, c in enumerate(candidates):
        if k in rejected:
            uncertain.append(UncertainV2(c.geom, "cruza_outra_linha", c.source, c.orig_ids, ";".join(c.history)))
        else:
            final.append(c)
    return final, uncertain, stats


# --------------------------------------------------------------------------------------------
# Orquestração
# --------------------------------------------------------------------------------------------


def run_postprocess_v2(
    lines: list[LineV2],
    field_boundary: BaseGeometry | None,
    spacing_m: float | None = None,
    config: PostprocessV2Config | None = None,
) -> PostprocessV2Result:
    cfg = config or PostprocessV2Config()
    if spacing_m is None:
        spacing_m = estimate_spacing([ln.geom for ln in lines])
    diag: dict[str, Any] = {"entrada": len(lines), "espacamento_m": spacing_m, "config": asdict(cfg)}
    uncertain: list[UncertainV2] = []

    lines, dup = remove_duplicates(lines, spacing_m, cfg)
    diag["duplicadas_removidas"] = dup

    lines, unc, st = clean_loops_and_hooks(lines, spacing_m, cfg)
    uncertain += unc
    diag["etapa1_loops_ganchos"] = st

    # Etapa 2: união cirúrgica de linhas quebradas copiando vizinhas (C1 suave)
    lines, unc, st = join_broken_lines(lines, spacing_m, cfg)
    uncertain += unc
    diag["etapa2_uniao"] = st

    if field_boundary is not None:
        lines, unc, st = extend_to_field_boundary(lines, field_boundary, spacing_m, cfg)
        uncertain += unc
        diag["etapa3_extensao"] = st
    else:
        diag["etapa3_extensao"] = "desativada: sem --field-boundary/--roi-is-field-boundary"

    # Suavização de quinas residuais em emendas
    lines, smoothed = smooth_residual_kinks(lines, cfg)
    diag["etapa3_quinas_suavizadas"] = smoothed

    lines, unc, st = final_validation(lines, spacing_m, cfg)
    uncertain += unc
    diag["etapa4_validacao"] = st
    diag["saida_completas"] = len(lines)
    diag["saida_incertas"] = len(uncertain)
    return PostprocessV2Result(lines, uncertain, diag)


# --------------------------------------------------------------------------------------------
# Entrada / saída
# --------------------------------------------------------------------------------------------


def lines_from_geodataframe(gdf: gpd.GeoDataFrame, source: str, clip_to: BaseGeometry | None = None) -> list[LineV2]:
    out: list[LineV2] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if clip_to is not None:
            geom = geom.intersection(clip_to)
        for part in _line_parts(geom):
            if part.length <= 0.05:
                continue
            sp = row.get("spacing_local_m")
            conf = row.get("confidence")
            out.append(
                LineV2(
                    geom=part,
                    source=source,
                    confidence=float(conf) if conf is not None and pd.notna(conf) else 0.9,
                    observed_len=float(part.length),
                    spacing=float(sp) if sp is not None and pd.notna(sp) and sp > 0 else None,
                    model_version=str(row.get("model_version") or "pos-v2"),
                    orig_ids=[str(row.get("row_id", ""))],
                    original_reason=str(row.get("repair_reason", "") or ""),
                )
            )
    return out


ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def read_gpkg_sqlite(path: Path) -> gpd.GeoDataFrame:
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        cur = con.cursor()
        table, geom_col, srs = cur.execute(
            "select table_name, column_name, srs_id from gpkg_geometry_columns"
        ).fetchone()
        cols = [r[1] for r in cur.execute(f'pragma table_info("{table}")')]
        data: dict[str, list] = {c: [] for c in cols if c != geom_col}
        geoms = []
        for rec in cur.execute(f'select * from "{table}"'):
            d = dict(zip(cols, rec, strict=False))
            blob = d.pop(geom_col)
            if blob is None:
                continue
            env = ENVELOPE_BYTES[(blob[3] >> 1) & 0b111]
            geoms.append(shapely.from_wkb(bytes(blob[8 + env :])))
            for c in data:
                data[c].append(d[c])
    finally:
        con.close()
    crs = CRS.from_epsg(int(srs))
    df = pd.DataFrame(data)
    return gpd.GeoDataFrame(df, geometry=geoms, crs=crs)


def _write_gpkg_sqlite(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    epsg = gdf.crs.to_epsg() if gdf.crs is not None else 32722
    if epsg is None:
        epsg = 32722
    srs_wkt = gdf.crs.to_wkt() if gdf.crs is not None else ""
    srs_name = gdf.crs.name if gdf.crs is not None else f"EPSG:{epsg}"

    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gpkg_spatial_ref_sys (
            srs_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL PRIMARY KEY,
            organization TEXT NOT NULL,
            organization_coordsys_id INTEGER NOT NULL,
            definition TEXT NOT NULL,
            description TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gpkg_contents (
            table_name TEXT NOT NULL PRIMARY KEY,
            data_type TEXT NOT NULL,
            identifier TEXT UNIQUE,
            description TEXT DEFAULT '',
            last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
            srs_id INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gpkg_geometry_columns (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL,
            m TINYINT NOT NULL,
            CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name)
        )
    """)
    cur.execute("""
        INSERT OR REPLACE INTO gpkg_spatial_ref_sys 
        VALUES (?, ?, 'EPSG', ?, ?, '')
    """, (srs_name, epsg, epsg, srs_wkt))

    cols = [c for c in gdf.columns if c != "geometry"]
    col_defs = []
    for c in cols:
        valid_series = gdf[c].dropna()
        sample = valid_series.iloc[0] if not valid_series.empty else ""
        if isinstance(sample, (int, np.integer)):
            ctype = "INTEGER"
        elif isinstance(sample, (float, np.floating)):
            ctype = "REAL"
        else:
            ctype = "TEXT"
        col_defs.append(f'"{c}" {ctype}')

    table_sql = f'CREATE TABLE "{layer}" (fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, geom BLOB, {", ".join(col_defs)})'
    cur.execute(table_sql)

    min_x, min_y, max_x, max_y = 0.0, 0.0, 0.0, 0.0
    if not gdf.empty:
        total_bounds = gdf.total_bounds
        min_x, min_y, max_x, max_y = float(total_bounds[0]), float(total_bounds[1]), float(total_bounds[2]), float(total_bounds[3])

    cur.execute("""
        INSERT OR REPLACE INTO gpkg_contents 
        (table_name, data_type, identifier, min_x, min_y, max_x, max_y, srs_id)
        VALUES (?, 'features', ?, ?, ?, ?, ?, ?)
    """, (layer, layer, min_x, min_y, max_x, max_y, epsg))

    cur.execute("""
        INSERT OR REPLACE INTO gpkg_geometry_columns
        (table_name, column_name, geometry_type_name, srs_id, z, m)
        VALUES (?, 'geom', 'LINESTRING', ?, 0, 0)
    """, (layer, epsg))

    insert_sql = f'INSERT INTO "{layer}" (geom, {", ".join([f"{c}" for c in cols])}) VALUES (?, {", ".join(["?" for _ in cols])})'

    rows_to_insert = []
    for _, row in gdf.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        wkb = shapely.to_wkb(g, byte_order=1)
        blob = b"GP\x00\x01" + int(epsg).to_bytes(4, "little") + wkb
        vals = [blob]
        for c in cols:
            v = row[c]
            if pd.isna(v):
                vals.append(None)
            elif isinstance(v, (np.integer, int)):
                vals.append(int(v))
            elif isinstance(v, (np.floating, float)):
                vals.append(float(v))
            else:
                vals.append(str(v))
        rows_to_insert.append(vals)

    cur.executemany(insert_sql, rows_to_insert)
    con.commit()
    con.close()


def _write_gpkg(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    try:
        import pyogrio

        pyogrio.write_dataframe(gdf, path, layer=layer, driver="GPKG", geometry_type="LineString", append=False)
    except Exception:
        try:
            gdf.to_file(path, layer=layer, driver="GPKG", index=False)
        except Exception:
            _write_gpkg_sqlite(gdf, path, layer)


def export_complete_v2(path: Path, lines: list[LineV2], crs: CRS | str, spacing_m: float) -> Path:
    crs_obj = CRS.from_user_input(crs)
    rows = []
    for ln in lines:
        total = ln.observed_len + ln.repaired_len
        rep = float(np.clip(ln.repaired_len / total, 0, 1)) if total > 0 else 0.0
        rows.append(
            {
                "row_id": _stable_id(ln.geom),
                "confidence": float(np.clip(ln.confidence, 0, 1)),
                "observed_ratio": 1.0 - rep,
                "repaired_ratio": rep,
                "spacing_local_m": ln.spacing if ln.spacing else spacing_m,
                "orientation_deg": _axial_orientation_deg(ln.geom),
                "topology_status": "valid",
                "repair_reason": json.dumps({"pos_v2": ln.history, "origem": ln.source, "ids_originais": ln.orig_ids}, ensure_ascii=False),
                "model_version": ln.model_version,
            }
        )
    df = pd.DataFrame(rows, columns=[
        "row_id", "confidence", "observed_ratio", "repaired_ratio", "spacing_local_m",
        "orientation_deg", "topology_status", "repair_reason", "model_version",
    ])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries([ln.geom for ln in lines], crs=crs_obj), crs=crs_obj)
    _write_gpkg(gdf, path, COMPLETE_V2_LAYER)
    return path


def export_uncertain_v2(path: Path, items: list[UncertainV2], crs: CRS | str) -> Path:
    crs_obj = CRS.from_user_input(crs)
    rows = [
        {
            "row_id": _stable_id(u.geom),
            "motivo_revisao": u.motivo,
            "origem": u.origem,
            "comprimento_m": float(u.geom.length),
            "ids_originais": ",".join(i for i in u.orig_ids if i),
            "detalhe": u.detalhe,
            "topology_status": "ambiguous",
        }
        for u in items
    ]
    df = pd.DataFrame(rows, columns=["row_id", "motivo_revisao", "origem", "comprimento_m", "ids_originais", "detalhe", "topology_status"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries([u.geom for u in items], crs=crs_obj), crs=crs_obj)
    _write_gpkg(gdf, path, UNCERTAIN_V2_LAYER)
    return path
