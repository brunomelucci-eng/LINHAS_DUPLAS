"""Prepare robust scene manifests with verified geographic groups and clean annotations.

Preserves original raw annotations in data_set/linha_raw, filters out non-planting
features, assigns verified geographic groups dynamically (< 50 km) to eliminate spatial
data leakage, computes SHA-256 hashes, and generates standardized YAML manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path
_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import geopandas as gpd  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import MultiPoint, box  # noqa: E402

from rowgraphnet.data.contracts import choose_local_metric_crs  # noqa: E402
from rowgraphnet.data.manifest import (  # noqa: E402
    IgnoreSpec,
    SceneManifest,
    VisibilitySpec,
    write_scene_manifest,
)

# Distância mínima (metros, CRS métrico) entre linhas distintas dentro de uma ROI.
# Pares de feições com distância estritamente menor invalidam a ROI inteira.
# Em linhas duplas: entrelinha estreita (0.60m a 0.90m), entrelinha larga (>= 1.00m).
# Linhas com distância < 0.40m configuram erro de digitalização, duplicação ou cruzamento espúrio.
MIN_LINE_SPACING_M = 0.40


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def extract_scene_id(filename: str) -> str:
    """Extract a scene identifier from a filename (supports 5-digit numbers, numeric stems, and names)."""
    stem = Path(filename).name
    # Strip aux.xml
    stem = re.sub(r"\.aux\.xml$", "", stem, flags=re.IGNORECASE)
    # Strip extension
    stem = Path(stem).stem
    # If standard 5-digit number is present anywhere
    match = re.search(r"(\d{5})", stem)
    if match:
        return match.group(1)
    # Otherwise clean standard vector prefixes and suffixes
    clean = re.sub(r"^(linhas[-_]|roi[-_]|linha[-_]|result[-_])", "", stem, flags=re.IGNORECASE)
    clean = re.sub(r"([-_]linhas|[-_]roi|[-_]talhoes|[-_]talhao)$", "", clean, flags=re.IGNORECASE)
    clean = clean.replace("-", "_").lower().strip()
    if clean in ("jp_1", "jp1"):
        return "jp1"
    if clean in ("jp_2", "jp2"):
        return "jp2"
    if clean in ("mauro1", "mauro_1"):
        return "mauro1"
    if clean in ("mauro2", "mauro_2"):
        return "mauro_2"
    if clean:
        return clean
    raise ValueError(f"Could not extract scene ID from filename: {filename}")


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometers between two geographic coordinates."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def compute_geographic_clusters(
    scene_ids: list[str],
    roi_files: dict[str, Path],
    max_distance_km: float = 50.0,
) -> dict[str, str]:
    """Cluster scenes dynamically so that any scenes closer than max_distance_km share a group."""
    centers: dict[str, tuple[float, float]] = {}
    for sid in scene_ids:
        gdf = gpd.read_file(roi_files[sid])
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union
        pt = geom.centroid
        centers[sid] = (float(pt.x), float(pt.y))

    # Build single-linkage graph of spatial proximity
    g = nx.Graph()
    g.add_nodes_from(scene_ids)

    for i in range(len(scene_ids)):
        for j in range(i + 1, len(scene_ids)):
            s1, s2 = scene_ids[i], scene_ids[j]
            d = haversine_km(centers[s1][0], centers[s1][1], centers[s2][0], centers[s2][1])
            if d < max_distance_km:
                g.add_edge(s1, s2, distance_km=d)

    components = sorted(
        list(nx.connected_components(g)),
        key=lambda c: (len(c), sorted(list(c))),
        reverse=True,
    )

    group_map: dict[str, str] = {}
    for comp in components:
        members = sorted(list(comp))
        group_name = f"group_cluster_{members[0]}"
        for m in members:
            group_map[m] = group_name

    return group_map


def allocate_group_splits(
    group_map: dict[str, str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    forced_test_scenes: list[str] | set[str] | None = None,
) -> dict[str, str]:
    """Deterministically assign entire geographic groups to train, validation, and test splits."""
    group_members = defaultdict(list)
    for sid, grp in group_map.items():
        group_members[grp].append(sid)

    sorted_groups = sorted(group_members.keys(), key=lambda gr: (-len(group_members[gr]), gr))
    total_scenes = len(group_map)
    target_train = round(total_scenes * train_ratio)
    target_val = max(1, round(total_scenes * val_ratio))

    splits: dict[str, str] = {}
    train_count = 0
    val_count = 0

    forced_groups = set()
    if forced_test_scenes:
        forced_set = set(forced_test_scenes)
        for sid, grp in group_map.items():
            if sid in forced_set:
                forced_groups.add(grp)
        for grp in forced_groups:
            splits[grp] = "test"

    for grp in sorted_groups:
        if grp in forced_groups:
            continue
        count = len(group_members[grp])
        if train_count + count <= target_train or train_count == 0:
            splits[grp] = "train"
            train_count += count
        elif val_count < target_val:
            splits[grp] = "validation"
            val_count += count
        else:
            splits[grp] = "test"

    # Ensure validation and test both have at least one group if possible
    assigned = set(splits.values())
    if "validation" not in assigned and len(sorted_groups) > 1:
        avail = [g for g in sorted_groups if g not in forced_groups]
        if avail:
            splits[avail[-1]] = "validation"
    if "test" not in assigned and len(sorted_groups) > 2:
        splits[sorted_groups[-2]] = "test"

    return splits


def safe_read_vector(file_path: Path, scene_id: str, kind: str) -> gpd.GeoDataFrame | None:
    """Safely read a vector file, trying multiple engines, without crashing the workflow."""
    engines = ["pyogrio", "fiona", None]
    last_err = None
    for eng in engines:
        try:
            kwargs = {"engine": eng} if eng else {}
            gdf = gpd.read_file(file_path, **kwargs)
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            return gdf
        except Exception as exc:
            last_err = exc
            continue
    print(f"[AVISO] Arquivo {kind} corrompido ou ilegível na cena {scene_id} ({file_path.name}): {last_err}")
    return None


def repair_line_geometry_detailed(geom: Any) -> tuple[Any | None, bool, bool]:
    """Validate, repair, and clean a line geometry, dropping degenerate points.

    Returns (repaired_geometry, was_repaired_flag, geometry_lost_flag).
    ``geometry_lost_flag`` is True when the repair deleted the whole feature or
    any non-empty component of it (degenerate parts, non-linear components).
    Pure clean-ups (Z removal, consecutive duplicate vertices) are not losses.
    """
    if geom is None or geom.is_empty:
        return None, False, False

    was_repaired = False
    lost = False
    try:
        from shapely import force_2d, make_valid

        if getattr(geom, "has_z", False):
            geom = force_2d(geom)
            was_repaired = True

        if not geom.is_valid:
            geom = make_valid(geom)
            was_repaired = True
            if geom is None or geom.is_empty:
                return None, True, True
    except Exception:
        try:
            from shapely import make_valid

            geom = make_valid(geom)
            was_repaired = True
        except Exception:
            return None, True, True

    from shapely.geometry import LineString, MultiLineString

    parts = []
    if geom.geom_type == "LineString":
        parts = [geom]
    elif geom.geom_type == "MultiLineString":
        parts = list(geom.geoms)
    elif hasattr(geom, "geoms"):
        was_repaired = True
        if any(
            g.geom_type not in {"LineString", "MultiLineString"} and not g.is_empty
            for g in geom.geoms
        ):
            lost = True
        parts = [g for g in geom.geoms if g.geom_type in {"LineString", "MultiLineString"}]
        flat = []
        for p in parts:
            if p.geom_type == "MultiLineString":
                flat.extend(list(p.geoms))
            else:
                flat.append(p)
        parts = flat

    valid_parts = []
    for p in parts:
        if p is None or p.is_empty:
            continue
        coords = list(p.coords)
        if len(coords) < 2:
            was_repaired = True
            lost = True
            continue
        dedup = [coords[0]]
        for c in coords[1:]:
            if c != dedup[-1]:
                dedup.append(c)
        if len(dedup) != len(coords):
            was_repaired = True
        if len(dedup) >= 2:
            ls = LineString(dedup)
            if ls.is_valid and ls.length > 1e-12:
                valid_parts.append(ls)
            else:
                was_repaired = True
                lost = True
        else:
            lost = True

    if not valid_parts:
        return None, True, True
    final_geom = valid_parts[0] if len(valid_parts) == 1 else MultiLineString(valid_parts)
    return final_geom, was_repaired, lost


def repair_line_geometry(geom: Any) -> tuple[Any | None, bool]:
    """Validate, repair, and clean a line geometry, dropping degenerate points.

    Returns (repaired_geometry, was_repaired_flag).
    """
    repaired, was_repaired, _lost = repair_line_geometry_detailed(geom)
    return repaired, was_repaired


def _flatten_linestrings(geom: Any) -> list[Any]:
    """Return the non-degenerate LineString components of any geometry."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom] if geom.length > 0 else []
    if hasattr(geom, "geoms"):
        out: list[Any] = []
        for part in geom.geoms:
            out.extend(_flatten_linestrings(part))
        return out
    return []


def _rois_hit_by_geometries(geoms: list[Any], roi_geoms: gpd.GeoSeries) -> set[Any]:
    """Return ROI index labels touched by any (possibly invalid) original geometry.

    Each geometry is probed by its repaired form and by its raw vertices, so a
    feature that the repair deleted can still be located. If a predicate
    cannot be evaluated for a ROI, that ROI is conservatively reported.
    """
    hit: set[Any] = set()
    for geom in geoms:
        if geom is None:
            continue
        try:
            if geom.is_empty:
                continue
        except Exception:
            continue
        probes: list[Any] = []
        try:
            probe = shapely.force_2d(geom)
            if not probe.is_valid:
                probe = shapely.make_valid(probe)
            if probe is not None and not probe.is_empty:
                probes.append(probe)
        except Exception:
            pass
        try:
            coords = shapely.get_coordinates(geom)
            coords = coords[np.isfinite(coords).all(axis=1)]
            if len(coords):
                probes.append(MultiPoint([tuple(c) for c in coords]))
        except Exception:
            pass
        if not probes:
            continue
        for label, roi_geom in roi_geoms.items():
            if label in hit:
                continue
            for probe in probes:
                try:
                    if roi_geom.intersects(probe):
                        hit.add(label)
                        break
                except Exception:
                    hit.add(label)
                    break
    return hit


def _rois_with_close_lines(
    lines: gpd.GeoDataFrame,
    rois: gpd.GeoDataFrame,
    min_spacing_m: float,
) -> dict[Any, float]:
    """Return {ROI label: min distance in m} for ROIs holding distinct lines closer than ``min_spacing_m``.

    Distances are measured in a local metric CRS between the parts of each
    line clipped to the ROI. Parts of the same feature are never compared
    against each other (they belong to the same row).
    """
    if lines.empty or rois.empty or len(lines) < 2:
        return {}
    metric_crs = choose_local_metric_crs(rois.crs, rois.geometry.union_all())
    rois_m = rois.to_crs(metric_crs)
    line_arr = np.asarray(lines.to_crs(metric_crs).geometry.values, dtype=object)
    tree = shapely.STRtree(line_arr)

    close: dict[Any, float] = {}
    for label, roi_geom in rois_m.geometry.items():
        candidates = tree.query(roi_geom, predicate="intersects")
        if len(candidates) < 2:
            continue
        clipped = []
        for geom in shapely.intersection(line_arr[candidates], roi_geom):
            parts = _flatten_linestrings(geom)
            if parts:
                clipped.append(parts[0] if len(parts) == 1 else shapely.MultiLineString(parts))
        if len(clipped) < 2:
            continue
        sub = np.asarray(clipped, dtype=object)
        left, right = shapely.STRtree(sub).query(sub, predicate="dwithin", distance=min_spacing_m)
        pair_mask = left < right
        if not pair_mask.any():
            continue
        distances = shapely.distance(sub[left[pair_mask]], sub[right[pair_mask]])
        too_close = distances < min_spacing_m
        if too_close.any():
            close[label] = float(distances[too_close].min())
    return close


def repair_roi_geometry(geom: Any) -> tuple[Any | None, bool]:
    """Validate and repair a polygon ROI geometry (fixing self-intersections, bowties, etc.).

    Returns (repaired_geometry, was_repaired_flag).
    """
    if geom is None or geom.is_empty:
        return None, False

    was_repaired = False
    try:
        from shapely import force_2d, make_valid

        if getattr(geom, "has_z", False):
            geom = force_2d(geom)
            was_repaired = True

        if not geom.is_valid:
            geom = make_valid(geom)
            was_repaired = True
            if geom is None or geom.is_empty or not geom.is_valid:
                geom = geom.buffer(0)
    except Exception:
        try:
            geom = geom.buffer(0)
            was_repaired = True
        except Exception:
            return None, True

    if geom is None or geom.is_empty:
        return None, True

    from shapely.geometry import MultiPolygon

    parts = []
    if geom.geom_type == "Polygon":
        parts = [geom]
    elif geom.geom_type == "MultiPolygon":
        parts = list(geom.geoms)
    elif hasattr(geom, "geoms"):
        was_repaired = True
        parts = [g for g in geom.geoms if g.geom_type in {"Polygon", "MultiPolygon"}]
        flat = []
        for p in parts:
            if p.geom_type == "MultiPolygon":
                flat.extend(list(p.geoms))
            else:
                flat.append(p)
        parts = flat

    valid_parts = [p for p in parts if p.is_valid and not p.is_empty and p.area > 0]
    if len(valid_parts) != len(parts):
        was_repaired = True
    if not valid_parts:
        return None, True
    final_geom = valid_parts[0] if len(valid_parts) == 1 else MultiPolygon(valid_parts)
    return final_geom, was_repaired


def sanitize_annotations(
    scene_id: str,
    raw_path: Path,
    clean_path: Path,
    roi_gdf: gpd.GeoDataFrame | None = None,
    img_path: Path | None = None,
    min_line_spacing_m: float = MIN_LINE_SPACING_M,
) -> tuple[gpd.GeoDataFrame | None, dict[str, Any]]:
    """Load raw annotations, auto-repair corrupted geometries, ensure unique row_id, and save.

    When ``roi_gdf`` is given, every ROI feature that contains distinct lines
    closer than ``min_line_spacing_m`` or a line geometry deleted (fully or
    partially) by the repair is discarded; its index label is reported in
    ``stats["discarded_roi_ids"]`` so the caller can rewrite the ROI file.
    The remaining ROIs of the scene are kept.
    """
    stats = {
        "scene_id": scene_id,
        "type": "annotations",
        "file": str(raw_path.name),
        "total_raw": 0,
        "valid_initial": 0,
        "repaired": 0,
        "lost_in_repair": 0,
        "dropped_invalid": 0,
        "dropped_outside_roi": 0,
        "dropped_outside_image": 0,
        "dropped_in_discarded_roi": 0,
        "partially_crossing_roi": 0,
        "discarded_roi_ids": [],
        "discarded_roi_ids_close_lines": [],
        "discarded_roi_ids_repair_loss": [],
        "close_lines_min_distance_m": {},
        "roi_quality_check_failed": False,
        "final_clean": 0,
        "corrupted_file": False,
    }
    gdf = safe_read_vector(raw_path, scene_id, "anotações de linha")
    if gdf is None or gdf.empty:
        stats["corrupted_file"] = True
        return None, stats

    stats["total_raw"] = len(gdf)

    # Filter out known non-planting features (e.g. contour terraces)
    if "LayerCAD" in gdf.columns:
        mask_lombada = gdf["LayerCAD"].astype(str).str.contains("LOMBADA", case=False, na=False)
        removed_count = int(mask_lombada.sum())
        if removed_count > 0:
            gdf = gdf.loc[~mask_lombada].copy()
            print(f"[Sanitize] Cena {scene_id}: Removidos {removed_count} elementos não-plantio ('LOMBADA')")

    original_geoms = list(gdf.geometry)
    repaired_geoms = []
    lost_flags = []
    for g in original_geoms:
        rep_g, rep_flag, lost_flag = repair_line_geometry_detailed(g)
        repaired_geoms.append(rep_g)
        lost_flags.append(lost_flag)
        if rep_flag:
            stats["repaired"] += 1
        else:
            stats["valid_initial"] += 1

    gdf["geometry"] = repaired_geoms

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        valid_geom = (
            gdf.geometry.notna()
            & (~gdf.geometry.is_empty)
            & gdf.geometry.is_valid
            & (gdf.geometry.length > 0)
            & gdf.geom_type.isin({"LineString", "MultiLineString"})
        )
    # Original geometries deleted (fully or partially) by the repair step; the
    # ROIs they touch are discarded below.
    valid_flags = valid_geom.to_numpy(dtype=bool)
    lost_originals = [
        orig
        for orig, lost_flag, is_valid in zip(original_geoms, lost_flags, valid_flags, strict=True)
        if lost_flag or not is_valid
    ]
    stats["lost_in_repair"] = len(lost_originals)
    dropped = int((~valid_geom).sum())
    stats["dropped_invalid"] = dropped
    gdf = gdf.loc[valid_geom].copy()

    # 1. Filtro espacial contra a imagem raster (se fornecida)
    if img_path is not None and Path(img_path).exists():
        try:
            with rasterio.open(img_path) as src:
                img_crs = src.crs
                img_box = box(*src.bounds)
            if img_crs is not None and gdf.crs is not None:
                gdf_img = gdf.to_crs(img_crs)
                mask_in_img = gdf_img.geometry.intersects(img_box)
            else:
                mask_in_img = gdf.geometry.intersects(img_box)
            dropped_img = int((~mask_in_img).sum())
            stats["dropped_outside_image"] = dropped_img
            if dropped_img > 0:
                print(f"[Filtro Espacial] Cena {scene_id}: {dropped_img} linhas fora da imagem descartadas.")
                gdf = gdf.loc[mask_in_img].copy()
        except Exception as exc:
            print(f"[AVISO] Não foi possível verificar sobreposição com imagem na cena {scene_id}: {exc}")

    # 2. Descarte de ROIs comprometidas: linhas distintas a menos de
    #    ``min_line_spacing_m`` ou geometrias apagadas pelo reparo. As demais
    #    ROIs da mesma cena são mantidas.
    if roi_gdf is not None and not roi_gdf.empty:
        try:
            roi_in_ann_crs = roi_gdf.to_crs(gdf.crs) if (gdf.crs and roi_gdf.crs) else roi_gdf
            repair_loss_rois = _rois_hit_by_geometries(lost_originals, roi_in_ann_crs.geometry)
            close_rois = _rois_with_close_lines(gdf, roi_in_ann_crs, min_line_spacing_m)
        except Exception as exc:
            stats["roi_quality_check_failed"] = True
            print(
                f"[AVISO] Cena {scene_id} descartada: não foi possível verificar espaçamento "
                f"mínimo/perdas de reparo por ROI: {exc}"
            )
            return None, stats

        def _native(label: Any) -> Any:
            return label.item() if isinstance(label, np.generic) else label

        discarded = repair_loss_rois | set(close_rois)
        stats["discarded_roi_ids_repair_loss"] = [_native(x) for x in sorted(repair_loss_rois, key=str)]
        stats["discarded_roi_ids_close_lines"] = [_native(x) for x in sorted(close_rois, key=str)]
        stats["discarded_roi_ids"] = [_native(x) for x in sorted(discarded, key=str)]
        stats["close_lines_min_distance_m"] = {
            str(label): round(dist, 4) for label, dist in sorted(close_rois.items(), key=lambda kv: str(kv[0]))
        }
        if discarded:
            for label in stats["discarded_roi_ids_close_lines"]:
                print(
                    f"[Qualidade ROI] Cena {scene_id}: ROI {label} descartada - linhas a "
                    f"{close_rois[label]:.3f} m (< {min_line_spacing_m:.2f} m)."
                )
            for label in stats["discarded_roi_ids_repair_loss"]:
                print(
                    f"[Qualidade ROI] Cena {scene_id}: ROI {label} descartada - geometria de "
                    f"linha apagada pelo reparo geométrico."
                )
            kept_roi = roi_in_ann_crs.drop(index=list(discarded))
            if kept_roi.empty:
                print(f"[AVISO] Cena {scene_id}: todas as ROIs foram descartadas por qualidade das linhas.")
                return None, stats
            if not gdf.empty:
                discarded_union = roi_in_ann_crs.loc[list(discarded)].geometry.union_all()
                kept_union = kept_roi.geometry.union_all()
                only_discarded = gdf.geometry.intersects(discarded_union) & ~gdf.geometry.intersects(kept_union)
                stats["dropped_in_discarded_roi"] = int(only_discarded.sum())
                gdf = gdf.loc[~only_discarded].copy()
            roi_gdf = kept_roi

    # 3. Filtro espacial contra o talhão / ROI (se fornecido)
    if roi_gdf is not None and not roi_gdf.empty:
        try:
            roi_in_ann_crs = roi_gdf.to_crs(gdf.crs) if (gdf.crs and roi_gdf.crs) else roi_gdf
            roi_geom = (
                roi_in_ann_crs.geometry.union_all()
                if hasattr(roi_in_ann_crs.geometry, "union_all")
                else roi_in_ann_crs.unary_union
            )
            mask_intersects_roi = gdf.geometry.intersects(roi_geom)
            dropped_roi = int((~mask_intersects_roi).sum())
            stats["dropped_outside_roi"] = dropped_roi
            if dropped_roi > 0:
                print(
                    f"[Filtro Espacial] Cena {scene_id}: {dropped_roi} linhas situadas totalmente fora do talhão (ROI) descartadas."
                )
                gdf = gdf.loc[mask_intersects_roi].copy()

            if not gdf.empty:
                mask_within_roi = gdf.geometry.within(roi_geom)
                part_crossing = int((~mask_within_roi).sum())
                stats["partially_crossing_roi"] = part_crossing
        except Exception as exc:
            print(f"[AVISO] Não foi possível verificar sobreposição com ROI na cena {scene_id}: {exc}")

    stats["final_clean"] = len(gdf)

    if gdf.empty:
        print(f"[AVISO] Cena {scene_id}: todas as geometrias de linhas eram inválidas/corrompidas ou externas e não puderam ser recuperadas!")
        return None, stats

    if (
        stats["repaired"] > 0
        or stats["dropped_invalid"] > 0
        or stats["dropped_outside_roi"] > 0
        or stats["dropped_outside_image"] > 0
        or stats["dropped_in_discarded_roi"] > 0
    ):
        details_dropped = []
        if stats["dropped_invalid"] > 0:
            details_dropped.append(f"{stats['dropped_invalid']} inválidas")
        if stats["dropped_outside_roi"] > 0:
            details_dropped.append(f"{stats['dropped_outside_roi']} fora do talhão")
        if stats["dropped_outside_image"] > 0:
            details_dropped.append(f"{stats['dropped_outside_image']} fora da imagem")
        if stats["dropped_in_discarded_roi"] > 0:
            details_dropped.append(f"{stats['dropped_in_discarded_roi']} em ROIs descartadas")
        dropped_str = f", {', '.join(details_dropped)} descartadas" if details_dropped else ""
        print(
            f"[Reparo Geométrico] Cena {scene_id} (linhas): {stats['repaired']} geometrias reparadas"
            f"{dropped_str}, {stats['final_clean']} salvas com sucesso."
        )

    # Ensure row_id column exists, is string, and is unique per feature
    row_id_field = "row_id"
    if row_id_field not in gdf.columns or gdf[row_id_field].isna().any():
        gdf[row_id_field] = [f"row_{i+1:04d}" for i in range(len(gdf))]
    else:
        gdf[row_id_field] = gdf[row_id_field].astype(str)
        if len(gdf[row_id_field].unique()) != len(gdf):
            gdf[row_id_field] = [f"row_{i+1:04d}" for i in range(len(gdf))]

    for col in gdf.columns:
        if col != "geometry" and any(hasattr(v, "__array__") or isinstance(v, (list, dict)) for v in gdf[col].dropna()):
            gdf[col] = gdf[col].astype(str)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text(gdf.to_json(), encoding="utf-8")
    return gdf, stats


def sanitize_roi(
    scene_id: str,
    raw_path: Path,
    clean_path: Path,
) -> tuple[gpd.GeoDataFrame | None, dict[str, Any]]:
    """Load raw ROI, auto-repair corrupted polygons, and save."""
    stats = {
        "scene_id": scene_id,
        "type": "roi",
        "file": str(raw_path.name),
        "total_raw": 0,
        "valid_initial": 0,
        "repaired": 0,
        "dropped_invalid": 0,
        "final_clean": 0,
        "corrupted_file": False,
    }
    gdf = safe_read_vector(raw_path, scene_id, "polígonos de ROI/Talhão")
    if gdf is None or gdf.empty:
        stats["corrupted_file"] = True
        return None, stats

    stats["total_raw"] = len(gdf)

    repaired_geoms = []
    for g in gdf.geometry:
        rep_g, rep_flag = repair_roi_geometry(g)
        repaired_geoms.append(rep_g)
        if rep_flag:
            stats["repaired"] += 1
        else:
            stats["valid_initial"] += 1

    gdf["geometry"] = repaired_geoms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        valid_roi = (
            gdf.geometry.notna()
            & (~gdf.geometry.is_empty)
            & gdf.geometry.is_valid
            & (gdf.geometry.area > 0)
            & gdf.geom_type.isin({"Polygon", "MultiPolygon"})
        )
    dropped = int((~valid_roi).sum())
    stats["dropped_invalid"] = dropped
    gdf = gdf.loc[valid_roi].copy()
    stats["final_clean"] = len(gdf)

    if gdf.empty:
        print(f"[AVISO] Cena {scene_id}: ROI inválida/corrompida e sem polígonos válidos!")
        return None, stats

    if stats["repaired"] > 0:
        print(
            f"[Reparo Geométrico] Cena {scene_id} (ROI): {stats['repaired']} polígonos reparados com sucesso!"
        )

    for col in gdf.columns:
        if col != "geometry" and any(hasattr(v, "__array__") or isinstance(v, (list, dict)) for v in gdf[col].dropna()):
            gdf[col] = gdf[col].astype(str)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_text(gdf.to_json(), encoding="utf-8")
    return gdf, stats


def _link_or_copy(src: Path, dst: Path) -> None:
    """Safely create a hardlink, or fallback to file copy if linking fails."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def sync_from_data_if_needed(data_path: Path) -> None:
    """Sync any new/missing files from data/IMAGE, data/LINHAS, data/TALHOES to data_set/."""
    img_dir = data_path / "imagem"
    linha_dir = data_path / "linha"
    linha_raw_dir = data_path / "linha_raw"
    roi_dir = data_path / "roi"
    roi_raw_dir = data_path / "roi_raw"

    img_dir.mkdir(parents=True, exist_ok=True)
    linha_dir.mkdir(parents=True, exist_ok=True)
    linha_raw_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_raw_dir.mkdir(parents=True, exist_ok=True)

    source_data = Path("data").resolve()
    source_img = source_data / "IMAGE"
    source_linha = source_data / "LINHAS"
    source_talhao = source_data / "TALHOES"

    if source_img.exists() and source_linha.exists() and source_talhao.exists():
        for f in source_img.iterdir():
            if f.is_file() and f.suffix.lower() in {".gpkg", ".tif", ".tiff"}:
                _link_or_copy(f, img_dir / f.name)
        for f in source_linha.iterdir():
            if f.is_file() and f.suffix.lower() in {".geojson", ".gpkg"}:
                _link_or_copy(f, linha_raw_dir / f.name)
                _link_or_copy(f, linha_dir / f.name)
        for f in source_talhao.iterdir():
            if f.is_file() and f.suffix.lower() in {".geojson", ".gpkg"}:
                _link_or_copy(f, roi_raw_dir / f.name)
                _link_or_copy(f, roi_dir / f.name)


def prepare_dataset_manifests(
    data_dir: str | Path = "data_set",
    output_manifest_dir: str | Path = "data_set/manifests",
    forced_test_scenes: list[str] | set[str] | None = None,
    excluded_scenes: list[str] | set[str] | None = None,
) -> list[Path]:
    data_path = Path(data_dir).resolve()
    manifest_path = Path(output_manifest_dir).resolve()
    manifest_path.mkdir(parents=True, exist_ok=True)

    img_dir = data_path / "imagem"
    linha_dir = data_path / "linha"
    linha_raw_dir = data_path / "linha_raw"
    roi_dir = data_path / "roi"
    roi_raw_dir = data_path / "roi_raw"

    img_dir.mkdir(parents=True, exist_ok=True)
    linha_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    # Sync from data/ if data_set is empty
    sync_from_data_if_needed(data_path)

    # Collect valid rasters strictly from data_set/imagem (ignoring auxiliary files)
    valid_raster_exts = {".gpkg", ".tif", ".tiff"}
    img_files: dict[str, Path] = {}
    for f in img_dir.iterdir():
        if f.is_file() and f.suffix.lower() in valid_raster_exts and not f.name.endswith(".aux.xml"):
            try:
                sid = extract_scene_id(f.name)
                img_files[sid] = f
            except ValueError:
                continue

    # Collect vector annotations strictly from data_set/linha (preferring linha_raw if present)
    valid_vector_exts = {".geojson", ".gpkg", ".shp", ".json"}
    linha_files: dict[str, Path] = {}
    source_linha_dir = linha_raw_dir if linha_raw_dir.exists() and any(linha_raw_dir.iterdir()) else linha_dir
    for f in source_linha_dir.iterdir():
        if f.is_file() and f.suffix.lower() in valid_vector_exts:
            try:
                sid = extract_scene_id(f.name)
                linha_files[sid] = f
            except ValueError:
                continue

    # Collect ROI polygons strictly from data_set/roi (preferring roi_raw if present)
    roi_files: dict[str, Path] = {}
    source_roi_dir = roi_raw_dir if roi_raw_dir.exists() and any(roi_raw_dir.iterdir()) else roi_dir
    for f in source_roi_dir.iterdir():
        if f.is_file() and f.suffix.lower() in valid_vector_exts:
            try:
                sid = extract_scene_id(f.name)
                roi_files[sid] = f
            except ValueError:
                continue

    # Candidate scenes MUST strictly be present in all 3 folders: imagem, linha, and roi
    candidate_scene_ids = sorted(
        list(set(img_files.keys()) & set(linha_files.keys()) & set(roi_files.keys()))
    )
    if excluded_scenes:
        excluded_set = set(excluded_scenes)
        candidate_scene_ids = [s for s in candidate_scene_ids if s not in excluded_set]
    print(
        f"[Manifests] Encontradas {len(candidate_scene_ids)} cenas válidas estritamente presentes "
        f"em imagem/, linha/ e roi/{f' (excluindo {excluded_scenes})' if excluded_scenes else ''}."
    )

    # Geometric verification and auto-repair of ROIs
    audit_reports: list[dict[str, Any]] = []
    clean_roi_files: dict[str, Path] = {}
    valid_roi_scene_ids: list[str] = []
    gdf_roi_cache: dict[str, gpd.GeoDataFrame] = {}
    roi_stats_cache: dict[str, dict[str, Any]] = {}

    for scene_id in candidate_scene_ids:
        src_roi_file = roi_files[scene_id]
        clean_roi_file = roi_dir / f"{src_roi_file.stem}.geojson"
        gdf_roi, roi_stats = sanitize_roi(scene_id, src_roi_file, clean_roi_file)
        audit_reports.append(roi_stats)
        roi_stats_cache[scene_id] = roi_stats
        if gdf_roi is not None and not gdf_roi.empty:
            clean_roi_files[scene_id] = clean_roi_file
            valid_roi_scene_ids.append(scene_id)
            gdf_roi_cache[scene_id] = gdf_roi
        else:
            print(f"[AVISO] Cena {scene_id} ignorada devido a ROI corrompida e irrecuperável.")

    # Geometric verification and auto-repair of Line Annotations
    clean_linha_files: dict[str, Path] = {}
    valid_scene_ids: list[str] = []
    gdf_clean_cache: dict[str, gpd.GeoDataFrame] = {}

    for scene_id in valid_roi_scene_ids:
        src_linha_file = linha_files[scene_id]
        clean_linha_file = linha_dir / f"{src_linha_file.stem}.geojson"
        roi_gdf = gdf_roi_cache.get(scene_id)
        img_file = img_files.get(scene_id)
        gdf_clean, line_stats = sanitize_annotations(
            scene_id,
            src_linha_file,
            clean_linha_file,
            roi_gdf=roi_gdf,
            img_path=img_file,
            min_line_spacing_m=MIN_LINE_SPACING_M,
        )
        audit_reports.append(line_stats)

        # Persist the ROI discards decided from the line quality checks.
        discarded_roi_ids = line_stats.get("discarded_roi_ids", [])
        roi_stats = roi_stats_cache[scene_id]
        roi_stats["discarded_close_lines"] = len(line_stats.get("discarded_roi_ids_close_lines", []))
        roi_stats["discarded_repair_loss"] = len(line_stats.get("discarded_roi_ids_repair_loss", []))
        roi_stats["discarded_quality"] = len(discarded_roi_ids)
        if discarded_roi_ids and roi_gdf is not None:
            kept_roi_gdf = roi_gdf.drop(index=discarded_roi_ids)
            roi_stats["final_clean"] = len(kept_roi_gdf)
            if gdf_clean is not None and not gdf_clean.empty and not kept_roi_gdf.empty:
                clean_roi_file = clean_roi_files[scene_id]
                clean_roi_file.write_text(kept_roi_gdf.to_json(), encoding="utf-8")
                gdf_roi_cache[scene_id] = kept_roi_gdf
                print(
                    f"[Qualidade ROI] Cena {scene_id}: {len(discarded_roi_ids)} ROI(s) descartada(s), "
                    f"{len(kept_roi_gdf)} mantida(s) em {clean_roi_file.name}."
                )

        if gdf_clean is not None and not gdf_clean.empty:
            clean_linha_files[scene_id] = clean_linha_file
            gdf_clean_cache[scene_id] = gdf_clean
            valid_scene_ids.append(scene_id)
        else:
            print(f"[AVISO] Cena {scene_id} ignorada devido a anotações corrompidas, irrecuperáveis ou ROIs descartadas.")

    print(f"[Manifests] Cenas 100% verificadas e aptas: {len(valid_scene_ids)} de {len(candidate_scene_ids)}")

    # Remove any manifest left over from earlier runs that is not among the valid scenes,
    # so training (which globs the manifest directory) cannot load stale data.
    valid_manifest_names = {f"scene_{sid}.yaml" for sid in valid_scene_ids}
    for m_file in manifest_path.glob("scene_*.yaml"):
        if m_file.name not in valid_manifest_names:
            m_file.unlink()
            print(f"[Manifests] Manifesto obsoleto removido: {m_file.name}")

    # Compute dynamic geographic clusters (< 50 km distance) based on repaired ROIs
    print("[Manifests] Calculando agrupamentos espaciais dinâmicos (< 50 km) sem vazamento...")
    geographic_groups = compute_geographic_clusters(valid_scene_ids, clean_roi_files, max_distance_km=50.0)
    group_splits = allocate_group_splits(
        geographic_groups,
        train_ratio=0.70,
        val_ratio=0.15,
        forced_test_scenes=forced_test_scenes,
    )

    unique_groups = sorted(set(geographic_groups.values()))
    print(f"[Manifests] Identificados {len(unique_groups)} clusters geográficos nas {len(valid_scene_ids)} cenas.")

    manifest_files: list[Path] = []
    split_counts: dict[str, int] = defaultdict(int)

    for scene_id in valid_scene_ids:
        img_file = img_files[scene_id]
        clean_linha_file = clean_linha_files[scene_id]
        clean_roi_file = clean_roi_files[scene_id]
        gdf_clean = gdf_clean_cache[scene_id]

        img_hash = sha256_file(img_file)
        roi_hash = sha256_file(clean_roi_file)
        linha_hash = sha256_file(clean_linha_file)

        group_id = geographic_groups[scene_id]
        split = group_splits[group_id]
        split_counts[split] += 1

        scene_name = f"scene_{scene_id}"

        manifest = SceneManifest(
            schema_version=1,
            scene_id=scene_name,
            group_id=group_id,
            split=split,
            image=img_file,
            bands=(1, 2, 3),
            roi=clean_roi_file,
            annotations=clean_linha_file,
            visibility=VisibilitySpec(
                field="visibility",
                observed_values=("observed",),
                occluded_values=("occluded",),
            ),
            ignore=IgnoreSpec(field="visibility", values=("ignore",)),
            sensor={
                "sensor": "orthomosaic",
                "scene_id": scene_id,
                "image_sha256": img_hash,
                "roi_sha256": roi_hash,
                "annotations_sha256": linha_hash,
                "feature_count": len(gdf_clean),
            },
            row_id_field="row_id",
        )

        out_path = manifest_path / f"{scene_name}.yaml"
        write_scene_manifest(manifest, out_path, relative_paths=True)
        manifest_files.append(out_path)

    print(f"[Manifests] Salvos com sucesso {len(manifest_files)} manifestos em {manifest_path}")
    print(f"[Manifests] Distribuição por split: {dict(split_counts)}")

    # Verify zero group leakage across splits
    group_to_split: dict[str, str] = {}
    for m_path in manifest_files:
        from rowgraphnet.data.manifest import load_scene_manifest

        m = load_scene_manifest(m_path)
        if m.group_id in group_to_split and group_to_split[m.group_id] != m.split:
            raise ValueError(
                f"VAZAMENTO DETECTADO: Grupo {m.group_id} possui múltiplos splits: {group_to_split[m.group_id]} e {m.split}"
            )
        group_to_split[m.group_id] = m.split
    print(f"[Manifests] Verificados {len(group_to_split)} grupos espaciais: ZERO vazamento entre splits!")

    # Write detailed geometry audit report
    report_file = data_path / "geometry_verification_report.json"
    total_checked = sum(r["total_raw"] for r in audit_reports)
    total_repaired = sum(r["repaired"] for r in audit_reports)
    total_dropped_invalid = sum(r["dropped_invalid"] for r in audit_reports)
    total_dropped_outside_roi = sum(r.get("dropped_outside_roi", 0) for r in audit_reports)
    total_dropped_outside_img = sum(r.get("dropped_outside_image", 0) for r in audit_reports)
    total_dropped_in_discarded_roi = sum(r.get("dropped_in_discarded_roi", 0) for r in audit_reports)
    total_dropped = (
        total_dropped_invalid
        + total_dropped_outside_roi
        + total_dropped_outside_img
        + total_dropped_in_discarded_roi
    )
    total_clean = sum(r["final_clean"] for r in audit_reports)
    total_rois_close_lines = sum(r.get("discarded_close_lines", 0) for r in audit_reports)
    total_rois_repair_loss = sum(r.get("discarded_repair_loss", 0) for r in audit_reports)
    total_rois_discarded = sum(r.get("discarded_quality", 0) for r in audit_reports)

    summary_data = {
        "total_files_audited": len(audit_reports),
        "total_features_checked": total_checked,
        "total_features_repaired": total_repaired,
        "total_features_dropped_invalid": total_dropped_invalid,
        "total_features_dropped_outside_roi": total_dropped_outside_roi,
        "total_features_dropped_outside_image": total_dropped_outside_img,
        "total_features_dropped_in_discarded_roi": total_dropped_in_discarded_roi,
        "total_features_dropped": total_dropped,
        "total_features_clean": total_clean,
        "min_line_spacing_m": MIN_LINE_SPACING_M,
        "total_rois_discarded": total_rois_discarded,
        "total_rois_discarded_close_lines": total_rois_close_lines,
        "total_rois_discarded_repair_loss": total_rois_repair_loss,
        "details": audit_reports,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(
        f"[Relatório Geométrico] Auditoria salva em {report_file.name}: "
        f"{total_checked} geometrias verificadas, {total_repaired} reparadas, "
        f"{total_dropped} descartadas ({total_dropped_invalid} inválidas, "
        f"{total_dropped_outside_roi} fora do ROI, {total_dropped_outside_img} fora da imagem, "
        f"{total_dropped_in_discarded_roi} em ROIs descartadas), "
        f"{total_clean} válidas mantidas. ROIs descartadas: {total_rois_discarded} "
        f"({total_rois_close_lines} por linhas < {MIN_LINE_SPACING_M:.2f} m, "
        f"{total_rois_repair_loss} por geometria apagada no reparo)."
    )

    return manifest_files


if __name__ == "__main__":
    prepare_dataset_manifests()
