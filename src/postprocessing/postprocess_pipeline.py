"""Single, reusable vector post-processing engine.

The inference CLI and the postprocess-only CLI must call this module instead
of maintaining independent copies of the geometry pipeline.  The engine is
deliberately free of file I/O so callers can decide how and where to persist
raw, final, and audit products.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, unary_union

from src.geospatial import (
    clip_predictions_to_roi,
    pixel_to_world,
    sample_line_probabilities,
    world_to_pixel,
)

from .curve_fitting import fit_branches_if_enabled
from .deduplication import _bearing_pca, _longitudinal_overlap, remove_duplicate_lines
from .double_row_refinement import refine_double_rows
from .double_row_validation import validate_double_rows
from .gap_bridging import bridge_gaps
from .line_extension import extend_lines_to_roi
from .lineage import (
    combine_record_source_raw_ids,
    parse_source_raw_ids,
    serialize_source_raw_ids,
    source_raw_ids_from_record,
)
from .network_cleanup import cleanup_network
from .pair_completion import audit_pair_completion
from .spur_removal import prune_spurs
from .regularized_centerline_fairing import (
    annotate_serration_status,
    classify_refinement_source,
    fair_centerlines_gdf,
    is_true,
)
from .continuous_fairing import continuous_fairing_gdf
from .topology_validation import analyze_topology


logger = logging.getLogger(__name__)

POSTPROCESS_VERSION = "1.2.0"


@dataclass(frozen=True)
class PostprocessResult:
    """Complete output contract of :func:`run_postprocessing`."""

    final_lines: gpd.GeoDataFrame
    raw_lines: gpd.GeoDataFrame
    repaired_segments: gpd.GeoDataFrame
    reconstructed_pairs: gpd.GeoDataFrame
    terminal_extensions: gpd.GeoDataFrame
    rejected_lines: gpd.GeoDataFrame
    manual_review: gpd.GeoDataFrame
    metrics: Mapping[str, Any]


def _empty_gdf(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def _concat_gdfs(
    frames: Sequence[gpd.GeoDataFrame], crs
) -> gpd.GeoDataFrame:
    present = [frame for frame in frames if frame is not None and not frame.empty]
    if not present:
        return _empty_gdf(crs)
    return gpd.GeoDataFrame(
        pd.concat(present, ignore_index=True, sort=False),
        geometry="geometry",
        crs=crs,
    )


def _debug_layer(
    debug_gdf: gpd.GeoDataFrame, layer_name: str, crs
) -> gpd.GeoDataFrame:
    if (
        debug_gdf is None
        or debug_gdf.empty
        or "debug_layer" not in debug_gdf.columns
    ):
        return _empty_gdf(crs)
    return debug_gdf.loc[debug_gdf["debug_layer"].eq(layer_name)].copy()


def _normalise_inputs(
    raw_lines_gdf: gpd.GeoDataFrame,
    roi_gdf: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not isinstance(raw_lines_gdf, gpd.GeoDataFrame):
        raise TypeError("raw_lines_gdf must be a GeoDataFrame.")
    if not isinstance(roi_gdf, gpd.GeoDataFrame):
        raise TypeError("roi_gdf must be a GeoDataFrame.")
    if raw_lines_gdf.crs is None:
        raise ValueError("Raw lines have no CRS.")
    if roi_gdf.crs is None:
        raise ValueError("ROI has no CRS.")

    lines = raw_lines_gdf.copy(deep=True).reset_index(drop=True)
    roi = roi_gdf.copy(deep=True)
    if roi.crs != lines.crs:
        roi = roi.to_crs(lines.crs)
    roi = roi.loc[roi.geometry.notna() & ~roi.geometry.is_empty].copy()
    if roi.empty:
        raise ValueError("ROI contains no usable geometry.")
    roi["geometry"] = roi.geometry.make_valid()

    lines = lines.loc[lines.geometry.notna() & ~lines.geometry.is_empty].copy()
    lines = lines.loc[lines.geometry.geom_type.eq("LineString")].reset_index(drop=True)
    lines = _canonicalise_lineage_frame(lines)
    return lines, roi


def _canonicalise_lineage_frame(
    lines_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Normalise lineage attributes without inventing IDs for legacy inputs."""
    result = lines_gdf.copy()
    if "source_raw_ids" not in result.columns and "raw_line_id" not in result.columns:
        return result
    result["source_raw_ids"] = [
        serialize_source_raw_ids(source_raw_ids_from_record(row))
        for _, row in result.iterrows()
    ]
    return result


def _apply_center_mass_guard(config: dict) -> None:
    """Make the shared safety block authoritative for every existing repair."""
    guard = config.get("center_mass_guard", {})
    if not bool(guard.get("enabled", False)):
        return
    fairing = config.setdefault("final_centerline_fairing", {})
    geometry = fairing.setdefault("geometry", {})
    probability = fairing.setdefault("probability", {})
    geometry["max_lateral_shift_m"] = float(
        guard.get("max_lateral_shift_m", geometry.get("max_lateral_shift_m", 0.10))
    )
    geometry["max_hausdorff_m"] = float(
        guard.get("max_hausdorff_m", geometry.get("max_hausdorff_m", 0.12))
    )
    probability["max_median_probability_loss"] = float(
        guard.get(
            "max_median_probability_loss",
            probability.get("max_median_probability_loss", 0.02),
        )
    )
    repair = config.setdefault("double_row_refinement", {}).setdefault(
        "reference_guided_repair", {}
    )
    repair["min_probability_fraction"] = float(
        guard.get(
            "min_supported_fraction",
            repair.get("min_probability_fraction", 0.55),
        )
    )


def _prevector_cleanup(
    raw_lines: gpd.GeoDataFrame,
    probability_center,
    probability_transform,
    config: dict,
) -> gpd.GeoDataFrame:
    """Run former pixel cleanup after the persisted post-inference boundary."""
    cleanup_cfg = config.get("postprocess_pipeline", {}).get(
        "prevector_cleanup", {}
    )
    source_stage = raw_lines.get(
        "source_stage", pd.Series("", index=raw_lines.index)
    )
    if (
        raw_lines.empty
        or probability_center is None
        or probability_transform is None
        or not bool(cleanup_cfg.get("enabled", True))
        or not source_stage.eq("post_inference").all()
    ):
        return raw_lines.copy()

    pixel_branches = [
        world_to_pixel(list(geometry.coords), probability_transform)
        for geometry in raw_lines.geometry
    ]
    endpoint_degree: Dict[Tuple[int, int], int] = {}
    for branch in pixel_branches:
        if len(branch) < 2:
            continue
        for node in (tuple(branch[0]), tuple(branch[-1])):
            endpoint_degree[node] = endpoint_degree.get(node, 0) + 1
    junctions = [node for node, degree in endpoint_degree.items() if degree > 2]
    endpoints = [node for node, degree in endpoint_degree.items() if degree == 1]
    gsd = float(
        np.sqrt(
            abs(
                probability_transform.a * probability_transform.e
                - probability_transform.b * probability_transform.d
            )
        )
    )
    pruned = prune_spurs(pixel_branches, junctions, endpoints, gsd, config)
    bridged = bridge_gaps(
        pruned,
        probability_center,
        probability_transform,
        gsd,
        config,
    )
    cleaned = cleanup_network(bridged, config, gsd=gsd)
    world_arrays = [
        np.asarray(pixel_to_world(branch, probability_transform), dtype=np.float64)
        for branch in cleaned
        if len(branch) >= 2
    ]
    fitted, fitting_attributes = fit_branches_if_enabled(
        world_arrays, config, return_metadata=True
    )
    candidates = [
        (geometry, attributes)
        for geometry, attributes in (
            (LineString(points), attributes)
            for points, attributes in zip(fitted, fitting_attributes)
            if len(points) >= 2
        )
        if not geometry.is_empty
    ]
    if not candidates:
        return raw_lines.iloc[0:0].copy()

    raw_geometries = list(raw_lines.geometry)
    raw_tree = STRtree(raw_geometries)
    lineage_tolerance = max(gsd * 1.5, 0.01)
    records = []
    for geometry, fitting_meta in candidates:
        source_indices = sorted(
            int(index)
            for index in raw_tree.query(
                geometry.buffer(lineage_tolerance), predicate="intersects"
            )
            if raw_geometries[int(index)].distance(geometry) <= lineage_tolerance
        )
        if not source_indices:
            nearest = raw_tree.query_nearest(geometry)
            nearest_array = np.atleast_1d(nearest)
            source_indices = [int(nearest_array[0])] if len(nearest_array) else []
        primary = source_indices[0] if source_indices else 0
        record = raw_lines.iloc[primary].to_dict()
        lineage = combine_record_source_raw_ids(
            raw_lines.iloc[index] for index in source_indices
        )
        if parse_source_raw_ids(lineage):
            record["source_raw_ids"] = lineage
        record.update(fitting_meta)
        record["geometry"] = geometry
        records.append(record)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=raw_lines.crs).reset_index(
        drop=True
    )


def _deduplication_match_score(
    candidate: LineString,
    retained: LineString,
    config: dict,
) -> Optional[float]:
    """Return the same mean-distance score used by deduplication, if valid."""
    dedup_cfg = config.get("deduplication", {})
    max_mean_distance = float(dedup_cfg.get("max_mean_distance_m", 0.20))
    max_angle = float(dedup_cfg.get("max_angle_difference_deg", 5.0))
    min_overlap = float(dedup_cfg.get("min_overlap_ratio", 0.70))
    coordinates = np.asarray(candidate.coords, dtype=np.float64)
    if not len(coordinates):
        return None
    sample_indices = np.linspace(
        0, len(coordinates) - 1, min(20, len(coordinates)), dtype=int
    )
    mean_distance = float(
        np.mean([Point(coordinates[index]).distance(retained) for index in sample_indices])
    )
    if mean_distance >= max_mean_distance:
        return None
    angle_difference = abs(_bearing_pca(candidate) - _bearing_pca(retained))
    angle_difference = min(angle_difference, 180.0 - angle_difference)
    if angle_difference > max_angle:
        return None
    shorter, longer = (
        (candidate, retained)
        if candidate.length <= retained.length
        else (retained, candidate)
    )
    if _longitudinal_overlap(shorter, longer) < min_overlap:
        return None
    return mean_distance


def _tree_query_indices(tree: STRtree, geometry, geometries) -> Sequence[int]:
    """Normalise STRtree results from Shapely 1.x and 2.x."""
    try:
        candidates = tree.query(geometry, predicate="intersects")
    except TypeError:  # pragma: no cover - compatibility with Shapely 1.x
        candidates = tree.query(geometry)
    by_identity = {id(item): index for index, item in enumerate(geometries)}
    indices = []
    for candidate in candidates:
        if isinstance(candidate, (int, np.integer)):
            indices.append(int(candidate))
        else:  # pragma: no cover - compatibility with Shapely 1.x
            index = by_identity.get(id(candidate))
            if index is not None:
                indices.append(index)
    return indices


def _deduplicate_gdf(lines_gdf: gpd.GeoDataFrame, config: dict) -> gpd.GeoDataFrame:
    if lines_gdf.empty:
        return lines_gdf.copy()
    working = _canonicalise_lineage_frame(lines_gdf)
    if not bool(config.get("deduplication", {}).get("enabled", True)):
        return working.reset_index(drop=True)
    geometries = list(working.geometry)
    deduplicated = remove_duplicate_lines(geometries, config)
    # Preserve the established inference semantics: on an exact-WKB duplicate,
    # the last matching attribute row is retained.
    source_by_wkb = {geometry.wkb: index for index, geometry in enumerate(geometries)}
    indices = [source_by_wkb[geometry.wkb] for geometry in deduplicated]
    result = working.iloc[indices].copy().reset_index(drop=True)
    result.geometry = deduplicated

    if "source_raw_ids" not in working.columns and "raw_line_id" not in working.columns:
        return result

    retained_by_identity = {
        id(geometry): index for index, geometry in enumerate(deduplicated)
    }
    retained_by_wkb: Dict[bytes, list[int]] = {}
    for index, geometry in enumerate(deduplicated):
        retained_by_wkb.setdefault(geometry.wkb, []).append(index)
    groups: list[list[Mapping[str, Any]]] = [[] for _ in deduplicated]
    retained_tree = STRtree(deduplicated) if deduplicated else None
    max_distance = float(
        config.get("deduplication", {}).get("max_mean_distance_m", 0.20)
    )
    max_candidates = int(
        config.get("deduplication", {}).get("max_candidates_per_line", 30)
    )

    for source_index, geometry in enumerate(geometries):
        retained_index = retained_by_identity.get(id(geometry))
        if retained_index is None:
            exact = retained_by_wkb.get(geometry.wkb, [])
            if exact:
                retained_index = exact[0]
        if retained_index is None and retained_tree is not None:
            candidates = _tree_query_indices(
                retained_tree,
                geometry.buffer(max_distance),
                deduplicated,
            )
            candidates = sorted(
                set(candidates), key=lambda index: geometry.distance(deduplicated[index])
            )[:max_candidates]
            scored = [
                (score, index)
                for index in candidates
                for score in [_deduplication_match_score(geometry, deduplicated[index], config)]
                if score is not None
            ]
            if scored:
                retained_index = min(scored)[1]
        if retained_index is not None:
            groups[retained_index].append(working.iloc[source_index])

    for retained_index, records in enumerate(groups):
        lineage = combine_record_source_raw_ids(records)
        if parse_source_raw_ids(lineage):
            result.at[retained_index, "source_raw_ids"] = lineage
    return result


def _merge_tracks_safely(
    lines_gdf: gpd.GeoDataFrame, config: dict
) -> gpd.GeoDataFrame:
    """Merge only track fragments whose union is already one safe LineString.

    Disconnected components are preserved for review.  In particular, this
    function never dissolves by pair_id and never inserts an unsupported fixed
    offset or connector.
    """
    merge_cfg = config.get("postprocess_pipeline", {}).get("track_merge", {})
    if (
        lines_gdf.empty
        or "track_id" not in lines_gdf.columns
        or not bool(merge_cfg.get("enabled", True))
    ):
        return lines_gdf.copy()

    rows = []
    consumed = set()
    for index, row in lines_gdf.iterrows():
        if index in consumed:
            continue
        track_id = row.get("track_id")
        if pd.isna(track_id):
            rows.append(row.to_dict())
            consumed.add(index)
            continue
        group_indices = [
            int(candidate)
            for candidate in lines_gdf.index[lines_gdf["track_id"].eq(track_id)]
        ]
        consumed.update(group_indices)
        if len(group_indices) == 1:
            rows.append(row.to_dict())
            continue

        group = lines_gdf.loc[group_indices]
        pair_values = group.get("pair_id", pd.Series(dtype="object")).dropna().unique()
        if len(pair_values) > 1:
            rows.extend(group.loc[candidate].to_dict() for candidate in group_indices)
            continue
        united = unary_union(list(group.geometry))
        merged = united if united.geom_type == "LineString" else linemerge(united)
        if (
            merged.geom_type != "LineString"
            or merged.is_empty
            or not merged.is_valid
            or not merged.is_simple
        ):
            rows.extend(group.loc[candidate].to_dict() for candidate in group_indices)
            continue

        primary_index = max(group_indices, key=lambda candidate: group.loc[candidate].geometry.length)
        record = group.loc[primary_index].to_dict()
        lineage = combine_record_source_raw_ids(
            member for _, member in group.iterrows()
        )
        if parse_source_raw_ids(lineage):
            record["source_raw_ids"] = lineage
        record["geometry"] = merged
        record["track_merged"] = True
        record["track_fragment_count"] = len(group_indices)
        rows.append(record)

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=lines_gdf.crs).reset_index(
        drop=True
    )


def _split_topology(
    lines_gdf: gpd.GeoDataFrame, config: dict
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Annotate topology once and retain rejected rows for audit."""
    if lines_gdf.empty:
        return lines_gdf.copy(), _empty_gdf(lines_gdf.crs)
    records = analyze_topology(list(lines_gdf.geometry), config)
    minimum_length = float(config.get("vector", {}).get("min_line_length_m", 3.0))
    accepted_indices = []
    rejected_records = []
    accepted_records = []
    for index, (geometry, topology) in enumerate(zip(lines_gdf.geometry, records)):
        reasons = [value for value in str(topology.get("rejection_reason", "")).split(";") if value]
        if geometry.length < minimum_length:
            reasons.append("below_minimum_length")
        record = dict(topology)
        record["rejection_reason"] = ";".join(dict.fromkeys(reasons))
        record["topology_status"] = "rejected" if reasons else "valid"
        if reasons:
            rejected = lines_gdf.iloc[index].to_dict()
            rejected.update(record)
            rejected["postprocess_rejection_stage"] = "topology"
            rejected_records.append(rejected)
        else:
            accepted_indices.append(index)
            accepted_records.append(record)

    accepted = lines_gdf.iloc[accepted_indices].copy().reset_index(drop=True)
    for column in accepted_records[0].keys() if accepted_records else []:
        accepted[column] = [record[column] for record in accepted_records]
    rejected = (
        gpd.GeoDataFrame(rejected_records, geometry="geometry", crs=lines_gdf.crs)
        if rejected_records
        else _empty_gdf(lines_gdf.crs)
    )
    logger.info(
        "Topology validation: kept %d valid lines from %d candidates.",
        len(accepted),
        len(lines_gdf),
    )
    return accepted, rejected


def _boolean_column(frame: gpd.GeoDataFrame, names: Sequence[str]) -> pd.Series:
    values = pd.Series(False, index=frame.index, dtype=bool)
    for name in names:
        if name in frame.columns:
            values |= frame[name].map(is_true).fillna(False).astype(bool)
    return values


def _positive_column(frame: gpd.GeoDataFrame, names: Sequence[str]) -> pd.Series:
    values = pd.Series(False, index=frame.index, dtype=bool)
    for name in names:
        if name in frame.columns:
            numeric = pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
            values |= numeric.gt(0.0)
    return values


def _add_provenance(
    lines_gdf: gpd.GeoDataFrame,
    config: dict,
) -> gpd.GeoDataFrame:
    result = lines_gdf.copy()
    version = str(
        config.get("postprocess_pipeline", {}).get(
            "version", POSTPROCESS_VERSION
        )
    )
    result["postprocess_version"] = version
    result["was_merged"] = _boolean_column(result, ("track_merged",))
    result["was_repaired"] = _boolean_column(
        result,
        ("reference_repair", "kink_repair", "splice_repaired"),
    )
    result["was_reconstructed"] = _boolean_column(
        result,
        ("was_reconstructed", "pair_reconstructed", "reconstructed_pair"),
    )
    result["was_terminal_extended"] = _positive_column(
        result,
        ("terminal_extended_m", "fairing_terminal_extended_m"),
    )
    if "refinement_source" in result.columns:
        result["source_type"] = result["refinement_source"].fillna("original")
    else:
        result["source_type"] = result.apply(classify_refinement_source, axis=1)
    result.loc[result["was_reconstructed"], "source_type"] = "reconstructed"
    return result


def _review_mask(lines_gdf: gpd.GeoDataFrame) -> pd.Series:
    mask = pd.Series(False, index=lines_gdf.index, dtype=bool)
    if "review_required" in lines_gdf.columns:
        mask |= lines_gdf["review_required"].map(is_true).fillna(False).astype(bool)
    if "pair_status" in lines_gdf.columns:
        mask |= ~lines_gdf["pair_status"].eq("valid_pair")
    if "serration_status" in lines_gdf.columns:
        mask |= lines_gdf["serration_status"].eq("failed")
    return mask


def _already_processed(raw_lines: gpd.GeoDataFrame, config: dict) -> bool:
    if raw_lines.empty or "postprocess_version" not in raw_lines.columns:
        return False
    expected = str(
        config.get("postprocess_pipeline", {}).get(
            "version", POSTPROCESS_VERSION
        )
    )
    return bool(raw_lines["postprocess_version"].astype(str).eq(expected).all())


def _result_without_processing(
    raw_lines: gpd.GeoDataFrame,
    *,
    status: str,
) -> PostprocessResult:
    final = raw_lines.copy(deep=True)
    empty = _empty_gdf(raw_lines.crs)
    manual = (
        final.loc[_review_mask(final)].copy()
        if not final.empty
        else _empty_gdf(raw_lines.crs)
    )
    return PostprocessResult(
        final_lines=final,
        raw_lines=raw_lines.copy(deep=True),
        repaired_segments=empty.copy(),
        reconstructed_pairs=empty.copy(),
        terminal_extensions=empty.copy(),
        rejected_lines=empty.copy(),
        manual_review=manual,
        metrics={
            "status": status,
            "input_line_count": len(raw_lines),
            "final_line_count": len(final),
            "production_gate_passed": status == "already_processed"
            and manual.empty,
        },
    )


def run_postprocessing(
    raw_lines_gdf: gpd.GeoDataFrame,
    probability_center,
    probability_transform,
    roi_gdf: gpd.GeoDataFrame,
    config: dict,
    debug: bool = False,
) -> PostprocessResult:
    """Run the complete vector post-processing pipeline exactly once."""
    raw_snapshot = raw_lines_gdf.copy(deep=True)
    pipeline_cfg = config.get("postprocess_pipeline", {})
    if not bool(pipeline_cfg.get("enabled", True)):
        return _result_without_processing(raw_snapshot, status="disabled")
    if _already_processed(raw_snapshot, config):
        return _result_without_processing(raw_snapshot, status="already_processed")

    lines, roi = _normalise_inputs(raw_snapshot, roi_gdf)
    roi_polygon = roi.geometry.union_all()
    working_config = deepcopy(config)
    _apply_center_mass_guard(working_config)

    lines = _prevector_cleanup(
        lines,
        probability_center,
        probability_transform,
        working_config,
    )
    lines = _deduplicate_gdf(lines, working_config)
    lines = validate_double_rows(lines, working_config, preliminary=True)

    # The final C1 fairing terminal is the sole owner when enabled.  Disable
    # the older refinement terminal only in this private config copy.
    final_terminal_enabled = bool(
        working_config.get("final_centerline_fairing", {})
        .get("terminal", {})
        .get("enabled", True)
    ) and bool(working_config.get("final_centerline_fairing", {}).get("enabled", False))
    if final_terminal_enabled:
        working_config.setdefault("double_row_refinement", {}).setdefault(
            "terminal_extension", {}
        )["enabled"] = False

    refined, refinement_debug = refine_double_rows(
        lines,
        roi_polygon,
        working_config,
        probability_raster=probability_center,
        transform=probability_transform,
        collect_debug=debug,
    )
    refined = _merge_tracks_safely(refined, working_config)
    fair_cfg = working_config.get("final_centerline_fairing", {})
    fair_engine = str(fair_cfg.get("engine", "legacy")).lower()
    if fair_engine == "continuous":
        faired = continuous_fairing_gdf(
            refined,
            config=working_config,
            enable_track_bridge=True,
        )
        fairing_debug = _empty_gdf(refined.crs)
    else:
        faired, fairing_debug = fair_centerlines_gdf(
            refined,
            roi_polygon,
            working_config,
            probability_raster=probability_center,
            transform=probability_transform,
            collect_debug=debug,
        )

    refinement_terminal_enabled = bool(
        working_config.get("double_row_refinement", {})
        .get("terminal_extension", {})
        .get("enabled", True)
    )
    use_legacy_extension = bool(
        working_config.get("line_extension", {}).get("enabled", False)
        and not refinement_terminal_enabled
        and not final_terminal_enabled
    )
    if use_legacy_extension:
        extended = extend_lines_to_roi(
            list(faired.geometry),
            roi_polygon,
            working_config,
            probability_raster=probability_center,
            transform=probability_transform,
        )
        faired = faired.copy()
        faired.geometry = extended

    clipped = clip_predictions_to_roi(faired, roi)
    topologically_valid, topology_rejected = _split_topology(
        clipped, working_config
    )
    final = validate_double_rows(topologically_valid, working_config)
    pair_completion = audit_pair_completion(final, working_config)
    final = pair_completion.lines
    if probability_center is not None and probability_transform is not None:
        spacing = float(
            working_config.get("metadata", {}).get(
                "probability_sample_spacing_m", 0.10
            )
        )
        final = sample_line_probabilities(
            final,
            probability_center,
            probability_transform,
            spacing_m=spacing,
        )
    final = _add_provenance(final, working_config)
    # This audit intentionally runs after every geometry-changing step.  Lines
    # that could not be repaired remain available in final_lines for diagnosis,
    # but are forced to manual review and cannot reach the production layers.
    final = annotate_serration_status(
        final, working_config, probability_transform
    )

    review_mask = _review_mask(final)
    final["postprocess_status"] = np.where(
        review_mask, "manual_review", "approved"
    )
    manual_review = final.loc[review_mask].copy()

    repaired_debug = _debug_layer(
        refinement_debug, "trechos_substituidos_pela_irma", final.crs
    )
    repaired_rows = final.loc[final["was_repaired"]].copy()
    repaired_segments = (
        repaired_debug if not repaired_debug.empty else repaired_rows
    )
    reconstructed_pairs = _concat_gdfs(
        (
            pair_completion.reconstructed_pairs,
            final.loc[final["was_reconstructed"]].copy(),
        ),
        final.crs,
    )
    terminal_extensions = final.loc[final["was_terminal_extended"]].copy()
    refinement_rejected = _debug_layer(
        refinement_debug, "linhas_rejeitadas", final.crs
    )
    rejected_lines = _concat_gdfs(
        (refinement_rejected, topology_rejected), final.crs
    )

    pair_status = (
        final["pair_status"]
        if "pair_status" in final.columns
        else pd.Series("unmatched", index=final.index)
    )
    unmatched_count = int((~pair_status.eq("valid_pair")).sum())
    final_crossing_count = int(
        final.get("has_crossing", pd.Series(False, index=final.index))
        .map(is_true)
        .fillna(False)
        .sum()
    )
    rejected_crossing_count = int(
        rejected_lines.get(
            "has_crossing", pd.Series(False, index=rejected_lines.index)
        )
        .map(is_true)
        .fillna(False)
        .sum()
    )
    crossing_count = final_crossing_count + rejected_crossing_count
    serration_failed_count = int(final["serration_status"].eq("failed").sum())
    production_line_count = int(
        ((~review_mask) & final["serration_status"].eq("clean")).sum()
    )
    gate_reasons = []
    if bool(pipeline_cfg.get("fail_on_unmatched", True)) and unmatched_count:
        gate_reasons.append("unmatched_lines")
    if bool(pipeline_cfg.get("fail_on_crossing", True)) and crossing_count:
        gate_reasons.append("crossings")
    if bool(pipeline_cfg.get("fail_on_serration", True)) and serration_failed_count:
        gate_reasons.append("residual_serration")
    production_gate_passed = not gate_reasons
    metrics: Dict[str, Any] = {
        "status": "completed" if production_gate_passed else "manual_review",
        "postprocess_version": str(
            pipeline_cfg.get("version", POSTPROCESS_VERSION)
        ),
        "input_line_count": len(raw_snapshot),
        "deduplicated_line_count": len(lines),
        "final_line_count": len(final),
        "repaired_count": int(final["was_repaired"].sum()),
        "reconstructed_count": int(final["was_reconstructed"].sum()),
        "terminal_extended_count": int(final["was_terminal_extended"].sum()),
        "rejected_count": len(rejected_lines),
        "manual_review_count": len(manual_review),
        "unmatched_count": unmatched_count,
        "crossing_count": crossing_count,
        "serration_failed_count": serration_failed_count,
        "production_line_count": production_line_count,
        "final_crossing_count": final_crossing_count,
        "rejected_crossing_count": rejected_crossing_count,
        "production_gate_passed": production_gate_passed,
        "production_gate_reasons": tuple(gate_reasons),
        **dict(pair_completion.metrics),
    }
    return PostprocessResult(
        final_lines=final,
        raw_lines=raw_snapshot,
        repaired_segments=repaired_segments,
        reconstructed_pairs=reconstructed_pairs,
        terminal_extensions=terminal_extensions,
        rejected_lines=rejected_lines,
        manual_review=manual_review,
        metrics=metrics,
    )


__all__ = ["POSTPROCESS_VERSION", "PostprocessResult", "run_postprocessing"]
