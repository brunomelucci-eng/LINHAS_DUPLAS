"""Persistence and lineage helpers for the two mandatory vector products."""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

from src.geospatial import clip_predictions_to_roi, sample_line_probabilities

from .lineage import parse_source_raw_ids, serialize_source_raw_ids
from .postprocess_pipeline import PostprocessResult
from .regularized_centerline_fairing import is_true


@dataclass(frozen=True)
class OutputLayout:
    run_dir: Path
    vectors_dir: Path
    rasters_dir: Path
    reports_dir: Path
    post_inference_path: Path
    postprocessing_path: Path
    probability_center_path: Path
    postprocess_report_path: Path
    comparison_report_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class PersistedTwoStageResult:
    raw_lines: gpd.GeoDataFrame
    postprocess_result: PostprocessResult
    final_lines: gpd.GeoDataFrame


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise_base_name(value: str, config: dict) -> str:
    name = Path(value).stem
    naming = config.get("outputs", {}).get("naming", {})
    for suffix in (
        str(naming.get("post_inference_suffix", "__pos_inferencia")),
        str(naming.get("postprocessing_suffix", "__pos_processamento")),
    ):
        if suffix and name.endswith(suffix):
            name = name[: -len(suffix)]
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in name)
    return safe.strip("_") or "resultado"


def create_output_layout(
    run_dir: str | os.PathLike[str],
    base_name: str,
    config: dict,
) -> OutputLayout:
    outputs_cfg = config.get("outputs", {})
    if outputs_cfg.get("always_save_post_inference", True) is not True:
        raise ValueError("The mandatory post-inference product cannot be disabled.")
    if outputs_cfg.get("always_save_postprocessing", True) is not True:
        raise ValueError("The mandatory post-processing product cannot be disabled.")
    if outputs_cfg.get("save_lineage", True) is not True:
        raise ValueError("Lineage cannot be disabled for the mandatory final product.")
    if outputs_cfg.get("save_comparison_report", True) is not True:
        raise ValueError("The mandatory comparison report cannot be disabled.")
    run_path = Path(run_dir)
    vectors = run_path / "vectors"
    rasters = run_path / "rasters"
    reports = run_path / "reports"
    for directory in (vectors, rasters, reports):
        directory.mkdir(parents=True, exist_ok=True)
    stem = _normalise_base_name(base_name, config)
    naming = outputs_cfg.get("naming", {})
    raw_suffix = str(naming.get("post_inference_suffix", "__pos_inferencia"))
    final_suffix = str(naming.get("postprocessing_suffix", "__pos_processamento"))
    return OutputLayout(
        run_dir=run_path,
        vectors_dir=vectors,
        rasters_dir=rasters,
        reports_dir=reports,
        post_inference_path=vectors / f"{stem}{raw_suffix}.gpkg",
        postprocessing_path=vectors / f"{stem}{final_suffix}.gpkg",
        probability_center_path=rasters / "probability_center.tif",
        postprocess_report_path=reports / "postprocess_report.json",
        comparison_report_path=reports / "comparison_report.html",
        manifest_path=run_path / "manifest.json",
    )


def _geometry_status(geometry) -> str:
    if geometry is None or geometry.is_empty:
        return "empty"
    if not geometry.is_valid:
        return "invalid"
    if not geometry.is_simple:
        return "valid_non_simple"
    return "valid_simple"


def prepare_post_inference_lines(
    raw_gdf: gpd.GeoDataFrame,
    probability_center,
    probability_transform,
    roi_gdf: gpd.GeoDataFrame,
    config: dict,
    *,
    run_id: str,
    checkpoint: str,
    created_at: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Apply only the non-destructive operations allowed on the raw product."""
    if raw_gdf.crs is None:
        raise ValueError("Post-inference vectors have no CRS.")
    roi = roi_gdf.to_crs(raw_gdf.crs) if roi_gdf.crs != raw_gdf.crs else roi_gdf
    lines = raw_gdf.copy(deep=True)
    lines = lines.loc[lines.geometry.notna() & ~lines.geometry.is_empty].copy()
    lines = clip_predictions_to_roi(lines, roi).reset_index(drop=True)
    lines = lines.loc[
        lines.geometry.notna()
        & ~lines.geometry.is_empty
        & lines.geometry.geom_type.eq("LineString")
    ].reset_index(drop=True)
    if probability_center is not None and probability_transform is not None:
        spacing = float(
            config.get("metadata", {}).get("probability_sample_spacing_m", 0.10)
        )
        lines = sample_line_probabilities(
            lines,
            probability_center,
            probability_transform,
            spacing_m=spacing,
        )
    timestamp = created_at or utc_now_iso()
    lines["run_id"] = str(run_id)
    if "raw_line_id" not in lines.columns:
        lines["raw_line_id"] = [f"raw_{index:06d}" for index in range(len(lines))]
    else:
        identifiers = lines["raw_line_id"].astype("string")
        if identifiers.isna().any() or identifiers.str.strip().eq("").any():
            raise ValueError("Existing raw_line_id values must be non-empty.")
        lines["raw_line_id"] = identifiers.astype(str)
    if lines["raw_line_id"].duplicated().any():
        raise ValueError("raw_line_id values must be unique.")
    lines["source_raw_ids"] = lines["raw_line_id"].map(
        serialize_source_raw_ids
    )
    lines["source_stage"] = "post_inference"
    lines["checkpoint"] = str(checkpoint)
    lines["config_hash"] = config_hash(config)
    lines["length_m"] = lines.geometry.length.astype(float)
    if "mean_probability" not in lines.columns:
        lines["mean_probability"] = np.nan
    if "median_probability" not in lines.columns:
        lines["median_probability"] = np.nan
    lines["geometry_status"] = lines.geometry.map(_geometry_status)
    lines["created_at"] = timestamp
    return lines


def _source_ids(value: Any) -> Sequence[str]:
    return parse_source_raw_ids(value)


def _serialise_source_ids(value: Any) -> str:
    return serialize_source_raw_ids(value)


def _assign_pair_sides(lines: gpd.GeoDataFrame) -> pd.Series:
    sides = pd.Series("unpaired", index=lines.index, dtype="object")
    if "pair_id" not in lines.columns:
        return sides
    for _, group in lines.dropna(subset=["pair_id"]).groupby("pair_id", sort=False):
        if len(group) != 2:
            sides.loc[group.index] = "unknown"
            continue
        first_index, second_index = list(group.index)
        first, second = lines.geometry.loc[first_index], lines.geometry.loc[second_index]
        coordinates = np.asarray(first.coords, dtype=np.float64)[:, :2]
        direction = coordinates[-1] - coordinates[0]
        if np.linalg.norm(direction) <= 1e-9:
            sides.loc[[first_index, second_index]] = ["unknown", "unknown"]
            continue
        if direction[0] < 0 or (abs(direction[0]) <= 1e-9 and direction[1] < 0):
            direction = -direction
        delta = np.asarray(second.centroid.coords[0])[:2] - np.asarray(first.centroid.coords[0])[:2]
        cross = float(direction[0] * delta[1] - direction[1] * delta[0])
        if abs(cross) <= 1e-9:
            sides.loc[[first_index, second_index]] = ["unknown", "unknown"]
        elif cross > 0:
            sides.loc[[first_index, second_index]] = ["right", "left"]
        else:
            sides.loc[[first_index, second_index]] = ["left", "right"]
    return sides


def prepare_final_lines(
    lines_gdf: gpd.GeoDataFrame,
    *,
    run_id: str,
    created_at: Optional[str] = None,
    valid_raw_line_ids: Optional[Sequence[str]] = None,
) -> gpd.GeoDataFrame:
    final = lines_gdf.copy().reset_index(drop=True)
    timestamp = created_at or utc_now_iso()
    final["run_id"] = str(run_id)
    final["final_line_id"] = [f"final_{index:06d}" for index in range(len(final))]
    if "track_id" not in final.columns:
        final["track_id"] = pd.Series(pd.NA, index=final.index, dtype="Int64")
    if "pair_id" not in final.columns:
        final["pair_id"] = pd.Series(pd.NA, index=final.index, dtype="Int64")
    final["pair_side"] = _assign_pair_sides(final)
    if "source_raw_ids" not in final.columns:
        final["source_raw_ids"] = final.get(
            "raw_line_id", pd.Series("", index=final.index)
        )
    final["source_raw_ids"] = final["source_raw_ids"].map(_serialise_source_ids)
    if len(final) and final["source_raw_ids"].eq("[]").any():
        raise ValueError("Every final line must reference at least one raw_line_id.")
    if valid_raw_line_ids is not None:
        allowed = {str(value) for value in valid_raw_line_ids}
        unknown = sorted(
            {
                identifier
                for value in final["source_raw_ids"]
                for identifier in _source_ids(value)
                if identifier not in allowed
            }
        )
        if unknown:
            raise ValueError(
                "Final source_raw_ids are absent from the post-inference product: "
                + ", ".join(unknown)
            )
    if "source_type" not in final.columns:
        final["source_type"] = "original"
    final["was_merged"] = (
        final.get("track_merged", pd.Series(False, index=final.index))
        .map(is_true)
        .fillna(False)
        .astype(bool)
    )
    for column in ("was_repaired", "was_reconstructed", "was_terminal_extended"):
        if column not in final.columns:
            final[column] = False
        final[column] = final[column].map(is_true).fillna(False).astype(bool)
    smoothed = pd.Series(False, index=final.index, dtype=bool)
    for column in ("smoothed", "fairing_applied"):
        if column in final.columns:
            smoothed |= final[column].map(is_true).fillna(False).astype(bool)
    final["was_smoothed"] = smoothed
    if "mean_probability" not in final.columns:
        final["mean_probability"] = np.nan
    if "median_probability" not in final.columns:
        final["median_probability"] = np.nan
    had_pair_status = "pair_status" in final.columns
    if not had_pair_status:
        final["pair_status"] = "unmatched"
    if "topology_status" not in final.columns:
        final["topology_status"] = "valid"
    if "review_required" not in final.columns:
        final["review_required"] = (
            final["pair_status"].ne("valid_pair") if had_pair_status else False
        )
    final["review_required"] = (
        final["review_required"].map(is_true).fillna(False).astype(bool)
    )
    # An absent pair_status means that double-row validation was deliberately
    # disabled upstream.  Do not turn that mode into a false manual-review
    # failure while preparing persistence metadata.  When pair annotations are
    # present, the strict valid-pair requirement remains authoritative.
    if had_pair_status:
        final["review_required"] |= final["pair_status"].ne("valid_pair")
    final["postprocess_status"] = np.where(
        final["review_required"], "manual_review", "approved"
    )
    if "postprocess_version" not in final.columns:
        final["postprocess_version"] = "unknown"
    final["source_stage"] = "postprocessing"
    final["created_at"] = timestamp
    return final


_RAW_REQUIRED_COLUMNS = (
    "run_id",
    "raw_line_id",
    "source_stage",
    "checkpoint",
    "config_hash",
    "length_m",
    "mean_probability",
    "median_probability",
    "geometry_status",
    "created_at",
)

_FINAL_REQUIRED_COLUMNS = (
    "run_id",
    "final_line_id",
    "track_id",
    "pair_id",
    "pair_side",
    "source_raw_ids",
    "source_type",
    "was_merged",
    "was_repaired",
    "was_reconstructed",
    "was_smoothed",
    "was_terminal_extended",
    "mean_probability",
    "median_probability",
    "pair_status",
    "topology_status",
    "review_required",
    "postprocess_status",
    "postprocess_version",
    "created_at",
)


def validate_vector_contract(
    frame: gpd.GeoDataFrame,
    *,
    stage: str,
    valid_raw_line_ids: Optional[Sequence[str]] = None,
) -> None:
    """Fail before persistence when a mandatory product is not auditable."""
    if frame.crs is None:
        raise ValueError(f"{stage} product has no CRS.")
    required = _RAW_REQUIRED_COLUMNS if stage == "post_inference" else _FINAL_REQUIRED_COLUMNS
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{stage} product is missing required columns: {missing}")
    id_column = "raw_line_id" if stage == "post_inference" else "final_line_id"
    identifiers = frame[id_column].astype("string")
    if identifiers.isna().any() or identifiers.str.strip().eq("").any():
        raise ValueError(f"{id_column} values must be non-empty.")
    if identifiers.duplicated().any():
        raise ValueError(f"{id_column} values must be unique.")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any():
        raise ValueError(f"{stage} product contains empty geometries.")
    if stage == "post_inference":
        if not frame["source_stage"].eq("post_inference").all():
            raise ValueError("Raw source_stage must be post_inference.")
        return
    if valid_raw_line_ids is not None:
        allowed = {str(value) for value in valid_raw_line_ids}
        unknown = {
            identifier
            for value in frame["source_raw_ids"]
            for identifier in _source_ids(value)
            if identifier not in allowed
        }
        if unknown:
            raise ValueError(f"Unknown source_raw_ids: {sorted(unknown)}")


def _prepare_target(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.stem}.partial-{uuid.uuid4().hex}{path.suffix}")


def relative_run_path(layout: OutputLayout, path: str | os.PathLike[str]) -> str:
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(layout.run_dir.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def _atomic_text(path: Path, content: str, *, overwrite: bool) -> None:
    _prepare_target(path, overwrite)
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_post_inference_product(
    lines_gdf: gpd.GeoDataFrame,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    export_epsg: Optional[int] = None,
) -> Path:
    target = Path(path)
    _prepare_target(target, overwrite)
    frame = lines_gdf.to_crs(export_epsg) if export_epsg and lines_gdf.crs else lines_gdf
    validate_vector_contract(frame, stage="post_inference")
    temporary = _temporary_sibling(target)
    try:
        frame.to_file(temporary, layer="post_inference_lines", driver="GPKG")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def select_production_lines(lines_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return only operationally approved, review-free, non-serrated lines."""
    review_required = (
        lines_gdf.get("review_required", pd.Series(False, index=lines_gdf.index))
        .map(is_true)
        .fillna(False)
        .astype(bool)
    )
    status = lines_gdf.get(
        "postprocess_status", pd.Series("approved", index=lines_gdf.index)
    )
    serration_status = lines_gdf.get(
        "serration_status", pd.Series("clean", index=lines_gdf.index)
    )
    return lines_gdf.loc[
        (~review_required)
        & status.eq("approved")
        & serration_status.eq("clean")
    ].copy()


def write_postprocess_product(
    result: PostprocessResult,
    path: str | os.PathLike[str],
    *,
    final_lines: Optional[gpd.GeoDataFrame] = None,
    overwrite: bool = False,
    export_epsg: Optional[int] = None,
) -> Path:
    target = Path(path)
    _prepare_target(target, overwrite)
    main = final_lines if final_lines is not None else result.final_lines
    validate_vector_contract(main, stage="postprocessing")
    approved = select_production_lines(main)
    production = approved.copy()
    layers = (
        ("final_lines", main),
        ("approved_lines", approved),
        ("production_lines", production),
        ("manual_review", result.manual_review),
        ("rejected_lines", result.rejected_lines),
        ("repaired_segments", result.repaired_segments),
        ("reconstructed_pairs", result.reconstructed_pairs),
        ("terminal_extensions", result.terminal_extensions),
    )
    temporary = _temporary_sibling(target)
    try:
        wrote = False
        for layer_name, frame in layers:
            if frame is None:
                frame = gpd.GeoDataFrame(
                    geometry=gpd.GeoSeries([], crs=main.crs), crs=main.crs
                )
            output = frame.to_crs(export_epsg) if export_epsg and frame.crs else frame
            output.to_file(
                temporary,
                layer=layer_name,
                driver="GPKG",
                mode="a" if wrote else "w",
            )
            wrote = True
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def write_compatibility_vector_product(
    lines_gdf: gpd.GeoDataFrame,
    path: str | os.PathLike[str],
    *,
    layer_name: str = "predicted_rows",
    overwrite: bool = False,
    export_epsg: Optional[int] = None,
) -> Path:
    """Write the legacy single-layer alias without weakening canonical I/O."""
    target = Path(path)
    _prepare_target(target, overwrite)
    production = select_production_lines(lines_gdf)
    frame = production.to_crs(export_epsg) if export_epsg and production.crs else production
    validate_vector_contract(frame, stage="postprocessing")
    temporary = _temporary_sibling(target)
    try:
        frame.to_file(temporary, layer=layer_name, driver="GPKG")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _record_postprocessing_failure(
    layout: OutputLayout,
    *,
    run_id: str,
    raw_lines: gpd.GeoDataFrame,
    checkpoint: str,
    config: dict,
    error: Exception,
    overwrite: bool,
    export_epsg: Optional[int],
) -> None:
    failure_manifest = {
        "run_id": str(run_id),
        "status": "postprocessing_failed",
        "post_inference": {
            "complete": True,
            "path": relative_run_path(layout, layout.post_inference_path),
            "layer": "post_inference_lines",
            "line_count": len(raw_lines),
            "crs": (
                f"EPSG:{int(export_epsg)}"
                if export_epsg is not None
                else str(raw_lines.crs)
            ),
        },
        "postprocessing": {
            "complete": False,
            "path": relative_run_path(layout, layout.postprocessing_path),
            "layer": "final_lines",
        },
        "checkpoint": str(checkpoint),
        "error_type": type(error).__name__,
        "error": str(error),
        "config_hash": config_hash(config),
        "created_at": utc_now_iso(),
    }
    _atomic_text(
        layout.manifest_path,
        json.dumps(failure_manifest, indent=2, ensure_ascii=False),
        overwrite=overwrite,
    )


def persist_two_stage_vector_outputs(
    raw_gdf: gpd.GeoDataFrame,
    probability_center,
    probability_transform,
    roi_gdf: gpd.GeoDataFrame,
    config: dict,
    *,
    layout: OutputLayout,
    run_id: str,
    checkpoint: str,
    debug: bool = False,
    export_epsg: Optional[int] = None,
    postprocess_fn: Optional[Callable[..., PostprocessResult]] = None,
    finalizer: Optional[Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame]] = None,
    timings: Optional[Dict[str, float]] = None,
) -> PersistedTwoStageResult:
    """Persist raw first, then run and persist post-processing.

    If ``postprocess_fn`` raises, the already-written post-inference GeoPackage
    is intentionally retained and no final GeoPackage is created.
    """
    if layout.post_inference_path.resolve() == layout.postprocessing_path.resolve():
        raise ValueError("Post-inference and post-processing paths must differ.")
    overwrite = bool(config.get("outputs", {}).get("overwrite_existing", False))
    # Validate both destinations before creating either product.  In
    # particular, a pre-existing final must not leave a new orphan raw file.
    _prepare_target(layout.post_inference_path, overwrite)
    _prepare_target(layout.postprocessing_path, overwrite)
    created_at = utc_now_iso()
    raw_lines = prepare_post_inference_lines(
        raw_gdf,
        probability_center,
        probability_transform,
        roi_gdf,
        config,
        run_id=run_id,
        checkpoint=checkpoint,
        created_at=created_at,
    )
    validate_vector_contract(raw_lines, stage="post_inference")
    write_post_inference_product(
        raw_lines,
        layout.post_inference_path,
        overwrite=overwrite,
        export_epsg=export_epsg,
    )

    if postprocess_fn is None:
        from .postprocess_pipeline import run_postprocessing

        postprocess_fn = run_postprocessing
    try:
        engine_started = time.perf_counter()
        try:
            result = postprocess_fn(
                raw_lines,
                probability_center,
                probability_transform,
                roi_gdf,
                config,
                debug=debug,
            )
        finally:
            if timings is not None:
                timings["postprocessing"] = time.perf_counter() - engine_started
        final_lines = prepare_final_lines(
            result.final_lines,
            run_id=run_id,
            created_at=created_at,
            valid_raw_line_ids=raw_lines["raw_line_id"].astype(str).tolist(),
        )
        if finalizer is not None:
            final_lines = finalizer(final_lines)
        validate_vector_contract(
            final_lines,
            stage="postprocessing",
            valid_raw_line_ids=raw_lines["raw_line_id"].astype(str).tolist(),
        )
        write_postprocess_product(
            result,
            layout.postprocessing_path,
            final_lines=final_lines,
            overwrite=overwrite,
            export_epsg=export_epsg,
        )
    except Exception as error:
        try:
            _record_postprocessing_failure(
                layout,
                run_id=run_id,
                raw_lines=raw_lines,
                checkpoint=checkpoint,
                config=config,
                error=error,
                overwrite=overwrite,
                export_epsg=export_epsg,
            )
        except Exception as manifest_error:
            logger.error(
                "Could not record post-processing failure in %s: %s",
                layout.manifest_path,
                manifest_error,
            )
        raise
    return PersistedTwoStageResult(raw_lines, result, final_lines)


def _connected_component_count(geometries: Sequence[Any]) -> int:
    if not geometries:
        return 0
    from shapely import STRtree

    tree = STRtree(list(geometries))
    parents = list(range(len(geometries)))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parents[root_second] = root_first

    for index, geometry in enumerate(geometries):
        for candidate in tree.query(geometry, predicate="intersects"):
            union(index, int(candidate))
    return len({find(index) for index in range(len(geometries))})


def _network_node_degrees(geometries: Sequence[Any]) -> Dict[tuple, int]:
    if not geometries:
        return {}
    noded = unary_union(list(geometries))
    if noded.geom_type == "LineString":
        segments = [noded]
    elif hasattr(noded, "geoms"):
        segments = [part for part in noded.geoms if part.geom_type == "LineString"]
    else:
        segments = []
    degrees: Dict[tuple, int] = {}
    for segment in segments:
        coordinates = list(segment.coords)
        if len(coordinates) < 2:
            continue
        for coordinate in (coordinates[0], coordinates[-1]):
            key = (round(float(coordinate[0]), 7), round(float(coordinate[1]), 7))
            degrees[key] = degrees.get(key, 0) + 1
    return degrees


def _network_summary(lines: gpd.GeoDataFrame, config: dict) -> Dict[str, Any]:
    if lines.empty:
        return {
            "line_count": 0,
            "total_length_m": 0.0,
            "fragment_feature_count": 0,
            "connected_component_count": 0,
            "unique_endpoint_count": 0,
            "closed_loop_count": 0,
            "degree_3_node_count": 0,
            "degree_4_plus_node_count": 0,
            "crossing_event_count": 0,
            "unmatched_line_count": 0,
            "valid_pair_count": 0,
            "valid_pair_line_count": 0,
            "reconstructed_line_count": 0,
            "repaired_segment_count": 0,
            "merged_line_count": 0,
            "terminal_extended_line_count": 0,
            "median_of_line_median_probability": None,
        }
    geometries = list(lines.geometry)
    node_degrees = _network_node_degrees(geometries)
    probabilities = pd.to_numeric(
        lines.get("median_probability", pd.Series(dtype=float)), errors="coerce"
    )
    finite = probabilities[np.isfinite(probabilities)]
    statuses = lines.get("pair_status", pd.Series("unmatched", index=lines.index))
    pair_ids = lines.get("pair_id", pd.Series(pd.NA, index=lines.index))
    valid_pair_ids = pair_ids.loc[statuses.eq("valid_pair")].dropna()
    valid_pair_count = int(
        sum(count == 2 for count in valid_pair_ids.value_counts().tolist())
    )
    repaired = _boolean_series(lines, "was_repaired")
    reconstructed = _boolean_series(lines, "was_reconstructed")
    merged = _boolean_series(lines, "was_merged") | _boolean_series(lines, "track_merged")
    terminal = _boolean_series(lines, "was_terminal_extended")
    return {
        "line_count": len(lines),
        "total_length_m": float(lines.geometry.length.sum()),
        "fragment_feature_count": len(lines),
        "connected_component_count": _connected_component_count(geometries),
        "unique_endpoint_count": int(sum(degree == 1 for degree in node_degrees.values())),
        "closed_loop_count": int(sum(bool(geometry.is_ring) for geometry in geometries)),
        "degree_3_node_count": int(sum(degree == 3 for degree in node_degrees.values())),
        "degree_4_plus_node_count": int(sum(degree >= 4 for degree in node_degrees.values())),
        "crossing_event_count": int(sum(degree >= 4 for degree in node_degrees.values())),
        "unmatched_line_count": int((~statuses.eq("valid_pair")).sum()),
        "valid_pair_count": valid_pair_count,
        "valid_pair_line_count": int(statuses.eq("valid_pair").sum()),
        "reconstructed_line_count": int(reconstructed.sum()),
        "repaired_segment_count": int(repaired.sum()),
        "merged_line_count": int(merged.sum()),
        "terminal_extended_line_count": int(terminal.sum()),
        "median_of_line_median_probability": (
            float(np.median(finite)) if len(finite) else None
        ),
    }


def _boolean_series(lines: gpd.GeoDataFrame, column: str) -> pd.Series:
    if column not in lines.columns:
        return pd.Series(False, index=lines.index, dtype=bool)
    return lines[column].map(is_true).fillna(False).astype(bool)


def _maximum_lineage_displacement(
    raw_lines: gpd.GeoDataFrame, final_lines: gpd.GeoDataFrame
) -> Optional[float]:
    if raw_lines.empty or final_lines.empty or "raw_line_id" not in raw_lines.columns:
        return None
    by_id = dict(zip(raw_lines["raw_line_id"].astype(str), raw_lines.geometry))
    maximum = 0.0
    measured = False
    for _, row in final_lines.iterrows():
        identifiers = _source_ids(row.get("source_raw_ids"))
        sources = [by_id[value] for value in identifiers if value in by_id]
        if not sources:
            continue
        source = unary_union(sources)
        sampled = row.geometry.segmentize(0.10)
        directed_distance = max(
            (Point(coordinate).distance(source) for coordinate in sampled.coords),
            default=0.0,
        )
        maximum = max(maximum, float(directed_distance))
        measured = True
    return maximum if measured else None


def build_comparison_report(
    raw_lines: gpd.GeoDataFrame,
    result: PostprocessResult,
    config: dict,
    *,
    run_id: str,
    inference_seconds: Optional[float] = None,
    postprocess_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    final = result.final_lines
    return {
        "run_id": str(run_id),
        "before_post_inference": _network_summary(raw_lines, config),
        "after_postprocessing": _network_summary(final, config),
        "changes": {
            "reconstructed_lines": len(result.reconstructed_pairs),
            "repaired_segments": len(result.repaired_segments),
            "terminal_extensions": len(result.terminal_extensions),
            "rejected_lines": len(result.rejected_lines),
            "manual_review_lines": len(result.manual_review),
            "maximum_directed_distance_to_lineage_m": _maximum_lineage_displacement(
                raw_lines, final
            ),
        },
        "timing_seconds": {
            "inference": inference_seconds,
            "postprocessing": postprocess_seconds,
        },
        "postprocess_metrics": dict(result.metrics),
        "created_at": utc_now_iso(),
    }


def write_reports(
    layout: OutputLayout,
    report: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    overwrite: bool = False,
) -> None:
    for path in (
        layout.postprocess_report_path,
        layout.comparison_report_path,
        layout.manifest_path,
    ):
        _prepare_target(path, overwrite)
    _atomic_text(
        layout.postprocess_report_path,
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        overwrite=overwrite,
    )
    before = report.get("before_post_inference", {})
    after = report.get("after_postprocessing", {})
    keys = list(dict.fromkeys([*before.keys(), *after.keys()]))
    rows = "".join(
        "<tr><th>{}</th><td>{}</td><td>{}</td></tr>".format(
            html.escape(str(key)),
            html.escape(str(before.get(key, ""))),
            html.escape(str(after.get(key, ""))),
        )
        for key in keys
    )
    changes = report.get("changes", {})
    timing = report.get("timing_seconds", {})
    postprocess_metrics = report.get("postprocess_metrics", {})

    def detail_rows(values: Mapping[str, Any]) -> str:
        return "".join(
            "<tr><th>{}</th><td>{}</td></tr>".format(
                html.escape(str(key)), html.escape(str(value))
            )
            for key, value in values.items()
        )

    _atomic_text(
        layout.comparison_report_path,
        "<!doctype html><meta charset='utf-8'><title>Comparacao vetorial</title>"
        "<h1>ANTES: pos-inferencia / DEPOIS: pos-processamento</h1>"
        "<table border='1'><thead><tr><th>Metrica</th><th>ANTES</th>"
        f"<th>DEPOIS</th></tr></thead><tbody>{rows}</tbody></table>"
        "<h2>Mudancas</h2><table border='1'><tbody>"
        f"{detail_rows(changes)}</tbody></table>"
        "<h2>Tempos (s)</h2><table border='1'><tbody>"
        f"{detail_rows(timing)}</tbody></table>"
        "<h2>Gate e auditoria</h2><table border='1'><tbody>"
        f"{detail_rows(postprocess_metrics)}</tbody></table>",
        overwrite=overwrite,
    )
    write_manifest(layout, manifest, overwrite=overwrite)


def write_manifest(
    layout: OutputLayout,
    manifest: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    _atomic_text(
        layout.manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        overwrite=overwrite,
    )


__all__ = [
    "OutputLayout",
    "PersistedTwoStageResult",
    "build_comparison_report",
    "config_hash",
    "create_output_layout",
    "prepare_final_lines",
    "prepare_post_inference_lines",
    "relative_run_path",
    "select_production_lines",
    "persist_two_stage_vector_outputs",
    "utc_now_iso",
    "validate_vector_contract",
    "write_post_inference_product",
    "write_manifest",
    "write_compatibility_vector_product",
    "write_postprocess_product",
    "write_reports",
]
