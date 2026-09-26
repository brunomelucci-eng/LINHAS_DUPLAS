"""Refine existing crop-row vectors without repeating neural inference."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

import geopandas as gpd
import rasterio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.geospatial import populate_metadata
from src.logging_utils import StageTimer
from src.postprocessing.output_products import (
    prepare_final_lines,
    write_postprocess_product,
)
from src.postprocessing.postprocess_pipeline import run_postprocessing
from src.postprocessing.regularized_centerline_fairing import (
    RasterProbabilitySampler,
)


logger = logging.getLogger("refine_existing_vectors")

_CANONICAL_INPUT_LAYER = "post_inference_lines"
_LEGACY_INPUT_LAYER = "predicted_rows"


def _first_text(gdf: gpd.GeoDataFrame, column: str, fallback: str) -> str:
    if column not in gdf.columns:
        return fallback
    values = gdf[column].dropna()
    return str(values.iloc[0]) if len(values) else fallback


def _resolved_path(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _validate_output_path(input_path: str, output_path: str) -> None:
    if _resolved_path(input_path) == _resolved_path(output_path):
        raise ValueError(
            "Input vectors and post-processing output must be different files."
        )
    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")


def _read_input_vectors(path: str, requested_layer: str) -> tuple[gpd.GeoDataFrame, str]:
    available = set(gpd.list_layers(path)["name"].astype(str))
    selected_layer = requested_layer
    if selected_layer not in available:
        if (
            requested_layer == _CANONICAL_INPUT_LAYER
            and _LEGACY_INPUT_LAYER in available
        ):
            selected_layer = _LEGACY_INPUT_LAYER
            logger.warning(
                "Canonical layer %s is absent; reading legacy layer %s from %s.",
                _CANONICAL_INPUT_LAYER,
                _LEGACY_INPUT_LAYER,
                path,
            )
        else:
            raise ValueError(
                f"Input layer {requested_layer!r} does not exist in {path!r}. "
                f"Available layers: {sorted(available)!r}"
            )
    return gpd.read_file(path, layer=selected_layer), selected_layer


def _raw_run_id(vectors: gpd.GeoDataFrame, input_path: str) -> str:
    if "run_id" in vectors.columns:
        values = [
            value
            for value in vectors["run_id"].dropna().astype(str).str.strip().unique()
            if value
        ]
        if len(values) > 1:
            raise ValueError("Input vectors contain more than one run_id.")
        if values:
            return values[0]
    fallback = f"legacy_{Path(input_path).stem}"
    logger.warning("Input vectors have no run_id; using compatibility id %s.", fallback)
    return fallback


def _ensure_raw_lineage(
    vectors: gpd.GeoDataFrame,
    *,
    run_id: str,
) -> gpd.GeoDataFrame:
    """Preserve canonical raw lineage and fill only legacy omissions in memory."""
    lines = vectors.copy(deep=True)
    generated_ids = [f"raw_{index:06d}" for index in range(len(lines))]
    if "raw_line_id" not in lines.columns:
        lines["raw_line_id"] = generated_ids
    else:
        lines["raw_line_id"] = lines["raw_line_id"].astype("object")
        missing = lines["raw_line_id"].isna() | (
            lines["raw_line_id"].astype(str).str.strip().eq("")
        )
        lines.loc[missing, "raw_line_id"] = [
            generated_ids[index]
            for index in range(len(lines))
            if bool(missing.iloc[index])
        ]
    raw_identifiers = lines["raw_line_id"].astype(str)
    if raw_identifiers.duplicated().any():
        raise ValueError("Input raw_line_id values must be unique.")
    if "source_raw_ids" not in lines.columns:
        lines["source_raw_ids"] = raw_identifiers
    else:
        lines["source_raw_ids"] = lines["source_raw_ids"].astype("object")
        missing = lines["source_raw_ids"].isna() | (
            lines["source_raw_ids"].astype(str).str.strip().eq("")
        )
        lines.loc[missing, "source_raw_ids"] = lines.loc[
            missing, "raw_line_id"
        ].astype(str)
    if "run_id" not in lines.columns:
        lines["run_id"] = run_id
    else:
        lines["run_id"] = lines["run_id"].astype("object")
        missing = lines["run_id"].isna() | lines["run_id"].astype(str).str.strip().eq("")
        lines.loc[missing, "run_id"] = run_id
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply regularized centerline fairing to existing vectors only."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-vectors", required=True)
    parser.add_argument("--input-layer", default=_CANONICAL_INPUT_LAYER)
    parser.add_argument("--probability-raster", required=True)
    parser.add_argument("--probability-band", type=int, default=1)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--roi-layer", default=None)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _validate_output_path(args.input_vectors, args.output)
    config = load_config(args.config)
    vectors, input_layer = _read_input_vectors(args.input_vectors, args.input_layer)
    if vectors.empty:
        raise ValueError("Input vector layer contains no crop-row geometries.")
    run_id = _raw_run_id(vectors, args.input_vectors)
    vectors = _ensure_raw_lineage(vectors, run_id=run_id)
    logger.info("Reading post-inference vectors from layer %s.", input_layer)
    roi = gpd.read_file(args.roi, layer=args.roi_layer)
    roi = roi.loc[~roi.geometry.is_empty].copy()
    roi["geometry"] = roi.geometry.make_valid()

    original_crs = vectors.crs
    with rasterio.open(args.probability_raster) as probability_source:
        if probability_source.crs is None:
            raise ValueError("Probability raster has no CRS.")
        if args.probability_band < 1 or args.probability_band > probability_source.count:
            raise ValueError("Invalid probability raster band index.")
        # The debug probability product currently has one band. Preserve the
        # generic CLI argument while exposing that band through the sampler.
        if args.probability_band != 1:
            raise ValueError("RasterProbabilitySampler currently supports probability band 1.")
        vectors = vectors.to_crs(probability_source.crs)
        roi = roi.to_crs(probability_source.crs)
        sampler = RasterProbabilitySampler(
            probability_source,
            max_cached_blocks=int(
                config.get("final_centerline_fairing", {})
                .get("io", {})
                .get("max_cached_probability_blocks", 128)
            ),
        )

        with StageTimer(
            "existing_vectors_postprocess",
            disk_path=os.path.dirname(args.output) or ".",
        ):
            postprocess_result = run_postprocessing(
                vectors,
                sampler,
                probability_source.transform,
                roi,
                config,
                debug=False,
            )
            refined = prepare_final_lines(
                postprocess_result.final_lines,
                run_id=run_id,
                valid_raw_line_ids=vectors["raw_line_id"].astype(str).tolist(),
            )
        with StageTimer("existing_vectors_metadata", disk_path=os.path.dirname(args.output) or "."):
            refined = populate_metadata(
                refined,
                config,
                model_name=_first_text(vectors, "model_name", "unet"),
                checkpoint_name=_first_text(vectors, "checkpoint_name", "best.pt"),
                source_id=_first_text(vectors, "source_id", os.path.basename(args.input_vectors)),
            )

    export_epsg = config.get("crs", {}).get("export_epsg")
    if export_epsg is None and original_crs is not None:
        export_epsg = original_crs.to_epsg()
    with StageTimer("existing_vectors_export", disk_path=os.path.dirname(args.output) or "."):
        write_postprocess_product(
            postprocess_result,
            args.output,
            final_lines=refined,
            overwrite=False,
            export_epsg=export_epsg,
        )
    logger.info(
        "Postprocess-only refinement completed: %d input lines -> %d output lines at %s",
        len(vectors),
        len(refined),
        args.output,
    )
    logger.info("Postprocess metrics: %s", dict(postprocess_result.metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
