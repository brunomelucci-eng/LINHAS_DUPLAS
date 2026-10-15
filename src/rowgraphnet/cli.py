"""Command-line interface for validation, inference, and synthetic QA."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from rowgraphnet import __version__
from rowgraphnet.data import prepare_metric_scene, validate_scene
from rowgraphnet.export import export_inference_bundle
from rowgraphnet.inference import (
    InferenceConfig,
    InputNormalization,
    TiledInferenceEngine,
    load_local_checkpoint,
    read_local_checkpoint_metadata,
)
from rowgraphnet.models import RowGraphNet
from rowgraphnet.pipeline import PostprocessConfig, postprocess_prediction
from rowgraphnet.resources import (
    DEFAULT_RESOURCE_FRACTION,
    apply_resource_caps,
    check_virtual_memory_safety,
)
from rowgraphnet.synthetic import generate_synthetic_scene

app = typer.Typer(
    name="rowgraphnet",
    help="Extração probabilística e reconstrução topológica de fileiras agrícolas.",
    no_args_is_help=True,
)


@dataclass(frozen=True, slots=True)
class InferenceFileConfig:
    """Strict optional overrides loaded from an inference YAML file."""

    tile_size_px: int | None = None
    overlap_fraction: float | None = None
    batch_size: int | None = None
    device: str | None = None
    resource_fraction: float | None = None
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    def __post_init__(self) -> None:
        if self.tile_size_px is not None and (
            isinstance(self.tile_size_px, bool)
            or not isinstance(self.tile_size_px, int)
            or self.tile_size_px < 64
        ):
            raise ValueError("tile_size_px must be an integer >= 64")
        if self.overlap_fraction is not None and (
            not isinstance(self.overlap_fraction, (int, float))
            or isinstance(self.overlap_fraction, bool)
            or not 0.0 <= float(self.overlap_fraction) < 1.0
        ):
            raise ValueError("overlap_fraction must lie in [0, 1)")
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        if self.device is not None and self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.resource_fraction is not None and (
            not isinstance(self.resource_fraction, (int, float))
            or isinstance(self.resource_fraction, bool)
            or not 0.1 <= float(self.resource_fraction) <= 1.0
        ):
            raise ValueError("resource_fraction must lie in [0.1, 1.0]")


def _bands(value: str) -> tuple[int, ...]:
    try:
        indexes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise typer.BadParameter("bandas devem ser índices inteiros separados por vírgula") from exc
    if not indexes or len(set(indexes)) != len(indexes) or any(index < 1 for index in indexes):
        raise typer.BadParameter("bandas devem ser índices positivos e únicos")
    return indexes


def _print_json(value: Mapping[str, Any]) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def _guard_outputs(output_dir: Path, overwrite: bool) -> None:
    expected = (
        "linhas_simples.gpkg",
        "trechos_reconstruidos.gpkg",
        "linhas_incertas.gpkg",
        "quality_report.json",
    )
    existing = [output_dir / name for name in expected if (output_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise typer.BadParameter(f"saídas já existem ({names}); use --overwrite conscientemente")


def _postprocess_from_mapping(raw: Mapping[str, Any]) -> PostprocessConfig:
    allowed = {item.name for item in fields(PostprocessConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise typer.BadParameter(f"chaves desconhecidas em postprocess: {unknown}")
    try:
        return PostprocessConfig(**dict(raw))
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"configuração de pós-processamento inválida: {exc}") from exc


def _load_inference_config(path: Path | None) -> InferenceFileConfig:
    if path is None:
        return InferenceFileConfig()
    try:
        decoded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise typer.BadParameter(f"configuração de pós-processamento inválida: {path}") from exc
    if not isinstance(decoded, Mapping):
        raise typer.BadParameter("arquivo de configuração deve conter um mapeamento YAML")

    postprocess_fields = {item.name for item in fields(PostprocessConfig)}
    if "postprocess" not in decoded and set(decoded).issubset(postprocess_fields):
        return InferenceFileConfig(postprocess=_postprocess_from_mapping(decoded))

    allowed_root = {
        "tile_size_px",
        "overlap_fraction",
        "batch_size",
        "device",
        "resource_fraction",
        "postprocess",
    }
    unknown_root = sorted(set(decoded) - allowed_root)
    if unknown_root:
        raise typer.BadParameter(f"chaves desconhecidas na configuração de inferência: {unknown_root}")
    raw_postprocess = decoded.get("postprocess", {})
    if not isinstance(raw_postprocess, Mapping):
        raise typer.BadParameter("chave postprocess deve conter um mapeamento")
    try:
        return InferenceFileConfig(
            tile_size_px=decoded.get("tile_size_px"),
            overlap_fraction=decoded.get("overlap_fraction"),
            batch_size=decoded.get("batch_size"),
            device=decoded.get("device"),
            resource_fraction=decoded.get("resource_fraction"),
            postprocess=_postprocess_from_mapping(raw_postprocess),
        )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"configuração de inferência inválida: {exc}") from exc


def _load_postprocess_config(path: Path | None) -> PostprocessConfig:
    """Backward-compatible helper used by integrations that only need this section."""

    return _load_inference_config(path).postprocess


def _resolve_inference_runtime(
    file_config: InferenceFileConfig,
    *,
    tile_size: int | None,
    overlap_fraction: float | None,
    batch_size: int | None,
    device: str | None,
) -> tuple[int, float, int, str]:
    """Apply defaults < YAML < explicit CLI precedence."""

    resolved_tile_size = (
        tile_size
        if tile_size is not None
        else file_config.tile_size_px
        if file_config.tile_size_px is not None
        else 512
    )
    resolved_overlap = (
        float(overlap_fraction)
        if overlap_fraction is not None
        else float(file_config.overlap_fraction)
        if file_config.overlap_fraction is not None
        else 0.5
    )
    resolved_batch_size = (
        batch_size
        if batch_size is not None
        else file_config.batch_size
        if file_config.batch_size is not None
        else 1
    )
    resolved_device = device or file_config.device or "auto"
    try:
        InferenceFileConfig(
            tile_size_px=resolved_tile_size,
            overlap_fraction=resolved_overlap,
            batch_size=resolved_batch_size,
            device=resolved_device,
        )
    except ValueError as exc:
        raise typer.BadParameter(f"opção de inferência inválida: {exc}") from exc
    return resolved_tile_size, resolved_overlap, resolved_batch_size, resolved_device


def _require_complete_vector_supervision(metadata: Any) -> None:
    architecture = metadata.raw.get("architecture", {})
    supervision = architecture.get("head_supervision") if isinstance(architecture, Mapping) else None
    allowed_statuses = {
        "center": {"direct"},
        "band": {"direct"},
        "orientation": {"direct", "derived_from_centerline"},
        "endpoints": {"direct"},
        "distance": {"direct", "derived_from_centerline"},
        "uncertainty": {"direct", "derived_from_distance_error"},
    }
    if isinstance(architecture, Mapping) and bool(architecture.get("refine_connectivity")):
        allowed_statuses["repair"] = {"direct"}
    required = set(allowed_statuses)
    if not isinstance(supervision, Mapping):
        raise typer.BadParameter(
            "checkpoint sem architecture.head_supervision; não é seguro assumir que todas "
            "as heads usadas na vetorização foram treinadas"
        )
    missing = sorted(required - set(supervision))
    if missing:
        raise typer.BadParameter(f"checkpoint não declara supervisão para as heads: {missing}")
    unsafe = {
        name: supervision[name]
        for name in sorted(required)
        if not isinstance(supervision[name], str)
        or str(supervision[name]).casefold() not in allowed_statuses[name]
    }
    if unsafe:
        raise typer.BadParameter(
            "checkpoint parcial não é seguro para a inferência vetorial completa; "
            f"supervisão insuficiente: {unsafe}. Complete os rótulos e retreine."
        )


def _load_field_boundary(path: Path, layer: str | None, target_crs: CRS):
    import geopandas as gpd

    try:
        frame = gpd.read_file(path, layer=layer)
    except Exception as exc:
        raise typer.BadParameter(f"não foi possível ler o limite do talhão: {path}") from exc
    if frame.empty or frame.crs is None:
        raise typer.BadParameter("limite do talhão deve ter geometrias e CRS definido")
    frame = frame.to_crs(target_crs)
    geometries = [
        geometry for geometry in frame.geometry if geometry is not None and not geometry.is_empty
    ]
    if not geometries or any(
        not isinstance(geometry, (Polygon, MultiPolygon)) for geometry in geometries
    ):
        raise typer.BadParameter("limite do talhão aceita somente Polygon ou MultiPolygon")
    boundary = unary_union(geometries)
    if (
        boundary.is_empty
        or not boundary.is_valid
        or not isinstance(boundary, (Polygon, MultiPolygon))
    ):
        raise typer.BadParameter("limite do talhão unido deve ser poligonal, válido e não vazio")
    return boundary


def _extract_roi_polygons(roi_path: Path) -> tuple[list, Any]:
    """Read a ROI file and return individual Polygon geometries + CRS.

    Used to detect multi-field ROIs and process each field independently
    to avoid excessive memory usage from large bounding boxes.
    """
    import geopandas as gpd

    gdf = gpd.read_file(roi_path)
    if gdf.empty or gdf.crs is None:
        return [], None

    file_crs = gdf.crs
    polygons: list = []

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            from shapely import make_valid

            geom = make_valid(geom)
            if not geom.is_valid:
                geom = geom.buffer(0)
        if geom.geom_type == "MultiPolygon":
            polygons.extend([p for p in geom.geoms if not p.is_empty and p.is_valid])
        elif geom.geom_type == "Polygon" and not geom.is_empty and geom.is_valid:
            polygons.append(geom)

    return polygons, file_crs



@app.command("version")
def version_command() -> None:
    """Exibir a versão do pacote."""

    typer.echo(__version__)


@app.command("validate-data")
def validate_data(
    raster: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    roi: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    bands: Annotated[str, typer.Option("--bands", help="Índices Rasterio 1-based.")] = "1,2,3",
    invalid_mask: Annotated[
        Path | None,
        typer.Option("--invalid-mask", exists=True, dir_okay=False, readable=True),
    ] = None,
) -> None:
    """Validar raster, ROI, bandas, GSD, nodata e CRS métrico de trabalho."""

    scene = validate_scene(
        raster,
        roi,
        bands=_bands(bands),
        invalid_mask_path=invalid_mask,
    )
    contract = scene.raster
    _print_json(
        {
            "status": "valid",
            "raster": str(contract.path),
            "shape": [contract.height, contract.width],
            "bands": list(contract.bands),
            "source_crs": contract.crs.to_string(),
            "metric_crs": contract.metric_crs.to_string(),
            "gsd_m": {"x": contract.gsd.x_m, "y": contract.gsd.y_m},
            "roi_was_clipped": scene.roi_was_clipped,
            "invalid_mask": str(scene.invalid_mask_path) if scene.invalid_mask_path else None,
        }
    )


@app.command("synthetic-demo")
def synthetic_demo(
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    kind: Annotated[
        str,
        typer.Option(help="straight, curve, s_curve, variable_spacing ou two_regimes."),
    ] = "straight",
    gaps: Annotated[bool, typer.Option("--gaps/--no-gaps")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Executar o pós-processamento em evidência sintética georreferenciada."""

    _guard_outputs(output_dir, overwrite)
    scene = generate_synthetic_scene(kind, with_gaps=gaps, with_nodata=True)
    model_version = "synthetic-oracle-v1"
    result = postprocess_prediction(
        scene.prediction,
        model_version=model_version,
        roi=scene.roi,
    )
    paths = export_inference_bundle(
        output_dir,
        scene.prediction,
        rows=result.refinement.rows,
        repairs=result.refinement.repairs,
        uncertain_rows=result.refinement.uncertain_rows,
        run_id=f"synthetic-{kind}",
        model_version=model_version,
        provenance={
            "validation_scope": "synthetic_geometry_only",
            "real_model_checkpoint": False,
            "confidence_calibration": "not_calibrated_on_real_data",
            "postprocess": result.diagnostics,
        },
        configuration=result.diagnostics["configuration"],
    )
    _print_json(
        {
            "status": "completed",
            "warning": "resultado sintético valida software, não acurácia agrícola",
            "valid_rows": len(result.refinement.rows),
            "uncertain_rows": len(result.refinement.uncertain_rows),
            "accepted_repairs": len(result.refinement.repairs),
            "artifacts": {name: str(path) for name, path in paths.items()},
        }
    )


def _model_from_metadata(metadata: Any) -> RowGraphNet:
    architecture = metadata.raw.get("architecture")
    if not isinstance(architecture, Mapping):
        raise typer.BadParameter(
            "checkpoint sem metadata.architecture; inclua encoder_config completo para "
            "reconstrução offline reproduzível"
        )
    encoder_config = architecture.get("encoder_config")
    if not isinstance(encoder_config, Mapping):
        raise typer.BadParameter(
            "checkpoint metadata.architecture.encoder_config deve ser um mapeamento completo"
        )
    return RowGraphNet(
        model_name=str(architecture.get("model_name", "nvidia/mit-b2")),
        pretrained=False,
        in_channels=metadata.in_channels,
        decoder_channels=int(architecture.get("decoder_channels", 128)),
        encoder_config=dict(encoder_config),
        refine_connectivity=bool(architecture.get("refine_connectivity", True)),
        refiner_hidden_channels=int(architecture.get("refiner_hidden_channels", 64)),
    )


@app.command("infer")
def infer(
    raster: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    roi: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    checkpoint: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    invalid_mask: Annotated[
        Path | None,
        typer.Option("--invalid-mask", exists=True, dir_okay=False, readable=True),
    ] = None,
    field_boundary: Annotated[
        Path | None,
        typer.Option("--field-boundary", exists=True, dir_okay=False, readable=True),
    ] = None,
    field_boundary_layer: Annotated[str | None, typer.Option("--field-boundary-layer")] = None,
    roi_is_field_boundary: Annotated[
        bool,
        typer.Option(
            "--roi-is-field-boundary",
            help="Declarar explicitamente que a ROI representa o limite físico do talhão.",
        ),
    ] = False,
    postprocess_config: Annotated[
        Path | None,
        typer.Option(
            "--inference-config",
            "--postprocess-config",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    tile_size: Annotated[int | None, typer.Option(min=64)] = None,
    overlap_fraction: Annotated[float | None, typer.Option(min=0.0, max=0.9)] = None,
    batch_size: Annotated[int | None, typer.Option(min=1)] = None,
    device: Annotated[str | None, typer.Option(help="auto, cpu ou cuda.")] = None,
    resource_fraction: Annotated[
        float,
        typer.Option(
            "--resource-fraction",
            min=0.1,
            max=1.0,
            help="Fração máxima da capacidade do computador (CPU/GPU/RAM) a utilizar (padrão: 0.70).",
        ),
    ] = DEFAULT_RESOURCE_FRACTION,
    allow_partial_checkpoint: Annotated[
        bool,
        typer.Option(
            "--allow-partial-checkpoint",
            help="Permitir execução com checkpoints que não possuem supervisão direta em todas as heads secundárias.",
        ),
    ] = False,
    export_enhanced: Annotated[
        bool,
        typer.Option(
            "--export-enhanced/--no-export-enhanced",
            help="Gerar adicionalmente os 2 pós-processamentos juntos: v1 (anterior) e v2 (cirúrgico).",
        ),
    ] = True,
    max_gap_distance_m: Annotated[
        float,
        typer.Option(
            "--max-gap-distance-m",
            help="Distância máxima de gap (em metros) no modo enhanced.",
        ),
    ] = 6.0,
    max_boundary_extension_m: Annotated[
        float,
        typer.Option(
            "--max-boundary-extension-m",
            help="Alcance máximo de extensão até o talhão (em metros) no modo enhanced.",
        ),
    ] = 8.0,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Executar checkpoint local, reconstrução e exportação completa."""

    if field_boundary is not None and roi_is_field_boundary:
        raise typer.BadParameter(
            "use --field-boundary ou --roi-is-field-boundary, nunca os dois simultaneamente"
        )
    if field_boundary_layer is not None and field_boundary is None:
        raise typer.BadParameter("--field-boundary-layer exige --field-boundary")
    _guard_outputs(output_dir, overwrite)
    inference_file_settings = _load_inference_config(postprocess_config)
    postprocess_settings = inference_file_settings.postprocess
    (
        effective_tile_size,
        effective_overlap_fraction,
        effective_batch_size,
        effective_device,
    ) = _resolve_inference_runtime(
        inference_file_settings,
        tile_size=tile_size,
        overlap_fraction=overlap_fraction,
        batch_size=batch_size,
        device=device,
    )
    effective_resource_fraction = (
        resource_fraction
        if resource_fraction != DEFAULT_RESOURCE_FRACTION
        else (inference_file_settings.resource_fraction or DEFAULT_RESOURCE_FRACTION)
    )
    status = apply_resource_caps(fraction=effective_resource_fraction, device=effective_device)
    cuda_info = (
        f"VRAM limitada a {status.cuda_vram_limit_gb:.1f} GB ({status.cuda_device_name})"
        if status.cuda_vram_limit_gb is not None
        else "modo CPU"
    )
    typer.echo(
        f"[Recursos] Padrão ativo de {effective_resource_fraction:.0%}: "
        f"{status.cpu_threads_allocated}/{status.cpu_cores} threads CPU, {cuda_info}."
    )
    for warning in check_virtual_memory_safety():
        typer.secho(f"[Aviso de Sistema] {warning}", fg=typer.colors.YELLOW)
    is_large_tile = effective_tile_size >= 512
    unsafe_batch = effective_batch_size > (4 if is_large_tile else 32)
    if unsafe_batch and effective_device in {"auto", "cuda"}:
        typer.secho(
            f"[Aviso de Desempenho] batch-size {effective_batch_size} com tile {effective_tile_size} "
            "pode causar sobrecarga de memória na GPU. Se o sistema ficar instável, reduza o batch-size.",
            fg=typer.colors.YELLOW,
        )
    metadata = read_local_checkpoint_metadata(checkpoint)
    if not allow_partial_checkpoint:
        _require_complete_vector_supervision(metadata)

    # ------------------------------------------------------------------
    # Detect multi-polygon ROI for per-field processing
    # ------------------------------------------------------------------
    roi_polygons, roi_file_crs = _extract_roi_polygons(roi)
    is_multi_talhao = len(roi_polygons) > 1

    # Build model and engine ONCE (reused across all talhões)
    model = _model_from_metadata(metadata)
    load_local_checkpoint(
        model,
        checkpoint,
        expected_bands=metadata.band_indexes,
        strict=True,
    )
    internal_bands = tuple(range(1, metadata.in_channels + 1))
    engine = TiledInferenceEngine(
        model,
        config=InferenceConfig(
            tile_size=effective_tile_size,
            overlap_fraction=effective_overlap_fraction,
            batch_size=effective_batch_size,
            device=effective_device,
            memory_budget_fraction=effective_resource_fraction,
            resource_fraction=effective_resource_fraction,
        ),
        normalization=InputNormalization.from_metadata(metadata.normalization),
        calibrators=metadata.build_calibrators(),
        expected_bands=internal_bands,
    )

    if not is_multi_talhao:
        # =============================================================
        # SINGLE TALHÃO — existing behavior, unchanged
        # =============================================================
        source = validate_scene(
            raster, roi, bands=metadata.band_indexes, invalid_mask_path=invalid_mask
        )
        working_path = output_dir / "_working" / "metric_input.tif"
        prepared = prepare_metric_scene(source, working_path, overwrite=overwrite)
        prediction = engine.predict(prepared.working)
        if roi_is_field_boundary:
            metric_field_boundary = prepared.working.roi_in_metric_crs
            boundary_source = "analysis_roi_explicitly_declared_physical_boundary"
        elif field_boundary is not None:
            metric_field_boundary = _load_field_boundary(
                field_boundary,
                field_boundary_layer,
                prepared.working.raster.crs,
            )
            boundary_source = str(field_boundary)
        else:
            metric_field_boundary = None
            boundary_source = None
        result = postprocess_prediction(
            prediction,
            model_version=metadata.model_version,
            roi=prepared.working.roi_in_metric_crs,
            field_boundary=metric_field_boundary,
            config=postprocess_settings,
        )
        final_rows = result.refinement.rows
        final_repairs = result.refinement.repairs
        final_uncertain = result.refinement.uncertain_rows
        final_prediction = prediction
        source_crs_str = source.raster.crs.to_string()
        working_crs_str = prepared.working.raster.crs.to_string()
        final_diagnostics = result.diagnostics
        final_metric_boundary = metric_field_boundary
        all_enhanced_rows: list = []
        if export_enhanced:
            import numpy as np

            from rowgraphnet.geometry.enhanced_refinement import (
                EnhancedRefinementConfig,
                apply_enhanced_refinement,
            )

            spacing_vals = [
                r.spacing_local_m
                for r in final_rows
                if r.spacing_local_m is not None
            ]
            spacing_m = float(np.median(spacing_vals)) if spacing_vals else 1.5
            cfg = EnhancedRefinementConfig(
                max_gap_distance_m=max_gap_distance_m,
                max_boundary_extension_m=max_boundary_extension_m,
            )
            all_enhanced_rows, _enh_diag = apply_enhanced_refinement(
                final_rows,
                spacing_m=spacing_m,
                field_boundary=final_metric_boundary,
                config=cfg,
                model_version=metadata.model_version,
            )
    else:
        # =============================================================
        # MULTI-TALHÃO — process each polygon individually, merge at end
        # =============================================================
        import torch

        typer.echo(
            f"\n[Multi-Talhão] ROI contém {len(roi_polygons)} polígonos separados "
            f"— processamento individual para evitar sobrecarga de memória."
        )

        all_rows: list = []
        all_repairs: list = []
        all_uncertain: list = []
        all_enhanced_rows: list = []
        all_metric_boundaries: list = []
        final_prediction = None
        source_crs_str = None
        working_crs_str = None
        boundary_source = None
        first_working_crs = None

        import pickle

        for idx, poly in enumerate(roi_polygons, 1):
            typer.echo(f"\n{'='*60}")
            typer.echo(f"[Talhão {idx}/{len(roi_polygons)}] Processando...")
            typer.echo(f"{'='*60}")

            cache_file = output_dir / "_working" / f"talhao_{idx}_cache.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f_cache:
                        cache_data = pickle.load(f_cache)
                    all_rows.extend(cache_data["rows"])
                    all_repairs.extend(cache_data["repairs"])
                    all_uncertain.extend(cache_data["uncertain"])
                    if export_enhanced and "enhanced_rows" in cache_data:
                        all_enhanced_rows.extend(cache_data["enhanced_rows"])
                    if cache_data.get("boundary") is not None:
                        all_metric_boundaries.append(cache_data["boundary"])
                    typer.secho(
                        f"  [Cache] Talhão {idx} recuperado com sucesso do cache em disco!",
                        fg=typer.colors.GREEN,
                    )
                    continue
                except Exception:
                    pass

            source_i = validate_scene(
                raster,
                poly,
                bands=metadata.band_indexes,
                invalid_mask_path=invalid_mask,
                roi_crs=roi_file_crs,
            )
            working_path = output_dir / "_working" / f"metric_input_talhao_{idx}.tif"
            typer.echo(f"  [1/4] Preparando cena métrica...")
            import sys; sys.stdout.flush()
            prepared_i = prepare_metric_scene(source_i, working_path, overwrite=True)

            # Verify CRS consistency across talhões
            this_crs = prepared_i.working.raster.crs.to_string()
            if first_working_crs is None:
                first_working_crs = this_crs
                source_crs_str = source_i.raster.crs.to_string()
                working_crs_str = this_crs
            elif this_crs != first_working_crs:
                typer.secho(
                    f"  [Aviso] CRS métrico diferente do primeiro talhão "
                    f"({this_crs} vs {first_working_crs}). "
                    f"Resultados podem ser inconsistentes.",
                    fg=typer.colors.YELLOW,
                )

            typer.echo(f"  [2/4] Executando predição neural na GPU (tiles Hann-blended)...")
            sys.stdout.flush()
            prediction_i = engine.predict(prepared_i.working)

            if roi_is_field_boundary:
                metric_fb_i = prepared_i.working.roi_in_metric_crs
                boundary_source = "analysis_roi_explicitly_declared_physical_boundary"
                all_metric_boundaries.append(metric_fb_i)
            elif field_boundary is not None:
                metric_fb_i = _load_field_boundary(
                    field_boundary,
                    field_boundary_layer,
                    prepared_i.working.raster.crs,
                )
                boundary_source = str(field_boundary)
            else:
                metric_fb_i = None
                boundary_source = None

            typer.echo(f"  [3/4] Vetorizando predições e reparando caminhos (A* otimizado + STRtree)...")
            sys.stdout.flush()
            result_i = postprocess_prediction(
                prediction_i,
                model_version=metadata.model_version,
                roi=prepared_i.working.roi_in_metric_crs,
                field_boundary=metric_fb_i,
                config=postprocess_settings,
            )

            n_valid = len(result_i.refinement.rows)
            n_uncertain = len(result_i.refinement.uncertain_rows)
            n_repairs = len(result_i.refinement.repairs)
            typer.secho(
                f"  Válidas: {n_valid} | Incertas: {n_uncertain} | Reparos: {n_repairs}",
                fg=typer.colors.GREEN,
            )
            sys.stdout.flush()

            all_rows.extend(result_i.refinement.rows)
            all_repairs.extend(result_i.refinement.repairs)
            all_uncertain.extend(result_i.refinement.uncertain_rows)

            enh_rows_i = []
            if export_enhanced:
                import numpy as np

                from rowgraphnet.geometry.enhanced_refinement import (
                    EnhancedRefinementConfig,
                    apply_enhanced_refinement,
                )

                spacing_vals_i = [
                    r.spacing_local_m
                    for r in result_i.refinement.rows
                    if r.spacing_local_m is not None
                ]
                spacing_m_i = float(np.median(spacing_vals_i)) if spacing_vals_i else 1.5
                cfg = EnhancedRefinementConfig(
                    max_gap_distance_m=max_gap_distance_m,
                    max_boundary_extension_m=max_boundary_extension_m,
                )
                try:
                    enh_rows_i, _ = apply_enhanced_refinement(
                        result_i.refinement.rows,
                        spacing_m=spacing_m_i,
                        field_boundary=metric_fb_i,
                        config=cfg,
                        model_version=metadata.model_version,
                    )
                    all_enhanced_rows.extend(enh_rows_i)
                except Exception as enh_err:
                    typer.secho(
                        f"  [Aviso] Refinamento avançado encontrou exceção no talhão {idx}: {enh_err}. "
                        "Mantendo linhas base.",
                        fg=typer.colors.YELLOW,
                    )
                    enh_rows_i = list(result_i.refinement.rows)
                    all_enhanced_rows.extend(enh_rows_i)

            # Persist per-talhão cache to avoid losing progress
            try:
                cache_data = {
                    "rows": result_i.refinement.rows,
                    "repairs": result_i.refinement.repairs,
                    "uncertain": result_i.refinement.uncertain_rows,
                    "enhanced_rows": enh_rows_i if export_enhanced else [],
                    "boundary": metric_fb_i,
                }
                with open(cache_file, "wb") as f_cache:
                    pickle.dump(cache_data, f_cache)
            except Exception as cache_err:
                typer.secho(
                    f"  [Aviso] Não foi possível gravar cache do talhão {idx}: {cache_err}",
                    fg=typer.colors.YELLOW,
                )

            # Free intermediate memory before next talhão
            if idx == len(roi_polygons):
                final_prediction = prediction_i
            else:
                del prediction_i
            del result_i, prepared_i, source_i
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        final_rows = all_rows
        final_repairs = all_repairs
        final_uncertain = all_uncertain
        final_diagnostics = {"multi_talhao_count": len(roi_polygons)}
        final_metric_boundary = (
            unary_union(all_metric_boundaries) if all_metric_boundaries else None
        )

        typer.echo(f"\n{'='*60}")
        typer.echo(
            f"Unificação de {len(roi_polygons)} talhões: "
            f"{len(final_rows)} válidas | {len(final_uncertain)} incertas | "
            f"{len(final_repairs)} reparos"
        )
        if export_enhanced:
            typer.echo(f"Total linhas completas v1 (enhanced): {len(all_enhanced_rows)}")
        typer.echo(f"{'='*60}")

    # =================================================================
    # Common export path (single or multi-talhão)
    # =================================================================
    inputs: dict[str, Path] = {
        "orthomosaic": raster,
        "roi": roi,
        "checkpoint": checkpoint,
    }
    if invalid_mask is not None:
        inputs["invalid_mask"] = invalid_mask
    if field_boundary is not None:
        inputs["field_boundary"] = field_boundary
    if postprocess_config is not None:
        inputs["postprocess_config"] = postprocess_config
    paths = export_inference_bundle(
        output_dir,
        final_prediction,
        rows=final_rows,
        repairs=final_repairs,
        uncertain_rows=final_uncertain,
        run_id=output_dir.name or "rowgraphnet-inference",
        model_version=metadata.model_version,
        provenance={
            "source_crs": source_crs_str,
            "working_crs": working_crs_str,
            "source_band_indexes": metadata.band_indexes,
            "checkpoint_calibrated_heads": sorted(metadata.calibration),
            "confidence_calibrated": bool(metadata.calibration),
            "field_boundary_source": boundary_source,
            "postprocess": final_diagnostics,
        },
        configuration={
            "tile_size": effective_tile_size,
            "overlap_fraction": effective_overlap_fraction,
            "batch_size": effective_batch_size,
            "device": str(engine.device),
            "architecture": metadata.raw.get("architecture"),
            "postprocess": asdict(postprocess_settings),
        },
        input_files=inputs,
    )
    if export_enhanced:
        import hashlib

        import numpy as np

        # 1) Pós-processamento 1: versão anterior (enhanced_refinement)
        from rowgraphnet.geometry.enhanced_refinement import export_enhanced_geopackage

        pos1_path = export_enhanced_geopackage(
            output_dir / "linhas_completas_v1_enhanced.gpkg",
            all_enhanced_rows,
            crs=working_crs_str,
            layer_name="linhas_completas_v1",
        )
        paths["linhas_completas_v1_enhanced"] = pos1_path

        # 2) Pós-processamento 2: nova versão cirúrgica (postprocess_v2)
        from rowgraphnet.geometry.postprocess_v2 import (
            PostprocessV2Config,
            export_complete_v2,
            export_uncertain_v2,
            lines_from_geodataframe,
            read_gpkg_sqlite,
            run_postprocess_v2,
        )

        simples_gpkg = paths.get("linhas_simples")
        if simples_gpkg and simples_gpkg.exists():
            gdf_s = read_gpkg_sqlite(simples_gpkg)
            v2_lines = lines_from_geodataframe(gdf_s, "simples")

            incertas_gpkg = paths.get("linhas_incertas")
            if incertas_gpkg and incertas_gpkg.exists():
                gdf_i = read_gpkg_sqlite(incertas_gpkg)
                if not gdf_i.empty:
                    reason_col = "repair_reason" if "repair_reason" in gdf_i.columns else ("motivo_revisao" if "motivo_revisao" in gdf_i.columns else None)
                    if reason_col:
                        mask = ~gdf_i[reason_col].fillna("").astype(str).str.contains("crosses|overlaps|duplicate")
                        filtered_i = gdf_i[mask]
                    else:
                        filtered_i = gdf_i
                    v2_lines.extend(lines_from_geodataframe(filtered_i, "incertas"))

            spacing_vals = [
                r.spacing_local_m
                for r in final_rows
                if r.spacing_local_m is not None
            ]
            sp_v2 = float(np.median(spacing_vals)) if spacing_vals else None

            v2_result = run_postprocess_v2(
                lines=v2_lines,
                field_boundary=final_metric_boundary,
                spacing_m=sp_v2,
                config=PostprocessV2Config(),
            )
            v2_path = export_complete_v2(
                output_dir / "linhas_completas_v2.gpkg",
                v2_result.lines,
                crs=working_crs_str,
                spacing_m=sp_v2 or 1.5,
            )
            unc2_path = export_uncertain_v2(
                output_dir / "linhas_incertas_v2.gpkg",
                v2_result.uncertain,
                crs=working_crs_str,
            )
            rep2_path = output_dir / "relatorio_pos_v2.json"
            rep2_path.write_text(
                json.dumps(v2_result.diagnostics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            paths["linhas_completas_v2"] = v2_path
            paths["linhas_incertas_v2"] = unc2_path
            paths["relatorio_pos_v2"] = rep2_path

        # Atualiza quality_report.json com os novos artefatos exportados
        qr_path = output_dir / "quality_report.json"
        if qr_path.exists():
            try:
                qr_data = json.loads(qr_path.read_text(encoding="utf-8"))
                outputs = qr_data.setdefault("artifacts", {}).setdefault("outputs", {})
                for k in ["linhas_completas_v1_enhanced", "linhas_completas_v2", "linhas_incertas_v2"]:
                    p = paths.get(k)
                    if p and p.exists():
                        outputs[k] = {
                            "path": str(p),
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                            "size_bytes": p.stat().st_size,
                        }
                qr_path.write_text(json.dumps(qr_data, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

    _print_json(
        {
            "status": "completed",
            **({"multi_talhao_count": len(roi_polygons)} if is_multi_talhao else {}),
            "valid_rows": len(final_rows),
            "uncertain_rows": len(final_uncertain),
            "accepted_repairs": len(final_repairs),
            "artifacts": {name: str(path) for name, path in paths.items()},
        }
    )

@app.command()
def train(
    manifests_dir: Annotated[
        Path,
        typer.Option("--manifests-dir", help="Diretório contendo manifestos YAML/JSON de cena."),
    ] = Path("data_set/manifests"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Diretório de saída para checkpoints."),
    ] = Path("checkpoints/exp_001"),
    model_name: Annotated[
        str,
        typer.Option("--model-name", help="Modelo de backbone SegFormer (ex: nvidia/mit-b2)."),
    ] = "nvidia/mit-b2",
    in_channels: Annotated[
        int,
        typer.Option("--in-channels", help="Número de canais de entrada da imagem."),
    ] = 3,
    tile_size: Annotated[
        int,
        typer.Option("--tile-size", help="Tamanho dos recortes (patches) em pixels."),
    ] = 512,
    stride: Annotated[
        int,
        typer.Option("--stride", help="Passo de deslocamento entre recortes."),
    ] = 256,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Tamanho do lote para o DataLoader."),
    ] = 2,
    epochs: Annotated[
        int,
        typer.Option("--epochs", help="Número de épocas de treinamento."),
    ] = 10,
    lr: Annotated[
        float,
        typer.Option("--lr", help="Taxa de aprendizado inicial."),
    ] = 1e-4,
    device: Annotated[
        str,
        typer.Option("--device", help="Dispositivo para execução (cuda, cpu, auto)."),
    ] = "auto",
    seed: Annotated[
        int,
        typer.Option("--seed", help="Semente determinística."),
    ] = 42,
) -> None:
    """Treinar o modelo RowGraphNet a partir de manifestos de cenas."""
    from rowgraphnet.training.runner import train_rowgraphnet

    best_ckpt = train_rowgraphnet(
        manifests_dir=manifests_dir,
        output_dir=output_dir,
        model_name=model_name,
        in_channels=in_channels,
        tile_size=tile_size,
        stride=stride,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        device_name=device,
        seed=seed,
    )
    _print_json(
        {
            "status": "completed",
            "best_checkpoint": str(best_ckpt),
            "output_dir": str(output_dir),
        }
    )


if __name__ == "__main__":  # pragma: no cover
    app()
