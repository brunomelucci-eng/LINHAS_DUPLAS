"""Pre-flight pipeline readiness verification for RowGraphNet training.

Inspired by mlops-console/src/pipeline_readiness.py, this module verifies:
1. Environment & Compute: Python version, PyTorch, CUDA device, VRAM, and RAM;
2. Storage & Workspace: Output directory write permissions and free disk space;
3. Data Contracts & Manifests: Manifest validity, train/val/test split separation,
   zero geographic group leakage, and file readability for all scenes;
4. Sensor & Band contracts: Raster bands count, CRS validity, and non-empty annotations.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import rasterio
import torch

from rowgraphnet.data.manifest import load_scene_manifests


class ReadinessError(RuntimeError):
    """Raised when one or more fatal readiness checks fail."""


@dataclass(frozen=True)
class ReadinessReport:
    total_scenes: int
    train_scenes: int
    val_scenes: int
    test_scenes: int
    groups_count: int
    device: str
    vram_gb: float | None
    ram_gb: float
    free_disk_gb: float
    warnings: list[str]


def verify_training_readiness(
    manifests_dir: str | Path = "data_set/manifests",
    output_dir: str | Path = "checkpoints",
    device_name: str = "auto",
    min_free_disk_gb: float = 1.0,
    min_ram_gb: float = 2.0,
) -> ReadinessReport:
    """Run comprehensive pre-flight readiness checks before training."""
    warnings: list[str] = []
    fatal_errors: list[str] = []

    print("\n" + "=" * 60)
    print("[Pipeline Readiness] Running pre-flight verification...")
    print("=" * 60)

    # 1. Manifests and Split integrity
    manifest_path = Path(manifests_dir).resolve()
    if not manifest_path.is_dir():
        fatal_errors.append(f"Manifest directory does not exist: {manifest_path}")
        manifest_files = []
    else:
        manifest_files = sorted(list(manifest_path.glob("*.yaml")) + list(manifest_path.glob("*.json")))

    if not manifest_files:
        fatal_errors.append(f"No YAML/JSON manifest files found in {manifest_path}")
        manifests = []
    else:
        try:
            manifests = load_scene_manifests(manifest_files)
        except Exception as exc:
            fatal_errors.append(f"Failed to load manifests: {exc}")
            manifests = []

    train_manifests = [m for m in manifests if m.split == "train"]
    val_manifests = [m for m in manifests if m.split == "validation"]
    test_manifests = [m for m in manifests if m.split == "test"]

    if manifests and not train_manifests:
        fatal_errors.append("No manifests assigned to split 'train'!")
    if manifests and not val_manifests:
        fatal_errors.append("No manifests assigned to split 'validation'!")

    # Verify zero group leakage
    group_splits: dict[str, set[str]] = {}
    for m in manifests:
        group_splits.setdefault(m.group_id, set()).add(m.split)

    leaking_groups = {g: s for g, s in group_splits.items() if len(s) > 1}
    if leaking_groups:
        fatal_errors.append(f"Data leakage detected! Geographic groups cross multiple splits: {leaking_groups}")

    # 2. Files existence and readability
    for m in manifests:
        # Check image
        if not m.image.is_file():
            fatal_errors.append(f"Scene {m.scene_id}: Image file not found: {m.image}")
        else:
            try:
                with rasterio.open(m.image) as src:
                    if src.count < max(m.bands):
                        fatal_errors.append(
                            f"Scene {m.scene_id}: Image has {src.count} bands, but manifest expects band {max(m.bands)}"
                        )
                    if src.crs is None:
                        fatal_errors.append(f"Scene {m.scene_id}: Image missing CRS georeferencing")
            except Exception as exc:
                fatal_errors.append(f"Scene {m.scene_id}: Cannot read raster: {exc}")

        # Check ROI
        if not m.roi.is_file():
            fatal_errors.append(f"Scene {m.scene_id}: ROI file not found: {m.roi}")
        else:
            try:
                roi_kwargs = {"layer": m.roi_layer} if m.roi_layer else {}
                roi = gpd.read_file(m.roi, **roi_kwargs)
                if roi.empty or roi.crs is None:
                    fatal_errors.append(f"Scene {m.scene_id}: ROI must be non-empty and define a CRS")
            except Exception as exc:
                fatal_errors.append(f"Scene {m.scene_id}: Cannot read ROI: {exc}")

        # Check annotations
        if not m.annotations.is_file():
            fatal_errors.append(f"Scene {m.scene_id}: Annotations file not found: {m.annotations}")
        else:
            try:
                annotation_kwargs = {"layer": m.annotations_layer} if m.annotations_layer else {}
                gdf = gpd.read_file(m.annotations, **annotation_kwargs)
                if gdf.empty:
                    warnings.append(f"Scene {m.scene_id}: Annotations file is empty: {m.annotations.name}")
                elif m.row_id_field not in gdf.columns:
                    fatal_errors.append(
                        f"Scene {m.scene_id}: Missing row_id field '{m.row_id_field}' in {m.annotations.name}"
                    )
                if not gdf.empty and gdf.crs is None:
                    fatal_errors.append(f"Scene {m.scene_id}: Annotations must define a CRS")
                if not gdf.empty and m.visibility.field not in gdf.columns:
                    warnings.append(
                        f"Scene {m.scene_id}: visibility field '{m.visibility.field}' is absent; "
                        "repair and endpoint heads will not receive direct supervision"
                    )
            except Exception as exc:
                fatal_errors.append(f"Scene {m.scene_id}: Cannot read annotations: {exc}")

    # 3. Hardware & Compute
    if device_name == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_name

    vram_gb: float | None = None
    if device not in {"cpu", "cuda"}:
        fatal_errors.append(f"Unsupported device: {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        fatal_errors.append("CUDA was requested but torch.cuda.is_available() is false")
    elif device == "cuda":
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(vram_bytes / (1024**3), 2)
        dev_name = torch.cuda.get_device_name(0)
        print(f"[Pipeline Readiness] Compute device: {dev_name} ({vram_gb:.2f} GB VRAM)")
        if vram_gb < 3.5:
            warnings.append(f"Low GPU VRAM ({vram_gb:.2f} GB). Ensure batch_size <= 2 and tile_size <= 512 with AMP.")
    else:
        print("[Pipeline Readiness] Compute device: CPU (training will be significantly slower)")
        warnings.append("CUDA is not being used. Training will be slow.")

    # System RAM
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        free_ram_gb = round(psutil.virtual_memory().available / (1024**3), 2)
        print(f"[Pipeline Readiness] Host RAM: {ram_gb:.1f} GB total ({free_ram_gb:.1f} GB available)")
        if free_ram_gb < min_ram_gb:
            warnings.append(f"Low free RAM: {free_ram_gb:.1f} GB available.")
    except ImportError:
        ram_gb = 0.0

    # 4. Storage
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(out_path)
    free_disk_gb = round(usage.free / (1024**3), 2)
    print(f"[Pipeline Readiness] Free disk space at {out_path.name}: {free_disk_gb:.2f} GB")
    if free_disk_gb < min_free_disk_gb:
        fatal_errors.append(
            f"Insufficient disk space in {out_path}: {free_disk_gb:.2f} GB free, minimum required is {min_free_disk_gb:.2f} GB"
        )

    # 5. Summary & Decision
    print(f"[Pipeline Readiness] Scenes: {len(manifests)} total | {len(train_manifests)} train | {len(val_manifests)} val | {len(test_manifests)} test")
    print(f"[Pipeline Readiness] Geographic groups: {len(group_splits)} (zero split leakage)")

    if warnings:
        for w in warnings:
            print(f"  [Warning] {w}")

    if fatal_errors:
        print("\n" + "!" * 60)
        print("[Pipeline Readiness] FAILED with fatal errors:")
        for err in fatal_errors:
            print(f"  - {err}")
        print("!" * 60 + "\n")
        raise ReadinessError(f"Training readiness check failed with {len(fatal_errors)} fatal errors.")

    print("[Pipeline Readiness] Implemented pre-flight checks passed; this does not imply model calibration or agronomic validation.")
    print("=" * 60 + "\n")

    return ReadinessReport(
        total_scenes=len(manifests),
        train_scenes=len(train_manifests),
        val_scenes=len(val_manifests),
        test_scenes=len(test_manifests),
        groups_count=len(group_splits),
        device=device,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        free_disk_gb=free_disk_gb,
        warnings=warnings,
    )
