"""Production training runner with strict group validation and resumability."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from rowgraphnet.data.dataset import RowGraphNetDataModule
from rowgraphnet.data.manifest import load_scene_manifests
from rowgraphnet.losses import RowGraphNetLoss
from rowgraphnet.models.rowgraphnet import RowGraphNet
from rowgraphnet.resources import DEFAULT_RESOURCE_FRACTION, apply_resource_caps
from rowgraphnet.training.checkpoint import (
    TrainingMetadata,
    architecture_from_model,
    load_training_checkpoint,
    save_training_checkpoint,
)
from rowgraphnet.training.core import RowGraphNetTrainingCore
from rowgraphnet.training.optim import AdamWConfig, SchedulerConfig
from rowgraphnet.training.readiness import verify_training_readiness
from rowgraphnet.training.reproducibility import seed_everything


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Typed effective configuration shared by YAML and command-line entrypoints."""

    manifests_dir: str = "data_set/manifests"
    output_dir: str = "checkpoints/rowgraphnet"
    model_name: str = "nvidia/mit-b2"
    in_channels: int = 3
    decoder_channels: int = 128
    refine_connectivity: bool = False
    tile_size: int = 256
    stride: int = 256
    batch_size: int = 2
    grad_accum_steps: int = 1
    num_workers: int = 0
    pin_memory: bool | None = None
    resource_fraction: float = DEFAULT_RESOURCE_FRACTION
    working_dir: str = "data_set/_working"
    min_valid_pixels: int = 10
    normalization: dict[str, Any] | None = None
    centerline_buffer_m: float | None = None
    proxy_band_width_m: float | None = None
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-2
    device_name: str = "auto"
    seed: int = 42
    pretrained: bool = True
    resume: str | None = None
    patience: int = 5
    no_amp: bool = False
    max_gradient_norm: float = 1.0
    scheduler_name: str = "cosine"
    save_best: bool = True
    save_final: bool = True
    run_metadata: dict[str, Any] = field(default_factory=dict)


_SECTION_FIELDS: dict[str, dict[str, str]] = {
    "hardware": {
        "device": "device_name",
        "use_amp": "use_amp",
        "num_workers": "num_workers",
        "pin_memory": "pin_memory",
        "resource_fraction": "resource_fraction",
    },
    "model": {
        "model_name": "model_name",
        "in_channels": "in_channels",
        "decoder_channels": "decoder_channels",
        "refine_connectivity": "refine_connectivity",
        "pretrained": "pretrained",
    },
    "dataset": {
        "manifests_dir": "manifests_dir",
        "working_dir": "working_dir",
        "tile_size": "tile_size",
        "stride": "stride",
        "min_valid_pixels": "min_valid_pixels",
        "normalization": "normalization",
        "centerline_buffer_m": "centerline_buffer_m",
        "proxy_band_width_m": "proxy_band_width_m",
    },
    "training": {
        "batch_size": "batch_size",
        "grad_accum_steps": "grad_accum_steps",
        "epochs": "epochs",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "optimizer": "optimizer",
        "patience": "patience",
        "max_gradient_norm": "max_gradient_norm",
        "scheduler": "scheduler_name",
    },
    "checkpoint": {
        "output_dir": "output_dir",
        "save_best": "save_best",
        "save_final": "save_final",
        "resume": "resume",
    },
}


def load_training_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> TrainingConfig:
    """Load strict hierarchical YAML and apply explicit CLI overrides last."""

    values: dict[str, Any] = {}
    run_metadata: dict[str, Any] = {}
    if path is not None:
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("training configuration root must be a mapping")
        unknown_sections = set(raw) - ({"experiment"} | set(_SECTION_FIELDS))
        if unknown_sections:
            raise ValueError(f"unknown configuration sections: {sorted(unknown_sections)}")

        experiment = raw.get("experiment", {})
        if not isinstance(experiment, dict):
            raise ValueError("experiment must be a mapping")
        unknown_experiment = set(experiment) - {"name", "description", "source", "seed"}
        if unknown_experiment:
            raise ValueError(f"unknown experiment settings: {sorted(unknown_experiment)}")
        run_metadata = {
            key: value for key, value in experiment.items() if key in {"name", "description", "source"}
        }
        if "seed" in experiment:
            values["seed"] = experiment["seed"]

        for section, mapping in _SECTION_FIELDS.items():
            section_value = raw.get(section, {})
            if not isinstance(section_value, dict):
                raise ValueError(f"{section} must be a mapping")
            unknown = set(section_value) - set(mapping)
            if unknown:
                raise ValueError(f"unknown {section} settings: {sorted(unknown)}")
            for source_name, value in section_value.items():
                destination = mapping[source_name]
                if destination == "use_amp":
                    values["no_amp"] = not bool(value)
                elif destination == "optimizer":
                    if str(value).casefold() != "adamw":
                        raise ValueError("only optimizer='adamw' is supported")
                else:
                    values[destination] = value

    values["run_metadata"] = run_metadata
    if overrides:
        unknown_overrides = set(overrides) - {item.name for item in TrainingConfig.__dataclass_fields__.values()}
        if unknown_overrides:
            raise ValueError(f"unknown CLI overrides: {sorted(unknown_overrides)}")
        values.update(overrides)
    return TrainingConfig(**values)


def accumulation_group_size(batch_index: int, total_batches: int, accumulation_steps: int) -> int:
    """Return the true divisor for a gradient-accumulation group."""

    if not 0 <= batch_index < total_batches:
        raise ValueError("batch_index must address an existing batch")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, total_batches - group_start)


def train_rowgraphnet(
    manifests_dir: str | Path = "data_set/manifests",
    output_dir: str | Path = "checkpoints/exp_matologia_rtx3050",
    model_name: str = "nvidia/mit-b2",
    in_channels: int = 3,
    decoder_channels: int = 128,
    refine_connectivity: bool = False,
    tile_size: int = 256,
    stride: int = 256,
    batch_size: int = 2,
    grad_accum_steps: int = 1,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    resource_fraction: float = DEFAULT_RESOURCE_FRACTION,
    working_dir: str | Path = "data_set/_working",
    min_valid_pixels: int = 10,
    normalization: dict[str, Any] | None = None,
    centerline_buffer_m: float | None = None,
    proxy_band_width_m: float | None = None,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    device_name: str = "auto",
    seed: int = 42,
    pretrained: bool = True,
    resume: str | Path | None = None,
    patience: int = 5,
    no_amp: bool = False,
    max_gradient_norm: float = 1.0,
    scheduler_name: str = "cosine",
    save_best: bool = True,
    save_final: bool = True,
    run_metadata: dict[str, Any] | None = None,
) -> Path:
    """Train RowGraphNet with strict data contracts and resumable state."""
    for name, value in {
        "in_channels": in_channels,
        "decoder_channels": decoder_channels,
        "tile_size": tile_size,
        "stride": stride,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "epochs": epochs,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if stride > tile_size:
        raise ValueError("stride must not exceed tile_size")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
        raise ValueError("patience must be a non-negative integer")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(max_gradient_norm) or max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be finite and positive")
    if centerline_buffer_m is not None and (
        not math.isfinite(float(centerline_buffer_m)) or centerline_buffer_m <= 0
    ):
        raise ValueError("centerline_buffer_m must be finite and positive")
    # 1. Pipeline Readiness pre-flight verification
    verify_training_readiness(
        manifests_dir=manifests_dir,
        output_dir=output_dir,
        device_name=device_name,
    )

    seed_everything(seed)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine compute device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    apply_resource_caps(fraction=resource_fraction, device=device)
    print(f"[RowGraphNet Train] Using device: {device}", flush=True)

    gpu_total_gb = 0.0
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_props = torch.cuda.get_device_properties(device)
        gpu_total_gb = gpu_props.total_memory / (1024**3)
        print(f"[RowGraphNet Train] GPU: {gpu_name} ({gpu_total_gb:.2f} GB VRAM)", flush=True)
        if gpu_total_gb < 5.0:
            print("[RowGraphNet Train] Low-VRAM GPU detected; monitor the measured peak.", flush=True)

    use_amp = (device.type == "cuda") and not no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[RowGraphNet Train] Mixed precision AMP enabled (GradScaler).", flush=True)

    # Load manifests
    manifest_dir_path = Path(manifests_dir).resolve()
    manifest_files = sorted(list(manifest_dir_path.glob("*.yaml")) + list(manifest_dir_path.glob("*.json")))

    if not manifest_files:
        raise FileNotFoundError(f"No manifest files found in {manifest_dir_path}")

    all_manifests = load_scene_manifests(manifest_files)
    train_manifests = [m for m in all_manifests if m.split == "train"]
    val_manifests = [m for m in all_manifests if m.split == "validation"]
    test_manifests = [m for m in all_manifests if m.split == "test"]

    if not train_manifests:
        raise ValueError(f"No manifests assigned to split 'train' in {manifest_dir_path}")
    if not val_manifests:
        raise ValueError(f"No manifests assigned to split 'validation' in {manifest_dir_path}")

    train_groups = {m.group_id for m in train_manifests}
    val_groups = {m.group_id for m in val_manifests}
    leakage = train_groups & val_groups
    if leakage:
        raise ValueError(f"Data leakage detected! Groups present in both train and validation: {sorted(leakage)}")
    band_contracts = {m.bands for m in (*train_manifests, *val_manifests)}
    if len(band_contracts) != 1:
        raise ValueError(f"All train/validation scenes must use the same ordered bands: {band_contracts}")
    band_indexes = next(iter(band_contracts))
    if len(band_indexes) != in_channels:
        raise ValueError(
            f"in_channels={in_channels} does not match manifest band contract {band_indexes}"
        )

    print(
        f"[RowGraphNet Train] Loaded {len(train_manifests)} train scenes ({len(train_groups)} groups) "
        f"and {len(val_manifests)} validation scenes ({len(val_groups)} groups).",
        flush=True,
    )
    if test_manifests:
        print(f"[RowGraphNet Train] {len(test_manifests)} test scenes reserved (untouched during training).", flush=True)

    # Build DataModule
    data_module = RowGraphNetDataModule(
        train_manifests=train_manifests,
        val_manifests=val_manifests,
        tile_size=tile_size,
        stride=stride,
        batch_size=batch_size,
        num_workers=num_workers,
        working_dir=working_dir,
        min_valid_pixels=min_valid_pixels,
        normalization=normalization,
        centerline_buffer_m=centerline_buffer_m,
        proxy_band_width_m=proxy_band_width_m,
        pin_memory=pin_memory,
        seed=seed,
    )

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"[RowGraphNet Train] Total training patches: {len(data_module.train_dataset)}", flush=True)
    print(f"[RowGraphNet Train] Total validation patches: {len(data_module.val_dataset)}", flush=True)
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("training and validation DataLoaders must both contain at least one batch")

    train_scenes = data_module.train_dataset.loaded_scenes
    has_visibility = any(scene.has_visibility_labels for scene in train_scenes)
    has_occluded = any(
        scene.has_visibility_labels
        and scene.annotations[scene.manifest.visibility.field]
        .astype(str)
        .str.casefold()
        .isin({value.casefold() for value in scene.manifest.visibility.occluded_values})
        .any()
        for scene in train_scenes
    )
    head_supervision = {
        "center": "direct",
        "orientation": "derived_from_centerline",
        "distance": "derived_from_centerline",
        "uncertainty": "derived_from_distance_error",
        "band": "metric_proxy" if proxy_band_width_m is not None else "unavailable",
        "endpoints": "weak_geometry_proxy" if has_visibility else "unavailable",
        "repair": "direct" if has_occluded else "unavailable",
    }

    # Instantiate Model & Loss
    model = RowGraphNet(
        model_name=model_name,
        pretrained=pretrained,
        in_channels=in_channels,
        decoder_channels=decoder_channels,
        refine_connectivity=refine_connectivity,
    )

    criterion = RowGraphNetLoss()

    optimizer_config = AdamWConfig(learning_rate=lr, weight_decay=weight_decay)
    scheduler_config = SchedulerConfig(
        name=scheduler_name,
        interval="epoch",
        total_steps=epochs,
    )
    training_core = RowGraphNetTrainingCore(
        model=model,
        criterion=criterion,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
    )
    training_core.to(device)

    optimizer_setup = training_core.configure_optimizers()

    best_val_loss = float("inf")
    start_epoch = 1
    epochs_without_improvement = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    best_ckpt_path = output_path / "best_model.ckpt"
    final_ckpt_path = output_path / "final_model.ckpt"

    # Handle resume
    if resume is not None:
        resume_file = Path(resume).resolve()
        if not resume_file.is_file():
            raise FileNotFoundError(f"Checkpoint to resume from not found: {resume_file}")
        print(f"[RowGraphNet Train] Resuming from checkpoint: {resume_file}", flush=True)
        resume_data = load_training_checkpoint(
            resume_file,
            model=model,
            criterion=criterion,
            optimizer=optimizer_setup.optimizer,
            scheduler=optimizer_setup.scheduler,
            scaler=scaler if use_amp else None,
            map_location=device,
        )
        if resume_data.get("epoch") is not None:
            start_epoch = int(resume_data["epoch"]) + 1
        extra = resume_data.get("extra_state", {})
        if "best_val_loss" in extra:
            best_val_loss = float(extra["best_val_loss"])
        epochs_without_improvement = int(extra.get("epochs_without_improvement", 0))
        global_step = int(resume_data.get("global_step") or 0)
        history_path = output_path / "history.json"
        if history_path.is_file():
            loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_history, list):
                raise ValueError("history.json must contain a list when resuming")
            history = loaded_history
        print(f"[RowGraphNet Train] Resumed at epoch {start_epoch}, previous best_val_loss: {best_val_loss:.4f}", flush=True)

    # Save run config
    effective_batch_size = batch_size * grad_accum_steps
    run_config = {
        "model_name": model_name,
        "in_channels": in_channels,
        "band_indexes": list(band_indexes),
        "decoder_channels": decoder_channels,
        "refine_connectivity": refine_connectivity,
        "tile_size": tile_size,
        "stride": stride,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": effective_batch_size,
        "num_workers": num_workers,
        "pin_memory": data_module.pin_memory,
        "working_dir": str(Path(working_dir)),
        "min_valid_pixels": min_valid_pixels,
        "centerline_buffer_m": centerline_buffer_m,
        "centerline_target_width_m": (
            None if centerline_buffer_m is None else 2.0 * centerline_buffer_m
        ),
        "proxy_band_width_m": proxy_band_width_m,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_gradient_norm": max_gradient_norm,
        "scheduler_name": scheduler_name,
        "device": str(device),
        "seed": seed,
        "pretrained": pretrained,
        "save_best": save_best,
        "save_final": save_final,
        "run_metadata": dict(run_metadata or {}),
        "head_supervision": head_supervision,
        "train_scenes": [m.scene_id for m in train_manifests],
        "val_scenes": [m.scene_id for m in val_manifests],
        "normalization": data_module.train_dataset.normalization,
    }
    with open(output_path / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print(
        f"Starting Training ({epochs} epochs, starting from {start_epoch}) | "
        f"Batch Size: {batch_size} (Effective: {effective_batch_size})",
        flush=True,
    )
    print("=" * 60, flush=True)

    start_time = time.time()
    last_epoch = start_epoch - 1
    avg_val_loss = float("nan")

    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        epoch_start = time.time()
        training_core.train()
        # Augmentation and shuffling vary by epoch but remain reproducible when
        # resuming at the same epoch.
        train_loader = data_module.train_dataloader(epoch=epoch)
        num_train_batches = len(train_loader)

        train_total_loss = 0.0
        train_batches = 0
        train_task_losses: dict[str, float] = {}

        optimizer_setup.optimizer.zero_grad(set_to_none=True)

        max_loader_retries = 3
        loader_attempts = 0
        train_iter = iter(train_loader)

        batch_idx = 0
        while batch_idx < num_train_batches:
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            except (SystemError, RuntimeError, OSError) as exc:
                loader_attempts += 1
                print(
                    f"  [WARNING] DataLoader worker error on batch {batch_idx} "
                    f"(attempt {loader_attempts}/{max_loader_retries}): {exc}",
                    flush=True,
                )
                if loader_attempts > max_loader_retries:
                    raise
                # Recreate the DataLoader and skip already-processed batches
                train_loader = data_module.train_dataloader(epoch=epoch)
                num_train_batches = len(train_loader)
                train_iter = iter(train_loader)
                for _ in range(batch_idx):
                    next(train_iter, None)
                continue

            batch_images = batch["image"].to(device)
            batch_targets = {k: v.to(device) for k, v in batch["targets"].items()}

            if grad_accum_steps <= 1:
                step_output = training_core.optimize_batch(
                    {"image": batch_images, "targets": batch_targets},
                    optimizer_setup=optimizer_setup,
                    max_gradient_norm=max_gradient_norm,
                    scaler=scaler if use_amp else None,
                    autocast_device="cuda" if use_amp else None,
                )
                global_step += 1
            else:
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        step_output = training_core.training_step(
                            {"image": batch_images, "targets": batch_targets}
                        )
                    group_size = accumulation_group_size(
                        batch_idx, num_train_batches, grad_accum_steps
                    )
                    scaled_loss = step_output.loss / group_size
                    scaler.scale(scaled_loss).backward()
                else:
                    step_output = training_core.training_step(
                        {"image": batch_images, "targets": batch_targets}
                    )
                    group_size = accumulation_group_size(
                        batch_idx, num_train_batches, grad_accum_steps
                    )
                    scaled_loss = step_output.loss / group_size
                    scaled_loss.backward()

                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_train_batches:
                    if use_amp:
                        scaler.unscale_(optimizer_setup.optimizer)
                        torch.nn.utils.clip_grad_norm_(training_core.parameters(), max_gradient_norm)
                        scaler.step(optimizer_setup.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(training_core.parameters(), max_gradient_norm)
                        optimizer_setup.optimizer.step()
                    optimizer_setup.optimizer.zero_grad(set_to_none=True)
                    if optimizer_setup.scheduler is not None and optimizer_setup.interval == "step":
                        optimizer_setup.step_scheduler()
                    global_step += 1

            train_total_loss += step_output.loss.item()
            train_batches += 1
            for task_name, task_val in step_output.losses.items():
                if task_name != "total":
                    train_task_losses[task_name] = train_task_losses.get(task_name, 0.0) + task_val.item()
            batch_idx += 1

        avg_train_loss = train_total_loss / max(1, train_batches)
        avg_train_tasks = {k: v / max(1, train_batches) for k, v in train_task_losses.items()}

        # Free training cache before validation to keep 4GB VRAM clean
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Validation loop
        training_core.eval()
        val_total_loss = 0.0
        val_batches = 0
        val_task_losses: dict[str, float] = {}

        with torch.no_grad():
            for batch in val_loader:
                batch_images = batch["image"].to(device)
                batch_targets = {k: v.to(device) for k, v in batch["targets"].items()}

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        val_output = training_core.validation_step(
                            {"image": batch_images, "targets": batch_targets}
                        )
                else:
                    val_output = training_core.validation_step(
                        {"image": batch_images, "targets": batch_targets}
                    )

                val_total_loss += val_output.loss.item()
                val_batches += 1
                for task_name, task_val in val_output.losses.items():
                    if task_name != "total":
                        val_task_losses[task_name] = val_task_losses.get(task_name, 0.0) + task_val.item()

        avg_val_loss = val_total_loss / max(1, val_batches)
        avg_val_tasks = {k: v / max(1, val_batches) for k, v in val_task_losses.items()}

        if optimizer_setup.scheduler is not None and optimizer_setup.interval == "epoch":
            metric = avg_val_loss if scheduler_name == "plateau" else None
            optimizer_setup.step_scheduler(metric)

        elapsed = time.time() - epoch_start

        # VRAM monitoring and cache cleanup
        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)
            vram_str = f" | Peak VRAM: {peak_vram:.2f} GB"
            torch.cuda.empty_cache()
        else:
            vram_str = ""

        # Detailed logging
        task_str = " | ".join(
            f"{k}: {v:.3f}"
            for k, v in sorted(avg_val_tasks.items())
            if k in ("center", "topology", "orientation", "endpoints", "distance", "band")
        )
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s{vram_str}] - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} ({task_str})",
            flush=True,
        )

        epoch_record = {
            "epoch": epoch,
            "elapsed_seconds": round(elapsed, 2),
            "train_loss": round(avg_train_loss, 5),
            "val_loss": round(avg_val_loss, 5),
            "train_tasks": {k: round(v, 5) for k, v in avg_train_tasks.items()},
            "val_tasks": {k: round(v, 5) for k, v in avg_val_tasks.items()},
        }
        history.append(epoch_record)
        with open(output_path / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            architecture = architecture_from_model(model)
            architecture["head_supervision"] = head_supervision
            metadata = TrainingMetadata.from_model(
                model,
                model_version=f"rowgraphnet-{model_name.replace('/', '_')}-best",
                band_indexes=band_indexes,
                normalization=data_module.train_dataset.normalization,
                architecture=architecture,
            )
            if save_best:
                save_training_checkpoint(
                    best_ckpt_path,
                    model,
                    metadata,
                    criterion=criterion,
                    optimizer=optimizer_setup.optimizer,
                    scheduler=optimizer_setup.scheduler,
                    scaler=scaler if use_amp else None,
                    epoch=epoch,
                    global_step=global_step,
                    overwrite=True,
                    extra_state={
                        "best_val_loss": best_val_loss,
                        "epoch": epoch,
                        "epochs_without_improvement": 0,
                    },
                )
                print(f"   --> Saved new best checkpoint to {best_ckpt_path.name} (Val Loss: {best_val_loss:.4f})", flush=True)
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"\n[RowGraphNet Train] Early stopping triggered after {patience} epochs without improvement.", flush=True)
                break

    # Save final model
    if last_epoch < start_epoch:
        raise ValueError(
            f"resume checkpoint is already at or beyond requested epochs={epochs}"
        )
    final_architecture = architecture_from_model(model)
    final_architecture["head_supervision"] = head_supervision
    final_metadata = TrainingMetadata.from_model(
        model,
        model_version=f"rowgraphnet-{model_name.replace('/', '_')}-final",
        band_indexes=band_indexes,
        normalization=data_module.train_dataset.normalization,
        architecture=final_architecture,
    )
    if save_final:
        save_training_checkpoint(
            final_ckpt_path,
            model,
            final_metadata,
            criterion=criterion,
            optimizer=optimizer_setup.optimizer,
            scheduler=optimizer_setup.scheduler,
            scaler=scaler if use_amp else None,
            epoch=last_epoch,
            global_step=global_step,
            overwrite=True,
            extra_state={
                "best_val_loss": best_val_loss,
                "last_val_loss": avg_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
            },
        )

    total_elapsed = time.time() - start_time
    print("=" * 60)
    print(f"Training completed in {total_elapsed / 60:.2f} minutes.")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Checkpoint:  {best_ckpt_path}")
    print(f"Final Checkpoint: {final_ckpt_path}")
    print("=" * 60)

    if save_best and best_ckpt_path.is_file():
        return best_ckpt_path
    if save_final and final_ckpt_path.is_file():
        return final_ckpt_path
    raise RuntimeError("training completed without a checkpoint because both save flags are false")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RowGraphNet multitask neural network.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--config", type=str, help="Path to strict YAML training configuration")
    parser.add_argument("--manifests-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--in-channels", type=int)
    parser.add_argument("--decoder-channels", type=int)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--working-dir", type=str)
    parser.add_argument("--min-valid-pixels", type=int)
    parser.add_argument(
        "--centerline-buffer-m",
        type=float,
        help="Per-side centerline target buffer in metres (0.10 means 20 cm total width)",
    )
    parser.add_argument("--proxy-band-width-m", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--device", dest="device_name", type=str)
    parser.add_argument("--resource-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max-gradient-norm", type=float)
    parser.add_argument("--scheduler", dest="scheduler_name", choices=("none", "cosine", "step", "plateau"))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-save-best", action="store_true")
    parser.add_argument("--no-save-final", action="store_true")

    parsed = vars(parser.parse_args())
    config_path = parsed.pop("config", None)
    if parsed.pop("no_pretrained", False):
        parsed["pretrained"] = False
    if parsed.pop("no_save_best", False):
        parsed["save_best"] = False
    if parsed.pop("no_save_final", False):
        parsed["save_final"] = False
    config = load_training_config(config_path, overrides=parsed)
    train_rowgraphnet(**asdict(config))


if __name__ == "__main__":
    main()
