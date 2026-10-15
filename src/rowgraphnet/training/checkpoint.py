"""Reproducible, inference-compatible RowGraphNet checkpoint creation."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class TrainingCheckpointError(ValueError):
    """Raised before writing a checkpoint with an incomplete provenance contract."""


def _plain_value(value: Any, *, path: str) -> Any:
    """Convert configuration objects to ``torch.load(weights_only=True)`` data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _plain_value(value.value, path=path)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_value(asdict(value), path=path)
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_value(item, path=f"{path}[]") for item in value]
    raise TrainingCheckpointError(
        f"{path} contains non-serializable value of type {type(value).__name__}"
    )


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingCheckpointError(f"{name} must be a mapping")
    converted = _plain_value(value, path=name)
    assert isinstance(converted, dict)
    return converted


def _model_input_channels(model: nn.Module) -> int | None:
    direct = getattr(model, "in_channels", None)
    if isinstance(direct, int):
        return direct
    encoder = getattr(model, "encoder", None)
    encoded = getattr(encoder, "in_channels", None)
    if isinstance(encoded, int):
        return encoded
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module.in_channels
    return None


def _encoder_configuration(model: nn.Module) -> dict[str, Any]:
    encoder = getattr(model, "encoder", None)
    config = getattr(encoder, "config", None)
    if config is None:
        raise TrainingCheckpointError(
            "cannot infer architecture.encoder_config from this model; pass a complete "
            "architecture mapping explicitly"
        )
    if isinstance(config, Mapping):
        raw = config
    elif callable(getattr(config, "to_dict", None)):
        raw = config.to_dict()
    else:
        raise TrainingCheckpointError("model encoder config must be a mapping or expose to_dict()")
    converted = _mapping(raw, name="architecture.encoder_config")
    if not converted:
        raise TrainingCheckpointError("architecture.encoder_config must not be empty")
    return converted


def _first_convolution_out_channels(module: nn.Module) -> int | None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.out_channels)
    return None


def architecture_from_model(model: nn.Module) -> dict[str, Any]:
    """Capture all settings needed by the offline CLI to rebuild RowGraphNet."""

    encoder_config = _encoder_configuration(model)
    encoder = getattr(model, "encoder", None)
    config = getattr(encoder, "config", None)
    model_name = str(getattr(config, "_name_or_path", "")).strip()
    if not model_name:
        model_name = type(getattr(encoder, "backbone", encoder)).__name__

    decoder = getattr(model, "decoder", None)
    decoder_channels = getattr(decoder, "decoder_channels", None)
    if not isinstance(decoder_channels, int) or decoder_channels < 1:
        raise TrainingCheckpointError("cannot infer positive decoder_channels from model")

    refiner = getattr(model, "connectivity_refiner", None)
    architecture: dict[str, Any] = {
        "model_name": model_name,
        "decoder_channels": decoder_channels,
        "refine_connectivity": refiner is not None,
        "encoder_config": encoder_config,
    }
    if refiner is not None:
        hidden_channels = _first_convolution_out_channels(refiner)
        if hidden_channels is None:
            raise TrainingCheckpointError("cannot infer refiner_hidden_channels from model")
        architecture["refiner_hidden_channels"] = hidden_channels
    return architecture


@dataclass(frozen=True, slots=True)
class TrainingMetadata:
    """Mandatory model, input and calibration provenance stored in every export."""

    model_version: str
    band_indexes: tuple[int, ...]
    in_channels: int
    architecture: Mapping[str, Any]
    normalization: Mapping[str, Any] = field(default_factory=dict)
    calibration: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model_version = str(self.model_version).strip()
        if not model_version:
            raise TrainingCheckpointError("model_version must not be empty")
        bands = tuple(self.band_indexes)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in bands):
            raise TrainingCheckpointError("band_indexes must contain integers")
        if not bands or any(value < 1 for value in bands) or len(set(bands)) != len(bands):
            raise TrainingCheckpointError("band_indexes must be unique positive integers")
        if isinstance(self.in_channels, bool) or not isinstance(self.in_channels, int):
            raise TrainingCheckpointError("in_channels must be an integer")
        if self.in_channels != len(bands):
            raise TrainingCheckpointError(
                "in_channels must equal the number of ordered band_indexes"
            )

        architecture = _mapping(self.architecture, name="architecture")
        encoder_config = architecture.get("encoder_config")
        if not isinstance(encoder_config, dict) or not encoder_config:
            raise TrainingCheckpointError(
                "architecture must contain a non-empty, complete encoder_config mapping"
            )
        configured_channels = encoder_config.get("num_channels", self.in_channels)
        if isinstance(configured_channels, bool) or int(configured_channels) != self.in_channels:
            raise TrainingCheckpointError(
                "architecture.encoder_config num_channels does not match in_channels"
            )
        # Always persist num_channels even when an injected test configuration
        # relied on the model constructor to add it.
        encoder_config["num_channels"] = self.in_channels

        normalization = _mapping(self.normalization, name="normalization")
        raw_calibration = _mapping(self.calibration, name="calibration")
        if any(not isinstance(config, dict) for config in raw_calibration.values()):
            raise TrainingCheckpointError("each calibration head must contain a mapping")

        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "band_indexes", bands)
        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "calibration", raw_calibration)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrainingMetadata:
        """Validate metadata supplied by Hydra, YAML or another application."""

        raw_bands = value.get("band_indexes", ())
        if not isinstance(raw_bands, Sequence) or isinstance(raw_bands, (str, bytes)):
            raise TrainingCheckpointError("band_indexes must be a sequence of integers")
        return cls(
            model_version=str(value.get("model_version", "")),
            band_indexes=tuple(raw_bands),
            in_channels=value.get("in_channels", 0),
            normalization=value.get("normalization", {}),
            calibration=value.get("calibration", {}),
            architecture=value.get("architecture", {}),
        )

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        model_version: str,
        band_indexes: Sequence[int],
        normalization: Mapping[str, Any] | None = None,
        calibration: Mapping[str, Mapping[str, Any]] | None = None,
        architecture: Mapping[str, Any] | None = None,
    ) -> TrainingMetadata:
        """Build validated metadata, serializing the complete encoder config."""

        bands = tuple(band_indexes)
        channels = _model_input_channels(model)
        if channels is None:
            channels = len(bands)
        return cls(
            model_version=model_version,
            band_indexes=bands,
            in_channels=channels,
            normalization=normalization or {},
            calibration=calibration or {},
            architecture=architecture or architecture_from_model(model),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh plain-data mapping suitable for safe torch loading."""

        return {
            "model_version": self.model_version,
            "band_indexes": list(self.band_indexes),
            "in_channels": self.in_channels,
            "normalization": _plain_value(self.normalization, path="normalization"),
            "calibration": _plain_value(self.calibration, path="calibration"),
            "architecture": _plain_value(self.architecture, path="architecture"),
        }


def build_checkpoint_metadata(
    model: nn.Module,
    *,
    model_version: str,
    band_indexes: Sequence[int],
    normalization: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Mapping[str, Any]] | None = None,
    architecture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Functional shortcut returning validated checkpoint metadata."""

    return TrainingMetadata.from_model(
        model,
        model_version=model_version,
        band_indexes=band_indexes,
        normalization=normalization,
        calibration=calibration,
        architecture=architecture,
    ).to_dict()


def _cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def build_checkpoint_payload(
    model: nn.Module,
    metadata: TrainingMetadata | Mapping[str, Any],
    *,
    criterion: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    scaler: Any | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a resumable payload whose ``state_dict`` is inference-compatible."""

    validated = (
        metadata
        if isinstance(metadata, TrainingMetadata)
        else TrainingMetadata.from_mapping(metadata)
    )
    model_channels = _model_input_channels(model)
    if model_channels is not None and model_channels != validated.in_channels:
        raise TrainingCheckpointError(
            f"model expects {model_channels} channels but metadata declares {validated.in_channels}"
        )
    for name, value in (("epoch", epoch), ("global_step", global_step)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise TrainingCheckpointError(f"{name} must be a non-negative integer")

    payload: dict[str, Any] = {
        "format_version": 1,
        "state_dict": _cpu_state_dict(model),
        "metadata": validated.to_dict(),
    }
    training_state: dict[str, Any] = {}
    if criterion is not None:
        training_state["criterion_state_dict"] = _cpu_state_dict(criterion)
    if optimizer is not None:
        training_state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        training_state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        training_state["scaler_state_dict"] = scaler.state_dict()
    if epoch is not None:
        training_state["epoch"] = epoch
    if global_step is not None:
        training_state["global_step"] = global_step
    if extra_state is not None:
        training_state["extra_state"] = dict(extra_state)
    if training_state:
        payload["training_state"] = training_state
    return payload


def save_training_checkpoint(
    path: str | PathLike[str],
    model: nn.Module,
    metadata: TrainingMetadata | Mapping[str, Any],
    *,
    criterion: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    scaler: Any | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    extra_state: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically save a local checkpoint after validating all metadata."""

    raw_path = str(path)
    if "://" in raw_path:
        raise TrainingCheckpointError("checkpoint path must be local, not a URL")
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_checkpoint_payload(
        model,
        metadata,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        global_step=global_step,
        extra_state=extra_state,
    )
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_training_checkpoint(
    path: str | PathLike[str],
    model: nn.Module,
    *,
    criterion: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    scaler: Any | None = None,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint payload, restore weights and training state, and return metadata/info."""
    destination = Path(path).expanduser().resolve()
    if not destination.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {destination}")
    payload = torch.load(destination, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise TrainingCheckpointError(f"invalid checkpoint format in {destination}")

    # Load model weights
    model.load_state_dict(payload["state_dict"])

    training_state = payload.get("training_state", {})
    if criterion is not None and "criterion_state_dict" in training_state:
        criterion.load_state_dict(training_state["criterion_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in training_state:
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in training_state:
        scheduler.load_state_dict(training_state["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in training_state and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(training_state["scaler_state_dict"])

    return {
        "metadata": payload.get("metadata", {}),
        "epoch": training_state.get("epoch"),
        "global_step": training_state.get("global_step"),
        "extra_state": training_state.get("extra_state", {}),
        "format_version": payload.get("format_version", 1),
    }


# Backwards-friendly concise aliases.
CheckpointMetadata = TrainingMetadata
save_checkpoint = save_training_checkpoint
load_checkpoint = load_training_checkpoint


__all__ = [
    "CheckpointMetadata",
    "TrainingCheckpointError",
    "TrainingMetadata",
    "architecture_from_model",
    "build_checkpoint_metadata",
    "build_checkpoint_payload",
    "load_checkpoint",
    "load_training_checkpoint",
    "save_checkpoint",
    "save_training_checkpoint",
]
