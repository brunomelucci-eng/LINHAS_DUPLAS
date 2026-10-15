"""Strict local checkpoint loading and band/normalization provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from rowgraphnet.inference.calibration import ProbabilityCalibrator, calibrator_from_config


class CheckpointContractError(RuntimeError):
    """Raised when checkpoint contents or input-band provenance are unsafe."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Metadata required to reproduce the checkpoint's input contract."""

    model_version: str
    band_indexes: tuple[int, ...]
    in_channels: int
    normalization: Mapping[str, Any] = field(default_factory=dict)
    calibration: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_mapping(cls, metadata: Mapping[str, Any]) -> CheckpointMetadata:
        model_version = str(metadata.get("model_version", "")).strip()
        if not model_version:
            raise CheckpointContractError("checkpoint metadata must define model_version")
        raw_bands = metadata.get("band_indexes", metadata.get("bands"))
        if not isinstance(raw_bands, Sequence) or isinstance(raw_bands, (str, bytes)):
            raise CheckpointContractError("checkpoint metadata must define band_indexes")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_bands):
            raise CheckpointContractError("checkpoint band indexes must be integers")
        bands = tuple(int(value) for value in raw_bands)
        if not bands or len(set(bands)) != len(bands) or any(value < 1 for value in bands):
            raise CheckpointContractError("checkpoint band indexes must be unique and positive")
        in_channels = int(metadata.get("in_channels", len(bands)))
        if in_channels != len(bands):
            raise CheckpointContractError(
                "checkpoint in_channels does not match the number of band indexes"
            )
        normalization = metadata.get("normalization", {})
        calibration = metadata.get("calibration", {})
        if not isinstance(normalization, Mapping) or not isinstance(calibration, Mapping):
            raise CheckpointContractError("normalization and calibration metadata must be mappings")
        if any(not isinstance(value, Mapping) for value in calibration.values()):
            raise CheckpointContractError("each calibration head must contain a mapping")
        return cls(
            model_version=model_version,
            band_indexes=bands,
            in_channels=in_channels,
            normalization=dict(normalization),
            calibration={str(key): dict(value) for key, value in calibration.items()},
            raw=dict(metadata),
        )

    def build_calibrators(self) -> dict[str, ProbabilityCalibrator]:
        return {
            head: calibrator_from_config(dict(config)) for head, config in self.calibration.items()
        }


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Result of loading weights into a caller-provided model."""

    path: Path
    metadata: CheckpointMetadata
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def assert_band_contract(actual: Sequence[int], expected: Sequence[int]) -> None:
    """Require exact band identity *and order*, not merely equal channel count."""

    actual_tuple = tuple(int(value) for value in actual)
    expected_tuple = tuple(int(value) for value in expected)
    if actual_tuple != expected_tuple:
        raise CheckpointContractError(
            f"raster bands {actual_tuple} do not match checkpoint bands {expected_tuple}"
        )


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


def _checkpoint_metadata(payload: Mapping[str, Any]) -> CheckpointMetadata:
    metadata = payload.get("metadata", payload.get("rowgraphnet_metadata"))
    if metadata is None and isinstance(payload.get("hyper_parameters"), Mapping):
        parameters = payload["hyper_parameters"]
        metadata = parameters.get("metadata", parameters.get("rowgraphnet_metadata"))
    if not isinstance(metadata, Mapping):
        raise CheckpointContractError(
            "checkpoint is missing RowGraphNet metadata; expected a 'metadata' mapping "
            "with model_version and band_indexes"
        )
    return CheckpointMetadata.from_mapping(metadata)


def _state_dict(payload: Mapping[str, Any]) -> dict[str, Tensor]:
    state = payload.get("state_dict", payload.get("model_state_dict"))
    if not isinstance(state, Mapping) or not state:
        raise CheckpointContractError("checkpoint must contain a non-empty state_dict")
    if any(
        not isinstance(key, str) or not isinstance(value, Tensor) for key, value in state.items()
    ):
        raise CheckpointContractError("checkpoint state_dict must map string keys to tensors")
    normalized = dict(state)
    for prefix in ("module.", "model."):
        if normalized and all(key.startswith(prefix) for key in normalized):
            normalized = {key[len(prefix) :]: value for key, value in normalized.items()}
    return normalized


def load_local_checkpoint(
    model: nn.Module,
    checkpoint_path: str | PathLike[str],
    *,
    expected_bands: Sequence[int] | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> LoadedCheckpoint:
    """Load a local checkpoint without model construction or network access."""

    raw_path = str(checkpoint_path)
    if "://" in raw_path:
        raise CheckpointContractError("checkpoint_path must be a local filesystem path, not a URL")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise CheckpointContractError(f"checkpoint file does not exist: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointContractError(f"could not safely read checkpoint: {path}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointContractError("checkpoint root must be a mapping")

    metadata = _checkpoint_metadata(payload)
    if expected_bands is not None:
        assert_band_contract(expected_bands, metadata.band_indexes)
    model_channels = _model_input_channels(model)
    if model_channels is not None and model_channels != metadata.in_channels:
        raise CheckpointContractError(
            f"model expects {model_channels} channels but checkpoint metadata declares "
            f"{metadata.in_channels}"
        )
    state = _state_dict(payload)
    try:
        incompatible = model.load_state_dict(state, strict=strict)
    except RuntimeError as exc:
        raise CheckpointContractError(
            f"checkpoint weights are incompatible with the model: {exc}"
        ) from exc
    return LoadedCheckpoint(
        path=path,
        metadata=metadata,
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )


def read_local_checkpoint_metadata(
    checkpoint_path: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> CheckpointMetadata:
    """Read only the reproducibility contract from a safe local checkpoint."""

    raw_path = str(checkpoint_path)
    if "://" in raw_path:
        raise CheckpointContractError("checkpoint_path must be a local filesystem path, not a URL")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise CheckpointContractError(f"checkpoint file does not exist: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointContractError(f"could not safely read checkpoint: {path}") from exc
    if not isinstance(payload, Mapping):
        raise CheckpointContractError("checkpoint root must be a mapping")
    return _checkpoint_metadata(payload)
