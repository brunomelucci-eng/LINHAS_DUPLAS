"""Offline-safe checkpoint loading, calibration, and tiled inference."""

from rowgraphnet.inference.calibration import (
    IsotonicCalibrator,
    ProbabilityCalibrator,
    TemperatureScaler,
    calibrator_from_config,
)
from rowgraphnet.inference.checkpoint import (
    CheckpointContractError,
    CheckpointMetadata,
    LoadedCheckpoint,
    assert_band_contract,
    load_local_checkpoint,
    read_local_checkpoint_metadata,
)
from rowgraphnet.inference.engine import (
    InferenceConfig,
    InferenceError,
    InputNormalization,
    TiledInferenceEngine,
    estimate_in_memory_prediction_bytes,
    predict_tiled,
    resolve_device,
)

__all__ = [
    "CheckpointContractError",
    "CheckpointMetadata",
    "InferenceConfig",
    "InferenceError",
    "InputNormalization",
    "IsotonicCalibrator",
    "LoadedCheckpoint",
    "ProbabilityCalibrator",
    "TemperatureScaler",
    "TiledInferenceEngine",
    "assert_band_contract",
    "calibrator_from_config",
    "estimate_in_memory_prediction_bytes",
    "load_local_checkpoint",
    "predict_tiled",
    "read_local_checkpoint_metadata",
    "resolve_device",
]
