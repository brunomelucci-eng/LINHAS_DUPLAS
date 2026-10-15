"""Batched CPU/GPU inference over overlapping geospatial tiles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil
import rasterio
import torch
import torch.nn.functional as F
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from torch import Tensor, nn

from rowgraphnet.data import HannBlender, RasterContract, ValidatedScene, generate_tiles
from rowgraphnet.data.normalization import InputNormalization
from rowgraphnet.domain import RasterPrediction
from rowgraphnet.inference.calibration import ProbabilityCalibrator
from rowgraphnet.inference.checkpoint import assert_band_contract
from rowgraphnet.resources import DEFAULT_RESOURCE_FRACTION, apply_resource_caps


class InferenceError(RuntimeError):
    """Raised for an invalid runtime/device/model-output inference contract."""


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Computational controls; none are agronomic geometry thresholds."""

    tile_size: int | tuple[int, int] = 1024
    overlap: int | tuple[int, int] | None = None
    overlap_fraction: float = 0.5
    batch_size: int = 1
    pad_multiple: int = 32
    device: str = "auto"
    use_amp: bool = True
    invalid_mask_nonzero_is_invalid: bool = True
    uncertainty_floor: float = 1e-6
    memory_budget_fraction: float = DEFAULT_RESOURCE_FRACTION
    resource_fraction: float = DEFAULT_RESOURCE_FRACTION

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if isinstance(self.pad_multiple, bool) or self.pad_multiple < 1:
            raise ValueError("pad_multiple must be positive")
        if not 0.0 <= self.overlap_fraction < 1.0:
            raise ValueError("overlap_fraction must be in [0, 1)")
        if not np.isfinite(self.uncertainty_floor) or self.uncertainty_floor <= 0.0:
            raise ValueError("uncertainty_floor must be finite and positive")
        if not 0.0 < self.memory_budget_fraction <= 1.0:
            raise ValueError("memory_budget_fraction must be in (0, 1]")
        if not 0.1 <= self.resource_fraction <= 1.0:
            raise ValueError("resource_fraction must be in [0.1, 1.0]")


def estimate_in_memory_prediction_bytes(height: int, width: int) -> int:
    """Conservative peak estimate for the current full-scene blending design."""

    if height < 1 or width < 1:
        raise ValueError("raster dimensions must be positive")
    pixels = int(height) * int(width)
    # Five scalar sums + five scalar weights + two orientation sums + one
    # orientation weight, all float32; seven finalized maps; two boolean masks;
    # plus a small allowance for normalization/temporary expressions.
    return pixels * ((13 + 7) * 4 + 2 + 8)


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve auto/cpu/cuda with an actionable unavailable-device error."""

    requested = (
        torch.device(device)
        if str(device) != "auto"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise InferenceError("CUDA was requested but torch.cuda.is_available() is false")
    if requested.type not in {"cpu", "cuda"}:
        raise InferenceError(f"unsupported inference device: {requested}")
    return requested


def _pair(value: int | tuple[int, int], *, name: str) -> tuple[int, int]:
    pair = (value, value) if isinstance(value, int) and not isinstance(value, bool) else value
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in pair)
    ):
        raise InferenceError(f"{name} must be a positive integer or (height, width) tuple")
    return pair


def _probability_head(
    output: Mapping[str, Tensor],
    name: str,
    calibrator: ProbabilityCalibrator | None,
) -> Tensor:
    logits_name = f"{name}_logits"
    if logits_name in output:
        logits = output[logits_name]
        probability = calibrator.calibrate_logits(logits) if calibrator else torch.sigmoid(logits)
    elif name in output:
        probability = output[name]
        if calibrator:
            probability = calibrator.calibrate_probabilities(probability)
    else:
        raise InferenceError(f"model output is missing {logits_name!r} or {name!r}")
    return probability.clamp(0.0, 1.0)


def _as_bchw(value: Tensor, name: str, channels: int, input_shape: tuple[int, ...]) -> Tensor:
    if not isinstance(value, Tensor):
        raise InferenceError(f"model output {name!r} is not a tensor")
    if value.ndim == 3 and channels == 1:
        value = value.unsqueeze(1)
    expected = (input_shape[0], channels, input_shape[-2], input_shape[-1])
    if tuple(value.shape) != expected:
        raise InferenceError(
            f"model output {name!r} has shape {tuple(value.shape)}; expected {expected}"
        )
    if not torch.isfinite(value).all():
        raise InferenceError(f"model output {name!r} contains non-finite values")
    return value.float()


def _activate_outputs(
    raw: Mapping[str, Tensor],
    input_shape: tuple[int, ...],
    calibrators: Mapping[str, ProbabilityCalibrator],
    uncertainty_floor: float,
) -> dict[str, Tensor]:
    probabilities = {
        name: _as_bchw(
            _probability_head(raw, name, calibrators.get(name)),
            name,
            1,
            input_shape,
        )
        for name in ("center", "band", "endpoints")
    }

    if "orientation" in raw:
        orientation = raw["orientation"]
    elif "orientation_raw" in raw:
        orientation = raw["orientation_raw"]
    elif "orientation_sin2" in raw and "orientation_cos2" in raw:
        orientation = torch.cat((raw["orientation_sin2"], raw["orientation_cos2"]), dim=1)
    else:
        raise InferenceError("model output is missing axial orientation channels")
    orientation = _as_bchw(orientation, "orientation", 2, input_shape)
    norm = torch.linalg.vector_norm(orientation, dim=1, keepdim=True)
    fallback = torch.zeros_like(orientation)
    fallback[:, 1:2] = 1.0
    orientation = torch.where(norm > 1e-8, orientation / norm.clamp_min(1e-8), fallback)

    if "distance" in raw:
        distance = raw["distance"].clamp_min(0.0)
    elif "distance_raw" in raw:
        distance = F.softplus(raw["distance_raw"])
    else:
        raise InferenceError("model output is missing 'distance' or 'distance_raw'")
    if "uncertainty" in raw:
        uncertainty = raw["uncertainty"].clamp_min(uncertainty_floor)
    elif "log_variance" in raw:
        uncertainty = F.softplus(raw["log_variance"]) + uncertainty_floor
    else:
        raise InferenceError("model output is missing 'uncertainty' or 'log_variance'")

    probabilities["orientation"] = orientation
    probabilities["distance"] = _as_bchw(distance, "distance", 1, input_shape)
    probabilities["uncertainty"] = _as_bchw(
        uncertainty,
        "uncertainty",
        1,
        input_shape,
    )
    return probabilities


class TiledInferenceEngine:
    """Read, batch, pad, predict and Hann-blend a validated raster scene."""

    def __init__(
        self,
        model: nn.Module,
        *,
        config: InferenceConfig | None = None,
        normalization: InputNormalization | None = None,
        calibrators: Mapping[str, ProbabilityCalibrator] | None = None,
        expected_bands: Sequence[int] | None = None,
    ) -> None:
        self.model = model
        self.config = config or InferenceConfig()
        self.normalization = normalization or InputNormalization()
        self.calibrators = dict(calibrators or {})
        unknown = set(self.calibrators) - {"band", "center", "endpoints"}
        if unknown:
            raise ValueError(f"calibrators target unsupported heads: {sorted(unknown)}")
        self.expected_bands = tuple(expected_bands) if expected_bands is not None else None
        self.device = resolve_device(self.config.device)
        apply_resource_caps(fraction=self.config.resource_fraction, device=self.device)

    def _overlap(self, tile_size: tuple[int, int]) -> int | tuple[int, int]:
        if self.config.overlap is not None:
            return self.config.overlap
        return tuple(math.floor(size * self.config.overlap_fraction) for size in tile_size)

    def predict(self, scene: ValidatedScene | RasterContract) -> RasterPrediction:
        raster = scene.raster if isinstance(scene, ValidatedScene) else scene
        if self.expected_bands is not None:
            assert_band_contract(raster.bands, self.expected_bands)
        tile_size = _pair(self.config.tile_size, name="tile_size")
        windows = generate_tiles(
            (raster.height, raster.width),
            tile_size,
            self._overlap(tile_size),
        )
        estimated_bytes = estimate_in_memory_prediction_bytes(raster.height, raster.width)
        vm = psutil.virtual_memory()
        available_bytes = int(vm.available)
        total_bytes = int(vm.total)
        allowed_bytes = int(min(available_bytes, total_bytes) * self.config.memory_budget_fraction)
        if estimated_bytes > allowed_bytes:
            raise InferenceError(
                "full-scene blending would require approximately "
                f"{estimated_bytes / 1024**3:.1f} GiB, above the configured "
                f"{self.config.memory_budget_fraction:.0%} share of system RAM "
                f"({allowed_bytes / 1024**3:.1f} GiB). Use a smaller ROI to avoid OOM or system freeze; "
                "disk-backed block streaming is not implemented yet."
            )
        scalar_blenders = {
            name: HannBlender((raster.height, raster.width), dtype=np.float32)
            for name in ("center", "band", "endpoints", "distance", "uncertainty")
        }
        orientation_blender = HannBlender(
            (2, raster.height, raster.width), dtype=np.float32
        )
        valid_mask = np.zeros((raster.height, raster.width), dtype=bool)
        roi_mask = np.ones_like(valid_mask)
        if isinstance(scene, ValidatedScene):
            roi_mask = geometry_mask(
                [mapping(scene.analysis_roi_in_raster_crs)],
                out_shape=valid_mask.shape,
                transform=raster.transform,
                invert=True,
            )

        self.model.to(self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            with ExitStack() as stack:
                dataset = stack.enter_context(rasterio.open(raster.path))
                mask_dataset = (
                    stack.enter_context(rasterio.open(scene.invalid_mask_path))
                    if isinstance(scene, ValidatedScene) and scene.invalid_mask_path
                    else None
                )
                for start in range(0, len(windows), self.config.batch_size):
                    batch_windows = windows[start : start + self.config.batch_size]
                    items: list[tuple[Any, np.ndarray, np.ndarray]] = []
                    for window in batch_windows:
                        raster_window = window.as_rasterio_window()
                        masked = dataset.read(raster.bands, window=raster_window, masked=True)
                        data = np.asarray(masked.filled(0))
                        invalid = np.ma.getmaskarray(masked).any(axis=0)
                        invalid |= ~np.isfinite(data).all(axis=0)
                        tile_valid = ~invalid
                        tile_valid &= roi_mask[window.slices]
                        if mask_dataset is not None:
                            external = mask_dataset.read(1, window=raster_window)
                            external_invalid = (
                                external != 0
                                if self.config.invalid_mask_nonzero_is_invalid
                                else external == 0
                            )
                            tile_valid &= ~external_invalid
                        normalized = self.normalization.apply(data)
                        normalized[:, ~tile_valid] = 0.0
                        items.append((window, normalized, tile_valid))
                        valid_mask[window.slices] |= tile_valid

                    activated = self._predict_batch([item[1] for item in items])
                    for batch_index, (window, _, _) in enumerate(items):
                        for name, blender in scalar_blenders.items():
                            tile = activated[name][batch_index, 0, : window.height, : window.width]
                            blender.add(tile.detach().cpu().numpy(), window)
                        orientation = activated["orientation"][
                            batch_index, :, : window.height, : window.width
                        ]
                        orientation_blender.add(orientation.detach().cpu().numpy(), window)

                    if self.device.type == "cuda" and (start // self.config.batch_size) % 25 == 0:
                        torch.cuda.empty_cache()
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.model.train(was_training)

        blended = {name: blender.finalize() for name, blender in scalar_blenders.items()}
        orientation = orientation_blender.finalize()
        norm = np.linalg.norm(orientation, axis=0)
        usable = np.isfinite(norm) & (norm > 1e-8)
        orientation[:, usable] /= norm[usable]
        orientation[0, ~usable] = 0.0
        orientation[1, ~usable] = 1.0

        for name in ("center", "band", "endpoints", "distance"):
            blended[name][~valid_mask] = 0.0
        orientation[0, ~valid_mask] = 0.0
        orientation[1, ~valid_mask] = 1.0
        blended["uncertainty"][~valid_mask] = np.inf
        return RasterPrediction(
            center=blended["center"],
            band=blended["band"],
            orientation_sin2=orientation[0],
            orientation_cos2=orientation[1],
            endpoints=blended["endpoints"],
            distance=blended["distance"],
            uncertainty=blended["uncertainty"],
            transform=raster.transform,
            crs=raster.crs,
            valid_mask=valid_mask,
        )

    def _predict_batch(self, images: list[np.ndarray]) -> dict[str, Tensor]:
        max_height = max(image.shape[-2] for image in images)
        max_width = max(image.shape[-1] for image in images)
        padded_height = math.ceil(max_height / self.config.pad_multiple) * self.config.pad_multiple
        padded_width = math.ceil(max_width / self.config.pad_multiple) * self.config.pad_multiple
        tensors: list[Tensor] = []
        channels = images[0].shape[0]
        for image in images:
            if image.shape[0] != channels:
                raise InferenceError("all tiles in a batch must have the same number of channels")
            tensor = torch.from_numpy(image)
            tensor = F.pad(
                tensor,
                (0, padded_width - image.shape[-1], 0, padded_height - image.shape[-2]),
            )
            tensors.append(tensor)
        batch = torch.stack(tensors).to(self.device, non_blocking=self.device.type == "cuda")
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.config.use_amp and self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            raw = self.model(batch)
            if not isinstance(raw, Mapping):
                raise InferenceError("model must return a mapping of named raster outputs")
            return _activate_outputs(
                raw,
                tuple(batch.shape),
                self.calibrators,
                self.config.uncertainty_floor,
            )


def predict_tiled(
    model: nn.Module,
    scene: ValidatedScene | RasterContract,
    **engine_kwargs: Any,
) -> RasterPrediction:
    """Functional shortcut around :class:`TiledInferenceEngine`."""

    return TiledInferenceEngine(model, **engine_kwargs).predict(scene)


__all__ = [
    "InferenceConfig",
    "InferenceError",
    "TiledInferenceEngine",
    "estimate_in_memory_prediction_bytes",
    "predict_tiled",
    "resolve_device",
]
