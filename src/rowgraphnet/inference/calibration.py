"""Probability calibration without a mandatory scikit-learn dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor


class ProbabilityCalibrator(Protocol):
    """Common interface used by inference probability heads."""

    def calibrate_logits(self, logits: Tensor) -> Tensor: ...

    def calibrate_probabilities(self, probabilities: Tensor) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class TemperatureScaler:
    """Apply scalar temperature scaling before the sigmoid activation."""

    temperature: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")

    def calibrate_logits(self, logits: Tensor) -> Tensor:
        return torch.sigmoid(logits / self.temperature)

    def calibrate_probabilities(self, probabilities: Tensor) -> Tensor:
        bounded = probabilities.clamp(
            torch.finfo(probabilities.dtype).eps, 1.0 - torch.finfo(probabilities.dtype).eps
        )
        logits = torch.logit(bounded)
        return self.calibrate_logits(logits)


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    """Piecewise-linear monotonic probability mapping.

    The optional :meth:`fit` implementation uses the pool-adjacent-violators
    algorithm directly, so inference and validation never require sklearn.
    """

    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        x = np.asarray(self.x_thresholds, dtype=np.float64)
        y = np.asarray(self.y_thresholds, dtype=np.float64)
        if x.ndim != 1 or len(x) < 1 or x.shape != y.shape:
            raise ValueError("isotonic thresholds must be non-empty 1-D arrays of equal length")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("isotonic thresholds must be finite")
        if len(x) > 1 and np.any(np.diff(x) <= 0.0):
            raise ValueError("isotonic x thresholds must be strictly increasing")
        if np.any(np.diff(y) < 0.0) or np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("isotonic y thresholds must be monotonic probabilities")

    def calibrate_logits(self, logits: Tensor) -> Tensor:
        return self.calibrate_probabilities(torch.sigmoid(logits))

    def calibrate_probabilities(self, probabilities: Tensor) -> Tensor:
        x = torch.as_tensor(
            self.x_thresholds,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        y = torch.as_tensor(
            self.y_thresholds,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        values = probabilities.clamp(0.0, 1.0)
        if x.numel() == 1:
            return torch.full_like(values, y[0])

        flat = values.reshape(-1).contiguous()
        upper = torch.searchsorted(x, flat, right=False)
        lower = (upper - 1).clamp(0, x.numel() - 1)
        upper = upper.clamp(0, x.numel() - 1)
        x0, x1 = x[lower], x[upper]
        y0, y1 = y[lower], y[upper]
        fraction = torch.where(x1 > x0, (flat - x0) / (x1 - x0), torch.zeros_like(flat))
        mapped = y0 + fraction.clamp(0.0, 1.0) * (y1 - y0)
        return mapped.reshape_as(values)

    @classmethod
    def fit(
        cls,
        probabilities: np.ndarray | list[float],
        targets: np.ndarray | list[float],
        *,
        sample_weight: np.ndarray | list[float] | None = None,
    ) -> IsotonicCalibrator:
        """Fit a non-decreasing mapping with weighted PAV."""

        scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        labels = np.asarray(targets, dtype=np.float64).reshape(-1)
        if scores.size == 0 or scores.shape != labels.shape:
            raise ValueError("probabilities and targets must have equal non-zero length")
        if not np.isfinite(scores).all() or not np.isfinite(labels).all():
            raise ValueError("calibration samples must be finite")
        if np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("probabilities must lie in [0, 1]")
        if np.any((labels < 0.0) | (labels > 1.0)):
            raise ValueError("targets must lie in [0, 1]")
        weights = (
            np.ones_like(scores)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        )
        if weights.shape != scores.shape or not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("sample weights must be finite, non-negative, and match samples")
        positive = weights > 0.0
        if not positive.any():
            raise ValueError("at least one sample must have positive weight")
        scores, labels, weights = scores[positive], labels[positive], weights[positive]

        order = np.argsort(scores, kind="stable")
        scores, labels, weights = scores[order], labels[order], weights[order]
        x, inverse = np.unique(scores, return_inverse=True)
        grouped_weight = np.bincount(inverse, weights=weights)
        grouped_sum = np.bincount(inverse, weights=weights * labels)

        # Blocks are [start, stop, weight, weighted_target_sum].
        blocks: list[list[float | int]] = [
            [index, index + 1, grouped_weight[index], grouped_sum[index]] for index in range(len(x))
        ]
        index = 0
        while index < len(blocks) - 1:
            current_mean = float(blocks[index][3]) / float(blocks[index][2])
            next_mean = float(blocks[index + 1][3]) / float(blocks[index + 1][2])
            if current_mean <= next_mean:
                index += 1
                continue
            blocks[index] = [
                blocks[index][0],
                blocks[index + 1][1],
                float(blocks[index][2]) + float(blocks[index + 1][2]),
                float(blocks[index][3]) + float(blocks[index + 1][3]),
            ]
            blocks.pop(index + 1)
            index = max(index - 1, 0)

        fitted = np.empty(len(x), dtype=np.float64)
        for start, stop, weight, target_sum in blocks:
            fitted[int(start) : int(stop)] = float(target_sum) / float(weight)
        return cls(tuple(float(value) for value in x), tuple(float(value) for value in fitted))


def calibrator_from_config(config: dict[str, Any]) -> ProbabilityCalibrator:
    """Build a calibrator from serializable checkpoint metadata."""

    kind = str(config.get("type", "")).casefold()
    if kind in {"temperature", "temperature_scaling"}:
        return TemperatureScaler(float(config["temperature"]))
    if kind in {"isotonic", "isotonic_regression"}:
        return IsotonicCalibrator(
            tuple(float(value) for value in config["x_thresholds"]),
            tuple(float(value) for value in config["y_thresholds"]),
        )
    raise ValueError(f"unsupported calibrator type: {config.get('type')!r}")
