"""Input normalization contracts shared across dataset and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _channel_values(
    values: float | Sequence[float] | None,
    channels: int,
    *,
    name: str,
    default: float,
) -> np.ndarray:
    if values is None:
        array = np.full(channels, default, dtype=np.float32)
    elif isinstance(values, (int, float)) and not isinstance(values, bool):
        array = np.full(channels, float(values), dtype=np.float32)
    else:
        array = np.asarray(tuple(values), dtype=np.float32)
        if array.shape != (channels,):
            raise ValueError(f"{name} must contain one value per input band")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} values must be finite")
    return array[:, None, None]


@dataclass(frozen=True, slots=True)
class InputNormalization:
    """Explicit channel normalization applied after optional integer scaling."""

    scale: float | tuple[float, ...] | None = None
    offset: float | tuple[float, ...] | None = None
    mean: float | tuple[float, ...] | None = None
    std: float | tuple[float, ...] | None = None
    scale_integer_to_unit: bool = True

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> InputNormalization:
        return cls(
            scale=metadata.get("scale"),
            offset=metadata.get("offset"),
            mean=metadata.get("mean"),
            std=metadata.get("std"),
            scale_integer_to_unit=bool(metadata.get("scale_integer_to_unit", True)),
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3:
            raise ValueError("input tile must have shape [channels, height, width]")
        original_dtype = image.dtype
        result = image.astype(np.float32, copy=True)
        channels = result.shape[0]
        if (
            self.scale is None
            and self.scale_integer_to_unit
            and np.issubdtype(original_dtype, np.integer)
        ):
            maximum = float(np.iinfo(original_dtype).max)
            result /= maximum
        else:
            result *= _channel_values(self.scale, channels, name="scale", default=1.0)
        result += _channel_values(self.offset, channels, name="offset", default=0.0)
        result -= _channel_values(self.mean, channels, name="mean", default=0.0)
        standard_deviation = _channel_values(self.std, channels, name="std", default=1.0)
        if np.any(standard_deviation <= 0.0):
            raise ValueError("std values must be positive")
        result /= standard_deviation
        if not np.isfinite(result).all():
            raise ValueError("normalization produced non-finite input values")
        return result


__all__ = ["InputNormalization"]
