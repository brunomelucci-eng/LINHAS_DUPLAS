"""High-resolution FPN decoder and calibrated-output-friendly task heads."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _group_count(channels: int, maximum: int = 8) -> int:
    """Choose a GroupNorm group count that also works for tiny test models."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    """Convolution followed by GroupNorm and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class MultiTaskDecoder(nn.Module):
    """Fuse SegFormer stages and emit all RowGraphNet raster predictions.

    The two orientation channels are ordered as ``sin(2 theta)`` and
    ``cos(2 theta)``. They are normalized to unit length per pixel, including a
    deterministic axial fallback for an exactly zero raw vector.
    """

    output_names = (
        "band",
        "center",
        "orientation",
        "endpoints",
        "distance",
        "uncertainty",
    )

    def __init__(
        self,
        feature_channels: Sequence[int],
        *,
        decoder_channels: int = 128,
        orientation_epsilon: float = 1e-6,
        uncertainty_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        feature_channels = tuple(int(value) for value in feature_channels)
        if not feature_channels or any(value < 1 for value in feature_channels):
            raise ValueError("feature_channels must contain positive channel counts")
        if decoder_channels < 1:
            raise ValueError("decoder_channels must be positive")
        if orientation_epsilon <= 0 or uncertainty_floor <= 0:
            raise ValueError("numerical stability constants must be positive")

        self.feature_channels = feature_channels
        self.decoder_channels = int(decoder_channels)
        self.orientation_epsilon = float(orientation_epsilon)
        self.uncertainty_floor = float(uncertainty_floor)

        self.lateral = nn.ModuleList(
            nn.Conv2d(channels, decoder_channels, kernel_size=1) for channels in feature_channels
        )
        self.smooth = nn.ModuleList(
            ConvNormAct(decoder_channels, decoder_channels) for _ in feature_channels
        )
        self.fusion = nn.Sequential(
            ConvNormAct(decoder_channels, decoder_channels),
            ConvNormAct(decoder_channels, decoder_channels),
        )

        self.band_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.center_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.orientation_head = nn.Conv2d(decoder_channels, 2, kernel_size=1)
        self.endpoints_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.distance_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

    def forward(
        self,
        features: Sequence[Tensor],
        *,
        output_size: tuple[int, int] | None = None,
        return_features: bool = False,
    ) -> dict[str, Tensor]:
        """Decode feature maps.

        ``output_size`` should be the input raster's ``(height, width)``. Every
        prediction is resized to that exact grid before its activation is
        applied, avoiding shape drift for dimensions not divisible by 32.
        """

        if len(features) != len(self.feature_channels):
            raise ValueError(
                f"expected {len(self.feature_channels)} feature maps, got {len(features)}"
            )
        batch_size = features[0].shape[0]
        lateral: list[Tensor] = []
        for index, (feature, expected_channels, projection) in enumerate(
            zip(features, self.feature_channels, self.lateral, strict=True)
        ):
            if feature.ndim != 4:
                raise ValueError(f"feature {index} must be BCHW")
            if feature.shape[0] != batch_size:
                raise ValueError("all feature stages must share the same batch size")
            if feature.shape[1] != expected_channels:
                raise ValueError(
                    f"feature {index} has {feature.shape[1]} channels; expected {expected_channels}"
                )
            lateral.append(projection(feature))

        pyramid = self.smooth[-1](lateral[-1])
        for index in range(len(lateral) - 2, -1, -1):
            pyramid = F.interpolate(
                pyramid,
                size=lateral[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pyramid = self.smooth[index](pyramid + lateral[index])
        decoded = self.fusion(pyramid)

        if output_size is None:
            output_size = tuple(int(value) for value in decoded.shape[-2:])
        if len(output_size) != 2 or min(output_size) < 1:
            raise ValueError("output_size must contain a positive height and width")

        def head_at_input_grid(head: nn.Module) -> Tensor:
            value = head(decoded)
            if value.shape[-2:] != output_size:
                value = F.interpolate(
                    value,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            return value

        band_logits = head_at_input_grid(self.band_head)
        center_logits = head_at_input_grid(self.center_head)
        endpoints_logits = head_at_input_grid(self.endpoints_head)
        distance_raw = head_at_input_grid(self.distance_head)
        log_variance = head_at_input_grid(self.uncertainty_head)
        orientation_raw = head_at_input_grid(self.orientation_head)
        orientation = self._normalize_orientation(orientation_raw)

        result = {
            "band_logits": band_logits,
            "band": torch.sigmoid(band_logits),
            "center_logits": center_logits,
            "center": torch.sigmoid(center_logits),
            # Before connectivity refinement the dense center head represents
            # directly visible evidence.  RowGraphNet replaces only the
            # complete-center aliases with the refined topology prediction.
            "center_observed_logits": center_logits,
            "center_observed": torch.sigmoid(center_logits),
            "center_complete_logits": center_logits,
            "center_complete": torch.sigmoid(center_logits),
            "orientation_raw": orientation_raw,
            "orientation": orientation,
            "orientation_sin2": orientation[:, 0:1],
            "orientation_cos2": orientation[:, 1:2],
            "endpoints_logits": endpoints_logits,
            "endpoints": torch.sigmoid(endpoints_logits),
            "distance_raw": distance_raw,
            "distance": F.softplus(distance_raw),
            "log_variance": log_variance,
            "uncertainty": F.softplus(log_variance) + self.uncertainty_floor,
        }
        if return_features:
            result["decoder_features"] = decoded
        return result

    def _normalize_orientation(self, raw: Tensor) -> Tensor:
        norm = torch.linalg.vector_norm(raw, dim=1, keepdim=True)
        normalized = raw / norm.clamp_min(self.orientation_epsilon)
        fallback = torch.zeros_like(raw)
        fallback[:, 1:2] = 1.0  # theta=0 -> [sin(0), cos(0)]
        return torch.where(norm > self.orientation_epsilon, normalized, fallback)
