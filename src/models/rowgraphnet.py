"""
RowGraphNet architecture ported from linhas_simples_pytorch.
Combines a SegFormer (MiT) encoder with a high-resolution MultiTaskDecoder
and learned residual ConnectivityRefiner for robust crop row centerline detection.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import segmentation_models_pytorch as smp

from .base import BaseRowModel


def _group_count(channels: int, maximum: int = 8) -> int:
    """Choose a GroupNorm group count that divides channels evenly."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    """Convolution followed by GroupNorm and GELU activation."""

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
    """High-resolution FPN decoder with multi-task heads for sugarcane row detection.

    Emits dense predictions for:
    - band: row occupancy mask probability
    - center: row centerline probability
    - orientation: normalized [sin(2*theta), cos(2*theta)] vector
    - endpoints: probability of line terminal points
    - distance: continuous distance to row centerline
    - uncertainty: learned epistemic/aleatoric variance
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

        # Multi-task heads matching linhas_simples_pytorch
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
        output_size: Optional[Tuple[int, int]] = None,
        return_features: bool = False,
    ) -> Dict[str, Tensor]:
        batch_size = features[0].shape[0]
        lateral: list[Tensor] = []
        for index, (feature, projection) in enumerate(zip(features, self.lateral)):
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
            "center_observed_logits": center_logits,
            "center_observed": torch.sigmoid(center_logits),
            "center_complete_logits": center_logits,
            "center_complete": torch.sigmoid(center_logits),
            "orientation_raw": orientation_raw,
            "orientation": orientation,
            "orientation_sin": orientation[:, 0:1],
            "orientation_cos": orientation[:, 1:2],
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
        fallback[:, 1:2] = 1.0  # default: direction along axis
        return torch.where(norm > self.orientation_epsilon, normalized, fallback)


class _ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            ConvNormAct(channels, channels, dilation=dilation),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.convolution(value))


class ConnectivityRefiner(nn.Module):
    """Predict a residual correction from center, direction, and uncertainty.

    The refiner consumes continuous evidence instead of a binarized skeleton.
    Its final layer is initialized to zero, so adding it starts as an exact no-op.
    """

    evidence_channels = 5  # center + sin/cos orientation + endpoints + uncertainty

    def __init__(
        self,
        *,
        context_channels: int = 0,
        hidden_channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.context_channels = int(context_channels)
        self.input_projection = ConvNormAct(
            self.evidence_channels + context_channels,
            hidden_channels,
        )
        self.blocks = nn.Sequential(
            *(_ResidualDilatedBlock(hidden_channels, value) for value in dilations)
        )
        self.repair_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        nn.init.zeros_(self.repair_head.weight)
        nn.init.zeros_(self.repair_head.bias)

    def forward(
        self,
        center_logits: Tensor,
        orientation: Tensor,
        endpoints_logits: Tensor,
        uncertainty: Tensor,
        *,
        context: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        target_size = center_logits.shape[-2:]
        processing_size = context.shape[-2:] if context is not None else target_size
        evidence = [center_logits, orientation, endpoints_logits, uncertainty]
        tensors = [
            F.interpolate(value, size=processing_size, mode="bilinear", align_corners=False)
            if value.shape[-2:] != processing_size
            else value
            for value in evidence
        ]
        if context is not None:
            tensors.append(context)

        hidden = self.blocks(self.input_projection(torch.cat(tensors, dim=1)))
        repair_logits = self.repair_head(hidden)
        if repair_logits.shape[-2:] != target_size:
            repair_logits = F.interpolate(
                repair_logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        refined_center_logits = center_logits + repair_logits
        return {
            "repair_logits": repair_logits,
            "repair_probability": torch.sigmoid(repair_logits),
            "refined_center_logits": refined_center_logits,
            "refined_center": torch.sigmoid(refined_center_logits),
        }


class RowGraphNetModel(BaseRowModel):
    """RowGraphNet architecture identical to linhas_simples_pytorch.

    Features:
    1. SegFormer (MixVisionTransformer - mit_b0, mit_b2, etc.) multiscale encoder.
    2. High-resolution FPN MultiTaskDecoder.
    3. Multi-task output heads: band, center, orientation, endpoints, distance, uncertainty.
    4. Optional learned ConnectivityRefiner for bridging breaks and gap repair.
    5. Full compatibility with the LINHAS_DUPLAS training and post-processing pipeline.
    """

    def __init__(
        self,
        encoder_name: str = "mit_b0",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        decoder_channels: int = 128,
        refine_connectivity: bool = True,
        refiner_hidden_channels: int = 64,
        out_channels: int = 4,
    ) -> None:
        super().__init__()
        weights = encoder_weights if encoder_weights else None
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            weights=weights,
        )
        # SegFormer encoder returns [in, dummy, stage1, stage2, stage3, stage4]
        # Multi-scale features start from index 2
        feature_channels = tuple(self.encoder.out_channels[2:])
        self.decoder = MultiTaskDecoder(
            feature_channels,
            decoder_channels=decoder_channels,
        )
        self.connectivity_refiner = (
            ConnectivityRefiner(
                context_channels=decoder_channels,
                hidden_channels=refiner_hidden_channels,
            )
            if refine_connectivity
            else None
        )
        self.out_channels = out_channels
        self.last_predictions: Optional[Dict[str, Tensor]] = None

    def forward_multitask(
        self,
        image: Tensor,
        *,
        return_features: bool = False,
    ) -> Dict[str, Tensor]:
        output_size = tuple(int(value) for value in image.shape[-2:])
        all_features = self.encoder(image)
        # Discard the 0-th (input) and 1-st (dummy) maps for SegFormer
        features = all_features[2:]

        predictions = self.decoder(
            features,
            output_size=output_size,
            return_features=self.connectivity_refiner is not None or return_features,
        )

        if self.connectivity_refiner is not None:
            coarse_logits = predictions["center_observed_logits"]
            coarse_center = predictions["center_observed"]
            refined = self.connectivity_refiner(
                coarse_logits,
                predictions["orientation"],
                predictions["endpoints_logits"],
                predictions["uncertainty"],
                context=predictions.get("decoder_features"),
            )
            predictions["coarse_center_logits"] = coarse_logits
            predictions["coarse_center"] = coarse_center
            predictions.update(refined)
            predictions["center_logits"] = refined["refined_center_logits"]
            predictions["center"] = refined["refined_center"]
            predictions["center_complete_logits"] = refined["refined_center_logits"]
            predictions["center_complete"] = refined["refined_center"]

        if not return_features:
            predictions.pop("decoder_features", None)

        self.last_predictions = predictions
        return predictions

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the training loss pipeline.

        Returns 4-channel tensor [band_logits, center_logits, orientation_sin, orientation_cos]
        compatible with the trainer and CombinedLoss.
        """
        preds = self.forward_multitask(x)
        # When refiner is enabled, center_logits already contains the refined logits
        return torch.cat(
            [
                preds["band_logits"],
                preds["center_logits"],
                preds["orientation"][:, 0:1],
                preds["orientation"][:, 1:2],
            ],
            dim=1,
        )

    def predict(self, x: Tensor) -> Tensor:
        """Inference pass returning 4 probability channels for sliding window and postprocessing:

        [band, center/refined_center, orientation_sin, orientation_cos]
        """
        preds = self.forward_multitask(x)
        return torch.cat(
            [
                preds["band"],
                preds["center"],
                preds["orientation"][:, 0:1],
                preds["orientation"][:, 1:2],
            ],
            dim=1,
        )

    def predict_multitask(self, x: Tensor) -> Dict[str, Tensor]:
        """Returns all rich multi-task prediction maps."""
        return self.forward_multitask(x)

    def get_output_spec(self) -> Dict[str, Any]:
        return {
            "channels": [
                "row_probability",
                "center_probability",
                "orientation_sin",
                "orientation_cos",
            ],
            "multitask_heads": list(MultiTaskDecoder.output_names) + ["refined_center"],
        }
