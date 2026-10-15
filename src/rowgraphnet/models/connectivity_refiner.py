"""Learned residual refinement for deliberately broken centerline evidence."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rowgraphnet.models.multitask_decoder import ConvNormAct, _group_count


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
    """Predict a residual correction from center, direction and uncertainty.

    The refiner consumes continuous evidence instead of a binarized skeleton.
    Its final layer is initialized to zero, so adding it to a trained decoder
    starts as an exact no-op and does not invent connections before training.
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
        if context_channels < 0 or hidden_channels < 1:
            raise ValueError("channel counts must be non-negative and hidden_channels positive")
        if not dilations or any(value < 1 for value in dilations):
            raise ValueError("dilations must contain positive integers")

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
        context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return repair evidence and a refined centerline prediction."""

        self._validate_inputs(
            center_logits,
            orientation,
            endpoints_logits,
            uncertainty,
            context,
        )
        target_size = center_logits.shape[-2:]
        # The decoder context normally lives at 1/4 resolution. Refining on
        # that native grid prevents a decoder_channels x full-resolution
        # allocation on large tiles; only the single-channel residual is
        # interpolated back to the source grid.
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

    def _validate_inputs(
        self,
        center_logits: Tensor,
        orientation: Tensor,
        endpoints_logits: Tensor,
        uncertainty: Tensor,
        context: Tensor | None,
    ) -> None:
        maps = {
            "center_logits": (center_logits, 1),
            "orientation": (orientation, 2),
            "endpoints_logits": (endpoints_logits, 1),
            "uncertainty": (uncertainty, 1),
        }
        reference_shape = center_logits.shape
        for name, (value, channels) in maps.items():
            if value.ndim != 4 or value.shape[1] != channels:
                raise ValueError(f"{name} must have shape [batch, {channels}, height, width]")
            if value.shape[0] != reference_shape[0] or value.shape[-2:] != reference_shape[-2:]:
                raise ValueError("all evidence maps must use the same batch and spatial grid")

        if self.context_channels:
            if context is None:
                raise ValueError(
                    f"context with {self.context_channels} channels is required by this refiner"
                )
            if context.ndim != 4 or context.shape[1] != self.context_channels:
                raise ValueError(
                    f"context must have shape [batch, {self.context_channels}, height, width]"
                )
            if context.shape[0] != reference_shape[0]:
                raise ValueError("context must share the evidence batch size")
        elif context is not None:
            raise ValueError("context was supplied to a refiner configured with context_channels=0")
