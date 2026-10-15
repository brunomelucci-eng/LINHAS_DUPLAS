"""SegFormer feature encoder used by RowGraphNet.

The Hugging Face dependency is imported lazily so that geometry-only commands
can import :mod:`rowgraphnet` without installing the optional ``model`` extra.
When instantiated, this class always uses the real ``transformers``
``SegformerModel`` implementation (unless an already-built compatible model is
explicitly injected for testing or advanced use).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn


def _segformer_classes() -> tuple[type[Any], type[nn.Module]]:
    """Load Transformers' SegFormer classes with an actionable error."""

    try:
        from transformers import SegformerConfig, SegformerModel
    except (
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
    ) as exc:  # pragma: no cover - env specific
        raise ImportError(
            "SegFormer requires the optional 'transformers' dependency. "
            "Install RowGraphNet with the 'model' extra."
        ) from exc
    return SegformerConfig, SegformerModel


class SegFormerEncoder(nn.Module):
    """Expose the four multiscale maps produced by a Transformers SegFormer.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier, normally ``nvidia/mit-b2`` or
        ``nvidia/mit-b3``.
    pretrained:
        Load pretrained weights. Set to ``False`` together with ``config`` for
        deterministic, offline construction.
    in_channels:
        Number of raster bands supplied to the model.
    config:
        Optional ``SegformerConfig`` or mapping. Supplying it avoids any Hub
        access and is particularly useful for tests and custom sensors.
    model:
        Optional pre-built SegFormer-compatible module. It must expose a
        ``config.hidden_sizes`` attribute and return hidden states.
    local_files_only:
        Forbid Hub downloads while resolving ``model_name``.
    """

    def __init__(
        self,
        model_name: str = "nvidia/mit-b2",
        *,
        pretrained: bool = True,
        in_channels: int = 3,
        config: Any | Mapping[str, Any] | None = None,
        model: nn.Module | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if model is not None and config is not None:
            raise ValueError("pass either model or config, not both")

        if model is None:
            config_cls, model_cls = _segformer_classes()
            if isinstance(config, Mapping):
                config = config_cls(**dict(config))

            if config is not None:
                if pretrained:
                    raise ValueError(
                        "pretrained=True cannot be combined with an explicit config; "
                        "construct the pretrained model first and pass it as model"
                    )
                config.num_channels = in_channels
                config.output_hidden_states = True
                config.return_dict = True
                model = model_cls(config)
            elif pretrained:
                model = model_cls.from_pretrained(
                    model_name,
                    num_channels=in_channels,
                    ignore_mismatched_sizes=in_channels != 3,
                    output_hidden_states=True,
                    local_files_only=local_files_only,
                )
            else:
                # SegformerConfig() is a genuine, compact MiT-style model and
                # keeps offline construction possible. Production B2/B3 runs
                # should use pretrained=True or pass an explicit config.
                default_config = config_cls(
                    num_channels=in_channels,
                    output_hidden_states=True,
                    return_dict=True,
                )
                model = model_cls(default_config)

        self.backbone = model
        backbone_config = getattr(model, "config", None)
        hidden_sizes = getattr(backbone_config, "hidden_sizes", None)
        if not hidden_sizes:
            raise TypeError("the SegFormer model must expose config.hidden_sizes")
        self.feature_channels = tuple(int(value) for value in hidden_sizes)
        self.in_channels = int(getattr(backbone_config, "num_channels", in_channels))

    @property
    def config(self) -> Any:
        """Return the underlying Transformers configuration."""

        return self.backbone.config

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, ...]:
        """Return channels-first feature maps from fine to coarse scale."""

        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape [batch, channels, height, width]")
        if pixel_values.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {pixel_values.shape[1]}"
            )

        output = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("SegFormer did not return hidden states")
        if len(hidden_states) < len(self.feature_channels):
            raise RuntimeError(
                f"expected {len(self.feature_channels)} feature stages, got {len(hidden_states)}"
            )

        selected = hidden_states[-len(self.feature_channels) :]
        maps = tuple(
            self._as_feature_map(
                state,
                channels=channels,
                input_hw=pixel_values.shape[-2:],
                stage_index=stage_index,
            )
            for stage_index, (state, channels) in enumerate(
                zip(selected, self.feature_channels, strict=True)
            )
        )
        return maps

    def _as_feature_map(
        self,
        state: Tensor,
        *,
        channels: int,
        input_hw: Sequence[int],
        stage_index: int,
    ) -> Tensor:
        """Normalize version-dependent hidden-state layouts to BCHW."""

        if state.ndim == 4:
            if state.shape[1] == channels:
                return state.contiguous()
            if state.shape[-1] == channels:
                return state.permute(0, 3, 1, 2).contiguous()
            raise RuntimeError(
                f"stage {stage_index} has incompatible shape {tuple(state.shape)}; "
                f"expected {channels} channels"
            )

        if state.ndim != 3:
            raise RuntimeError(
                f"stage {stage_index} hidden state must be 3D or 4D, got {state.ndim}D"
            )

        # Transformers releases have represented token maps as either BNC or
        # BCN. Resolve both without depending on a specific library version.
        if state.shape[-1] == channels:
            tokens = state.transpose(1, 2)
        elif state.shape[1] == channels:
            tokens = state
        else:
            raise RuntimeError(
                f"stage {stage_index} has incompatible token shape {tuple(state.shape)}"
            )

        token_count = tokens.shape[-1]
        height, width = self._expected_stage_hw(input_hw, stage_index)
        if height * width != token_count:
            height, width = self._factor_spatial_shape(
                token_count,
                aspect=float(input_hw[0]) / float(input_hw[1]),
            )
        return tokens.reshape(tokens.shape[0], channels, height, width).contiguous()

    def _expected_stage_hw(self, input_hw: Sequence[int], stage_index: int) -> tuple[int, int]:
        patch_sizes = tuple(getattr(self.config, "patch_sizes", (7, 3, 3, 3)))
        strides = tuple(getattr(self.config, "strides", (4, 2, 2, 2)))
        height, width = int(input_hw[0]), int(input_hw[1])
        for index in range(stage_index + 1):
            kernel = int(patch_sizes[index])
            stride = int(strides[index])
            padding = kernel // 2
            height = (height + 2 * padding - kernel) // stride + 1
            width = (width + 2 * padding - kernel) // stride + 1
        return height, width

    @staticmethod
    def _factor_spatial_shape(token_count: int, aspect: float) -> tuple[int, int]:
        """Infer a plausible HxW pair if a custom backbone omits map shapes."""

        if token_count < 1:
            raise RuntimeError("feature maps cannot contain zero spatial tokens")
        candidates: list[tuple[float, int, int]] = []
        for height in range(1, math.isqrt(token_count) + 1):
            if token_count % height:
                continue
            width = token_count // height
            for candidate_h, candidate_w in ((height, width), (width, height)):
                ratio_error = abs(math.log((candidate_h / candidate_w) / aspect))
                candidates.append((ratio_error, candidate_h, candidate_w))
        _, height, width = min(candidates, key=lambda item: item[0])
        return height, width
