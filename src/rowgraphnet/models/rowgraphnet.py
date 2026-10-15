"""End-to-end RowGraphNet raster model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from rowgraphnet.models.connectivity_refiner import ConnectivityRefiner
from rowgraphnet.models.encoder import SegFormerEncoder
from rowgraphnet.models.multitask_decoder import MultiTaskDecoder


class RowGraphNet(nn.Module):
    """SegFormer + high-resolution multitask decoder + connectivity refiner.

    The graph scorer is intentionally a separate module because candidate
    endpoints arise after raster segment extraction. This class covers the
    dense prediction stage and guarantees that every public raster map matches
    the input height and width.
    """

    def __init__(
        self,
        *,
        model_name: str = "nvidia/mit-b2",
        pretrained: bool = True,
        in_channels: int = 3,
        decoder_channels: int = 128,
        encoder_config: Any | Mapping[str, Any] | None = None,
        encoder: SegFormerEncoder | None = None,
        refine_connectivity: bool = True,
        refiner_hidden_channels: int = 64,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder or SegFormerEncoder(
            model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            config=encoder_config,
            local_files_only=local_files_only,
        )
        self.decoder = MultiTaskDecoder(
            self.encoder.feature_channels,
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

    def forward(self, image: Tensor, *, return_features: bool = False) -> dict[str, Tensor]:
        if image.ndim != 4:
            raise ValueError("image must have shape [batch, channels, height, width]")
        output_size = tuple(int(value) for value in image.shape[-2:])
        features = self.encoder(image)
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
                context=predictions["decoder_features"],
            )
            predictions["coarse_center_logits"] = coarse_logits
            predictions["coarse_center"] = coarse_center
            predictions.update(refined)
            # Downstream inference can consistently consume `center`, whether
            # the optional refiner is enabled or not.
            predictions["center_logits"] = refined["refined_center_logits"]
            predictions["center"] = refined["refined_center"]
            predictions["center_complete_logits"] = refined["refined_center_logits"]
            predictions["center_complete"] = refined["refined_center"]

        if not return_features:
            predictions.pop("decoder_features", None)
        return predictions
