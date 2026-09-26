import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .base import BaseRowModel
from typing import Dict, Any

class RowDeepLab(BaseRowModel):
    def __init__(
        self, 
        encoder_name: str = "resnet50", 
        encoder_weights: str = "imagenet", 
        in_channels: int = 3, 
        out_channels: int = 4
    ):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights if encoder_weights else None,
            in_channels=in_channels,
            classes=out_channels
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        probs = torch.sigmoid(logits[:, 0:2, :, :])
        orientations = torch.tanh(logits[:, 2:4, :, :])
        return torch.cat([probs, orientations], dim=1)

    def get_output_spec(self) -> Dict[str, Any]:
        return {
            'channels': [
                'row_probability',
                'center_probability',
                'orientation_sin',
                'orientation_cos'
            ]
        }
