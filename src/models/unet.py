import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .base import BaseRowModel
from typing import Dict, Any

class RowUNet(BaseRowModel):
    def __init__(
        self, 
        encoder_name: str = "resnet34", 
        encoder_weights: str = "imagenet", 
        in_channels: int = 3, 
        out_channels: int = 4
    ):
        super().__init__()
        self.model = smp.Unet(
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
        # Apply Sigmoid to the first 2 channels (row mask, centerline mask)
        probs = torch.sigmoid(logits[:, 0:2, :, :])
        # Apply Tanh to the last 2 channels (orientation sin, cos) to bound them to [-1, 1]
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
