import torch
import torch.nn as nn
from typing import Dict, Any

class BaseRowModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs model inference, applying appropriate activation functions
        (e.g., sigmoid for probabilities, raw/tanh for orientations).
        """
        raise NotImplementedError

    def get_output_spec(self) -> Dict[str, Any]:
        """
        Returns metadata about the output channels.
        """
        raise NotImplementedError
