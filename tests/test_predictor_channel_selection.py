import numpy as np
import torch

from src.inference.predictor import RowPredictor


class _FakeModel:
    def predict(self, patches):
        batch, _, height, width = patches.shape
        channels = [
            torch.full((batch, height, width), float(index), device=patches.device)
            for index in range(4)
        ]
        return torch.stack(channels, dim=1)


def test_predictor_selects_requested_channels_before_cpu_result():
    predictor = RowPredictor.__new__(RowPredictor)
    predictor.config = {'inference': {'mixed_precision': False}}
    predictor.device = torch.device('cpu')
    predictor.model = _FakeModel()
    patches = np.zeros((2, 3, 8, 8), dtype=np.float32)

    selected = predictor.predict_batch(patches, output_indices=[1])

    assert selected.shape == (2, 1, 8, 8)
    assert np.all(selected == 1.0)
