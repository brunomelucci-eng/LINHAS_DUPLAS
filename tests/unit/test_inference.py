from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from shapely.geometry import box
from torch import nn

from rowgraphnet.data import validate_scene
from rowgraphnet.inference import (
    CheckpointContractError,
    InferenceConfig,
    IsotonicCalibrator,
    TemperatureScaler,
    TiledInferenceEngine,
    estimate_in_memory_prediction_bytes,
    load_local_checkpoint,
)


def test_prediction_memory_estimate_scales_with_pixel_count() -> None:
    assert estimate_in_memory_prediction_bytes(20, 30) == 2 * estimate_in_memory_prediction_bytes(
        10, 30
    )


class _LogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen: list[tuple[torch.device, tuple[int, ...]]] = []

    def forward(self, image):
        self.seen.append((image.device, tuple(image.shape)))
        base = image[:, :1] * 0.0 + self.anchor
        orientation = torch.cat((base + 3.0, base + 4.0), dim=1)
        return {
            "center_logits": base + 2.0,
            "band_logits": base,
            "endpoints_logits": base - 2.0,
            "orientation_raw": orientation,
            "distance_raw": base - 1.0,
            "log_variance": base,
        }


def _scene(tmp_path):
    path = tmp_path / "ortho.tif"
    height, width = 35, 43
    data = np.full((3, height, width), 100, dtype=np.uint8)
    data[:, 12, 13] = 0
    transform = from_origin(500_000, 7_400_000, 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:32723",
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(data)
    roi = box(500_005, 7_399_970, 500_038, 7_399_998)
    return validate_scene(path, roi, roi_crs="EPSG:32723", bands=(1, 2, 3))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_tiled_batched_inference_padding_activations_blending_and_mask(tmp_path, device):
    scene = _scene(tmp_path)
    model = _LogitModel()
    engine = TiledInferenceEngine(
        model,
        config=InferenceConfig(
            tile_size=(20, 22),
            overlap=(6, 7),
            batch_size=3,
            pad_multiple=16,
            device=device,
            use_amp=False,
        ),
        expected_bands=(1, 2, 3),
    )

    prediction = engine.predict(scene)

    assert prediction.center.shape == (35, 43)
    assert prediction.valid_mask is not None
    assert prediction.valid_mask[10, 10]
    assert not prediction.valid_mask[0, 0]
    assert not prediction.valid_mask[12, 13]
    assert prediction.center[10, 10] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())
    assert prediction.band[10, 10] == pytest.approx(0.5)
    assert prediction.endpoints[10, 10] == pytest.approx(torch.sigmoid(torch.tensor(-2.0)).item())
    assert prediction.distance[10, 10] == pytest.approx(
        torch.nn.functional.softplus(torch.tensor(-1.0)).item()
    )
    assert prediction.uncertainty[10, 10] > 0.0
    np.testing.assert_allclose(prediction.orientation_sin2[prediction.valid_mask], 0.6, atol=1e-6)
    np.testing.assert_allclose(prediction.orientation_cos2[prediction.valid_mask], 0.8, atol=1e-6)
    assert prediction.center[0, 0] == 0.0
    assert prediction.orientation_sin2[0, 0] == 0.0
    assert prediction.orientation_cos2[0, 0] == 1.0
    assert math.isinf(float(prediction.uncertainty[0, 0]))
    assert model.training
    assert model.seen
    assert all(seen_device.type == device for seen_device, _ in model.seen)
    assert all(
        shape[0] <= 3 and shape[-2] % 16 == 0 and shape[-1] % 16 == 0 for _, shape in model.seen
    )


def test_temperature_and_isotonic_calibration_are_dependency_free_and_monotonic():
    temperature = TemperatureScaler(2.0)
    logits = torch.tensor([-2.0, 0.0, 2.0])
    torch.testing.assert_close(temperature.calibrate_logits(logits), torch.sigmoid(logits / 2.0))

    isotonic = IsotonicCalibrator.fit(
        [0.1, 0.2, 0.3, 0.4, 0.8, 0.9],
        [0.0, 1.0, 0.0, 0.5, 1.0, 1.0],
    )
    calibrated = isotonic.calibrate_probabilities(torch.linspace(0, 1, 21))
    assert torch.all(calibrated[1:] >= calibrated[:-1])
    assert torch.all((calibrated >= 0) & (calibrated <= 1))


def test_engine_applies_temperature_to_logits_before_sigmoid(tmp_path):
    scene = _scene(tmp_path)
    result = TiledInferenceEngine(
        _LogitModel(),
        config=InferenceConfig(tile_size=64, overlap=0, device="cpu"),
        calibrators={"center": TemperatureScaler(2.0)},
    ).predict(scene)

    assert result.center[10, 10] == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item())


def test_checkpoint_loader_requires_local_metadata_and_exact_band_order(tmp_path):
    source = nn.Conv2d(3, 2, kernel_size=1)
    with torch.no_grad():
        source.weight.fill_(0.25)
        source.bias.fill_(-0.5)
    path = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": {f"model.{key}": value for key, value in source.state_dict().items()},
            "metadata": {
                "model_version": "test-v1",
                "band_indexes": [1, 2, 3],
                "in_channels": 3,
                "normalization": {"mean": [0.1, 0.2, 0.3], "std": [1, 1, 1]},
                "calibration": {"center": {"type": "temperature", "temperature": 1.5}},
            },
        },
        path,
    )
    target = nn.Conv2d(3, 2, kernel_size=1)

    loaded = load_local_checkpoint(target, path, expected_bands=(1, 2, 3))

    assert loaded.metadata.model_version == "test-v1"
    assert isinstance(loaded.metadata.build_calibrators()["center"], TemperatureScaler)
    torch.testing.assert_close(target.weight, source.weight)
    torch.testing.assert_close(target.bias, source.bias)

    with pytest.raises(CheckpointContractError, match="do not match"):
        load_local_checkpoint(target, path, expected_bands=(3, 2, 1))


def test_checkpoint_loader_has_clear_errors_for_missing_file_and_metadata(tmp_path):
    model = nn.Conv2d(3, 1, 1)
    with pytest.raises(CheckpointContractError, match="does not exist"):
        load_local_checkpoint(model, tmp_path / "missing.ckpt")
    with pytest.raises(CheckpointContractError, match="local filesystem"):
        load_local_checkpoint(model, "https://example.invalid/model.ckpt")

    path = tmp_path / "no_metadata.ckpt"
    torch.save({"state_dict": model.state_dict()}, path)
    with pytest.raises(CheckpointContractError, match="metadata"):
        load_local_checkpoint(model, path)


def test_engine_rejects_checkpoint_band_contract_mismatch_before_reading(tmp_path):
    scene = _scene(tmp_path)
    engine = TiledInferenceEngine(
        _LogitModel(),
        config=InferenceConfig(tile_size=32, overlap=8, device="cpu"),
        expected_bands=(3, 2, 1),
    )

    with pytest.raises(CheckpointContractError, match="do not match"):
        engine.predict(scene)
