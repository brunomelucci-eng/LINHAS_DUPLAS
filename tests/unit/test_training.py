from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from rowgraphnet.inference import load_local_checkpoint, read_local_checkpoint_metadata
from rowgraphnet.losses import RowGraphNetLoss
from rowgraphnet.training import (
    LIGHTNING_AVAILABLE,
    AdamWConfig,
    RowGraphNetLightningModule,
    RowGraphNetTrainingCore,
    SchedulerConfig,
    TrainingCheckpointError,
    TrainingMetadata,
    accumulation_group_size,
    architecture_from_model,
    load_training_config,
    save_training_checkpoint,
    seed_everything,
    seeded_generator,
)


class _TinyCenterModel(nn.Module):
    in_channels = 3

    def __init__(self) -> None:
        super().__init__()
        self.center_head = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, image):
        logits = self.center_head(image)
        return {"center_logits": logits, "center": torch.sigmoid(logits)}


class _FakeEncoder(nn.Module):
    in_channels = 3

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 1, kernel_size=1)
        self.config = SimpleNamespace(
            _name_or_path="offline/tiny-segformer",
            to_dict=lambda: {
                "num_channels": 3,
                "depths": [1, 1, 1, 1],
                "hidden_sizes": [8, 16, 32, 64],
                "patch_sizes": [7, 3, 3, 3],
                "strides": [4, 2, 2, 2],
                "num_attention_heads": [1, 1, 2, 4],
                "sr_ratios": [8, 4, 2, 1],
                "mlp_ratios": [2, 2, 2, 2],
            },
        )


class _RowLikeModel(_TinyCenterModel):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _FakeEncoder()
        self.decoder = nn.Identity()
        self.decoder.decoder_channels = 16
        self.connectivity_refiner = None


def test_all_shipped_training_configs_follow_the_strict_runner_schema() -> None:
    config_paths = sorted(Path("configs/train").glob("*.yaml"))
    assert config_paths
    loaded = [load_training_config(path) for path in config_paths]
    assert all(config.centerline_buffer_m == pytest.approx(0.10) for config in loaded)


def _batch() -> dict[str, object]:
    image = torch.randn(2, 3, 9, 11)
    center = torch.zeros(2, 1, 9, 11)
    center[:, :, 4, 2:9] = 1.0
    valid = torch.ones(2, 9, 11, dtype=torch.bool)
    return {
        "image": image,
        "targets": {
            # ``center`` preserves compatibility with the baseline loss while
            # the explicit observed/complete targets exercise the new contract.
            "center": center,
            "center_observed": center,
            "center_complete": center,
            "valid_mask": valid,
            "center_observed_valid_mask": valid,
            "center_complete_valid_mask": valid,
        },
    }


def _architecture() -> dict[str, object]:
    return {
        "model_name": "offline/tiny",
        "decoder_channels": 8,
        "refine_connectivity": False,
        "encoder_config": {
            "num_channels": 3,
            "depths": [1],
            "hidden_sizes": [8],
            "patch_sizes": [3],
            "strides": [2],
        },
    }


def test_plain_pytorch_training_and_validation_steps_use_multitask_loss() -> None:
    model = _TinyCenterModel()
    core = RowGraphNetTrainingCore(
        model,
        criterion=RowGraphNetLoss(learned_weighting=False, cldice_iterations=2),
        optimizer_config=AdamWConfig(learning_rate=1e-2, weight_decay=0.0),
        scheduler_config=SchedulerConfig(name="cosine", interval="step", total_steps=2),
    )
    setup = core.configure_optimizers()
    before = model.center_head.weight.detach().clone()

    train_result = core.optimize_batch(_batch(), setup, max_gradient_norm=1.0)

    assert train_result.loss.ndim == 0
    assert torch.isfinite(train_result.loss)
    assert train_result.batch_size == 2
    assert "train/total" in train_result.metrics("train")
    assert not torch.equal(model.center_head.weight.detach(), before)
    assert setup.optimizer.param_groups[0]["lr"] < 1e-2

    core.eval()
    validation_result = core.validation_step(_batch())
    assert not validation_result.loss.requires_grad
    assert "val/total" in validation_result.metrics("val")


def test_seed_controls_python_numpy_torch_and_dataloader_generator() -> None:
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        first_state = seed_everything(1234, deterministic=True)
        first = (random.random(), np.random.rand(), torch.rand(3))
        seed_everything(1234, deterministic=True)
        second = (random.random(), np.random.rand(), torch.rand(3))

        assert first_state.seed == 1234
        assert first_state.deterministic_algorithms
        assert first[0] == second[0]
        assert first[1] == second[1]
        torch.testing.assert_close(first[2], second[2])
        left = torch.rand(4, generator=seeded_generator(77))
        right = torch.rand(4, generator=seeded_generator(77))
        torch.testing.assert_close(left, right)
    finally:
        torch.use_deterministic_algorithms(was_deterministic)


def test_architecture_capture_keeps_complete_encoder_configuration() -> None:
    architecture = architecture_from_model(_RowLikeModel())

    assert architecture["model_name"] == "offline/tiny-segformer"
    assert architecture["decoder_channels"] == 16
    assert architecture["refine_connectivity"] is False
    encoder_config = architecture["encoder_config"]
    assert encoder_config["hidden_sizes"] == [8, 16, 32, 64]
    assert encoder_config["num_attention_heads"] == [1, 1, 2, 4]


def test_checkpoint_is_inference_compatible_and_contains_required_metadata(tmp_path) -> None:
    source = _TinyCenterModel()
    metadata = TrainingMetadata.from_model(
        source,
        model_version="rowgraphnet-test-v1",
        band_indexes=(1, 2, 3),
        normalization={"mean": [0.1, 0.2, 0.3], "std": [1.0, 1.0, 1.0]},
        calibration={"center": {"type": "temperature", "temperature": 1.2}},
        architecture=_architecture(),
    )
    path = save_training_checkpoint(tmp_path / "model.ckpt", source, metadata)

    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw["metadata"]) >= {
        "model_version",
        "band_indexes",
        "in_channels",
        "normalization",
        "calibration",
        "architecture",
    }
    assert raw["metadata"]["architecture"]["encoder_config"]["depths"] == [1]
    loaded_metadata = read_local_checkpoint_metadata(path)
    assert loaded_metadata.model_version == "rowgraphnet-test-v1"
    assert loaded_metadata.band_indexes == (1, 2, 3)

    target = _TinyCenterModel()
    load_local_checkpoint(target, path, expected_bands=(1, 2, 3))
    for expected, actual in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)
    with pytest.raises(FileExistsError):
        save_training_checkpoint(path, source, metadata)

    # Verify that overwrite=True succeeds without FileExistsError
    save_training_checkpoint(path, source, metadata, overwrite=True)

    # Verify load_training_checkpoint
    from rowgraphnet.training.checkpoint import load_training_checkpoint
    restored = _TinyCenterModel()
    info = load_training_checkpoint(path, restored)
    assert info["metadata"]["model_version"] == "rowgraphnet-test-v1"
    for expected, actual in zip(source.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


@pytest.mark.real_data
def test_training_readiness_passes_on_current_manifests() -> None:
    from rowgraphnet.training.readiness import verify_training_readiness
    report = verify_training_readiness("data_set/manifests", "checkpoints")
    assert report.total_scenes > 0
    assert report.train_scenes > 0
    assert report.val_scenes > 0
    assert report.groups_count > 0


def test_checkpoint_metadata_rejects_missing_encoder_config_and_channel_mismatch() -> None:
    model = _TinyCenterModel()
    with pytest.raises(TrainingCheckpointError, match="encoder_config"):
        TrainingMetadata.from_model(
            model,
            model_version="v1",
            band_indexes=(1, 2, 3),
            architecture={"decoder_channels": 8},
        )
    with pytest.raises(TrainingCheckpointError, match="number of ordered band_indexes"):
        TrainingMetadata(
            model_version="v1",
            band_indexes=(1, 2),
            in_channels=3,
            architecture=_architecture(),
        )


def test_missing_lightning_dependency_has_actionable_error() -> None:
    if LIGHTNING_AVAILABLE:
        pytest.skip("Lightning optional dependency is installed")
    metadata = TrainingMetadata.from_model(
        _TinyCenterModel(),
        model_version="v1",
        band_indexes=(1, 2, 3),
        architecture=_architecture(),
    )
    with pytest.raises(ImportError, match=r"\[train\]"):
        RowGraphNetLightningModule(_TinyCenterModel(), metadata=metadata)


def test_strict_yaml_uses_cli_overrides_last_and_rejects_unknown_keys(tmp_path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "training:\n  epochs: 100\n  batch_size: 2\nmodel:\n  in_channels: 3\n",
        encoding="utf-8",
    )
    config = load_training_config(config_path, overrides={"epochs": 1, "tile_size": 64})
    assert config.epochs == 1
    assert config.tile_size == 64
    assert config.batch_size == 2

    config_path.write_text("training:\n  typo_epochs: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="typo_epochs"):
        load_training_config(config_path)


def test_centerline_buffer_is_loaded_as_per_side_radius(tmp_path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "dataset:\n  centerline_buffer_m: 0.10\n",
        encoding="utf-8",
    )
    config = load_training_config(config_path)
    assert config.centerline_buffer_m == pytest.approx(0.10)


def test_incomplete_accumulation_group_uses_its_actual_size() -> None:
    assert [accumulation_group_size(i, 10, 4) for i in range(10)] == [4] * 8 + [2, 2]
