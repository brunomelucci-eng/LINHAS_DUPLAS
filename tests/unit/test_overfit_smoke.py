"""Overfit and end-to-end training smoke test for RowGraphNet."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from rowgraphnet.data.dataset import RowGraphNetDataset
from rowgraphnet.data.manifest import load_scene_manifests
from rowgraphnet.losses import RowGraphNetLoss
from rowgraphnet.models.rowgraphnet import RowGraphNet
from rowgraphnet.training.checkpoint import (
    TrainingMetadata,
    load_training_checkpoint,
    save_training_checkpoint,
)
from rowgraphnet.training.core import RowGraphNetTrainingCore
from rowgraphnet.training.optim import AdamWConfig

pytestmark = [pytest.mark.integration, pytest.mark.real_data, pytest.mark.slow]


def test_multitask_loss_and_overfit_on_real_tiles() -> None:
    """Verify that all multitask loss heads produce finite gradients and overfit on small tile batch."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_files = sorted(list(Path("data_set/manifests").glob("*.yaml")))
    if not manifest_files:
        pytest.skip("private real-data manifests are not installed")
    manifests = load_scene_manifests(manifest_files)
    train_manifests = [m for m in manifests if m.split == "train"][:1]
    if not train_manifests:
        pytest.skip("no train manifests available")

    dataset = RowGraphNetDataset(
        train_manifests,
        tile_size=128,
        stride=128,
        min_valid_pixels=10,
        split="train",
        augment=False,
    )
    assert len(dataset) >= 2, "Expected at least 2 patches from scene 41807"

    sample0 = dataset[0]
    sample1 = dataset[1]

    batch_images = torch.stack([sample0["image"], sample1["image"]]).to(device)
    batch_targets = {
        k: torch.stack([sample0["targets"][k], sample1["targets"][k]]).to(device)
        for k in sample0["targets"]
    }

    # Lightweight model for smoke test
    model = RowGraphNet(
        pretrained=False,
        in_channels=3,
        decoder_channels=32,
        refine_connectivity=True,
    ).to(device)

    criterion = RowGraphNetLoss()
    optimizer_config = AdamWConfig(learning_rate=1e-3, weight_decay=1e-4)
    training_core = RowGraphNetTrainingCore(
        model=model,
        criterion=criterion,
        optimizer_config=optimizer_config,
    ).to(device)

    optimizer_setup = training_core.configure_optimizers()

    # Step 1: Initial forward & loss
    initial_output = training_core.optimize_batch(
        {"image": batch_images, "targets": batch_targets},
        optimizer_setup=optimizer_setup,
        max_gradient_norm=1.0,
    )
    initial_loss = initial_output.loss.item()
    assert torch.isfinite(initial_output.loss), f"Loss is non-finite: {initial_loss}"

    # The legacy real annotations contain centerlines but no visibility or
    # cultivated-strip truth. Verify that useful heads remain trainable while
    # unsupported heads are explicitly masked instead of learning fake zeros.
    assert "center" in initial_output.losses
    assert "topology" in initial_output.losses
    assert "orientation" in initial_output.losses
    assert "distance" in initial_output.losses
    assert torch.count_nonzero(batch_targets["band_valid_mask"]) == 0
    assert torch.count_nonzero(batch_targets["endpoints_valid_mask"]) == 0
    assert torch.count_nonzero(batch_targets["repair_valid_mask"]) == 0

    # Run 10 optimization steps to verify loss decrease
    final_loss = initial_loss
    for _ in range(10):
        step_output = training_core.optimize_batch(
            {"image": batch_images, "targets": batch_targets},
            optimizer_setup=optimizer_setup,
            max_gradient_norm=1.0,
        )
        assert torch.isfinite(step_output.loss)
        final_loss = step_output.loss.item()

    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    # Verify atomic checkpoint saving and restoring
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "smoke_best.ckpt"
        metadata = TrainingMetadata.from_model(
            model,
            model_version="rowgraphnet-smoke-test",
            band_indexes=(1, 2, 3),
            normalization=dataset.normalization,
        )

        # Save checkpoint
        save_training_checkpoint(
            ckpt_path,
            model,
            metadata,
            criterion=criterion,
            optimizer=optimizer_setup.optimizer,
            scheduler=optimizer_setup.scheduler,
            epoch=1,
            overwrite=True,
            extra_state={"best_val_loss": final_loss},
        )
        assert ckpt_path.is_file()

        # Overwrite again to ensure no FileExistsError
        save_training_checkpoint(
            ckpt_path,
            model,
            metadata,
            epoch=2,
            overwrite=True,
        )

        # Load checkpoint and verify restored model
        restored_model = RowGraphNet(
            pretrained=False,
            in_channels=3,
            decoder_channels=32,
            refine_connectivity=True,
        ).to(device)

        load_info = load_training_checkpoint(ckpt_path, restored_model, map_location=device)
        assert load_info["epoch"] == 2
        for p1, p2 in zip(model.parameters(), restored_model.parameters(), strict=True):
            torch.testing.assert_close(p1, p2)
