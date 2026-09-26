import json

import torch
from torch.utils.data import DataLoader, Dataset

from src.training.threshold_calibration import (
    apply_checkpoint_threshold,
    calibrate_center_threshold,
    calibration_path,
)


class _ValidationDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        image = torch.zeros((3, 4, 4), dtype=torch.float32)
        image[0, 1:3, 1:3] = 0.85
        target = torch.zeros((1, 4, 4), dtype=torch.float32)
        target[:, 1:3, 1:3] = 1.0
        return image, {
            "center_mask": target,
            "valid_mask": torch.ones_like(target),
        }


class _CalibratedModel(torch.nn.Module):
    def predict(self, images):
        output = torch.zeros(
            (images.shape[0], 4, images.shape[2], images.shape[3]),
            device=images.device,
        )
        output[:, 1] = images[:, 0]
        return output


def test_calibration_sidecar_is_applied_to_inference_config(tmp_path):
    checkpoint = str(tmp_path / "best.pt")
    config = {
        "training": {
            "threshold_calibration": {
                "min_threshold": 0.30,
                "max_threshold": 0.90,
                "threshold_step": 0.10,
                "minimum_validation_dice": 0.90,
                "max_predicted_fraction": 0.40,
            }
        },
        "inference": {"threshold_calibration": {"enabled": True, "required": True}},
        "postprocessing": {"center_threshold": 0.50},
    }
    result = calibrate_center_threshold(
        _CalibratedModel(),
        DataLoader(_ValidationDataset(), batch_size=1),
        torch.device("cpu"),
        config,
        checkpoint,
    )

    assert result["status"] == "approved"
    assert result["validation_dice"] == 1.0
    with open(calibration_path(checkpoint), encoding="utf-8") as handle:
        assert json.load(handle)["center_threshold"] == result["center_threshold"]

    applied = apply_checkpoint_threshold(config, checkpoint)
    assert applied["status"] == "approved"
    assert config["postprocessing"]["center_threshold"] == result["center_threshold"]


def test_rejected_calibration_cannot_be_used(tmp_path):
    checkpoint = str(tmp_path / "best.pt")
    with open(calibration_path(checkpoint), "w", encoding="utf-8") as handle:
        json.dump({"status": "rejected", "failure_reason": "low dice"}, handle)
    config = {
        "inference": {"threshold_calibration": {"enabled": True, "required": True}}
    }
    try:
        apply_checkpoint_threshold(config, checkpoint)
    except ValueError as error:
        assert "not approved" in str(error)
    else:
        raise AssertionError("Rejected calibration should block inference")
