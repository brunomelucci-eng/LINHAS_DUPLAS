"""Validation-based calibration for the centerline probability threshold."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import numpy as np
import torch


logger = logging.getLogger(__name__)


def calibration_path(checkpoint_path: str) -> str:
    return f"{checkpoint_path}.threshold.json"


def calibrate_center_threshold(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    config: dict,
    checkpoint_path: str,
) -> Dict[str, Any]:
    """Choose a threshold using the complete validation split, not inference data."""
    cfg = config.get("training", {}).get("threshold_calibration", {})
    threshold_min = float(cfg.get("min_threshold", 0.30))
    threshold_max = float(cfg.get("max_threshold", 0.90))
    threshold_step = float(cfg.get("threshold_step", 0.02))
    thresholds = np.arange(
        threshold_min,
        threshold_max + threshold_step / 2.0,
        threshold_step,
        dtype=np.float64,
    )
    true_positive = np.zeros_like(thresholds)
    predicted_positive = np.zeros_like(thresholds)
    target_positive = 0.0
    valid_pixels = 0.0

    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            center_target = targets["center_mask"].to(device) >= 0.5
            valid = targets["valid_mask"].to(device) >= 0.5
            center_target &= valid
            probabilities = model.predict(images)[:, 1:2]

            target_positive += float(center_target.sum().item())
            valid_pixels += float(valid.sum().item())
            for index, threshold in enumerate(thresholds):
                prediction = (probabilities >= float(threshold)) & valid
                predicted_positive[index] += float(prediction.sum().item())
                true_positive[index] += float((prediction & center_target).sum().item())

    if valid_pixels <= 0 or target_positive <= 0:
        raise ValueError(
            "Threshold calibration requires valid positive centerline pixels in validation."
        )

    dice = (2.0 * true_positive) / np.maximum(
        predicted_positive + target_positive, 1.0
    )
    predicted_fraction = predicted_positive / valid_pixels
    target_fraction = target_positive / valid_pixels
    predicted_to_target = predicted_positive / max(target_positive, 1.0)

    max_predicted_fraction = float(cfg.get("max_predicted_fraction", 0.25))
    max_predicted_to_target = float(cfg.get("max_predicted_to_target_ratio", 3.0))
    min_predicted_to_target = float(cfg.get("min_predicted_to_target_ratio", 0.25))
    eligible = (
        (predicted_fraction <= max_predicted_fraction)
        & (predicted_to_target <= max_predicted_to_target)
        & (predicted_to_target >= min_predicted_to_target)
    )
    if eligible.any():
        candidate_scores = np.where(eligible, dice, -1.0)
        best_index = int(np.argmax(candidate_scores))
        eligibility_failure = None
    else:
        best_index = int(np.argmax(dice))
        eligibility_failure = "no_threshold_satisfied_occupancy_limits"

    minimum_dice = float(cfg.get("minimum_validation_dice", 0.30))
    approved = eligibility_failure is None and float(dice[best_index]) >= minimum_dice
    result = {
        "status": "approved" if approved else "rejected",
        "center_threshold": round(float(thresholds[best_index]), 6),
        "validation_dice": float(dice[best_index]),
        "predicted_fraction": float(predicted_fraction[best_index]),
        "target_fraction": float(target_fraction),
        "predicted_to_target_ratio": float(predicted_to_target[best_index]),
        "minimum_validation_dice": minimum_dice,
        "failure_reason": eligibility_failure,
        "checkpoint": os.path.abspath(checkpoint_path),
        "threshold_candidates": [
            {
                "threshold": round(float(threshold), 6),
                "dice": float(dice[index]),
                "predicted_fraction": float(predicted_fraction[index]),
                "predicted_to_target_ratio": float(predicted_to_target[index]),
                "eligible": bool(eligible[index]),
            }
            for index, threshold in enumerate(thresholds)
        ],
    }
    output_path = calibration_path(checkpoint_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    logger.info(
        "Center threshold calibration: status=%s threshold=%.3f dice=%.4f "
        "predicted_fraction=%.4f target_fraction=%.4f report=%s",
        result["status"],
        result["center_threshold"],
        result["validation_dice"],
        result["predicted_fraction"],
        result["target_fraction"],
        output_path,
    )
    return result


def apply_checkpoint_threshold(config: dict, checkpoint_path: str) -> Dict[str, Any] | None:
    """Load an approved sidecar and apply its threshold to a run config."""
    cfg = config.get("inference", {}).get("threshold_calibration", {})
    if not cfg.get("enabled", False):
        return None
    path = calibration_path(checkpoint_path)
    required = bool(cfg.get("required", False))
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"Required threshold calibration report not found: {path}"
            )
        logger.warning("Threshold calibration report not found: %s", path)
        return None
    with open(path, encoding="utf-8") as handle:
        calibration = json.load(handle)
    if calibration.get("status") != "approved":
        raise ValueError(
            "Checkpoint threshold calibration is not approved: "
            f"{path} ({calibration.get('failure_reason') or 'low validation Dice'})"
        )
    threshold = float(calibration["center_threshold"])
    post_cfg = config.setdefault("postprocessing", {})
    post_cfg["threshold_mode"] = "single"
    post_cfg["center_threshold"] = threshold
    post_cfg["center_threshold_high"] = threshold
    post_cfg["center_threshold_low"] = threshold
    logger.info("Applied calibrated center threshold %.3f from %s", threshold, path)
    return calibration
