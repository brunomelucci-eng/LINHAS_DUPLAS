"""Framework-independent training steps for RowGraphNet.

The core deliberately depends only on PyTorch.  Lightning integration lives in
``rowgraphnet.training.lightning`` and delegates to the same batch/loss
contract, so a missing optional training dependency never prevents unit tests
or inference-only installations from importing this package.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from rowgraphnet.losses import MultiTaskLoss, RowGraphNetLoss
from rowgraphnet.training.optim import (
    AdamWConfig,
    OptimizerSetup,
    SchedulerConfig,
    configure_adamw,
)

IMAGE_KEYS = ("image", "images", "pixel_values")


@dataclass(slots=True)
class TrainingStepOutput:
    """Result shared by the plain PyTorch and Lightning training paths."""

    loss: Tensor
    losses: dict[str, Tensor]
    predictions: Mapping[str, Tensor]
    batch_size: int

    def metrics(self, prefix: str, *, detach: bool = True) -> dict[str, Tensor]:
        """Return loss tensors under stable ``train/...`` or ``val/...`` names."""

        normalized = prefix.strip().rstrip("/")
        if not normalized:
            raise ValueError("metric prefix must not be empty")
        return {
            f"{normalized}/{name}": value.detach() if detach else value
            for name, value in self.losses.items()
        }


def unpack_training_batch(batch: Any) -> tuple[Tensor, Mapping[str, Tensor]]:
    """Extract images and multitask targets from common DataLoader layouts.

    Accepted layouts are ``(images, targets)``, ``{"image": ..., "targets":
    {...}}``, and a flat mapping where all entries except the image are targets.
    Keeping this adaptation here makes the loss contract identical for plain
    PyTorch and Lightning callers.
    """

    images: Any
    targets: Any
    if isinstance(batch, (tuple, list)):
        if len(batch) != 2:
            raise ValueError("sequence batches must contain exactly (images, targets)")
        images, targets = batch
    elif isinstance(batch, Mapping):
        image_key = next((key for key in IMAGE_KEYS if key in batch), None)
        if image_key is None:
            raise KeyError(f"mapping batch must contain one image key from {IMAGE_KEYS}")
        images = batch[image_key]
        if "targets" in batch:
            targets = batch["targets"]
        else:
            targets = {
                str(key): value
                for key, value in batch.items()
                if key != image_key and isinstance(value, Tensor)
            }
    else:
        raise TypeError("batch must be a mapping or an (images, targets) sequence")

    if not isinstance(images, Tensor):
        raise TypeError("batch images must be a torch.Tensor")
    if images.ndim < 1:
        raise ValueError("batch images must include a batch dimension")
    if not isinstance(targets, Mapping):
        raise TypeError("batch targets must be a mapping")
    if any(not isinstance(key, str) for key in targets):
        raise TypeError("target names must be strings")
    return images, targets


def compute_training_step(
    model: nn.Module,
    criterion: RowGraphNetLoss | MultiTaskLoss,
    batch: Any,
) -> TrainingStepOutput:
    """Run the model and composite loss without changing optimizer state."""

    images, targets = unpack_training_batch(batch)
    predictions = model(images)
    if not isinstance(predictions, Mapping):
        raise TypeError("RowGraphNet model must return a mapping of named predictions")
    if any(
        not isinstance(key, str) or not isinstance(value, Tensor)
        for key, value in predictions.items()
    ):
        raise TypeError("model predictions must map string names to tensors")

    losses = criterion(predictions, targets)
    if not isinstance(losses, Mapping):
        raise TypeError("multitask criterion must return a mapping of named losses")
    normalized_losses = dict(losses)
    total = normalized_losses.get("total")
    if not isinstance(total, Tensor) or total.ndim != 0:
        raise ValueError("multitask criterion must return a scalar tensor named 'total'")
    if not bool(torch.isfinite(total).detach().cpu()):
        raise FloatingPointError("non-finite total loss encountered")
    if any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in normalized_losses.items()
    ):
        raise TypeError("criterion losses must map string names to tensors")
    return TrainingStepOutput(
        loss=total,
        losses=normalized_losses,
        predictions=predictions,
        batch_size=int(images.shape[0]),
    )


class RowGraphNetTrainingCore(nn.Module):
    """Small, directly testable PyTorch training module.

    ``training_step`` computes differentiable losses but intentionally leaves
    zeroing gradients, backward, stepping and mixed precision to the caller.
    ``optimize_batch`` is provided for the common eager PyTorch case.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        criterion: RowGraphNetLoss | MultiTaskLoss | None = None,
        optimizer_config: AdamWConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion if criterion is not None else RowGraphNetLoss()
        self.optimizer_config = optimizer_config or AdamWConfig()
        self.scheduler_config = scheduler_config or SchedulerConfig()

    def forward(self, images: Tensor, **kwargs: Any) -> Mapping[str, Tensor]:
        output = self.model(images, **kwargs)
        if not isinstance(output, Mapping):
            raise TypeError("RowGraphNet model must return a mapping of named predictions")
        return output

    def training_step(self, batch: Any) -> TrainingStepOutput:
        """Compute one differentiable training step."""

        return compute_training_step(self.model, self.criterion, batch)

    @torch.no_grad()
    def validation_step(self, batch: Any) -> TrainingStepOutput:
        """Compute one validation step without constructing an autograd graph."""

        return compute_training_step(self.model, self.criterion, batch)

    def configure_optimizers(self) -> OptimizerSetup:
        """Create AdamW over both model and learned multitask-loss parameters."""

        return configure_adamw(
            self,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
        )

    def optimize_batch(
        self,
        batch: Any,
        optimizer_setup: OptimizerSetup,
        *,
        max_gradient_norm: float | None = None,
        scaler: Any | None = None,
        autocast_device: str | None = None,
    ) -> TrainingStepOutput:
        """Execute a complete eager PyTorch update and return detached metrics."""

        if max_gradient_norm is not None and max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive when provided")
        self.train()
        optimizer_setup.optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(autocast_device or "cuda"):
                result = self.training_step(batch)
            scaler.scale(result.loss).backward()
            if max_gradient_norm is not None:
                scaler.unscale_(optimizer_setup.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_gradient_norm)
            scaler.step(optimizer_setup.optimizer)
            scaler.update()
        else:
            result = self.training_step(batch)
            result.loss.backward()
            if max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_gradient_norm)
            optimizer_setup.optimizer.step()

        if optimizer_setup.scheduler is not None and optimizer_setup.interval == "step":
            optimizer_setup.step_scheduler()
        return result


# Concise alias for callers not tied to the project name.
TrainingCore = RowGraphNetTrainingCore


__all__ = [
    "RowGraphNetTrainingCore",
    "TrainingCore",
    "TrainingStepOutput",
    "compute_training_step",
    "unpack_training_batch",
]
