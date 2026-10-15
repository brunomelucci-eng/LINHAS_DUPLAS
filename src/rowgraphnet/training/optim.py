"""Optimizer and learning-rate scheduler construction."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)

SchedulerName = Literal["none", "cosine", "step", "plateau"]
SchedulerInterval = Literal["step", "epoch"]


@dataclass(frozen=True, slots=True)
class AdamWConfig:
    """Serializable AdamW settings."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    amsgrad: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if len(self.betas) != 2 or any(
            not math.isfinite(value) or not 0 <= value < 1 for value in self.betas
        ):
            raise ValueError("betas must contain two values in [0, 1)")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Settings shared by the supported schedulers.

    ``total_steps`` is used by cosine annealing, ``step_size`` by StepLR, and
    ``patience`` by ReduceLROnPlateau. These are optimization controls rather
    than agronomic geometry thresholds.
    """

    name: SchedulerName = "cosine"
    interval: SchedulerInterval = "epoch"
    total_steps: int = 100
    minimum_learning_rate: float = 0.0
    step_size: int = 10
    gamma: float = 0.1
    patience: int = 5
    monitor: str = "val/total"

    def __post_init__(self) -> None:
        if self.name not in {"none", "cosine", "step", "plateau"}:
            raise ValueError("unsupported scheduler name")
        if self.interval not in {"step", "epoch"}:
            raise ValueError("interval must be 'step' or 'epoch'")
        if self.total_steps < 1 or self.step_size < 1 or self.patience < 0:
            raise ValueError("scheduler step counts must be positive and patience non-negative")
        if not math.isfinite(self.minimum_learning_rate) or self.minimum_learning_rate < 0:
            raise ValueError("minimum_learning_rate must be finite and non-negative")
        if not math.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if self.name == "plateau" and not self.monitor.strip():
            raise ValueError("plateau scheduling requires a monitor metric")
        if self.name == "plateau" and self.interval != "epoch":
            raise ValueError("plateau scheduling must use interval='epoch'")


@dataclass(slots=True)
class OptimizerSetup:
    """Optimizer plus optional scheduler and its execution contract."""

    optimizer: Optimizer
    scheduler: LRScheduler | ReduceLROnPlateau | None
    interval: SchedulerInterval
    monitor: str | None

    def step_scheduler(self, metric: Tensor | float | None = None) -> None:
        """Advance the scheduler, requiring a metric for plateau scheduling."""

        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError("plateau scheduler requires its monitored validation metric")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def as_lightning_config(self) -> Optimizer | dict[str, object]:
        """Convert to the structure expected by Lightning."""

        if self.scheduler is None:
            return self.optimizer
        scheduler_config: dict[str, object] = {
            "scheduler": self.scheduler,
            "interval": self.interval,
            "frequency": 1,
        }
        if isinstance(self.scheduler, ReduceLROnPlateau):
            scheduler_config["monitor"] = self.monitor
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": scheduler_config,
        }


def configure_adamw(
    module_or_parameters: nn.Module | Iterable[Tensor],
    *,
    optimizer_config: AdamWConfig | None = None,
    scheduler_config: SchedulerConfig | None = None,
) -> OptimizerSetup:
    """Build AdamW and an optional scheduler for plain PyTorch or Lightning."""

    optimizer_config = optimizer_config or AdamWConfig()
    scheduler_config = scheduler_config or SchedulerConfig()
    if isinstance(module_or_parameters, nn.Module):
        parameters = [
            parameter for parameter in module_or_parameters.parameters() if parameter.requires_grad
        ]
    else:
        parameters = [parameter for parameter in module_or_parameters if parameter.requires_grad]
    if not parameters:
        raise ValueError("cannot configure an optimizer without trainable parameters")

    optimizer = AdamW(
        parameters,
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        betas=optimizer_config.betas,
        eps=optimizer_config.epsilon,
        amsgrad=optimizer_config.amsgrad,
    )
    scheduler: LRScheduler | ReduceLROnPlateau | None
    if scheduler_config.name == "none":
        scheduler = None
    elif scheduler_config.name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.total_steps,
            eta_min=scheduler_config.minimum_learning_rate,
        )
    elif scheduler_config.name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_config.step_size,
            gamma=scheduler_config.gamma,
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.gamma,
            patience=scheduler_config.patience,
            min_lr=scheduler_config.minimum_learning_rate,
        )
    return OptimizerSetup(
        optimizer=optimizer,
        scheduler=scheduler,
        interval=scheduler_config.interval,
        monitor=scheduler_config.monitor if scheduler_config.name == "plateau" else None,
    )


__all__ = [
    "AdamWConfig",
    "OptimizerSetup",
    "SchedulerConfig",
    "configure_adamw",
]
