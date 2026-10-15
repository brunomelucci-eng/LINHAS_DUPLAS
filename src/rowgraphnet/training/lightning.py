"""Optional Lightning adapter for the framework-independent training core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from torch import Tensor, nn

from rowgraphnet.losses import MultiTaskLoss, RowGraphNetLoss
from rowgraphnet.training.checkpoint import TrainingMetadata, save_training_checkpoint
from rowgraphnet.training.core import TrainingStepOutput, compute_training_step
from rowgraphnet.training.optim import AdamWConfig, SchedulerConfig, configure_adamw

_LIGHTNING_IMPORT_ERROR: BaseException | None = None
try:  # pragma: no cover - availability depends on the installed extras
    import lightning as _lightning
except (ImportError, ModuleNotFoundError, RuntimeError) as exc:  # pragma: no cover
    _LIGHTNING_IMPORT_ERROR = exc
    _LightningModuleBase = nn.Module
    LIGHTNING_AVAILABLE = False
else:  # pragma: no cover - exercised only when the optional extra is installed
    _LightningModuleBase = _lightning.LightningModule
    LIGHTNING_AVAILABLE = True


def require_lightning() -> None:
    """Raise an actionable error when the optional training extra is absent."""

    if not LIGHTNING_AVAILABLE:
        raise ImportError(
            "Lightning training requires the optional 'lightning' dependency. "
            'Install RowGraphNet with `pip install -e ".[train]"`.'
        ) from _LIGHTNING_IMPORT_ERROR


class RowGraphNetLightningModule(_LightningModuleBase):  # type: ignore[misc, valid-type]
    """Lightning adapter using :class:`RowGraphNetLoss` for train and validation."""

    def __init__(
        self,
        model: nn.Module,
        *,
        metadata: TrainingMetadata | Mapping[str, Any],
        criterion: RowGraphNetLoss | MultiTaskLoss | None = None,
        optimizer_config: AdamWConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
    ) -> None:
        require_lightning()
        super().__init__()
        self.model = model
        self.criterion = criterion if criterion is not None else RowGraphNetLoss()
        self.metadata = (
            metadata
            if isinstance(metadata, TrainingMetadata)
            else TrainingMetadata.from_mapping(metadata)
        )
        self.optimizer_config = optimizer_config or AdamWConfig()
        self.scheduler_config = scheduler_config or SchedulerConfig()
        # Plain dictionaries keep Lightning checkpoints portable and let the
        # inference loader find the same contract in hyper_parameters.
        self.save_hyperparameters(
            {
                "metadata": self.metadata.to_dict(),
                "optimizer_config": asdict(self.optimizer_config),
                "scheduler_config": asdict(self.scheduler_config),
            }
        )

    def forward(self, images: Tensor, **kwargs: Any) -> Mapping[str, Tensor]:
        output = self.model(images, **kwargs)
        if not isinstance(output, Mapping):
            raise TypeError("RowGraphNet model must return a mapping of named predictions")
        return output

    def _step(self, batch: Any, prefix: str) -> TrainingStepOutput:
        result = compute_training_step(self.model, self.criterion, batch)
        for name, value in result.losses.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=prefix == "train",
                on_epoch=True,
                prog_bar=name == "total",
                batch_size=result.batch_size,
                sync_dist=True,
            )
        return result

    def training_step(self, batch: Any, batch_idx: int = 0) -> Tensor:
        del batch_idx
        return self._step(batch, "train").loss

    def validation_step(self, batch: Any, batch_idx: int = 0) -> Tensor:
        del batch_idx
        return self._step(batch, "val").loss

    def configure_optimizers(self) -> Any:
        setup = configure_adamw(
            self,
            optimizer_config=self.optimizer_config,
            scheduler_config=self.scheduler_config,
        )
        return setup.as_lightning_config()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Make mandatory provenance available at the checkpoint root."""

        checkpoint["metadata"] = self.metadata.to_dict()

    def export_inference_checkpoint(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Export model-only weights accepted directly by the inference loader."""

        trainer = getattr(self, "_trainer", None)
        schedulers = trainer.lr_scheduler_configs if trainer is not None else []
        scheduler = schedulers[0].scheduler if len(schedulers) == 1 else None
        optimizers = trainer.optimizers if trainer is not None else []
        optimizer = optimizers[0] if len(optimizers) == 1 else None
        return save_training_checkpoint(
            path,
            self.model,
            self.metadata,
            criterion=self.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=int(self.current_epoch),
            global_step=int(self.global_step),
            overwrite=overwrite,
        )


# Alternative spelling used by some Lightning projects.
LightningRowGraphNet = RowGraphNetLightningModule


__all__ = [
    "LIGHTNING_AVAILABLE",
    "LightningRowGraphNet",
    "RowGraphNetLightningModule",
    "require_lightning",
]
