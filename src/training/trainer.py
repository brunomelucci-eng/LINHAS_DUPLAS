import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import logging
import datetime
import re
from typing import Dict, Any, Optional, List
from .checkpoint import save_checkpoint, load_checkpoint
from .evaluator import Evaluator
from .threshold_calibration import calibrate_center_threshold

logger = logging.getLogger(__name__)


def _dataset_image_count(config: dict, output_dir: str) -> int:
    """Return the number of source images actually recorded for the dataset."""
    manifest_path = os.path.join(output_dir, 'dataset', 'manifest.json')
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding='utf-8') as handle:
                manifest = json.load(handle)
            inputs = manifest.get('inputs')
            if isinstance(inputs, list) and inputs:
                return len(inputs)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Could not read dataset image count from %s: %s", manifest_path, exc)

    inputs = config.get('inputs')
    if isinstance(inputs, list) and inputs:
        return len(inputs)
    return 1 if config.get('input') else 0


def _unique_checkpoint_paths(checkpoint_dir: str, image_count: int, now=None):
    """Allocate non-overwriting best/last checkpoint names for one training run."""
    now = now or datetime.datetime.now()
    date_text = now.strftime('%d_%m_%Y')
    stem = f"{date_text}_{image_count}_imagens"
    pattern = re.compile(
        rf"^(?:best|last)_{re.escape(stem)}_run_(\d{{3}})\.pt$"
    )
    existing_runs = []
    if os.path.isdir(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            match = pattern.match(filename)
            if match:
                existing_runs.append(int(match.group(1)))
    run_number = max(existing_runs, default=0) + 1
    suffix = f"{stem}_run_{run_number:03d}.pt"
    return (
        os.path.join(checkpoint_dir, f"best_{suffix}"),
        os.path.join(checkpoint_dir, f"last_{suffix}"),
    )

class Trainer:
    def __init__(self, config: dict, model: torch.nn.Module, loss_fn: torch.nn.Module):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        
        train_cfg = config.get('training', {})
        self.epochs = train_cfg.get('epochs', 100)
        self.batch_size = train_cfg.get('batch_size', 8)
        self.gradient_accumulation_steps = max(1, train_cfg.get('gradient_accumulation_steps', 1))
        self.mixed_precision = train_cfg.get('mixed_precision', True)
        self.grad_clip = train_cfg.get('gradient_clip_norm', 1.0)
        self.patience = train_cfg.get('early_stopping_patience', 15)
        self.monitor = train_cfg.get('monitor', 'val_center_cldice')
        self.monitor_mode = train_cfg.get('monitor_mode', 'max')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        opt_type = train_cfg.get('optimizer', 'adamw').lower()
        lr = train_cfg.get('learning_rate', 3e-4)
        wd = train_cfg.get('weight_decay', 1e-4)
        
        if opt_type == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
            
        # LR Scheduler
        sched_type = train_cfg.get('scheduler', 'plateau')
        if isinstance(sched_type, dict):
            sched_type = sched_type.get('type', 'plateau')
        sched_type = str(sched_type).lower()

        if sched_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=1e-6
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min' if 'loss' in self.monitor else 'max', 
                factor=0.5, 
                patience=5
            )
        
        self.scaler = GradScaler('cuda', enabled=self.mixed_precision)
        
        self.output_dir = config.get('project', {}).get('output_dir', 'outputs')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        image_count = _dataset_image_count(config, self.output_dir)
        self.best_checkpoint_path, self.last_checkpoint_path = _unique_checkpoint_paths(
            self.checkpoint_dir, image_count
        )
        logger.info("Best checkpoint for this run: %s", self.best_checkpoint_path)
        logger.info("Last checkpoint for this run: %s", self.last_checkpoint_path)
        
        self.evaluator = Evaluator(self.device, loss_fn)
        
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> List[Dict[str, Any]]:
        best_metric = -float('inf') if self.monitor_mode == 'max' else float('inf')
        epochs_no_improve = 0
        history = []
        
        logger.info(f"Starting training on device: {self.device}")
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError("Train and validation loaders must both contain at least one batch.")
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")
            self.optimizer.zero_grad(set_to_none=True)
            for batch_index, (images, targets) in enumerate(pbar, start=1):
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Use autocast for mixed precision
                with autocast('cuda', enabled=self.mixed_precision):
                    logits = self.model(images)
                    loss_dict = self.loss_fn(logits, targets)
                    loss = loss_dict['loss']
                    scaled_loss = loss / self.gradient_accumulation_steps
                    
                self.scaler.scale(scaled_loss).backward()
                
                if batch_index % self.gradient_accumulation_steps == 0 or batch_index == len(train_loader):
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                
                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                
            avg_train_loss = train_loss / len(train_loader)
            
            # Run evaluation on validation set
            val_metrics = self.evaluator.evaluate(self.model, val_loader)
            val_loss = val_metrics['val_loss']
            
            # Compute monitored metric (needed for early stopping regardless of scheduler)
            monitor_val = val_metrics.get(self.monitor, val_loss)

            # Step scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(monitor_val)
            else:
                self.scheduler.step()
            
            logger.info(
                f"Epoch {epoch} summary: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_row_dice={val_metrics.get('val_row_dice', 0.0):.4f}, "
                f"val_center_cldice={val_metrics.get('val_center_cldice', 0.0):.4f}, "
                f"val_orientation_loss={val_metrics.get('val_orientation_loss', 0.0):.4f}"
            )
            
            epoch_history = {
                'epoch': epoch,
                'train_loss': avg_train_loss,
                **val_metrics
            }
            history.append(epoch_history)
            
            # Check for improvement on target metric
            is_better = False
            if self.monitor_mode == 'max':
                if monitor_val > best_metric:
                    best_metric = monitor_val
                    is_better = True
            else:
                if monitor_val < best_metric:
                    best_metric = monitor_val
                    is_better = True
                    
            if is_better:
                epochs_no_improve = 0
                logger.info(f"New best model found based on '{self.monitor}'={monitor_val:.4f}. Saving checkpoint.")
                save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    metrics=epoch_history,
                    config=self.config,
                    filepath=self.best_checkpoint_path
                )
            else:
                epochs_no_improve += 1
                
            save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                metrics=epoch_history,
                config=self.config,
                filepath=self.last_checkpoint_path
            )
            
            if epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs of no improvement.")
                break
                
        # Write training history
        history_path = os.path.join(self.output_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        calibration_cfg = self.config.get('training', {}).get(
            'threshold_calibration', {}
        )
        if calibration_cfg.get('enabled', False):
            load_checkpoint(self.best_checkpoint_path, self.model)
            calibration = calibrate_center_threshold(
                self.model,
                val_loader,
                self.device,
                self.config,
                self.best_checkpoint_path,
            )
            if (
                calibration.get('status') != 'approved'
                and calibration_cfg.get('fail_on_rejected', True)
            ):
                raise RuntimeError(
                    'Training produced a checkpoint with rejected center-threshold '
                    'calibration. Inference is blocked; inspect the calibration JSON.'
                )
            
        logger.info(f"Training finished. History written to {history_path}")
        return history
