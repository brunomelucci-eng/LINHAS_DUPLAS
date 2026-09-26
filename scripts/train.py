import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader
import logging
import json

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.seed import set_seed
from src.logging_utils import setup_logging
from src.data.dataset import SugarcaneDataset
from src.data.augmentations import RowAugmentor
from src.models import build_model
from src.losses import CombinedLoss
from src.training import Trainer

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train the Sugarcane Row AI model.")
    parser.add_argument('--config', type=str, default='configs/unet_resnet34.yaml', help="Path to config YAML.")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint .pt to resume training from.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = config.get('project', {}).get('output_dir', 'outputs')
    setup_logging(output_dir)
    
    set_seed(config.get('project', {}).get('seed', 42))
    
    if torch.cuda.is_available():
        # Reserve 20% VRAM for Windows display driver and IDE stability (prevents TDR and Electron crash)
        torch.cuda.set_per_process_memory_fraction(0.80)
        torch.cuda.empty_cache()
    
    logger.info("Initializing training components...")
    
    # 1. Dataset directories
    dataset_dir = os.path.join(output_dir, 'dataset')
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Dataset splits not found in {dataset_dir}. "
            "Please run scripts/prepare_dataset.py first."
        )
    inspection_path = os.path.join(output_dir, 'dataset_inspection_report.json')
    if not os.path.exists(inspection_path):
        raise FileNotFoundError("Dataset inspection report is missing. Run scripts/inspect_data.py first.")
    with open(inspection_path, encoding='utf-8') as handle:
        inspection = json.load(handle)
    if inspection.get('status') != 'approved':
        raise ValueError("Dataset inspection is not approved; training is blocked.")
        
    # 2. Augmentor and Datasets
    augmentor = RowAugmentor(config)
    
    train_dataset = SugarcaneDataset(train_dir, augmentor=augmentor)
    val_dataset = SugarcaneDataset(val_dir, augmentor=None) # No augmentation for validation
    
    # 3. DataLoaders
    train_cfg = config.get('training', {})
    batch_size = train_cfg.get('batch_size', 8)
    num_workers = train_cfg.get('num_workers', 4)
    
    # On Windows, num_workers=0 and pin_memory=False prevents multiprocessing lockups
    # and avoids C-level memory corruption (0xc0000005) in Python 3.12.
    pin_memory = True
    if os.name == 'nt':
        logger.info("Windows detected: setting DataLoader num_workers=0 and pin_memory=False to prevent driver heap corruption.")
        num_workers = 0
        pin_memory = False
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Loaded {len(train_dataset)} training tiles and {len(val_dataset)} validation tiles.")
    
    # 4. Model and Loss
    model = build_model(config)
    loss_fn = CombinedLoss(config)
    
    # 5. Trainer
    trainer = Trainer(config, model, loss_fn, resume_checkpoint=args.resume)
    
    # 6. Fit
    trainer.fit(train_loader, val_loader)
    logger.info("Model training completed successfully!")

if __name__ == '__main__':
    main()
