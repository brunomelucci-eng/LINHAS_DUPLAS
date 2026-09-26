import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader
import logging

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.logging_utils import setup_logging
from src.data.dataset import SugarcaneDataset
from src.models import build_model
from src.losses import CombinedLoss
from src.training.evaluator import Evaluator
from src.training.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Validate the Sugarcane Row AI model checkpoint.")
    parser.add_argument('--config', type=str, default='configs/unet_resnet34.yaml', help="Path to config YAML.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint file (.pt).")
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = config.get('project', {}).get('output_dir', 'outputs')
    setup_logging(output_dir)
    
    logger.info("Initializing validation evaluation...")
    
    dataset_dir = os.path.join(output_dir, 'dataset')
    val_dir = os.path.join(dataset_dir, 'val')
    
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        
    val_dataset = SugarcaneDataset(val_dir, augmentor=None)
    
    num_workers = 0
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('training', {}).get('batch_size', 8),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config)
    
    # Load weights
    load_checkpoint(args.checkpoint, model)
    model.to(device)
    
    loss_fn = CombinedLoss(config)
    evaluator = Evaluator(device, loss_fn)
    
    logger.info(f"Evaluating model on {len(val_dataset)} validation patches...")
    val_metrics = evaluator.evaluate(model, val_loader)
    
    logger.info("=== EVALUATION METRICS ===")
    for k, v in val_metrics.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("==========================")

if __name__ == '__main__':
    main()
