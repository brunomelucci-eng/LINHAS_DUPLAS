import argparse
import sys
import os
import torch
import logging

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.models import build_model
from src.training.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Export weights from checkpoint.")
    parser.add_argument('--config', type=str, default='configs/unet_resnet34.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help="Checkpoint file path.")
    parser.add_argument('--output', type=str, required=True, help="Export output weights path.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    model = build_model(config)
    
    load_checkpoint(args.checkpoint, model)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Model weights successfully exported to {args.output}")

if __name__ == '__main__':
    main()
