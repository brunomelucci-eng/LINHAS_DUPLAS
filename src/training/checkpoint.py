import torch
import os
import sys
import datetime
import logging
import json
import hashlib
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def save_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    metrics: Dict[str, Any],
    config: dict,
    filepath: str
):
    import numpy as np
    import rasterio
    import geopandas as gpd
    import shapely
    
    output_dir = config.get('project', {}).get('output_dir', 'outputs')
    manifest_path = os.path.join(output_dir, 'dataset', 'manifest.json')
    manifest_hash = None
    gsd_report = None
    if os.path.exists(manifest_path):
        with open(manifest_path, 'rb') as handle:
            manifest_hash = hashlib.sha256(handle.read()).hexdigest()
    gsd_path = os.path.join(output_dir, 'dataset_gsd_report.csv')
    if os.path.exists(gsd_path):
        with open(gsd_path, encoding='utf-8') as handle:
            gsd_report = handle.read()
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'config': config,
        'dataset_manifest_hash': manifest_hash,
        'gsd_report': gsd_report,
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
            'rasterio_version': rasterio.__version__,
            'geopandas_version': gpd.__version__,
            'shapely_version': shapely.__version__,
            'seed': config.get('project', {}).get('seed', 42)
        }
    }
    
    torch.save(state, filepath)
    logger.debug(f"Saved checkpoint to {filepath}")

def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None
) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
    state = torch.load(filepath, map_location='cpu', weights_only=False)
    
    model.load_state_dict(state['model_state_dict'])
    logger.info(f"Loaded model weights from checkpoint: {filepath} (Epoch {state['epoch']})")
    
    if optimizer and state.get('optimizer_state_dict'):
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if scheduler and state.get('scheduler_state_dict'):
        scheduler.load_state_dict(state['scheduler_state_dict'])
    if scaler and state.get('scaler_state_dict') and scaler:
        scaler.load_state_dict(state['scaler_state_dict'])
        
    return state
