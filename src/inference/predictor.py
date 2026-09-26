import torch
import numpy as np
from typing import Optional, Sequence
from ..models import build_model
from ..training.checkpoint import load_checkpoint

class RowPredictor:
    def __init__(self, config: dict, checkpoint_path: Optional[str] = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = build_model(config)
        
        if checkpoint_path:
            load_checkpoint(checkpoint_path, self.model)
            
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(
        self,
        patches: np.ndarray,
        output_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """
        Args:
            patches: numpy array of shape (B, C, H, W) normalized to [0, 1]
        Returns:
            numpy array of shape (B, 4, H, W) containing predictions
        """
        patches_t = torch.from_numpy(patches)
        if self.device.type == 'cuda':
            patches_t = patches_t.pin_memory().to(self.device, non_blocking=True)
        else:
            patches_t = patches_t.to(self.device)
        
        inf_cfg = self.config.get('inference', {})
        mixed_precision = inf_cfg.get('mixed_precision', True)
        device_type = self.device.type
        
        from torch.amp import autocast
        with torch.no_grad():
            with autocast(device_type=device_type, enabled=(device_type == 'cuda' and mixed_precision)):
                outputs = self.model.predict(patches_t)
                # Select only requested channels while the tensor is still on
                # the GPU, avoiding needless PCIe transfers of unused maps.
                if output_indices is not None:
                    outputs = outputs[:, list(output_indices)]
                outputs = outputs.to('cpu', non_blocking=(device_type == 'cuda'))
                if device_type == 'cuda':
                    torch.cuda.current_stream(self.device).synchronize()
                outputs = outputs.numpy()
                
        return outputs

    def predict_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Args:
            patch: numpy array of shape (C, H, W) normalized to [0, 1]
        Returns:
            numpy array of shape (4, H, W) containing predictions
        """
        patches = np.expand_dims(patch, axis=0)
        outputs = self.predict_batch(patches)
        return outputs[0]
