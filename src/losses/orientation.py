import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MaskedOrientationLoss(nn.Module):
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self, 
        pred_sin: torch.Tensor, 
        pred_cos: torch.Tensor, 
        target_sin: torch.Tensor, 
        target_cos: torch.Tensor, 
        weight_mask: torch.Tensor, 
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate Masked Orientation Loss.
        Args:
            pred_sin: prediction tensor (B, 1, H, W)
            pred_cos: prediction tensor (B, 1, H, W)
            target_sin: target tensor (B, 1, H, W)
            target_cos: target tensor (B, 1, H, W)
            weight_mask: mask indicating positive row region (B, 1, H, W)
            valid_mask: optional valid pixel mask (B, 1, H, W)
        """
        total_mask = weight_mask.clone()
        if valid_mask is not None:
            total_mask = total_mask * valid_mask
            
        mask_sum = total_mask.sum()
        if mask_sum < 1e-5:
            # Return dummy loss with gradient support to avoid PyTorch errors
            return (pred_sin * 0.0 + pred_cos * 0.0).sum()
            
        if self.loss_type == 'mse':
            loss_sin = F.mse_loss(pred_sin, target_sin, reduction='none')
            loss_cos = F.mse_loss(pred_cos, target_cos, reduction='none')
            loss = loss_sin + loss_cos
        elif self.loss_type == 'smooth_l1':
            loss_sin = F.smooth_l1_loss(pred_sin, target_sin, reduction='none')
            loss_cos = F.smooth_l1_loss(pred_cos, target_cos, reduction='none')
            loss = loss_sin + loss_cos
        elif self.loss_type == 'cosine':
            eps = 1e-8
            pred_norm = torch.sqrt(pred_sin**2 + pred_cos**2 + eps)
            pred_sin_n = pred_sin / pred_norm
            pred_cos_n = pred_cos / pred_norm
            
            target_norm = torch.sqrt(target_sin**2 + target_cos**2 + eps)
            target_sin_n = target_sin / target_norm
            target_cos_n = target_cos / target_norm
            
            cos_sim = pred_sin_n * target_sin_n + pred_cos_n * target_cos_n
            # Loss range [0, 2]
            loss = 1.0 - cos_sim
        else:
            raise ValueError(f"Unknown loss type: '{self.loss_type}'")
            
        weighted_loss = loss * total_mask
        return weighted_loss.sum() / (mask_sum + 1e-8)
