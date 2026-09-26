import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean', from_logits: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate Focal Loss.
        Args:
            pred: predictions tensor (B, 1, H, W)
            target: target tensor (B, 1, H, W)
            mask: optional valid pixel mask (B, 1, H, W)
        """
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            pred_prob = torch.sigmoid(pred)
        else:
            bce = F.binary_cross_entropy(pred, target, reduction='none')
            pred_prob = pred
            
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        loss = bce * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
            
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
                
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
