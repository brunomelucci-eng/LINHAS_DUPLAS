import torch
import torch.nn as nn
from typing import Optional

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.eps = eps
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate Dice Loss.
        Args:
            pred: predictions tensor (B, 1, H, W)
            target: target tensor (B, 1, H, W)
            mask: optional valid pixel mask (B, 1, H, W)
        """
        if self.from_logits:
            pred = torch.sigmoid(pred)
        
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        if mask is not None:
            mask = mask.view(mask.size(0), -1)
            pred = pred * mask
            target = target * mask
            
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()
