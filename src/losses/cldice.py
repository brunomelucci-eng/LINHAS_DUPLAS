import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def soft_erode(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) != 4:
        raise ValueError("Input to soft_erode must be (B, C, H, W)")
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)

def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    if len(img.shape) != 4:
        raise ValueError("Input to soft_dilate must be (B, C, H, W)")
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))

def soft_skeletonize(img: torch.Tensor, iters: int = 5) -> torch.Tensor:
    img = torch.clamp(img, 0.0, 1.0)
    skeleton = torch.zeros_like(img)
    for _ in range(iters):
        erode = soft_erode(img)
        opened = soft_open(img)
        diff = F.relu(img - opened)
        skeleton = torch.max(skeleton, diff)
        img = erode
    return skeleton

class SoftClDiceLoss(nn.Module):
    def __init__(self, iters: int = 5, eps: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.iters = iters
        self.eps = eps
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate clDice loss.
        Args:
            pred: predictions tensor (B, 1, H, W)
            target: target tensor (B, 1, H, W)
            mask: optional valid pixel mask (B, 1, H, W)
        """
        if self.from_logits:
            pred = torch.sigmoid(pred)
            
        if mask is not None:
            pred = pred * mask
            target = target * mask
            
        s_pred = soft_skeletonize(pred, self.iters)
        s_target = soft_skeletonize(target, self.iters)
        
        s_pred_flat = s_pred.view(s_pred.size(0), -1)
        s_target_flat = s_target.view(s_target.size(0), -1)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        tprec = (s_pred_flat * target_flat).sum(dim=1) / (s_pred_flat.sum(dim=1) + self.eps)
        trec = (s_target_flat * pred_flat).sum(dim=1) / (s_target_flat.sum(dim=1) + self.eps)
        
        cldice = 2.0 * tprec * trec / (tprec + trec + self.eps)
        
        return 1.0 - cldice.mean()
