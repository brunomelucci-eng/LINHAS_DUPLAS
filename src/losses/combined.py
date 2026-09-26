import torch
import torch.nn as nn
from .dice import DiceLoss
from .focal import FocalLoss
from .cldice import SoftClDiceLoss
from .orientation import MaskedOrientationLoss
from typing import Dict

class CombinedLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        loss_cfg = config.get('loss', {})
        
        self.row_bce_weight = loss_cfg.get('row_bce_weight', 0.20)
        self.row_dice_weight = loss_cfg.get('row_dice_weight', 0.20)
        self.center_focal_weight = loss_cfg.get('center_focal_weight', 0.15)
        self.center_dice_weight = loss_cfg.get('center_dice_weight', 0.20)
        self.center_false_positive_weight = loss_cfg.get(
            'center_false_positive_weight', 0.0
        )
        self.cldice_weight = loss_cfg.get('cldice_weight', 0.15)
        self.orientation_weight = loss_cfg.get('orientation_weight', 0.10)
        
        # Initialize sub-losses
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice = DiceLoss()
        self.cldice = SoftClDiceLoss(iters=5)
        self.orientation = MaskedOrientationLoss(loss_type='mse')

    def forward(self, logits: torch.Tensor, targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: Output tensor from forward pass (B, 4, H, W)
                logits[:, 0, :, :] -> Row mask logit
                logits[:, 1, :, :] -> Centerline mask logit
                logits[:, 2, :, :] -> Orientation sin component logit/raw
                logits[:, 3, :, :] -> Orientation cos component logit/raw
            targets: Dictionary containing the ground truth tensors.
        """
        row_logit = logits[:, 0:1, :, :]
        center_logit = logits[:, 1:2, :, :]
        pred_sin = logits[:, 2:3, :, :]
        pred_cos = logits[:, 3:4, :, :]
        
        t_row = targets['row_mask']
        t_center = targets['center_mask']
        t_sin = targets['orientation_sin']
        t_cos = targets['orientation_cos']
        valid_mask = targets['valid_mask']
        
        # 1. Row mask losses
        loss_row_focal = self.focal(row_logit, t_row, valid_mask)
        loss_row_dice = self.dice(row_logit, t_row, valid_mask)
        
        # 2. Centerline mask losses
        loss_center_focal = self.focal(center_logit, t_center, valid_mask)
        loss_center_dice = self.dice(center_logit, t_center, valid_mask)
        loss_center_cldice = self.cldice(center_logit, t_center, valid_mask)
        center_probability = torch.sigmoid(center_logit)
        center_background = (1.0 - t_center) * valid_mask
        background_pixels = center_background.sum().clamp_min(1.0)
        loss_center_false_positive = (
            center_probability * center_background
        ).sum() / background_pixels
        
        # 3. Orientation loss masked by row mask and valid mask
        # Values predicted directly are fed; we apply tanh boundary scaling during predict,
        # but during loss backprop, raw logits are compared.
        # However, for orientation, we compare logits directly since they are linear values.
        loss_orient = self.orientation(pred_sin, pred_cos, t_sin, t_cos, t_row, valid_mask)
        
        # Total weighted loss
        total_loss = (
            self.row_bce_weight * loss_row_focal +
            self.row_dice_weight * loss_row_dice +
            self.center_focal_weight * loss_center_focal +
            self.center_dice_weight * loss_center_dice +
            self.center_false_positive_weight * loss_center_false_positive +
            self.cldice_weight * loss_center_cldice +
            self.orientation_weight * loss_orient
        )
        
        return {
            'loss': total_loss,
            'row_focal': loss_row_focal,
            'row_dice': loss_row_dice,
            'center_focal': loss_center_focal,
            'center_dice': loss_center_dice,
            'center_false_positive': loss_center_false_positive,
            'center_cldice': loss_center_cldice,
            'orientation': loss_orient
        }
