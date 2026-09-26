import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict

class Evaluator:
    def __init__(self, device: torch.device, loss_fn: nn.Module):
        self.device = device
        self.loss_fn = loss_fn

    def evaluate(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        model.eval()
        val_loss = 0.0
        val_row_focal = 0.0
        val_row_dice = 0.0
        val_center_focal = 0.0
        val_center_dice = 0.0
        val_center_false_positive = 0.0
        val_center_cldice = 0.0
        val_orientation = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                logits = model(images)
                loss_dict = self.loss_fn(logits, targets)
                
                val_loss += loss_dict['loss'].item()
                val_row_focal += loss_dict['row_focal'].item()
                val_row_dice += loss_dict['row_dice'].item()
                val_center_focal += loss_dict['center_focal'].item()
                val_center_dice += loss_dict['center_dice'].item()
                val_center_false_positive += loss_dict.get(
                    'center_false_positive', torch.tensor(0.0)
                ).item()
                val_center_cldice += loss_dict['center_cldice'].item()
                val_orientation += loss_dict['orientation'].item()
                
        n = len(val_loader)
        if n == 0:
            return {}
            
        # Convert Dice and clDice loss values back to standard positive coefficients [0, 1]
        # (Since DiceLoss = 1 - Dice, Dice = 1 - DiceLoss)
        return {
            'val_loss': val_loss / n,
            'val_row_focal': val_row_focal / n,
            'val_row_dice': 1.0 - (val_row_dice / n),
            'val_center_focal': val_center_focal / n,
            'val_center_dice': 1.0 - (val_center_dice / n),
            'val_center_false_positive': val_center_false_positive / n,
            'val_center_cldice': 1.0 - (val_center_cldice / n),
            'val_orientation_loss': val_orientation / n
        }
