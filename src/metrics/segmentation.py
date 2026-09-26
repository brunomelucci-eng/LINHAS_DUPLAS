import numpy as np
from skimage.morphology import skeletonize
from typing import Dict

def compute_binary_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Computes standard binary classification metrics.
    Inputs are numpy arrays of shape (H, W) in range {0, 1}.
    """
    pred_b = (pred > 0).astype(bool)
    target_b = (target > 0).astype(bool)
    
    tp = np.sum(pred_b & target_b)
    fp = np.sum(pred_b & ~target_b)
    fn = np.sum(~pred_b & target_b)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
        'dice': float(dice)
    }

def compute_cldice(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Computes hard centerline Dice (clDice) using skeletonization.
    """
    pred_b = (pred > 0).astype(bool)
    target_b = (target > 0).astype(bool)
    
    if not np.any(pred_b) and not np.any(target_b):
        return 1.0
    if not np.any(pred_b) or not np.any(target_b):
        return 0.0
        
    s_pred = skeletonize(pred_b)
    s_target = skeletonize(target_b)
    
    sum_s_pred = s_pred.sum()
    sum_s_target = s_target.sum()
    
    t_prec = np.sum(s_pred & target_b) / (sum_s_pred + 1e-8)
    t_rec = np.sum(s_target & pred_b) / (sum_s_target + 1e-8)
    
    cldice = 2.0 * t_prec * t_rec / (t_prec + t_rec + 1e-8)
    return float(cldice)
