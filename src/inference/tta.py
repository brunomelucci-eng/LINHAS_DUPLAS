import numpy as np
from .predictor import RowPredictor

def predict_with_tta(predictor: RowPredictor, patch: np.ndarray) -> np.ndarray:
    """
    Run Test-Time Augmentation (TTA) on a patch.
    Averages predictions over:
      1. Original patch
      2. Horizontally flipped patch (with orientation correction)
      3. Vertically flipped patch (with orientation correction)
    """
    # 1. Original prediction
    pred_orig = predictor.predict_patch(patch)
    
    # 2. Horizontal Flip
    patch_hf = np.flip(patch, axis=-1).copy()
    pred_hf = predictor.predict_patch(patch_hf)
    # Undo spatial flip
    pred_hf = np.flip(pred_hf, axis=-1).copy()
    # Correct values: sin -> -sin, cos -> cos
    pred_hf[2] = -pred_hf[2]
    
    # 3. Vertical Flip
    patch_vf = np.flip(patch, axis=-2).copy()
    pred_vf = predictor.predict_patch(patch_vf)
    # Undo spatial flip
    pred_vf = np.flip(pred_vf, axis=-2).copy()
    # Correct values: sin -> -sin, cos -> cos
    pred_vf[2] = -pred_vf[2]
    
    # Average the predictions
    avg_pred = (pred_orig + pred_hf + pred_vf) / 3.0
    return avg_pred
