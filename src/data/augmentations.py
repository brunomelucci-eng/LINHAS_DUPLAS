import albumentations as A
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RowAugmentor:
    def __init__(self, config: dict):
        cfg = config.get('augmentation', {})
        self.enabled = cfg.get('enabled', True)
        if not self.enabled:
            return
            
        # Non-spatial transforms using Albumentations
        self.pixel_transform = A.Compose([
            A.RandomBrightnessContrast(p=cfg.get('brightness_contrast_p', 0.4)),
            A.RandomGamma(p=cfg.get('gamma_p', 0.3)),
            A.GaussNoise(p=cfg.get('noise_p', 0.2)),
            A.Blur(blur_limit=3, p=cfg.get('blur_p', 0.15)),
        ])
        
    def __call__(
        self, 
        image: np.ndarray, 
        row_mask: np.ndarray, 
        center_mask: np.ndarray, 
        orientation_sin: np.ndarray, 
        orientation_cos: np.ndarray, 
        valid_mask: np.ndarray
    ) -> tuple:
        """
        Apply augmentations. Handles spatial changes (flips and transposes) manually
        to guarantee correct analytical adjustment of the orientation vectors,
        and applies Albumentations for pixel-level augmentations.
        
        Args:
            image: numpy array of shape (C, H, W)
            row_mask: numpy array of shape (H, W)
            center_mask: numpy array of shape (H, W)
            orientation_sin: numpy array of shape (H, W)
            orientation_cos: numpy array of shape (H, W)
            valid_mask: numpy array of shape (H, W)
        """
        if not self.enabled:
            return image, row_mask, center_mask, orientation_sin, orientation_cos, valid_mask
            
        # 1. Apply pixel-level augmentations
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            # Convert to (H, W, C) for Albumentations
            image_hwc = np.transpose(image, (1, 2, 0))
        else:
            image_hwc = image
            
        augmented = self.pixel_transform(image=image_hwc)
        image_aug = augmented['image']
        
        if image_aug.ndim == 3:
            # Convert back to (C, H, W)
            image_aug = np.transpose(image_aug, (2, 0, 1))
            
        # 2. Apply spatial augmentations manually
        
        # Horizontal Flip
        if np.random.rand() < 0.5:
            image_aug = np.flip(image_aug, axis=-1).copy()
            row_mask = np.flip(row_mask, axis=-1).copy()
            center_mask = np.flip(center_mask, axis=-1).copy()
            valid_mask = np.flip(valid_mask, axis=-1).copy()
            
            # For 2theta: sin(2(pi - theta)) = -sin(2theta), cos(2(pi - theta)) = cos(2theta)
            orientation_sin = -np.flip(orientation_sin, axis=-1).copy()
            orientation_cos = np.flip(orientation_cos, axis=-1).copy()
            
        # Vertical Flip
        if np.random.rand() < 0.5:
            image_aug = np.flip(image_aug, axis=-2).copy()
            row_mask = np.flip(row_mask, axis=-2).copy()
            center_mask = np.flip(center_mask, axis=-2).copy()
            valid_mask = np.flip(valid_mask, axis=-2).copy()
            
            # For 2theta: sin(2(-theta)) = -sin(2theta), cos(2(-theta)) = cos(2theta)
            orientation_sin = -np.flip(orientation_sin, axis=-2).copy()
            orientation_cos = np.flip(orientation_cos, axis=-2).copy()
            
        # Transpose (Swap X and Y)
        if np.random.rand() < 0.5:
            if image_aug.ndim == 3:
                image_aug = np.transpose(image_aug, (0, 2, 1)).copy()
            else:
                image_aug = np.transpose(image_aug, (1, 0)).copy()
                
            row_mask = np.transpose(row_mask, (1, 0)).copy()
            center_mask = np.transpose(center_mask, (1, 0)).copy()
            valid_mask = np.transpose(valid_mask, (1, 0)).copy()
            
            # For 2theta: sin(2(pi/2 - theta)) = sin(2theta), cos(2(pi/2 - theta)) = -cos(2theta)
            orig_sin = orientation_sin.copy()
            orig_cos = orientation_cos.copy()
            orientation_sin = np.transpose(orig_sin, (1, 0)).copy()
            orientation_cos = -np.transpose(orig_cos, (1, 0)).copy()
            
        return image_aug, row_mask, center_mask, orientation_sin, orientation_cos, valid_mask
