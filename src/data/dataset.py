import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

class SugarcaneDataset(Dataset):
    def __init__(self, dataset_dir: str, augmentor: Optional[Any] = None):
        """
        Args:
            dataset_dir: Path to directory containing .npz files
            augmentor: Optional RowAugmentor instance
        """
        self.dataset_dir = dataset_dir
        self.augmentor = augmentor
        
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
            
        self.filenames = [
            f for f in os.listdir(dataset_dir)
            if f.endswith('.npz')
        ]
        self.filenames.sort()
        
    def __len__(self) -> int:
        return len(self.filenames)
        
    def __getitem__(self, idx: int) -> tuple:
        filepath = os.path.join(self.dataset_dir, self.filenames[idx])
        data = np.load(filepath)
        
        image = data['image'] # (C, H, W)
        row_mask = data['row_mask'] # (H, W)
        center_mask = data['center_mask'] # (H, W)
        orientation_sin = data['orientation_sin'] # (H, W)
        orientation_cos = data['orientation_cos'] # (H, W)
        valid_mask = data['valid_mask'] # (H, W)
        
        if self.augmentor is not None:
            image, row_mask, center_mask, orientation_sin, orientation_cos, valid_mask = self.augmentor(
                image, row_mask, center_mask, orientation_sin, orientation_cos, valid_mask
            )
            
        # Convert to PyTorch tensors
        image_t = torch.from_numpy(image).float()
        
        targets = {
            'row_mask': torch.from_numpy(row_mask).float().unsqueeze(0),
            'center_mask': torch.from_numpy(center_mask).float().unsqueeze(0),
            'orientation_sin': torch.from_numpy(orientation_sin).float().unsqueeze(0),
            'orientation_cos': torch.from_numpy(orientation_cos).float().unsqueeze(0),
            'valid_mask': torch.from_numpy(valid_mask).float().unsqueeze(0),
        }
        
        return image_t, targets
