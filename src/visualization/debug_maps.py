import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional

def save_probability_maps(
    row_prob: Optional[np.ndarray],
    center_prob: np.ndarray,
    output_path: str
):
    """
    Saves available row-band and centerline probability plots.

    ``row_prob`` can be absent when inference is configured to retain only
    the center channel, which is the memory-efficient default for large ROIs.
    """
    if row_prob is None:
        fig, axis = plt.subplots(figsize=(7, 7))
        image = axis.imshow(center_prob, cmap='magma', vmin=0.0, vmax=1.0)
        axis.set_title("Centerline Probability")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        image = axes[0].imshow(row_prob, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[0].set_title("Row Band Probability")
        fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].axis('off')

        image = axes[1].imshow(center_prob, cmap='magma', vmin=0.0, vmax=1.0)
        axes[1].set_title("Centerline Probability")
        fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].axis('off')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def save_orientation_quiver(
    orientation_sin: np.ndarray,
    orientation_cos: np.ndarray,
    center_prob: np.ndarray,
    output_path: str,
    stride: int = 16
):
    """
    Saves a quiver vector field plot demonstrating local orientation angles.
    """
    H, W = center_prob.shape
    y, x = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing='ij')
    
    u = orientation_cos[y, x]
    v = orientation_sin[y, x]
    
    # Show vectors only where centerline probability exceeds threshold
    mask = center_prob[y, x] > 0.35
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(center_prob, cmap='gray', alpha=0.4)
    
    if np.any(mask):
        ax.quiver(x[mask], y[mask], u[mask], v[mask], color='red', scale=25, width=0.004)
        
    ax.set_title("Local Orientation Vector Field")
    ax.axis('off')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
