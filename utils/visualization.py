

import os
import matplotlib.pyplot as plt
from pathlib import Path
import torch


def plot_training_history(train_losses, val_losses, save_path):
   
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Sophisticated Hybrid UNet Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def create_visualization(predictions, targets, target_bands, results_dir, metrics=None):
   
    print("\nCreating visualizations...")
    vis_dir = os.path.join(results_dir, 'water_bodies_visualizations')
    Path(vis_dir).mkdir(exist_ok=True)
    
    num_bands = predictions.shape[1]
    
    # Band comparison for first sample
    fig, axes = plt.subplots(2, num_bands, figsize=(18, 6))
    
    for band_idx in range(num_bands):
        # Target
        target_img = targets[0, band_idx].numpy()
        axes[0, band_idx].imshow(target_img, cmap='viridis', vmin=0, vmax=1)
        axes[0, band_idx].set_title(f'Target B{target_bands[band_idx]}')
        axes[0, band_idx].axis('off')
        
        # Prediction
        pred_img = predictions[0, band_idx].numpy()
        axes[1, band_idx].imshow(pred_img, cmap='viridis', vmin=0, vmax=1)
        axes[1, band_idx].set_title(f'Pred B{target_bands[band_idx]}')
        axes[1, band_idx].axis('off')
    
    if metrics:
        metrics_text = f'MSE: {metrics["mse"]:.4f}, MAE: {metrics["mae"]:.4f}, PSNR: {metrics["psnr"]:.1f} dB'
        plt.suptitle(f'Sophisticated Hybrid UNet - Water Body Sample 1\n{metrics_text}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'sample_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Average spectral profile
    avg_target = targets.mean(dim=(0, 2, 3))
    avg_pred = predictions.mean(dim=(0, 2, 3))
    
    plt.figure(figsize=(10, 6))
    band_indices = range(num_bands)
    
    plt.plot(band_indices, avg_target.numpy(), 'b-o', linewidth=3, markersize=8, label='Target')
    plt.plot(band_indices, avg_pred.numpy(), 'r--s', linewidth=3, markersize=8, label='Predicted')
    plt.fill_between(band_indices, avg_target.numpy(), avg_pred.numpy(), alpha=0.2, color='gray')
    
    plt.xlabel('Band Index')
    plt.ylabel('Normalized Intensity')
    plt.title('Average Spectral Profile - Sophisticated Hybrid UNet', fontsize=14)
    plt.xticks(band_indices, [f'B{b}' for b in target_bands])
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'spectral_profile.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualizations saved to: {vis_dir}")
