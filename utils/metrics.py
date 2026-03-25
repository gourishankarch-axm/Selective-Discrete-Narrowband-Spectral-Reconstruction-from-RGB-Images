
import torch
import numpy as np


def calculate_metrics(predictions, targets):
   
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    
    if mse > 0:
        psnr = 10 * torch.log10(torch.tensor(1.0) / mse).item()
    else:
        psnr = float('inf')
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'psnr': psnr
    }


def save_metrics(metrics, save_path, num_samples, target_bands=None):
   
    with open(save_path, 'w') as f:
        f.write("Water Bodies Evaluation - Sophisticated Hybrid UNet\n")
        f.write("="*50 + "\n")
        f.write(f"Samples evaluated: {num_samples}\n\n")
        f.write(f"MSE:  {metrics['mse']:.6f}\n")
        f.write(f"MAE:  {metrics['mae']:.6f}\n")
        f.write(f"RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
        
        if target_bands:
            f.write("\nBand-wise Metrics:\n")
            f.write("-"*30 + "\n")
            
            # Calculate per-band metrics
            for band_idx, band in enumerate(target_bands):
                band_pred = predictions[:, band_idx, :, :]
                band_target = targets[:, band_idx, :, :]
                band_mse = torch.mean((band_pred - band_target) ** 2).item()
                f.write(f"Band {band}: MSE = {band_mse:.6f}\n")
