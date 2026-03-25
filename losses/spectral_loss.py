
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralCorrelationLoss(nn.Module):
   
    def __init__(self, weight=0.3):
        super(SpectralCorrelationLoss, self).__init__()
        self.weight = weight
    
    def forward(self, pred, target):
       
        # Pixel-wise MSE loss
        mse_loss = F.mse_loss(pred, target)
        
        # Spectral correlation loss
        b, c, h, w = pred.shape
        
        # Reshape to [batch, channels, pixels]
        pred_flat = pred.view(b, c, -1)
        target_flat = target.view(b, c, -1)
        
        # Calculate correlation matrices
        pred_corr = torch.bmm(pred_flat, pred_flat.transpose(1, 2)) / (h * w)
        target_corr = torch.bmm(target_flat, target_flat.transpose(1, 2)) / (h * w)
        
        # Correlation matrix loss
        corr_loss = F.mse_loss(pred_corr, target_corr)
        
        # Combined loss
        total_loss = mse_loss + self.weight * corr_loss
        
        return total_loss, mse_loss.item(), corr_loss.item()
