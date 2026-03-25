
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleBlock(nn.Module):
    """
    Multi-scale processing block with four branches:
    1. 1x1 -> 3x3 convolution
    2. 1x1 -> 3x3 dilated convolution (dilation=2)
    3. 1x1 -> 5x5 convolution
    4. Global average pooling -> 1x1 convolution -> upsampling
    """
    def __init__(self, channels):
        super(MultiScaleBlock, self).__init__()
        
        # Branch 1: 1x1 -> 3x3
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels//4, channels//4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Branch 2: 1x1 -> 3x3 dilated
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels//4, channels//4, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Branch 3: 1x1 -> 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels//4, channels//4, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Branch 4: Global context
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels//4),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = F.interpolate(self.branch4(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fusion(out)
        
        return out + x  # Residual connection
