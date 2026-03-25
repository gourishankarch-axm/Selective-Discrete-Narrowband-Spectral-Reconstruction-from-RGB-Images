

import torch.nn as nn
from models.attention import ChannelAttention, SpatialAttention


class EnhancedResidualBlock(nn.Module):
    """
    Advanced residual block with channel attention and dilated convolutions
    """
    def __init__(self, channels, dilation=1, reduction=8):
        super(EnhancedResidualBlock, self).__init__()
        
        # Main convolution path
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                              padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
        
        # Activation
        self.activation = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        identity = x
        
        # First convolution
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Apply attention mechanisms
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        
        # Residual connection
        out = out + identity
        out = self.activation(out)
        
        return out
