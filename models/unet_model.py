
import torch
import torch.nn as nn
from config.config import config
from models.blocks import EnhancedResidualBlock
from models.multi_scale import MultiScaleBlock
from models.attention import ChannelAttention
from models.transformer import LightweightTransformer


class SRHybridUNet(nn.Module):
    """
    Hybrid U-Net architecture:
    - Input: 3-channel RGB (64x64)
    - Encoder: 3 downsampling stages (64→32→16→8) with channel doubling
    - Transformer bottleneck at 8x8 (64 tokens)
    - Decoder: 3 upsampling stages with skip connections
    - Output: 6 target spectral bands
    """
    def __init__(self, input_channels=3, output_channels=6, base_features=16):
        super(SRHybridUNet, self).__init__()
        
        self.base_features = base_features
        
        print(f"Sophisticated Hybrid UNet Configuration (Text-compliant):")
        print(f"  Input channels: {input_channels} (RGB)")
        print(f"  Output channels: {output_channels}")
        print(f"  Base features: {base_features}")
        print(f"  Encoder channels: {base_features} → {base_features*2} → {base_features*4} → {base_features*8}")
        
        # =-----------INITIAL
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_features),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # -----------ENCODER 
        # Stage 1: 64x64 → 32x32, channels base_features → base_features*2
        self.enc1_conv = nn.Sequential(
            EnhancedResidualBlock(base_features),
            MultiScaleBlock(base_features)
        )
        self.enc1_down = nn.Conv2d(base_features, base_features*2, kernel_size=3, stride=2, padding=1)
        
        # Stage 2: 32x32 → 16x16, channels base_features*2 → base_features*4
        self.enc2_conv = nn.Sequential(
            EnhancedResidualBlock(base_features*2),
            MultiScaleBlock(base_features*2)
        )
        self.enc2_down = nn.Conv2d(base_features*2, base_features*4, kernel_size=3, stride=2, padding=1)
        
        # Stage 3: 16x16 → 8x8, channels base_features*4 → base_features*8
        self.enc3_conv = nn.Sequential(
            EnhancedResidualBlock(base_features*4),
            MultiScaleBlock(base_features*4)
        )
        self.enc3_down = nn.Conv2d(base_features*4, base_features*8, kernel_size=3, stride=2, padding=1)
        
        # =------- BOTTLENECK 
        # Two EnhancedResidualBlocks (one with dilation)
        self.bottleneck_conv = nn.Sequential(
            EnhancedResidualBlock(base_features*8),
            EnhancedResidualBlock(base_features*8, dilation=2)
        )
        
        # Transformer projection and processing
        self.bottleneck_proj = nn.Conv2d(base_features*8, config.transformer_dim, kernel_size=1, bias=False)
        self.transformer = LightweightTransformer(
            dim=config.transformer_dim,
            num_tokens=config.num_tokens,
            num_heads=4,
            num_layers=2
        )
        self.bottleneck_proj_back = nn.Conv2d(config.transformer_dim, base_features*8, kernel_size=1, bias=False)
        
        # =------- DECODER 
        # Decoder 3: 8x8 → 16x16
        self.dec3_up = nn.ConvTranspose2d(base_features*8, base_features*4, kernel_size=2, stride=2)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(base_features*8, base_features*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_features*4),
            nn.LeakyReLU(0.1, inplace=True),
            EnhancedResidualBlock(base_features*4),
            MultiScaleBlock(base_features*4)
        )
        
   
        self.dec2_up = nn.ConvTranspose2d(base_features*4, base_features*2, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(base_features*4, base_features*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_features*2),
            nn.LeakyReLU(0.1, inplace=True),
            EnhancedResidualBlock(base_features*2),
            MultiScaleBlock(base_features*2)
        )
        
       
        self.dec1_up = nn.ConvTranspose2d(base_features*2, base_features, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(base_features*2, base_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_features),
            nn.LeakyReLU(0.1, inplace=True),
            EnhancedResidualBlock(base_features),
            MultiScaleBlock(base_features)
        )
        
        # ===-----FINAL OUTPUT 
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_features, base_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_features),
            nn.LeakyReLU(0.1, inplace=True),
            ChannelAttention(base_features),
            nn.Conv2d(base_features, output_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {total_params:,}")
    
    def forward(self, x):
        # Initial convolution
        x0 = self.initial_conv(x)
        
        # =-------------ENCODER
        enc1 = self.enc1_conv(x0)           # [B, base_features, 64, 64]
        pool1 = self.enc1_down(enc1)        # [B, base_features*2, 32, 32]
        
        enc2 = self.enc2_conv(pool1)        # [B, base_features*2, 32, 32]
        pool2 = self.enc2_down(enc2)        # [B, base_features*4, 16, 16]
        
        enc3 = self.enc3_conv(pool2)        # [B, base_features*4, 16, 16]
        pool3 = self.enc3_down(enc3)        # [B, base_features*8, 8, 8]
        
        # ---------------BOTTLENECK 
        bottleneck = self.bottleneck_conv(pool3)            
        bottleneck_in = self.bottleneck_proj(bottleneck)     
        transformer_out = self.transformer(bottleneck_in)     
        bottleneck_out = self.bottleneck_proj_back(transformer_out)  
        
        # --------------DECODER
        up3 = self.dec3_up(bottleneck_out)                   
        up3 = torch.cat([up3, enc3], dim=1)                  
        dec3 = self.dec3_conv(up3)                            
        
        up2 = self.dec2_up(dec3)                             
        up2 = torch.cat([up2, enc2], dim=1)                   
        dec2 = self.dec2_conv(up2)                           
        
        up1 = self.dec1_up(dec2)                              # [B, base_features, 64, 64]
        up1 = torch.cat([up1, enc1], dim=1)                   
        dec1 = self.dec1_conv(up1)                           
        
        # ---------------FINAL OUT
        output = self.output_conv(dec1)                      
        
        return output
