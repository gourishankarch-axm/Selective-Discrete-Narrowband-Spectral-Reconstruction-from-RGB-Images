
import math
import torch
import torch.nn as nn


class LightweightTransformer(nn.Module):
 
    def __init__(self, dim=96, num_tokens=64, num_heads=4, num_layers=2):
        super(LightweightTransformer, self).__init__()
        
        self.num_tokens = num_tokens
        self.dim = dim
        
        # Token creation through adaptive pooling
        self.token_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(dim, num_heads) for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Create tokens through adaptive pooling
        grid_size = int(math.sqrt(self.num_tokens))
        tokens = nn.functional.adaptive_avg_pool2d(x, (grid_size, grid_size))
        tokens = self.token_proj(tokens)
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, num_tokens, C]
        
        # Add positional encoding
        tokens = tokens + self.pos_embedding
        
        # Apply transformer layers
        for layer in self.layers:
            tokens = layer(tokens)
        
        # Final normalization
        tokens = self.norm(tokens)
        
        # Reshape back to feature map
        tokens = tokens.transpose(1, 2).reshape(b, self.dim, grid_size, grid_size)
        tokens = nn.functional.interpolate(tokens, size=(h, w), mode='bilinear', align_corners=False)
        
        return tokens


class TransformerLayer(nn.Module):
    
    def __init__(self, dim, num_heads):
        super(TransformerLayer, self).__init__()
        
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Lightweight feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(dim // 2, dim)
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x
