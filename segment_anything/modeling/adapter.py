import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAdapter(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # x shape: B, C, H, W
        # Convert to B, H, W, C for transformer
        x = x.permute(0, 2, 3, 1)
        # Apply adapter
        x = x + self.adapter(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x 