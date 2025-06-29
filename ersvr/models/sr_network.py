import torch
import torch.nn as nn
from .upsampling import UpsamplingBlock

class SRNetwork(nn.Module):
    """Super Resolution Network with ESPCN-like backbone"""
    def __init__(self, in_channels=64, out_channels=3):
        super(SRNetwork, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.upsampling = UpsamplingBlock(64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, bicubic):
        x = self.conv_layers(x)
        
        print(f"Before upsampling: {x.shape}")
        x = self.upsampling(x)
        print(f"After upsampling: {x.shape}")
        print(f"Bicubic shape: {bicubic.shape}")
        
        x = self.final_conv(x)
        x = x + bicubic
        return x