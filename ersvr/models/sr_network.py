import torch
import torch.nn as nn
from .upsampling import UpsamplingBlock

class SRNetwork(nn.Module):
    """Super Resolution Network with ESPCN-like backbone"""
    def __init__(self, in_channels=64, out_channels=3):
        super(SRNetwork, self).__init__()
        
        # Initial feature extraction
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
        
        # Upsampling blocks for 4x upscaling
        self.upsampling = UpsamplingBlock(64)
        
        # Final convolution for RGB output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, bicubic):
        # Process features
        x = self.conv_layers(x)
        
        # Print shapes for debugging
        print(f"Before upsampling: {x.shape}")
        
        # Upsampling blocks for 4x upscaling
        x = self.upsampling(x)
        
        print(f"After upsampling: {x.shape}")
        print(f"Bicubic shape: {bicubic.shape}")
        
        # Final convolution for RGB output
        x = self.final_conv(x)
        
        # Add residual connection with bicubic upsampled input
        x = x + bicubic
        
        return x 