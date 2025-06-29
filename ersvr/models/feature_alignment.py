import torch.nn as nn
from .mbd import MBDModule

class FeatureAlignmentBlock(nn.Module):
    """Feature Alignment Block for processing concatenated frames"""
    def __init__(self, in_channels=9, out_channels=64):
        super(FeatureAlignmentBlock, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.mbd = MBDModule(out_channels, out_channels)
        
    def forward(self, x):
        # Input shape: (B, 9, H, W) - concatenated frames
        x = self.conv_layers(x)
        x = self.mbd(x)
        return x