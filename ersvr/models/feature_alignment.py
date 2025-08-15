import torch.nn as nn

from .mbd import MBDModule


class FeatureAlignmentBlock(nn.Module):
    """Feature Alignment Block for processing concatenated frames"""

    def __init__(self, in_channels=9, out_channels=64):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.mbd = MBDModule(out_channels, out_channels)

    def forward(self, x):
        # Input shape: (B, 9, H, W) - concatenated frames
        x = self.conv_layers(x)
        x = self.mbd(x)
        return x

# 1. Sequential Processing
# Conv layers first: Extract basic temporal features
# MBD second: Enhance and align those features
# Progressive refinement: Each stage builds upon the previous

# 2. Feature Transformation Pipeline
# Raw Frames (9 channels) 
#     ↓
# Conv Layers → Basic Features (64 channels) 
#     ↓
# MBD Module → Enhanced Features (64 channels) 
#     ↓
# Output Features 

# 3. Complementary Functions
# Conv layers: Learn temporal patterns from concatenated frames
# MBD module: Apply multi-scale processing for better feature alignment
# Combined effect: Better temporal understanding and feature quality

# Data Shape Transformation:

# Input: [B, 9, H, W] - 3 concatenated RGB frames
#   ↓
# After Conv: [B, 64, H, W] - 64-channel features
#   ↓
# After MBD: [B, 64, H, W] - Enhanced 64-channel features