import torch.nn as nn

class SubpixelUpsampling(nn.Module):
    """Subpixel Upsampling Module using PixelShuffle"""
    def __init__(self, in_channels, scale_factor=2):
        super(SubpixelUpsampling, self).__init__()
        
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * (scale_factor ** 2),
            kernel_size=3,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class UpsamplingBlock(nn.Module):
    """Block for 4x upsampling using two SubpixelUpsampling modules"""
    def __init__(self, in_channels):
        super(UpsamplingBlock, self).__init__()
        
        self.upsample1 = SubpixelUpsampling(in_channels)
        self.upsample2 = SubpixelUpsampling(in_channels)
        
    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        return x