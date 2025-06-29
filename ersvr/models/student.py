import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution Block for efficiency.
    Consists of a depthwise convolution followed by a pointwise convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class StudentSRNet(nn.Module):
    """
    Ultra-lightweight Student Model for Video Super-Resolution.
    - Input: (B, 3, 3, H, W)  # 3 frames, 3 channels each
    - Output: (B, 3, H*4, W*4)  # Super-resolved center frame
    Designed for real-time, mobile/edge deployment.
    """
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.input_conv = nn.Conv2d(9, 16, 3, padding=1)
        self.block1 = DepthwiseSeparableConv(16, 32)
        self.block2 = DepthwiseSeparableConv(32, 32)
        self.block3 = DepthwiseSeparableConv(32, 16)
        self.upsample1 = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.output_conv = nn.Conv2d(16, 3, 3, padding=1)
    
    def forward(self, x):
        # x: (B, 3, 3, H, W) -> (B, 9, H, W)
        b, n, c, h, w = x.shape
        x = x.reshape(b, n * c, h, w)
        x = self.input_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.output_conv(x)
        return x 