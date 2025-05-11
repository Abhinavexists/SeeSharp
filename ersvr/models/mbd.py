import torch
import torch.nn as nn

class MBDModule(nn.Module):
    """Multi-Branch Dilated Convolution Module"""
    def __init__(self, in_channels, out_channels):
        super(MBDModule, self).__init__()
        
        # Pointwise convolution for channel reduction
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Parallel dilated convolutions with different dilation rates
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     padding=d, dilation=d) for d in [1, 2, 4]
        ])
        
        # Final 1x1 convolution for feature fusion
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Pointwise convolution
        x = self.pointwise(x)
        
        # Apply parallel dilated convolutions
        dilated_outputs = []
        for conv in self.dilated_convs:
            dilated_outputs.append(conv(x))
        
        # Concatenate all dilated outputs
        x = torch.cat(dilated_outputs, dim=1)
        
        # Final fusion
        x = self.fusion(x)
        
        return x 