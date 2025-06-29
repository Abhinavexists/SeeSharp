import torch
import torch.nn as nn

class MBDModule(nn.Module):
    """Multi-Branch Dilated Convolution Module"""
    def __init__(self, in_channels, out_channels):
        super(MBDModule, self).__init__()
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     padding=d, dilation=d) for d in [1, 2, 4]
        ])
        
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.pointwise(x)
        
        dilated_outputs = []
        for conv in self.dilated_convs:
            dilated_outputs.append(conv(x))
        
        x = torch.cat(dilated_outputs, dim=1)
        x = self.fusion(x)
        
        return x 