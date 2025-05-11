import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .feature_alignment import FeatureAlignmentBlock
from .sr_network import SRNetwork

class ERSVR(nn.Module):
    """Real-time Video Super Resolution Network using Recurrent Multi-Branch Dilated Convolutions"""
    def __init__(self, scale_factor=4):
        super(ERSVR, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Feature alignment block
        self.feature_alignment = FeatureAlignmentBlock(in_channels=9, out_channels=64)
        
        # SR network
        self.sr_network = SRNetwork(in_channels=64, out_channels=3)
        
    def forward(self, x):
        # Input shape: (B, 3, 3, H, W) - batch of 3 RGB frames
        batch_size, num_frames, channels, height, width = x.shape
        
        # Rearrange input to (B, 9, H, W)
        x = rearrange(x, 'b n c h w -> b (n c) h w')
        
        # Extract center frame for residual connection
        center_frame = x[:, 3:6, :, :]  # RGB channels of center frame
        
        # Bicubic upsampling of center frame for residual connection
        bicubic = F.interpolate(
            center_frame,
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        )
        
        # Feature alignment
        features = self.feature_alignment(x)
        
        # SR network
        output = self.sr_network(features, bicubic)
        
        # Ensure output and bicubic have the same dimensions
        if output.shape != bicubic.shape:
            print(f"Output shape: {output.shape}, Bicubic shape: {bicubic.shape}")
            raise ValueError("Output and bicubic tensors must have the same dimensions")
        
        return output 