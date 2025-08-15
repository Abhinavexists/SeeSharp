import torch
import torch.nn as nn


class MBDModule(nn.Module):
    """Multi-Branch Dilated Convolution Module"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.dilated_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=d, dilation=d
                )
                for d in [1, 2, 4]
            ]
        )

        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.pointwise(x)

        dilated_outputs = []
        for conv in self.dilated_convs:
            dilated_outputs.append(conv(x))

        x = torch.cat(dilated_outputs, dim=1)
        x = self.fusion(x)

        return x


# This is how dilated convolutions work on input features [X]

# dilation = 1

# X X X
# X X X
# X X X

# (3×3, immediate neighbors)

# dilation = 2

# X . X . X
# . . . . .
# X . X . X
# . . . . .
# X . X . X

# (5×5 area, gaps between samples)

# dilation = 4

# X . . . X . . . X
# . . . . . . . . .
# . . . . . . . . .
# . . . . . . . . .
# X . . . X . . . X
# . . . . . . . . .
# . . . . . . . . .
# . . . . . . . . .
# X . . . X . . . X

# (9×9 area, bigger gaps between samples)

# When you dilate, the kernel covers a wider area, so without padding, the output would shrink.
# For dilation=d, padding must be at least d to keep spatial size same (in “same convolution” mode).

# So, for dilation=1, padding=1, dilation=2, padding=2, dilation=4, padding=4.

# In the dilated convolution, the kernel is expanded by the dilation factor, but the input is not padded.
# This means that the output will be smaller than the input, and the output will be smaller than the input.

# So, for dilation=1, padding=1, dilation=2, padding=2, dilation=4, padding=4.
