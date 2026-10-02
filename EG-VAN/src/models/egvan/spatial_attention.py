"""Paper-described spatial path used by SCGA and MFF."""

import torch
from torch import nn
from torch.nn import functional as F


class SpatialAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.conv64 = nn.Conv2d(channels, 64, 1)
        self.conv16 = nn.Conv2d(64, 16, 1, dilation=2)
        self.conv8 = nn.Conv2d(16, 8, 1)
        self.mask_conv = nn.Conv2d(8, 1, 1)
        # Figure 4 shows a parallel 1x1 projection of the input feature path.
        self.input_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.register_buffer("fixed_channel_projection", torch.ones(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.relu(self.conv64(x))
        mask = torch.relu(self.conv16(mask))
        mask = torch.relu(self.conv8(mask))
        mask = torch.sigmoid(self.mask_conv(mask))
        refined = mask * F.adaptive_avg_pool2d(mask, 1)
        expanded = F.conv2d(refined, self.fixed_channel_projection)
        return self.input_projection(x) * expanded
