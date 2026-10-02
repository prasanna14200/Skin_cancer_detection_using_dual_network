"""Spatial attention followed by group-wise horizontal/vertical mean-max attention."""

import torch
from torch import nn

from .spatial_attention import SpatialAttention


class GroupMeanMaxAttention(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        if channels < 1 or groups < 1 or channels % groups:
            raise ValueError("channels must be divisible by positive groups")
        self.groups = groups
        self.group_channels = channels // groups
        self.horizontal = nn.Conv2d(2 * self.group_channels, self.group_channels, 1)
        self.vertical = nn.Conv2d(2 * self.group_channels, self.group_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.groups * self.group_channels:
            raise ValueError("GMA input channels differ from construction")
        grouped = x.reshape(b * self.groups, self.group_channels, h, w)
        horizontal = torch.cat((grouped.mean(dim=3, keepdim=True), grouped.amax(dim=3, keepdim=True)), dim=1)
        vertical = torch.cat((grouped.mean(dim=2, keepdim=True), grouped.amax(dim=2, keepdim=True)), dim=1)
        ah = torch.sigmoid(self.horizontal(horizontal))
        av = torch.sigmoid(self.vertical(vertical))
        score = torch.sigmoid(ah + av)
        return (grouped * score).reshape(b, c, h, w)


class SCGA(nn.Module):
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.spatial = SpatialAttention(channels)
        self.gma = GroupMeanMaxAttention(channels, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gma(self.spatial(x))
