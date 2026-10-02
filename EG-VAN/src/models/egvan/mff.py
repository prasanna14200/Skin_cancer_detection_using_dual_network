"""Paired-scale multi-scale fusion from the EG-VAN paper description."""

import torch
from torch import nn
from torch.nn import functional as F

from .scga import GroupMeanMaxAttention, SCGA
from .spatial_attention import SpatialAttention


class MFF(nn.Module):
    def __init__(self, efficient_channels: int, resnet_channels: int, out_channels: int = 128, groups: int = 8, carry_channels: int = 0):
        super().__init__()
        if min(efficient_channels, resnet_channels, out_channels) < 1 or carry_channels < 0:
            raise ValueError("channel counts must be positive")
        self.carry_channels = carry_channels
        self.efficient_attention = SpatialAttention(efficient_channels)
        self.resnet_attention = GroupMeanMaxAttention(resnet_channels, groups)
        joined = efficient_channels + resnet_channels + carry_channels
        self.depthwise = nn.Conv2d(joined, joined, 3, stride=2, padding=1, groups=joined, bias=False)
        self.pointwise = nn.Conv2d(joined, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scga = SCGA(out_channels, groups)

    def forward(self, efficient: torch.Tensor, resnet: torch.Tensor, carry: torch.Tensor | None = None) -> torch.Tensor:
        if efficient.shape[0] != resnet.shape[0]:
            raise ValueError("MFF inputs must share batch size")
        if (carry is None) != (self.carry_channels == 0):
            raise ValueError("MFF carry input does not match configured carry channels")
        if carry is not None and (carry.shape[0] != efficient.shape[0] or carry.shape[1] != self.carry_channels):
            raise ValueError("MFF carry batch/channels mismatch")
        efficient = self.efficient_attention(efficient)
        resnet = self.resnet_attention(resnet)
        if efficient.shape[-2:] != resnet.shape[-2:]:
            target = tuple(min(a, b) for a, b in zip(efficient.shape[-2:], resnet.shape[-2:]))
            efficient = F.interpolate(efficient, size=target, mode="bilinear", align_corners=False)
            resnet = F.interpolate(resnet, size=target, mode="bilinear", align_corners=False)
        parts = [efficient, resnet]
        if carry is not None:
            if carry.shape[-2:] != efficient.shape[-2:]:
                carry = F.interpolate(carry, size=efficient.shape[-2:], mode="bilinear", align_corners=False)
            parts.append(carry)
        joined = torch.cat(parts, dim=1)
        return self.scga(self.bn(self.pointwise(self.depthwise(joined))))
