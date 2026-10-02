"""Embedded-Gaussian non-local block with subsampled keys/values."""

import torch
from torch import nn
from torch.nn import functional as F


class NonLocalBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 2, key_stride: int = 2):
        super().__init__()
        if channels < 1 or reduction < 1 or key_stride < 1:
            raise ValueError("channels, reduction and key_stride must be positive")
        inner = max(1, channels // reduction)
        self.theta = nn.Conv2d(channels, inner, 1, bias=False)
        self.phi = nn.Conv2d(channels, inner, 1, bias=False)
        self.g = nn.Conv2d(channels, inner, 1, bias=False)
        self.project = nn.Conv2d(inner, channels, 1, bias=False)
        self.key_stride = key_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        kv = F.avg_pool2d(x, self.key_stride) if h >= self.key_stride and w >= self.key_stride else x
        q = self.theta(x).flatten(2).transpose(1, 2)
        k = self.phi(kv).flatten(2)
        v = self.g(kv).flatten(2).transpose(1, 2)
        attention = torch.softmax(torch.bmm(q, k), dim=-1)
        attended = torch.bmm(attention, v).transpose(1, 2).reshape(b, -1, h, w)
        return x + self.project(attended)
