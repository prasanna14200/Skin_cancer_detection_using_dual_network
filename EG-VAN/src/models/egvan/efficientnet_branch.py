"""Torchvision EfficientNetV2S intermediate feature taps."""

import torch
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


class EfficientNetBranch(nn.Module):
    tap_indices = (2, 3, 5, 7)
    output_channels = (48, 64, 160, 1280)

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.features = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None).features

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        taps = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.tap_indices:
                taps.append(x)
        return tuple(taps)
