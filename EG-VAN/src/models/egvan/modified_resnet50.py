"""Torchvision ResNet50 with stage-level SCGA and non-local insertion."""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .non_local import NonLocalBlock
from .scga import SCGA


class ModifiedResNet50(nn.Module):
    output_channels = (256, 512, 1024, 2048)

    def __init__(self, pretrained: bool = False, groups: int = 8):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.scga1 = SCGA(256, groups)
        self.scga2 = SCGA(512, groups)
        self.nonlocal3 = NonLocalBlock(1024)
        self.nonlocal4 = NonLocalBlock(2048)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x1 = self.scga1(self.layer1(x))
        x2 = self.scga2(self.layer2(x1))
        x3 = self.nonlocal3(self.layer3(x2))
        x4 = self.nonlocal4(self.layer4(x3))
        return x1, x2, x3, x4
