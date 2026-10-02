"""Dual-branch EG-VAN reconstruction with four paired MFF scales."""

import torch
from torch import nn
from torch.nn import functional as F

from .efficientnet_branch import EfficientNetBranch
from .mff import MFF
from .modified_resnet50 import ModifiedResNet50


class EGVAN(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained_efficient: bool = False, pretrained_resnet: bool = False, fusion_channels: int = 128):
        super().__init__()
        if num_classes < 2 or fusion_channels < 1 or fusion_channels % 8:
            raise ValueError("num_classes >= 2 and fusion_channels divisible by 8 required")
        self.efficient = EfficientNetBranch(pretrained_efficient)
        self.resnet = ModifiedResNet50(pretrained_resnet)
        self.fusions = nn.ModuleList([
            MFF(ec, rc, fusion_channels, carry_channels=0 if index == 0 else fusion_channels)
            for index, (ec, rc) in enumerate(zip(self.efficient.output_channels, self.resnet.output_channels))
        ])
        self.aggregate = nn.Conv2d(4 * fusion_channels, fusion_channels, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(fusion_channels, num_classes)

    def forward_features(self, x: torch.Tensor) -> dict[str, tuple[torch.Tensor, ...] | torch.Tensor]:
        efficient = self.efficient(x)
        resnet = self.resnet(x)
        fused_list = []
        carry = None
        for module, a, b in zip(self.fusions, efficient, resnet):
            carry = module(a, b, carry)
            fused_list.append(carry)
        fused = tuple(fused_list)
        target = tuple(min(t.shape[-d] for t in fused) for d in (2, 1))
        aligned = [F.interpolate(t, size=target, mode="bilinear", align_corners=False) if t.shape[-2:] != target else t for t in fused]
        combined = self.aggregate(torch.cat(aligned, dim=1))
        return {"efficient": efficient, "resnet": resnet, "fused": fused, "combined": combined}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.classifier(self.pool(features["combined"]).flatten(1))
