"""Plain ImageNet-pretrained EfficientNetV2S baseline."""

from __future__ import annotations

from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

NUM_CLASSES = 7


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> tuple[nn.Module, EfficientNet_V2_S_Weights | None]:
    """Build EfficientNetV2S with only its classifier replaced."""
    # This is intentionally a baseline: no EG-VAN attention branches, fusion
    # modules, or paper-specific heads are added here.
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    input_features = model.classifier[-1].in_features
    # Replace only the ImageNet classifier so the backbone features and
    # torchvision preprocessing contract remain intact.
    model.classifier[-1] = nn.Linear(input_features, num_classes)
    return model, weights
