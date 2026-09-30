"""Minimal Grad-CAM utility for frozen classification models."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


def gradcam(
    model: nn.Module,
    image: Tensor,
    target_layer: nn.Module,
    class_index: int,
) -> Tensor:
    """Return a normalized 2D Grad-CAM map for one NCHW image and class."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM expects one NCHW image")
    activations: list[Tensor] = []

    def capture_activation(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise ValueError("Target layer must output a spatial NCHW tensor")
        activations.append(output)

    handle = target_layer.register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image)
        if len(activations) != 1:
            raise RuntimeError(f"Expected one target-layer activation, found {len(activations)}")
        if class_index < 0 or class_index >= logits.shape[1]:
            raise ValueError(f"Class index outside model output: {class_index}")
        activation = activations[0]
        gradient = torch.autograd.grad(
            outputs=logits[0, class_index],
            inputs=activation,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((channel_weights * activation).sum(dim=1, keepdim=True))
        cam = functional.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam_min, cam_max = cam.min(), cam.max()
        if float((cam_max - cam_min).detach()) > 0.0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        return cam.detach()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
