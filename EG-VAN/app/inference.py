"""Single-image inference for the immutable Stage 23 research classifier.

No dataset loaders, network services, calibration transform, or decision gate.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "models" / "frozen_stage23", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from image_quality import compute_image_quality  # noqa: E402
from explainability.gradcam import gradcam  # noqa: E402
from load_frozen import (  # noqa: E402
    EXPECTED_CLASSES,
    EXPECTED_SHA,
    load_classifier,
    predict_rgb_image,
    verify_registry,
)

MODEL_ID = "EGVAN_STAGE23_FINAL"


def load_pipeline(device: str | torch.device | None = None):
    """Verify the frozen registry/checkpoint before strict-loading one model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_classifier(device)


def entropy_scores(probabilities: list[float] | np.ndarray) -> tuple[float, float]:
    values = np.asarray(probabilities, dtype=np.float64)
    if (values.shape != (7,) or not np.isfinite(values).all() or
            (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-5)):
        raise ValueError("Expected seven finite probabilities summing to one")
    positive = values[values > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    return entropy, entropy / math.log(7)


def _explain(model, transform, rgb: Image.Image, class_index: int) -> tuple[np.ndarray, Image.Image]:
    """Grad-CAM on the verified ResNet nonlocal3 layer; no parameter update."""
    if model.training:
        raise ValueError("Grad-CAM requires eval mode")
    device = next(model.parameters()).device
    tensor = transform(rgb).unsqueeze(0).to(device)
    target = model.resnet.nonlocal3
    with torch.enable_grad():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                cam = gradcam(model, tensor, target, class_index)
        else:
            cam = gradcam(model, tensor, target, class_index)
    if tuple(cam.shape) != (384, 384) or not torch.isfinite(cam).all().item():
        raise ValueError("Grad-CAM map is invalid")
    heatmap = cam.cpu().numpy().astype(np.float32)
    # The frozen transform resizes directly to 384x384. Render over that view,
    # preserving the original uploaded image separately in the caller/UI.
    resized = np.asarray(rgb.resize((384, 384), Image.Resampling.BILINEAR), dtype=np.float32)
    red = np.zeros_like(resized)
    red[..., 0] = 255.0
    overlay = Image.fromarray(np.uint8(np.clip(0.65 * resized + 0.35 * heatmap[..., None] * red, 0, 255)))
    return heatmap, overlay


def analyze_image(image: Image.Image, model, transform, classes=EXPECTED_CLASSES,
                  *, include_gradcam: bool = False) -> dict:
    """Analyze one user image; results are research output, not a diagnosis."""
    if tuple(classes) != EXPECTED_CLASSES:
        raise ValueError("Class order differs from frozen registry")
    if image.width < 16 or image.height < 16:
        raise ValueError("Image must be at least 16x16 pixels")
    rgb = image.convert("RGB")
    prediction = predict_rgb_image(model, transform, rgb, classes)
    probability = prediction["probabilities"]
    values = [probability[name] for name in classes]
    entropy, normalized = entropy_scores(values)
    # Stage 20 quality measurements were evaluated on processed images. Use
    # the same 384x384 view here, not a raw-image accept/reject gate.
    quality_view = np.asarray(rgb.resize((384, 384), Image.Resampling.BILINEAR), dtype=np.uint8)
    quality = compute_image_quality(quality_view)
    result = {
        "predicted_class": prediction["predicted_class"],
        "probabilities": probability,
        "confidence": max(values),
        "predictive_entropy": entropy,
        "normalized_predictive_entropy": normalized,
        "review_recommendation": None,
        "review_rule_status": "No Stage 23 validation-derived review threshold is registered",
        "image_quality": quality,
        "image_quality_status": "Descriptive technical measurements on resized image; no validated quality gate",
        "gradcam_heatmap": None,
        "gradcam_overlay": None,
        "gradcam_status": "Not requested",
        "model_identifier": MODEL_ID,
        "selected_epoch": 14,
        "checkpoint_sha256": EXPECTED_SHA,
    }
    if include_gradcam:
        index = classes.index(result["predicted_class"])
        heatmap, overlay = _explain(model, transform, rgb, index)
        result["gradcam_heatmap"] = heatmap
        result["gradcam_overlay"] = overlay
        result["gradcam_status"] = (
            "Degenerate zero-contrast map; no spatial interpretation available"
            if float(heatmap.max() - heatmap.min()) <= 1e-6 else
            "Qualitative attention visualization for predicted class; not localization ground truth"
        )
    return result
