"""CPU-only tests for local integration; no HAM test or PH2 fixtures."""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference import (  # noqa: E402
    EXPECTED_CLASSES,
    analyze_image,
    entropy_scores,
    load_pipeline,
)
from src.explainability.gradcam import gradcam  # noqa: E402


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def test_entropy_limits_and_validation():
    one_hot = [1.0] + [0.0] * 6
    assert entropy_scores(one_hot) == (0.0, 0.0)
    uniform = [1 / 7] * 7
    h, normalized = entropy_scores(uniform)
    assert h == pytest.approx(math.log(7))
    assert normalized == pytest.approx(1.0)
    with pytest.raises(ValueError):
        entropy_scores([float("nan")] + [0.0] * 6)


class TinyCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.target = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, 7)

    def forward(self, x):
        return self.head(self.target(x).relu().mean(dim=(2, 3)))


def test_gradcam_utility_finite_and_no_mutation():
    torch.manual_seed(4)
    model = TinyCamModel().eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    cam = gradcam(model, torch.rand(1, 3, 32, 32), model.target, 4)
    assert tuple(cam.shape) == (32, 32)
    assert torch.isfinite(cam).all().item()
    assert cam.min() >= 0 and cam.max() <= 1
    assert not model.training
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


@pytest.fixture(scope="module")
def frozen():
    torch.set_num_threads(2)
    return load_pipeline("cpu")


def test_frozen_pipeline_synthetic_and_checkpoint_unchanged(frozen):
    checkpoint = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt"
    before = _digest(checkpoint)
    model, transform, classes = frozen
    assert not model.training
    assert tuple(classes) == EXPECTED_CLASSES
    rgb = Image.new("RGB", (384, 384), (127, 109, 92))
    transformed = transform(rgb)
    assert tuple(transformed.shape) == (3, 384, 384)
    expected = (np.array([127, 109, 92]) / 255 - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
    assert np.allclose(transformed[:, 0, 0].numpy(), expected, atol=1e-5)
    result = analyze_image(rgb, model, transform, classes)
    assert result["predicted_class"] in classes
    assert tuple(result["probabilities"]) == classes
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-5
    assert all(np.isfinite(v) for v in result["image_quality"].values())
    assert result["review_recommendation"] is None
    assert result["gradcam_heatmap"] is None
    assert result["checkpoint_sha256"] == before
    assert _digest(checkpoint) == before


def test_local_app_has_no_external_api_dependency():
    paths = (ROOT / "app/inference.py", ROOT / "app/streamlit_app.py")
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "requests." not in text and "openai" not in text.lower()
    assert "stage20" not in text.lower()
