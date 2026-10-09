"""Hash-checked Stage 23 classifier loader; no dataset access or weight download."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import EfficientNet_V2_S_Weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGISTRY = HERE / "model_registry.json"
EXPECTED_ID = "EGVAN_STAGE23_FINAL"
EXPECTED_SHA = "42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab"
EXPECTED_CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_registry(root: Path = ROOT) -> dict:
    """Verify the reference and every recorded source without mutating files."""
    root = root.resolve()
    if root != ROOT.resolve():
        raise ValueError("Use the pinned EG-VAN repository root")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if (registry["model_identifier"] != EXPECTED_ID or
            registry["selected_epoch"] != 14 or
            registry["checkpoint"]["sha256"] != EXPECTED_SHA or
            tuple(registry["class_order"]) != EXPECTED_CLASSES or
            registry["class_to_index"] != {name: i for i, name in enumerate(EXPECTED_CLASSES)} or
            registry["input_size"] != [3, 384, 384] or
            registry["preprocessing"]["evaluation_transform_order"] !=
            ["Resize((384,384))", "ToTensor()", "Normalize(mean,std)"]):
        raise ValueError("Frozen registry identity, class mapping, or input protocol changed")
    if registry["checkpoint"]["path"] != (
            "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt"):
        raise ValueError("Frozen checkpoint reference changed")
    for relative, expected in registry["source_artifact_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen source artifact missing or changed: {relative}")
    return registry


def load_classifier(device: str | torch.device = "cpu", *, root: Path = ROOT):
    """Return the strict-loaded eval model, deterministic transform, and class order."""
    registry = verify_registry(root)
    src = str(root / "src")
    policy = str(root / "analysis/stage13_melanoma_ablation")
    for entry in (src, policy):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    from models.egvan import EGVAN
    from stage13c_abc_selective_qk_fp32_gate import install_selective_qk_policy
    spec = importlib.util.spec_from_file_location("_egvan_frozen_src_train", root / "src/train.py")
    if spec is None or spec.loader is None:
        raise ImportError("Frozen preprocessing source cannot be loaded")
    preprocessing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocessing_module)

    state = torch.load(root / registry["checkpoint"]["path"],
                       map_location="cpu", weights_only=False, mmap=True)
    if (state.get("epoch") != 14 or state.get("best_epoch") != 14 or
            state.get("class_order") != list(EXPECTED_CLASSES) or
            state.get("architecture") != registry["architecture"]["identifier"]):
        raise ValueError("Frozen checkpoint metadata differs from registry")
    model = EGVAN(num_classes=7, pretrained_efficient=False, pretrained_resnet=False)
    model.load_state_dict(state["model_state"], strict=True)
    del state
    install_selective_qk_policy(model)
    model.to(device).eval()
    _, transform = preprocessing_module.make_transforms(EfficientNet_V2_S_Weights.DEFAULT)
    steps = transform.transforms
    expected = registry["preprocessing"]
    if ([type(step).__name__ for step in steps] != ["Resize", "ToTensor", "Normalize"] or
            tuple(steps[0].size) != tuple(expected["resize_pixels"]) or
            list(steps[2].mean) != expected["normalization_mean"] or
            list(steps[2].std) != expected["normalization_std"]):
        raise ValueError("Frozen validation preprocessing changed")
    return model, transform, EXPECTED_CLASSES


def predict_rgb_image(model, transform, image: Image.Image,
                      classes: tuple[str, ...] = EXPECTED_CLASSES) -> dict:
    """Single-image deterministic argmax; no threshold, calibration, or test loader."""
    if tuple(classes) != EXPECTED_CLASSES:
        raise ValueError("Frozen class order changed")
    device = next(model.parameters()).device
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(tensor)
        else:
            logits = model(tensor)
        if logits.shape != (1, 7) or not torch.isfinite(logits).all().item():
            raise ValueError("Frozen model returned invalid logits")
        probability = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()
    index = max(range(7), key=probability.__getitem__)
    return {"predicted_class": classes[index], "probabilities": dict(zip(classes, probability))}
