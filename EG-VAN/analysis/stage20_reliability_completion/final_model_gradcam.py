"""Bounded frozen-model Grad-CAM preparation/Colab execution; no training."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def normalized_cam_array(cam):
    """Detach the explanation from autograd before CPU/NumPy rendering."""
    detached = cam.detach()
    minimum, maximum = detached.min(), detached.max()
    if bool((maximum > minimum).item()):
        return ((detached - minimum) / (maximum - minimum)).cpu().numpy()
    return np.zeros(tuple(detached.shape), dtype=np.float32)


def choose_cases(root: Path, output: Path) -> None:
    source = root / "analysis/stage16_final_evaluation/final_ham_test_predictions.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1014 or len({r["image_id"] for r in rows}) != 1014:
        raise ValueError("Unexpected frozen HAM prediction IDs/count")
    groups = {
        "correct_mel": lambda r: r["true_class"] == "mel" and r["predicted_class"] == "mel",
        "mel_to_nv": lambda r: r["true_class"] == "mel" and r["predicted_class"] == "nv",
        "correct_nv": lambda r: r["true_class"] == "nv" and r["predicted_class"] == "nv",
        "correct_bkl": lambda r: r["true_class"] == "bkl" and r["predicted_class"] == "bkl",
        "correct_bcc": lambda r: r["true_class"] == "bcc" and r["predicted_class"] == "bcc",
    }
    # Fixed deterministic rule: lexicographically first two for each key clinical
    # group, then one per other-class group; no image was visualized to choose IDs.
    counts = {"correct_mel": 2, "mel_to_nv": 2, "correct_nv": 2,
              "correct_bkl": 1, "correct_bcc": 1}
    cases = []
    for group, condition in groups.items():
        matches = sorted((r for r in rows if condition(r)), key=lambda r: r["image_id"])
        if len(matches) < counts[group]:
            raise ValueError(f"Not enough cases for {group}")
        for row in matches[:counts[group]]:
            cases.append({"group": group, "image_id": row["image_id"],
                          "true_class": row["true_class"], "predicted_class": row["predicted_class"],
                          "saved_probability": float(row[f"p_{row['predicted_class']}"])})
    payload = {"status": "CASES_FROZEN_BEFORE_VISUALIZATION", "source_predictions_sha256": sha256(source),
               "selection_rule": "Lexicographically first IDs: 2 correct MEL, 2 MEL->NV, 2 correct NV, 1 correct BKL, 1 correct BCC",
               "target_layer": "model.resnet.nonlocal3", "target_class": "saved predicted class",
               "cases": cases, "checkpoint_sha256":
               "85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5"}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(f"Selected {len(cases)} cases; manifest SHA256 {sha256(output)}")


def run(root: Path, selection: Path, output: Path) -> None:
    import torch
    from torchvision.transforms.functional import resize as image_resize
    from torchvision.transforms import InterpolationMode

    sys.path.insert(0, str(root / "experiments/stage16_final_evaluation"))
    from common import CLASSES, CHECKPOINT_SHA, load_model, preflight, require_t4

    require_t4()
    frozen = json.loads(selection.read_text(encoding="utf-8"))
    source = root / "analysis/stage16_final_evaluation/final_ham_test_predictions.csv"
    if (frozen["source_predictions_sha256"] != sha256(source) or
        frozen["checkpoint_sha256"] != CHECKPOINT_SHA or
        len(frozen["cases"]) != 8):
        raise ValueError("Selection provenance mismatch")
    if output.exists():
        raise FileExistsError("Grad-CAM output already exists")
    preflight(root)
    gc.collect()
    model, transform = load_model(root)
    target_layer = model.resnet.nonlocal3
    if frozen["target_layer"] != "model.resnet.nonlocal3":
        raise ValueError("Target layer mismatch")
    output.mkdir(parents=True)
    records = []
    for case in frozen["cases"]:
        path = root / "data/processed/images" / f"{case['image_id']}.jpg"
        image_hash = sha256(path)
        original = Image.open(path).convert("RGB")
        x = transform(original).unsqueeze(0).cuda()
        activations = []
        logits = probabilities = activation = grad = weighted = cam = None

        def capture(_module, _inputs, tensor):
            activations.append(tensor)

        hook = target_layer.register_forward_hook(capture)
        try:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad(), torch.autocast("cuda"):
                logits = model(x)
                if len(activations) != 1 or not torch.isfinite(logits).all():
                    raise ValueError(f"Invalid final-model forward: {case['image_id']}")
                probabilities = torch.softmax(logits.float(), dim=1)[0]
                predicted = CLASSES[int(probabilities.argmax())]
                p_saved = case["saved_probability"]
                p_replayed = float(probabilities[CLASSES.index(case["predicted_class"])].detach().item())
                if predicted != case["predicted_class"] or abs(p_replayed - p_saved) > 5e-4:
                    raise ValueError(f"Saved-output mismatch: {case['image_id']} {p_replayed} vs {p_saved}")
                activation = activations[0]
                grad = torch.autograd.grad(logits[0, CLASSES.index(predicted)], activation)[0]
                if not torch.isfinite(activation).all() or not torch.isfinite(grad).all():
                    raise ValueError(f"Non-finite activation/gradient: {case['image_id']}")
                weighted = torch.relu((grad.float().mean(dim=(2, 3), keepdim=True) * activation.float()).sum(dim=1))
                cam = torch.nn.functional.interpolate(weighted.unsqueeze(1), size=(384, 384),
                                                      mode="bilinear", align_corners=False)[0, 0]
                if not torch.isfinite(cam).all():
                    raise ValueError(f"Non-finite map: {case['image_id']}")
                cam = normalized_cam_array(cam)
            rgb = np.asarray(image_resize(original, [384, 384], interpolation=InterpolationMode.BILINEAR), dtype=np.float32) / 255
            heat = np.stack((cam, np.zeros_like(cam), 1 - cam), axis=2)
            overlay = np.clip(.65 * rgb + .35 * heat, 0, 1)
            name = f"{case['group']}_{case['image_id']}"
            Image.fromarray(np.uint8(cam * 255)).save(output / f"{name}_map.png")
            Image.fromarray(np.uint8(overlay * 255)).save(output / f"{name}_overlay.png")
            records.append({**case, "processed_image_sha256": image_hash, "replayed_probability": p_replayed,
                            "map_min": float(np.min(cam)), "map_max": float(np.max(cam)),
                            "map_sha256": sha256(output / f"{name}_map.png"),
                            "overlay_sha256": sha256(output / f"{name}_overlay.png")})
        finally:
            hook.remove()
            model.zero_grad(set_to_none=True)
            del x, activations, logits, probabilities, activation, grad, weighted, cam
            gc.collect()
            torch.cuda.empty_cache()
    result = {"status": "MAPS_GENERATED_REVIEW_PENDING", "selection_manifest_sha256": sha256(selection),
              "checkpoint_sha256": CHECKPOINT_SHA, "cases": records,
              "interpretation_limit": "Grad-CAM is descriptive, not causal or clinically validated",
              "training_performed": False, "ph2_accessed": False}
    with (output / "gradcam_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"GRADCAM_COMPLETE cases={len(records)} output={output}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = p.parse_args()
    root = args.project_root.resolve()
    base = root / "analysis/stage20_reliability_completion"
    selection = base / "gradcam_selection.json"
    if args.prepare:
        choose_cases(root, selection)
    else:
        # v1 may contain incomplete evidence from the rendering failure.
        # Never delete or overwrite that directory during the bounded rerun.
        run(root, selection, base / "final_model_gradcam_output_v2")


if __name__ == "__main__":
    main()
