"""Inference-only PH2 evaluation for the frozen HAM10000 checkpoint."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from dataset import CLASS_TO_INDEX
from models.baseline_effnet import build_model


PH2_TO_HAM10000 = {
    "common nevus": "nv",
    "melanoma": "mel",
}
# Atypical nevus is deliberately excluded because the repository has no
# approved HAM10000 equivalent label for it.
EXCLUDED_PH2_LABELS = {"atypical nevus"}


def load_verified_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load a manually verified external manifest; reject unknown mappings."""
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_path", "external_label", "ham_label"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"External manifest must contain {sorted(required)}")
    for row in rows:
        expected = PH2_TO_HAM10000.get(row["external_label"].strip().lower())
        if expected is None or row["ham_label"] != expected or expected not in CLASS_TO_INDEX:
            # Fail closed: a bad external mapping would create fabricated labels.
            raise ValueError(f"Unapproved or ambiguous PH2 mapping: {row}")
    return rows


def build_manifest_template(output_path: str | Path) -> None:
    """Create an empty manifest template; does not download or infer labels."""
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "external_label", "ham_label"])
        writer.writeheader()


def evaluate_ph2(manifest_path: str | Path, checkpoint_path: str | Path, output_dir: str | Path) -> dict:
    """Run frozen-checkpoint inference on mapped PH2 samples only."""
    if not torch.cuda.is_available():
        # Keep PH2 evaluation aligned with the approved Colab/CUDA baseline
        # runtime and avoid silently producing a new local-CPU condition.
        raise RuntimeError("PH2 evaluation requires the CUDA Colab runtime; CPU inference is not permitted.")
    rows = load_verified_manifest(manifest_path)
    # Only the direct label overlap is evaluated: common nevus -> nv and
    # melanoma -> mel. Atypical nevus rows must be absent from this manifest.
    rows = [row for row in rows if row["ham_label"] in {"nv", "mel"}]
    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, weights = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    if model.classifier[-1].out_features != 7:
        raise ValueError("Checkpoint classifier does not have seven outputs")
    model.to(device).eval()
    preprocess = weights.transforms()
    predictions = []
    matrix = np.zeros((2, 2), dtype=np.int64)
    index = {"nv": 0, "mel": 1}
    with torch.inference_mode():
        for row in rows:
            image = preprocess(Image.open(row["image_path"]).convert("RGB")).unsqueeze(0).to(device)
            prediction = int(model(image).argmax(dim=1).item())
            predicted_label = {4: "mel", 5: "nv"}.get(prediction, "other")
            actual = row["ham_label"]
            if predicted_label in index:
                matrix[index[actual], index[predicted_label]] += 1
            predictions.append({"image_id": row["image_path"], "actual": actual, "predicted": predicted_label})
    total = len(rows)
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    recalls = {label: float(matrix[i, i] / matrix[i, :].sum()) if matrix[i, :].sum() else 0.0 for i, label in enumerate(("nv", "mel"))}
    f1 = []
    for i in range(2):
        tp = matrix[i, i]; precision = tp / matrix[:, i].sum() if matrix[:, i].sum() else 0.0
        f1.append(2 * precision * recalls[("nv", "mel")[i]] / (precision + recalls[("nv", "mel")[i]]) if precision + recalls[("nv", "mel")[i]] else 0.0)
    result = {"evaluated_images": total, "accuracy": accuracy, "macro_f1": float(np.mean(f1)), "per_class_recall": recalls, "confusion_matrix": matrix.tolist(), "prediction_counts": {label: sum(p["predicted"] == label for p in predictions) for label in ("nv", "mel", "other")}}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "confusion_matrix.json").write_text(json.dumps(matrix.tolist(), indent=2), encoding="utf-8")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "actual", "predicted"]); writer.writeheader(); writer.writerows(predictions)
    (out / "evaluation_config.json").write_text(json.dumps({"checkpoint": str(checkpoint_path), "input_size": [384, 384], "normalization": "torchvision EfficientNetV2S weights", "device": str(device), "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("manifest"); parser.add_argument("checkpoint"); parser.add_argument("output_dir"); args = parser.parse_args()
    print(json.dumps(evaluate_ph2(args.manifest, args.checkpoint, args.output_dir), indent=2))
