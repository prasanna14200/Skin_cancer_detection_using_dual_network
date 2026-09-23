"""Checkpoint evaluation for the Phase 4C EfficientNetV2S baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, HAM10000Dataset
from models.baseline_effnet import build_model
from train import IMAGE_SIZE
from torchvision import transforms


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    project_root: str | Path,
    split_csv_name: str,
    split: str = "test",
    batch_size: int = 16,
) -> dict:
    """Evaluate an existing checkpoint without training or split changes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model, weights = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
    model.to(device).eval()
    # Rebuild the same deterministic test transform used by the baseline:
    # 384x384 resize plus ImageNet normalization from the EfficientNet weights.
    normalization = weights.transforms().normalize
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(normalization.mean, normalization.std),
    ])
    root = Path(project_root)
    dataset = HAM10000Dataset(root / "data/processed/images", root / "data/splits" / split_csv_name, split, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    predictions, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    # Macro-F1 is computed over all seven classes, which keeps minority-class
    # behavior visible instead of letting the dominant NV class control the score.
    total = len(targets)
    accuracy = float(sum(target == prediction for target, prediction in zip(targets, predictions)) / total)
    recalls = []
    f1_scores = []
    for class_index in range(len(CLASS_NAMES)):
        true_positive = matrix[class_index, class_index]
        actual = matrix[class_index, :].sum()
        predicted = matrix[:, class_index].sum()
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        recalls.append(recall)
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "split": split,
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "per_class_recall": {name: float(value) for name, value in zip(CLASS_NAMES, recalls)},
        "confusion_matrix": matrix.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("project_root")
    parser.add_argument("split_csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.project_root, args.split_csv, args.split)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
