"""Post-hoc uncertainty analysis for the frozen leakage-aware checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
FEATURES = ("brightness", "contrast", "sharpness", "saturation", "dark_pixel_ratio", "bright_pixel_ratio", "entropy", "illumination_variation")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def entropy(probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def macro_f1(targets: list[int], predictions: list[int]) -> float:
    values = []
    for cls in range(len(CLASS_NAMES)):
        tp = sum(t == cls and p == cls for t, p in zip(targets, predictions))
        fp = sum(t != cls and p == cls for t, p in zip(targets, predictions))
        fn = sum(t == cls and p != cls for t, p in zip(targets, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(values))


def calibration_bins(records: list[dict], bin_count: int = 10) -> list[dict]:
    result = []
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        selected = [row for row in records if lower <= row["confidence"] and (row["confidence"] < upper or (index == bin_count - 1 and row["confidence"] <= upper))]
        result.append({
            "bin": index,
            "lower": lower,
            "upper": upper,
            "count": len(selected),
            "mean_confidence": float(np.mean([row["confidence"] for row in selected])) if selected else None,
            "empirical_accuracy": float(np.mean([row["correct"] for row in selected])) if selected else None,
            "absolute_gap": float(abs(np.mean([row["confidence"] for row in selected]) - np.mean([row["correct"] for row in selected]))) if selected else None,
        })
    return result


def selective_prediction(records: list[dict], thresholds: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)) -> list[dict]:
    result = []
    for threshold in thresholds:
        retained = [row for row in records if row["confidence"] >= threshold]
        result.append({"threshold": threshold, "retained": len(retained), "coverage": len(retained) / len(records), "accuracy": float(np.mean([row["correct"] for row in retained])) if retained else None})
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def save_plots(records: list[dict], bins: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    correct_records = [row for row in records if row["correct"]]
    incorrect_records = [row for row in records if not row["correct"]]
    plt.figure(figsize=(8, 5)); plt.hist([row["confidence"] for row in correct_records], bins=20, alpha=0.6, label="Correct"); plt.hist([row["confidence"] for row in incorrect_records], bins=20, alpha=0.6, label="Incorrect"); plt.xlabel("Model confidence"); plt.ylabel("Count"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / "confidence_distribution.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 5)); plt.boxplot([[row["confidence"] for row in correct_records], [row["confidence"] for row in incorrect_records]], tick_labels=["Correct", "Incorrect"], showfliers=False); plt.ylabel("Model confidence"); plt.tight_layout(); plt.savefig(output_dir / "confidence_correct_vs_incorrect.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5)); plt.hist([row["entropy"] for row in correct_records], bins=20, alpha=0.6, label="Correct"); plt.hist([row["entropy"] for row in incorrect_records], bins=20, alpha=0.6, label="Incorrect"); plt.xlabel("Predictive entropy"); plt.ylabel("Count"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / "entropy_distribution.png", dpi=150); plt.close()
    populated = [row for row in bins if row["count"]]
    plt.figure(figsize=(7, 5)); plt.plot([row["mean_confidence"] for row in populated], [row["empirical_accuracy"] for row in populated], "o-", label="Observed"); plt.plot([0, 1], [0, 1], "--", label="Perfect calibration"); plt.xlabel("Mean confidence"); plt.ylabel("Empirical accuracy"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / "reliability_diagram.png", dpi=150); plt.close()


def quality_uncertainty_rows(root: Path, output_dir: Path, records: list[dict]) -> list[dict]:
    quality_path = root / "experiments/image_quality/image_quality.csv"
    if not quality_path.is_file():
        return []
    quality = {row["image_id"]: row for row in read_csv(quality_path)}
    merged = [dict(record, **{f"quality_{feature}": float(quality[record["image_id"]][feature]) for feature in FEATURES}) for record in records if record["image_id"] in quality]
    output = []
    for feature in FEATURES:
        quality_key = f"quality_{feature}"
        order = np.argsort([row[quality_key] for row in merged], kind="stable")
        for index, positions in enumerate(np.array_split(order, 3)):
            selected = [merged[int(position)] for position in positions]
            output.append({"feature": feature, "quantile": index + 1, "quantile_name": ("lowest", "middle", "highest")[index], "count": len(selected), "quality_min": min(row[quality_key] for row in selected), "quality_max": max(row[quality_key] for row in selected), "accuracy": float(np.mean([row["correct"] for row in selected])), "error_rate": float(1.0 - np.mean([row["correct"] for row in selected])), "mean_confidence": float(np.mean([row["confidence"] for row in selected])), "mean_predictive_entropy": float(np.mean([row["entropy"] for row in selected]))})
    write_csv(output_dir / "quality_uncertainty.csv", output)
    return output


def postprocess_saved_outputs(root: Path, output_dir: Path) -> None:
    """Recompute calibration/quality summaries from saved predictions only."""
    records = read_csv(output_dir / "predictions.csv")
    for row in records:
        row["confidence"] = float(row["confidence"])
        row["entropy"] = float(row["entropy"])
        row["correct"] = row["correct"].lower() == "true"
        row["probabilities"] = json.loads(row["probabilities"])
    if len(records) != 1014 or len({row["image_id"] for row in records}) != 1014:
        raise ValueError("Saved predictions must contain 1,014 unique test images")
    bins = calibration_bins(records)
    write_csv(output_dir / "calibration_bins.csv", bins)
    summary_path = output_dir / "uncertainty_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["ece"] = float(sum(row["count"] / len(records) * (row["absolute_gap"] or 0.0) for row in bins))
    summary["calibration_bin_count"] = len(bins)
    summary["calibration_bin_sample_total"] = sum(row["count"] for row in bins)
    summary["postprocessing_note"] = "Calibration bins, ECE, and quality tercile analysis recomputed from saved predictions; model inference was not rerun."
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_plots(records, bins, output_dir)
    quality_uncertainty_rows(root, output_dir, records)


def run_inference(root: Path, checkpoint_path: Path, output_dir: Path) -> int:
    import torch
    split_rows = read_csv(root / "data/splits/split_leakage_aware.csv")
    test_rows = [row for row in split_rows if row["split"] == "test"]
    if len(test_rows) != 1014 or len({row["image_id"] for row in test_rows}) != 1014:
        raise ValueError("Expected 1,014 unique leakage-aware test rows")
    for row in test_rows:
        image_path = root / "data/processed/images" / f"{row['image_id']}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    checkpoint_preview = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classifier_shape = tuple(checkpoint_preview["model_state"]["classifier.1.weight"].shape)
    if classifier_shape[0] != 7:
        raise ValueError(f"Checkpoint classifier has unexpected shape: {classifier_shape}")
    if not torch.cuda.is_available():
        print(json.dumps({"status": "BLOCKED", "reason": "CUDA unavailable; CPU inference is not permitted.", "torch": torch.__version__, "test_rows": len(test_rows), "classifier_shape": classifier_shape}, indent=2))
        return 2
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    import sys
    sys.path.insert(0, str(root / "src"))
    from models.baseline_effnet import build_model

    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model, weights = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    if model.classifier[-1].out_features != 7:
        raise ValueError("Checkpoint classifier does not have seven outputs")
    model.cuda().eval()
    transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    class TestDataset(Dataset):
        def __len__(self): return len(test_rows)
        def __getitem__(self, index):
            row = test_rows[index]
            image = transform(Image.open(root / "data/processed/images" / f"{row['image_id']}.jpg").convert("RGB"))
            return image, CLASS_NAMES.index(row["dx"]), row["image_id"]

    loader = DataLoader(TestDataset(), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    records = []
    with torch.inference_mode():
        for images, targets, image_ids in loader:
            probabilities = torch.softmax(model(images.cuda(non_blocking=True)), dim=1).cpu().numpy()
            for probability, target, image_id in zip(probabilities, targets.tolist(), image_ids):
                prediction = int(np.argmax(probability))
                records.append({"image_id": image_id, "true_label": CLASS_NAMES[target], "predicted_label": CLASS_NAMES[prediction], "correct": prediction == target, "confidence": float(probability.max()), "entropy": entropy(probability), "probabilities": json.dumps(probability.tolist(), separators=(",", ":"))})

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "predictions.csv", records)
    bins = calibration_bins(records)
    write_csv(output_dir / "calibration_bins.csv", bins)
    selective = selective_prediction(records)
    write_csv(output_dir / "selective_prediction.csv", selective)
    confusion = [[sum(row["true_label"] == CLASS_NAMES[i] and row["predicted_label"] == CLASS_NAMES[j] for row in records) for j in range(7)] for i in range(7)]
    per_class = {}
    for label in CLASS_NAMES:
        group = [row for row in records if row["true_label"] == label]
        per_class[label] = {"count": len(group), "mean_confidence": float(np.mean([row["confidence"] for row in group])), "mean_entropy": float(np.mean([row["entropy"] for row in group])), "accuracy": float(np.mean([row["correct"] for row in group]))}
    correct = [row for row in records if row["correct"]]
    incorrect = [row for row in records if not row["correct"]]
    ece = float(sum((row["count"] / len(records)) * (row["absolute_gap"] or 0.0) for row in bins))
    summary = {"status": "COMPLETE", "test_samples": len(records), "accuracy": float(np.mean([row["correct"] for row in records])), "macro_f1": macro_f1([CLASS_NAMES.index(row["true_label"]) for row in records], [CLASS_NAMES.index(row["predicted_label"]) for row in records]), "mean_confidence": float(np.mean([row["confidence"] for row in records])), "median_confidence": float(np.median([row["confidence"] for row in records])), "mean_entropy": float(np.mean([row["entropy"] for row in records])), "median_entropy": float(np.median([row["entropy"] for row in records])), "correct_confidence": float(np.mean([row["confidence"] for row in correct])), "incorrect_confidence": float(np.mean([row["confidence"] for row in incorrect])), "correct_entropy": float(np.mean([row["entropy"] for row in correct])), "incorrect_entropy": float(np.mean([row["entropy"] for row in incorrect])), "ece": ece, "confusion_matrix": confusion, "per_class": per_class, "selective_prediction": selective, "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "python": platform.python_version()}}
    (output_dir / "uncertainty_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "per_class_uncertainty.csv", [{"true_label": label, **values} for label, values in per_class.items()])
    save_plots(records, bins, output_dir)
    quality_uncertainty_rows(root, output_dir, records)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--postprocess-existing", action="store_true", help="Recompute derivative artifacts from predictions.csv without model inference")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir or root / "experiments/uncertainty"
    if args.postprocess_existing:
        postprocess_saved_outputs(root, output_dir)
        print("POSTPROCESS_COMPLETE: saved predictions reused; inference not run")
        return 0
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    return run_inference(root, args.checkpoint, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
