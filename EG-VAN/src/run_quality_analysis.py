"""Run Phase 6 image-quality analysis without changing frozen inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from image_quality import QUALITY_FEATURES, compute_image_quality

CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_quality_table(root: Path, output_dir: Path, split_name: str) -> Path:
    metadata = {row["image_id"]: row for row in read_csv(root / "data/processed/metadata_clean.csv")}
    split_rows = read_csv(root / "data/splits" / split_name)
    if len(split_rows) != 10015 or len({row["image_id"] for row in split_rows}) != 10015:
        raise ValueError("Frozen split row count or image uniqueness is invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "lesion_id", "split", "dx", *QUALITY_FEATURES]
    output = output_dir / "image_quality.csv"
    failures = []
    seen = set()
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split_row in split_rows:
            image_id = split_row["image_id"]
            if image_id in seen:
                raise ValueError(f"Duplicate image ID: {image_id}")
            seen.add(image_id)
            if image_id not in metadata:
                raise ValueError(f"Split image missing from metadata: {image_id}")
            image_path = root / "data/processed/images" / f"{image_id}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            try:
                quality = compute_image_quality(image_path)
            except Exception as exc:
                failures.append({"image_id": image_id, "reason": str(exc)})
                continue
            writer.writerow({"image_id": image_id, "lesion_id": split_row["lesion_id"], "split": split_row["split"], "dx": split_row["dx"], **quality})
    if failures:
        (output_dir / "quality_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} quality measurements failed")
    if len(seen) != 10015:
        raise ValueError("Quality table did not cover all frozen split rows")
    return output


def summarize(rows: list[dict[str, str]], group_keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items()):
        item = {k: value for k, value in zip(group_keys, key)}
        item["count"] = len(group)
        for feature in QUALITY_FEATURES:
            values = [float(row[feature]) for row in group]
            item[f"{feature}_mean"] = statistics.fmean(values)
            item[f"{feature}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            item[f"{feature}_median"] = statistics.median(values)
            item[f"{feature}_q25"] = float(np.percentile(values, 25))
            item[f"{feature}_q75"] = float(np.percentile(values, 75))
        result.append(item)
    return result


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows: list[dict[str, str]], output_dir: Path) -> None:
    splits = ("train", "val", "test")
    for filename, group_field, groups in (("quality_by_split.png", "split", splits), ("quality_by_class.png", "dx", CLASS_NAMES)):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for axis, feature in zip(axes.flat, QUALITY_FEATURES):
            data = [[float(row[feature]) for row in rows if row[group_field] == group] for group in groups]
            axis.boxplot(data, tick_labels=groups, showfliers=False)
            axis.set_title(feature)
            axis.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)


def f1_macro(targets: list[int], predictions: list[int], class_count: int = 7) -> float:
    scores = []
    for cls in range(class_count):
        tp = sum(t == cls and p == cls for t, p in zip(targets, predictions))
        fp = sum(t != cls and p == cls for t, p in zip(targets, predictions))
        fn = sum(t == cls and p != cls for t, p in zip(targets, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(statistics.fmean(scores))


def model_error_analysis(root: Path, output_dir: Path, checkpoint: Path, split_name: str) -> str:
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        import sys
        sys.path.insert(0, str(root / "src"))
        from models.baseline_effnet import build_model
    except Exception as exc:
        return f"BLOCKED: model runtime import failed: {exc}"
    if not torch.cuda.is_available():
        return "BLOCKED: CUDA unavailable; model-error analysis was not run locally."
    try:
        checkpoint_data = torch.load(checkpoint, map_location="cuda", weights_only=False)
        model, weights = build_model(pretrained=False)
        model.load_state_dict(checkpoint_data["model_state"])
        model.cuda().eval()
        normalization = weights.transforms().normalize
        transform = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), transforms.Normalize(normalization.mean, normalization.std)])
        rows = [row for row in read_csv(root / "data/splits" / split_name) if row["split"] == "test"]
        quality = {row["image_id"]: row for row in read_csv(output_dir / "image_quality.csv")}
        records = []
        with torch.inference_mode():
            for row in rows:
                image = transform(Image.open(root / "data/processed/images" / f"{row['image_id']}.jpg").convert("RGB")).unsqueeze(0).cuda()
                probabilities = torch.softmax(model(image), dim=1)[0]
                prediction = int(probabilities.argmax())
                records.append({"image_id": row["image_id"], "target": CLASS_NAMES.index(row["dx"]), "predicted": prediction, "confidence": float(probabilities.max()), "correct": prediction == CLASS_NAMES.index(row["dx"])})
        with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        for record in records:
            record.update({feature: float(quality[record["image_id"]][feature]) for feature in QUALITY_FEATURES})
        (output_dir / "quality_error_analysis.json").write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
        return "COMPLETE"
    except Exception as exc:
        return f"BLOCKED: inference failed: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = root / "experiments/image_quality"
    table = build_quality_table(root, output_dir, "split_leakage_aware.csv")
    rows = read_csv(table)
    write_rows(output_dir / "quality_stats_by_split.csv", summarize(rows, ("split",)))
    write_rows(output_dir / "quality_stats_by_split_class.csv", summarize(rows, ("split", "dx")))
    make_plots(rows, output_dir)
    error_status = "BLOCKED: checkpoint not provided"
    if args.checkpoint:
        error_status = model_error_analysis(root, output_dir, args.checkpoint, "split_leakage_aware.csv")
    report = {"images_analyzed": len(rows), "quality_features": list(QUALITY_FEATURES), "error_analysis": error_status, "split": "split_leakage_aware.csv", "runtime": {"python": platform.python_version()}}
    (output_dir / "analysis_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
