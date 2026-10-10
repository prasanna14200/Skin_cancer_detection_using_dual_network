"""Frozen EG-VAN HAM test evaluation. --preflight does not run inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_S_Weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CHECKPOINT_SHA = "60f34ea4cfa6cbf3dce69e8d8bcc82292d8d8813af30f33b2f666a4f06340de0"
SPLIT_SHA = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
EXP5_SHA = "81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a"
COUNTS = {"akiec": 40, "bcc": 58, "bkl": 104, "df": 11, "mel": 107, "nv": 676, "vasc": 18}
OUTPUTS = ("test_predictions.csv", "test_metrics.json", "per_class_metrics.csv",
           "confusion_matrix.csv", "confusion_matrix.png", "normalized_confusion_matrix.png",
           "egvan_vs_exp5_comparison.csv", "stage10_internal_test_report.md")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def local_imports(root: Path):
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from dataset import CLASS_NAMES, HAM10000Dataset, validate_split_integrity
    from models.egvan import EGVAN
    from train import make_transforms
    return CLASS_NAMES, HAM10000Dataset, validate_split_integrity, EGVAN, make_transforms


def preflight(root: Path) -> dict:
    root = root.resolve()
    ckpt = root / "experiments/egvan_reconstruction_controlled_exp1/best_checkpoint.pt"
    split = root / "data/splits/split_leakage_aware.csv"
    exp5 = root / "experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt"
    hashes = {"egvan_best_checkpoint": sha256(ckpt), "frozen_split": sha256(split),
              "experiment5_checkpoint": sha256(exp5)}
    if hashes != {"egvan_best_checkpoint": CHECKPOINT_SHA, "frozen_split": SPLIT_SHA,
                  "experiment5_checkpoint": EXP5_SHA}:
        raise ValueError(f"Protected hash mismatch: {hashes}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    if state["epoch"] != 15 or state["best_epoch"] != 15 or tuple(state["class_order"]) != CLASSES:
        raise ValueError("Selected checkpoint epoch or class order mismatch")
    run_cfg = state["training_configuration"]
    if (run_cfg["image_size"] != 384 or run_cfg["architecture"] != "stage8b_egvan_four_mff_serial_terminal_branches"
            or run_cfg["split_sha256"] != SPLIT_SHA or run_cfg["effective_batch_size"] != 16):
        raise ValueError("Checkpoint configuration differs from frozen Stage 9 protocol")
    del state
    names, Dataset, integrity_fn, _, make_transforms = local_imports(root)
    if tuple(names) != CLASSES:
        raise ValueError("Dataset class order mismatch")
    integrity = integrity_fn(split, require_lesion_isolation=True)
    if integrity["crossing_lesions"] or integrity["rows"] != 10015:
        raise ValueError("Frozen split integrity mismatch")
    partitions = defaultdict(set)
    counts = Counter()
    rows = read_csv(split)
    for row in rows:
        if row["split"] not in ("train", "val", "test") or row["dx"] not in CLASSES:
            raise ValueError("Unexpected split partition or class")
        partitions[row["split"]].add(row["image_id"])
        if row["split"] == "test":
            counts[row["dx"]] += 1
    if dict(counts) != COUNTS or len(partitions["test"]) != 1014:
        raise ValueError(f"Frozen test counts mismatch: {counts}")
    if partitions["test"] & (partitions["train"] | partitions["val"]):
        raise ValueError("Test image ID overlaps train or validation")
    missing = [image_id for image_id in partitions["test"]
               if not (root / "data/processed/images" / f"{image_id}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing test images: {missing[:5]} (total {len(missing)})")
    _, eval_tf = make_transforms(EfficientNet_V2_S_Weights.DEFAULT)
    steps = [type(t).__name__ for t in eval_tf.transforms]
    if steps != ["Resize", "ToTensor", "Normalize"] or tuple(eval_tf.transforms[0].size) != (384, 384):
        raise ValueError(f"Evaluation preprocessing drift: {steps}")
    weight_preprocess = EfficientNet_V2_S_Weights.DEFAULT.transforms()
    if (tuple(eval_tf.transforms[2].mean) != tuple(weight_preprocess.mean)
            or tuple(eval_tf.transforms[2].std) != tuple(weight_preprocess.std)):
        raise ValueError("ImageNet normalization drift")
    if len(Dataset(root / "data/processed/images", split, "test", eval_tf)) != 1014:
        raise ValueError("Test loader length mismatch")
    return {"status": "PASS", "hashes": hashes, "checkpoint_epoch": 15,
            "class_order": list(CLASSES), "test_sample_count": 1014,
            "test_class_counts": {c: counts[c] for c in CLASSES},
            "split_integrity": integrity, "missing_test_images": 0,
            "evaluation_preprocessing": "processed RGB JPEG -> Resize((384,384)) -> ToTensor -> EfficientNet_V2_S_Weights.DEFAULT mean/std; no augmentation or TTA",
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}


def matrix_from_rows(rows: list[dict]) -> list[list[int]]:
    matrix = [[0] * 7 for _ in CLASSES]
    for row in rows:
        matrix[CLASSES.index(row["true_label"])][CLASSES.index(row["predicted_label"])] += 1
    return matrix


def metrics_from_matrix(matrix: list[list[int]]) -> dict:
    support = [sum(row) for row in matrix]
    total = sum(support)
    per = {}
    for i, name in enumerate(CLASSES):
        tp = matrix[i][i]
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support[i] if support[i] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[name] = {"support": support[i], "predicted_count": predicted,
                     "precision": precision, "recall": recall, "f1": f1}
    correct = sum(matrix[i][i] for i in range(7))
    result = {"sample_count": total, "class_order": list(CLASSES),
              "accuracy": correct / total, "balanced_accuracy": sum(v["recall"] for v in per.values()) / 7,
              "macro_precision": sum(v["precision"] for v in per.values()) / 7,
              "macro_recall": sum(v["recall"] for v in per.values()) / 7,
              "macro_f1": sum(v["f1"] for v in per.values()) / 7,
              "weighted_precision": sum(v["precision"] * v["support"] for v in per.values()) / total,
              "weighted_recall": sum(v["recall"] * v["support"] for v in per.values()) / total,
              "weighted_f1": sum(v["f1"] * v["support"] for v in per.values()) / total,
              "per_class": per, "confusion_matrix": matrix}
    return result


def verify_predictions(rows: list[dict], split_rows: list[dict], saved: dict) -> dict:
    test = {r["image_id"]: r["dx"] for r in split_rows if r["split"] == "test"}
    forbidden = {r["image_id"] for r in split_rows if r["split"] != "test"}
    ids = [r["image_id"] for r in rows]
    if len(rows) != 1014 or len(set(ids)) != 1014 or set(ids) != set(test) or set(ids) & forbidden:
        raise ValueError("Prediction identity/count integrity failure")
    for row in rows:
        if row["true_label"] != test[row["image_id"]] or row["predicted_label"] not in CLASSES:
            raise ValueError(f"Prediction label mismatch: {row['image_id']}")
        probabilities = json.loads(row["probabilities"])
        if (len(probabilities) != 7 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or v > 1 for v in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-4)):
            raise ValueError(f"Invalid probabilities: {row['image_id']}")
        if CLASSES[max(range(7), key=lambda i: probabilities[i])] != row["predicted_label"]:
            raise ValueError(f"Argmax mismatch: {row['image_id']}")
        if (row["correct"] == "True") != (row["true_label"] == row["predicted_label"]):
            raise ValueError(f"Correctness mismatch: {row['image_id']}")
    recomputed = metrics_from_matrix(matrix_from_rows(rows))
    if saved != recomputed:
        raise ValueError("Recomputed metrics do not exactly match saved metrics")
    return {"status": "PASS", "prediction_rows": len(rows), "unique_ids": len(set(ids)),
            "probabilities_valid": True, "recomputed_metrics_equal_saved": True}


def write_csv_new(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def plot_matrix(path: Path, matrix: list[list[float]], title: str, fmt: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(7), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(7), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, format(matrix[i][j], fmt), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def compare(metrics: dict, exp5: dict, exp5_rows: list[dict], rows: list[dict]) -> tuple[list[dict], dict]:
    if exp5["sample_count"] != 1014 or exp5["class_order"] != list(CLASSES):
        raise ValueError("Experiment #5 saved metrics not on frozen test protocol")
    exp5_matrix_metrics = metrics_from_matrix(exp5["confusion_matrix"])
    if matrix_from_rows(exp5_rows) != exp5["confusion_matrix"]:
        raise ValueError("Experiment #5 prediction file does not reproduce saved confusion matrix")
    if any((r["correct"] == "True") != (r["true_label"] == r["predicted_label"]) for r in exp5_rows):
        raise ValueError("Experiment #5 correctness flags inconsistent")
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        if not math.isclose(exp5_matrix_metrics[key], exp5[key], abs_tol=1e-12):
            raise ValueError("Experiment #5 saved metrics internally inconsistent")
    fields = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    values = {k: (exp5_matrix_metrics[k], metrics[k]) for k in fields}
    for key, label in (("precision", "melanoma_precision"), ("recall", "melanoma_recall"), ("f1", "melanoma_f1")):
        values[label] = (exp5_matrix_metrics["per_class"]["mel"][key], metrics["per_class"]["mel"][key])
    values["nevus_recall"] = (exp5_matrix_metrics["per_class"]["nv"]["recall"], metrics["per_class"]["nv"]["recall"])
    comparison = [{"metric": name, "experiment5": a, "egvan": b, "egvan_minus_experiment5": b - a}
                  for name, (a, b) in values.items()]
    old = {r["image_id"]: r for r in exp5_rows}
    new = {r["image_id"]: r for r in rows}
    if len(old) != 1014 or set(old) != set(new):
        raise ValueError("Experiment #5 and EG-VAN prediction IDs differ")
    paired = {"both_correct": 0, "egvan_only_correct": 0,
              "experiment5_only_correct": 0, "both_wrong": 0}
    for image_id in new:
        if old[image_id]["true_label"] != new[image_id]["true_label"]:
            raise ValueError("Paired test ground truth differs")
        e = new[image_id]["correct"] == "True"
        b = old[image_id]["correct"] == "True"
        paired["both_correct" if e and b else "egvan_only_correct" if e else
               "experiment5_only_correct" if b else "both_wrong"] += 1
    if sum(paired.values()) != 1014:
        raise ValueError("Paired correctness count mismatch")
    return comparison, paired


def run(root: Path, workers: int, batch_size: int) -> None:
    report = preflight(root)
    print(json.dumps(report, indent=2), flush=True)
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Stage 10 inference requires a CUDA Tesla T4; preflight did not run inference")
    if workers < 0 or batch_size < 1:
        raise ValueError("Invalid batch size or worker count")
    output = root / "analysis/egvan_internal_test_exp1"
    if any((output / name).exists() for name in OUTPUTS):
        raise FileExistsError("Stage 10 test outputs already exist; refusing repeat inference/overwrite")
    names, Dataset, _, EGVAN, make_transforms = local_imports(root)
    _, transform = make_transforms(EfficientNet_V2_S_Weights.DEFAULT)
    ds = Dataset(root / "data/processed/images", root / "data/splits/split_leakage_aware.csv", "test", transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    state = torch.load(root / "experiments/egvan_reconstruction_controlled_exp1/best_checkpoint.pt",
                       map_location="cpu", weights_only=False, mmap=True)
    model = EGVAN(num_classes=7, pretrained_efficient=False, pretrained_resnet=False)
    model.load_state_dict(state["model_state"], strict=True)
    del state
    model.cuda().eval()
    rows = []
    with torch.inference_mode():
        for images, labels in loader:
            with torch.autocast("cuda"):
                logits = model(images.cuda(non_blocking=True))
            probs = torch.softmax(logits.float(), dim=1).cpu().tolist()
            for j, probability in enumerate(probs):
                source = ds.rows[len(rows)]
                prediction = CLASSES[max(range(7), key=lambda k: probability[k])]
                rows.append({"image_id": source["image_id"], "true_label": source["dx"],
                             "predicted_label": prediction, "correct": str(source["dx"] == prediction),
                             "probabilities": json.dumps(probability)})
    metrics = metrics_from_matrix(matrix_from_rows(rows))
    split_rows = read_csv(root / "data/splits/split_leakage_aware.csv")
    integrity = verify_predictions(rows, split_rows, metrics)
    exp5_metrics = json.loads((root / "experiments/efficientnetv2s_controlled_exp5/test_metrics.json").read_text())
    exp5_rows = read_csv(root / "experiments/efficientnetv2s_controlled_exp5/test_predictions.csv")
    comparison, paired = compare(metrics, exp5_metrics, exp5_rows, rows)
    write_csv_new(output / "test_predictions.csv", rows,
                  ["image_id", "true_label", "predicted_label", "correct", "probabilities"])
    write_json_new(output / "test_metrics.json", metrics)
    # Read the actual saved files and recompute, independent of the in-memory rows.
    verify_predictions(read_csv(output / "test_predictions.csv"), split_rows,
                       json.loads((output / "test_metrics.json").read_text()))
    write_csv_new(output / "per_class_metrics.csv",
                  [{"class": name, **metrics["per_class"][name]} for name in CLASSES],
                  ["class", "support", "predicted_count", "precision", "recall", "f1"])
    write_csv_new(output / "confusion_matrix.csv",
                  [{"true_label": name, **dict(zip(CLASSES, metrics["confusion_matrix"][i]))}
                   for i, name in enumerate(CLASSES)], ["true_label", *CLASSES])
    matrix = metrics["confusion_matrix"]
    normalized = [[value / sum(row) if sum(row) else 0.0 for value in row] for row in matrix]
    plot_matrix(output / "confusion_matrix.png", matrix, "EG-VAN HAM test confusion matrix", "d")
    plot_matrix(output / "normalized_confusion_matrix.png", normalized,
                "EG-VAN HAM test normalized confusion matrix", ".2f")
    write_csv_new(output / "egvan_vs_exp5_comparison.csv", comparison,
                  ["metric", "experiment5", "egvan", "egvan_minus_experiment5"])
    errors = sorted(((matrix[i][j], CLASSES[i], CLASSES[j]) for i in range(7)
                     for j in range(7) if i != j), reverse=True)
    report_text = ("# Stage 10 frozen HAM internal test evaluation\n\n"
        f"EG-VAN checkpoint `{CHECKPOINT_SHA}` at epoch 15; frozen split `{SPLIT_SHA}`. "
        "This test set was used for evaluation only. No training, threshold tuning, checkpoint reselection, or PH² access occurred.\n\n"
        f"Test samples: {metrics['sample_count']}; prediction integrity: {integrity['status']}.\n\n"
        f"Accuracy {metrics['accuracy']:.6f}; balanced accuracy {metrics['balanced_accuracy']:.6f}; "
        f"macro F1 {metrics['macro_f1']:.6f}; weighted F1 {metrics['weighted_f1']:.6f}.\n\n"
        f"Melanoma precision {metrics['per_class']['mel']['precision']:.6f}, "
        f"recall {metrics['per_class']['mel']['recall']:.6f}, F1 {metrics['per_class']['mel']['f1']:.6f}; "
        f"NV recall {metrics['per_class']['nv']['recall']:.6f}.\n\n"
        f"Experiment #5 accuracy {exp5_metrics['accuracy']:.6f}; EG-VAN minus Experiment #5 "
        f"{metrics['accuracy'] - exp5_metrics['accuracy']:+.6f}. Descriptive comparison only; no significance claim.\n\n"
        f"Paired correctness: {paired}. McNemar exact test not performed.\n\n"
        f"Largest off-diagonal transitions (count, true, predicted): {errors[:10]}. "
        f"MEL→NV {matrix[CLASSES.index('mel')][CLASSES.index('nv')]}; "
        f"NV→MEL {matrix[CLASSES.index('nv')][CLASSES.index('mel')]}.\n")
    with (output / "stage10_internal_test_report.md").open("x", encoding="utf-8") as f:
        f.write(report_text)
    source_files = {
        "egvan_best_checkpoint": root / "experiments/egvan_reconstruction_controlled_exp1/best_checkpoint.pt",
        "experiment5_checkpoint": root / "experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt",
        "frozen_split": root / "data/splits/split_leakage_aware.csv",
        "experiment5_test_metrics": root / "experiments/efficientnetv2s_controlled_exp5/test_metrics.json",
        "experiment5_test_predictions": root / "experiments/efficientnetv2s_controlled_exp5/test_predictions.csv",
        "egvan_config": root / "experiments/egvan_reconstruction_controlled_exp1/config.json",
        "evaluation_script": Path(__file__), "train_transforms_source": root / "src/train.py",
        "egvan_model_source": root / "src/models/egvan/egvan.py"}
    if (sha256(source_files["egvan_best_checkpoint"]) != CHECKPOINT_SHA
            or sha256(source_files["experiment5_checkpoint"]) != EXP5_SHA
            or sha256(source_files["frozen_split"]) != SPLIT_SHA):
        raise ValueError("Protected source hash changed during evaluation")
    manifest = {"status": "COMPLETE", "egvan_checkpoint_path": str(source_files["egvan_best_checkpoint"].relative_to(root)),
                "egvan_checkpoint_sha256": CHECKPOINT_SHA, "selected_epoch": 15,
                "frozen_split_sha256": SPLIT_SHA, "class_order": list(CLASSES), "test_sample_count": 1014,
                "runtime": report["runtime"], "gpu": report["runtime"]["gpu"],
                "evaluation_preprocessing": report["evaluation_preprocessing"],
                "source_artifact_hashes": {name: sha256(path) for name, path in source_files.items()},
                "generated_artifact_hashes": {name: sha256(output / name) for name in OUTPUTS},
                "prediction_integrity": integrity, "paired_correctness": paired,
                "mcnemar_exact_test": "not_performed", "training_performed": False,
                "fine_tuning_performed": False, "threshold_tuning": False,
                "checkpoint_selection_using_test": False, "ph2_accessed": False}
    manifest_path = output / "stage10_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") != "READY_FOR_COLAB_EVALUATION":
            raise FileExistsError("Existing Stage 10 manifest is not preflight handoff")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        write_json_new(manifest_path, manifest)
    print(json.dumps({"status": "COMPLETE", "metrics": metrics, "paired": paired, "integrity": integrity}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.project_root), indent=2))
    else:
        run(args.project_root.resolve(), args.num_workers, args.batch_size)


if __name__ == "__main__":
    main()
