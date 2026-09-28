"""Colab/Tesla-T4 runner for controlled experiment 1.

Training is refused unless the frozen split, baseline checkpoint, plan, and
CUDA Tesla T4 preflight all pass. This runner never reads PH2 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset import CLASS_NAMES, validate_split_integrity  # noqa: E402
from models.baseline_effnet import build_model  # noqa: E402
from train import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    IMAGE_SIZE,
    LEARNING_RATE,
    SEED,
    build_run_components,
    configuration,
    focal_loss,
    make_optimizer_and_scheduler,
    set_seed,
)

EXPECTED_CHECKPOINT_SHA256 = "f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848"
EXPECTED_SPLIT_SHA256 = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
EXPECTED_CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
SPLIT_NAME = "split_leakage_aware.csv"
EXPERIMENT_NAME = "efficientnetv2s_controlled_exp1"
PLAN_NAME = "experiment_plan.json"
GENERATED_OUTPUTS = (
    "best_checkpoint.pt",
    "training_history.json",
    "config.json",
    "validation_metrics.json",
    "test_metrics.json",
    "experiment_manifest.json",
    "validation_predictions.csv",
    "test_predictions.csv",
    "baseline_validation_metrics.json",
    "baseline_test_metrics.json",
    "baseline_validation_predictions.csv",
    "baseline_test_predictions.csv",
    "comparison_metrics.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def class_counts_from_split(split_csv: Path, class_names: Sequence[str]) -> dict[str, dict[str, int]]:
    counts = {part: Counter() for part in ("train", "val", "test")}
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "lesion_id", "dx", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Split CSV missing required columns: {sorted(required)}")
        for row in reader:
            part = row["split"]
            label = row["dx"]
            if part not in counts:
                raise ValueError(f"Unexpected split label: {part!r}")
            if label not in class_names:
                raise ValueError(f"Unexpected diagnosis label: {label!r}")
            counts[part][label] += 1
    return {
        part: {label: int(counts[part][label]) for label in class_names}
        for part in ("train", "val", "test")
    }


def compute_class_weights_from_train(
    split_csv: Path, class_names: Sequence[str] = EXPECTED_CLASS_NAMES
) -> Tensor:
    """Return mean-one inverse-frequency weights using TRAIN rows only."""
    if tuple(class_names) != EXPECTED_CLASS_NAMES:
        raise ValueError(f"Class order must be exactly {EXPECTED_CLASS_NAMES}")
    counts = Counter()
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"dx", "split"}.issubset(reader.fieldnames):
            raise ValueError("Split CSV must contain dx and split columns")
        for row in reader:
            if row["split"] == "train":
                if row["dx"] not in class_names:
                    raise ValueError(f"Unknown training diagnosis: {row['dx']!r}")
                counts[row["dx"]] += 1
    missing = [name for name in class_names if counts[name] <= 0]
    if missing:
        raise ValueError(f"Training partition has no samples for classes: {missing}")
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(class_names) * counts[name]) for name in class_names],
        dtype=torch.float32,
    )
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Computed class weights must be finite and positive")
    return weights


def class_balanced_focal_loss(
    logits: Tensor,
    targets: Tensor,
    class_weights: Tensor | None,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
) -> Tensor:
    """Apply a fixed target-class multiplier to the baseline focal objective."""
    if class_weights is None:
        return focal_loss(logits, targets, alpha=alpha, gamma=gamma)
    if class_weights.ndim != 1 or class_weights.numel() != len(EXPECTED_CLASS_NAMES):
        raise ValueError("class_weights must have one value in the frozen class order")
    if not torch.isfinite(class_weights).all() or not (class_weights > 0).all():
        raise ValueError("class_weights must be finite and positive")
    probabilities = torch.softmax(logits, dim=1)
    target_probability = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    cross_entropy = nn.functional.cross_entropy(logits, targets, reduction="none")
    target_weights = class_weights.to(device=targets.device, dtype=logits.dtype)[targets]
    per_sample = alpha * (1.0 - target_probability).pow(gamma) * cross_entropy
    return (target_weights * per_sample).mean()


def expected_training_configuration() -> dict:
    return {
        "model": "EfficientNetV2S",
        "pretrained_weight_source": "torchvision EfficientNet_V2_S_Weights.DEFAULT",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adamax",
        "learning_rate": LEARNING_RATE,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 1},
        "seed": SEED,
        "classes": list(EXPECTED_CLASS_NAMES),
        "split_csv": SPLIT_NAME,
        "checkpoint_selection_metric": "validation_loss_minimize",
        "class_weight_formula": "N_train / (K * n_train[class])",
    }


def preflight(project_root: Path = ROOT, require_t4: bool = False) -> dict:
    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")
    if tuple(CLASS_NAMES) != EXPECTED_CLASS_NAMES:
        raise RuntimeError(f"Source class order differs from plan: {CLASS_NAMES}")

    split_csv = project_root / "data" / "splits" / SPLIT_NAME
    checkpoint_path = (
        project_root / "experiments" / "efficientnetv2s_leakage_aware" / "best_checkpoint.pt"
    )
    baseline_config_path = (
        project_root / "experiments" / "efficientnetv2s_leakage_aware" / "config.json"
    )
    plan_path = project_root / "experiments" / EXPERIMENT_NAME / PLAN_NAME
    output_dir = plan_path.parent
    for path in (split_csv, checkpoint_path, baseline_config_path, plan_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment preflight file is missing: {path}")

    split_hash = sha256_file(split_csv)
    checkpoint_hash = sha256_file(checkpoint_path)
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"Frozen split SHA256 mismatch: {split_hash}")
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Baseline checkpoint SHA256 mismatch: {checkpoint_hash}")

    plan = read_json(plan_path)
    if plan.get("baseline_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Experiment plan baseline checkpoint hash does not match")
    if plan.get("split_reference", {}).get("sha256") != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Experiment plan split hash does not match")
    if plan.get("class_order") != list(EXPECTED_CLASS_NAMES):
        raise RuntimeError("Experiment plan class order does not match source")
    if plan.get("training_configuration") != {
        "model": "EfficientNetV2S",
        "pretrained_weight_source": "torchvision EfficientNet_V2_S_Weights.DEFAULT",
        "image_size": 384,
        "batch_size": 16,
        "epochs": 25,
        "optimizer": "Adamax",
        "learning_rate": 0.001,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "intervention_weight_formula": "N_train / (K * n_train[class])",
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 1},
        "mixed_precision": True,
        "device_requirement": "CUDA Tesla T4",
        "early_stopping": False,
    }:
        raise RuntimeError("Experiment plan training configuration is inconsistent")

    integrity = validate_split_integrity(split_csv, require_lesion_isolation=True)
    if integrity != {"rows": 10015, "unique_lesions": 7470, "crossing_lesions": 0}:
        raise RuntimeError(f"Frozen split integrity differs from baseline: {integrity}")
    class_counts = class_counts_from_split(split_csv, EXPECTED_CLASS_NAMES)
    if class_counts != plan.get("class_counts"):
        raise RuntimeError("Split class counts do not match the saved experiment plan")
    processed_images_dir = project_root / "data" / "processed" / "images"
    if not processed_images_dir.is_dir():
        raise FileNotFoundError(f"Processed HAM10000 image directory is missing: {processed_images_dir}")
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        image_ids = [row["image_id"] for row in csv.DictReader(handle)]
    missing_images = [
        image_id
        for image_id in image_ids
        if not (processed_images_dir / f"{image_id}.jpg").is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} frozen-split processed HAM images are missing; "
            f"examples: {missing_images[:5]}"
        )

    baseline_config = read_json(baseline_config_path)
    for key, expected in {
        "seed": SEED,
        "model": "EfficientNetV2S",
        "pretrained_weight_source": "torchvision EfficientNet_V2_S_Weights.DEFAULT",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adamax",
        "learning_rate": LEARNING_RATE,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 1},
        "split_csv": SPLIT_NAME,
        "classes": list(EXPECTED_CLASS_NAMES),
        "early_stopping": False,
    }.items():
        if baseline_config.get(key) != expected:
            raise RuntimeError(f"Baseline configuration mismatch for {key}")

    existing_outputs = [name for name in GENERATED_OUTPUTS if (output_dir / name).exists()]
    if existing_outputs:
        raise FileExistsError(f"Experiment would overwrite existing outputs: {existing_outputs}")

    gpu_name = None
    if require_t4:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; training is refused.")
        gpu_name = torch.cuda.get_device_name(0)
        if "Tesla T4" not in gpu_name:
            raise RuntimeError(f"Expected a Tesla T4, detected {gpu_name!r}; training is refused.")

    report = {
        "preflight": "PASS",
        "root": str(project_root),
        "split_path": str(split_csv),
        "split_sha256": split_hash,
        "split_integrity": integrity,
        "class_counts": class_counts,
        "processed_image_count": len(image_ids),
        "missing_processed_images": 0,
        "class_order": list(CLASS_NAMES),
        "baseline_checkpoint": str(checkpoint_path),
        "baseline_checkpoint_sha256": checkpoint_hash,
        "baseline_config_matches": True,
        "plan_matches_frozen_inputs": True,
        "output_directory": str(output_dir),
        "existing_training_outputs": existing_outputs,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name if gpu_name else (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "training_configuration": expected_training_configuration(),
    }
    return report


def binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(int(value) for value in labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[index][0]:
            stop += 1
        average_rank = ((index + 1) + stop) / 2.0
        positive_rank_sum += average_rank * sum(int(label) for _, label in ordered[index:stop])
        index = stop
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def metrics_from_predictions(
    targets: Sequence[int], probabilities: np.ndarray, class_names: Sequence[str]
) -> dict:
    prediction_indices = probabilities.argmax(axis=1)
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for target, predicted in zip(targets, prediction_indices):
        matrix[int(target), int(predicted)] += 1
    per_class = {}
    f1_values = []
    recalls = []
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "support": support,
            "predicted_count": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        recalls.append(recall)
        f1_values.append(f1)
    total = len(targets)
    correct = int(np.trace(matrix))
    melanoma_index = class_names.index("mel")
    melanoma_labels = [int(target == melanoma_index) for target in targets]
    nevus_index = class_names.index("nv")
    melanoma_nevus_indices = [
        index
        for index, target in enumerate(targets)
        if target in (melanoma_index, nevus_index)
    ]
    return {
        "sample_count": total,
        "class_order": list(class_names),
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "per_class": per_class,
        "melanoma_roc_auc_ovr": binary_roc_auc(
            melanoma_labels, probabilities[:, melanoma_index].tolist()
        ),
        "melanoma_vs_nevus_roc_auc": binary_roc_auc(
            [melanoma_labels[index] for index in melanoma_nevus_indices],
            probabilities[melanoma_nevus_indices, melanoma_index].tolist(),
        ),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    target_indices: list[int] = []
    probability_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
            probability_rows.extend(torch.softmax(logits.float(), dim=1).cpu().numpy())
            target_indices.extend(labels.tolist())
    probabilities = np.asarray(probability_rows, dtype=np.float64)
    metrics = metrics_from_predictions(target_indices, probabilities, EXPECTED_CLASS_NAMES)
    rows = []
    dataset_rows = loader.dataset.rows
    for row, target, probability in zip(dataset_rows, target_indices, probabilities):
        predicted = int(np.argmax(probability))
        rows.append(
            {
                "image_id": row["image_id"],
                "true_label": EXPECTED_CLASS_NAMES[target],
                "predicted_label": EXPECTED_CLASS_NAMES[predicted],
                "correct": target == predicted,
                "probabilities": [float(value) for value in probability],
            }
        )
    return metrics, rows


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_predictions(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image_id", "true_label", "predicted_label", "correct", "probabilities"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "probabilities": json.dumps(row["probabilities"], separators=(",", ":")),
                }
            )


def run_training(
    project_root: Path = ROOT,
    workers: int = 2,
    preflight_report: dict | None = None,
) -> dict:
    if preflight_report is None:
        preflight_report = preflight(project_root, require_t4=True)
    root = project_root.resolve()
    output_dir = root / "experiments" / EXPERIMENT_NAME
    split_csv = root / "data" / "splits" / SPLIT_NAME
    device = torch.device("cuda")
    set_seed(SEED)

    model, train_dataset, val_dataset, test_dataset = build_run_components(root, SPLIT_NAME)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True
    )
    class_weights = compute_class_weights_from_train(split_csv).to(device)
    model.to(device)
    optimizer, scheduler = make_optimizer_and_scheduler(model)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    baseline_config = read_json(
        root / "experiments" / "efficientnetv2s_leakage_aware" / "config.json"
    )
    experiment_config = {
        **baseline_config,
        "software": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
        },
        "experiment": EXPERIMENT_NAME,
        "intervention": {
            "loss": "class-balanced focal loss",
            "class_weight_formula": "N_train / (K * n_train[class])",
            "weight_source_split": "train",
            "class_weights": {
                name: float(class_weights[index].item())
                for index, name in enumerate(EXPECTED_CLASS_NAMES)
            },
        },
        "checkpoint_selection_metric": "validation_loss_minimize",
        "ph2_accessed": False,
    }
    write_json(output_dir / "config.json", experiment_config)

    history = []
    best_val_loss = math.inf
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        train_matrix = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(images)
                loss = class_balanced_focal_loss(logits, labels, class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            predictions = logits.argmax(dim=1)
            for target, predicted in zip(labels.detach().cpu(), predictions.detach().cpu()):
                train_matrix[int(target), int(predicted)] += 1
            count = labels.size(0)
            total_loss += float(loss.detach().float().cpu()) * count
            total_items += count
        train_metrics = metrics_from_predictions_from_matrix(train_matrix)

        val_loss_sum = 0.0
        val_items = 0
        val_matrix = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
        model.eval()
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=True):
                    logits = model(images)
                    loss = class_balanced_focal_loss(logits, labels, class_weights)
                predictions = logits.argmax(dim=1)
                for target, predicted in zip(labels.cpu(), predictions.cpu()):
                    val_matrix[int(target), int(predicted)] += 1
                count = labels.size(0)
                val_loss_sum += float(loss.float().cpu()) * count
                val_items += count
        val_loss = val_loss_sum / val_items
        val_metrics = metrics_from_predictions_from_matrix(val_matrix)
        scheduler.step(val_loss)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / total_items,
            "train": train_metrics,
            "val_loss": val_loss,
            "validation": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "class_weights": experiment_config["intervention"]["class_weights"],
        }
        history.append(record)
        write_json(output_dir / "training_history.json", {"epochs": history})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": experiment_config,
                    "class_order": list(EXPECTED_CLASS_NAMES),
                    "class_weights": experiment_config["intervention"]["class_weights"],
                },
                best_checkpoint_path,
            )

    checkpoint_hash = sha256_file(best_checkpoint_path)
    checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    selected_epoch = int(checkpoint["epoch"])

    validation_metrics, validation_rows = evaluate_model(model, val_loader, device)
    test_metrics, test_rows = evaluate_model(model, test_loader, device)
    validation_metrics["loss"] = float(checkpoint["val_loss"])
    validation_metrics["selected_epoch"] = selected_epoch
    validation_metrics["checkpoint_selection_metric"] = "validation_loss_minimize"
    test_metrics["selected_epoch"] = selected_epoch
    test_metrics["checkpoint_sha256"] = checkpoint_hash
    write_json(output_dir / "validation_metrics.json", validation_metrics)
    write_json(output_dir / "test_metrics.json", test_metrics)
    write_predictions(output_dir / "validation_predictions.csv", validation_rows)
    write_predictions(output_dir / "test_predictions.csv", test_rows)

    baseline_path = (
        root / "experiments" / "efficientnetv2s_leakage_aware" / "best_checkpoint.pt"
    )
    baseline_checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=True)
    baseline_model, _ = build_model(pretrained=False)
    baseline_model.load_state_dict(baseline_checkpoint["model_state"], strict=True)
    baseline_model.to(device).eval()
    base_val_metrics, base_val_rows = evaluate_model(baseline_model, val_loader, device)
    base_test_metrics, base_test_rows = evaluate_model(baseline_model, test_loader, device)
    base_val_metrics["loss"] = float(baseline_checkpoint["val_loss"])
    base_val_metrics["checkpoint_sha256"] = EXPECTED_CHECKPOINT_SHA256
    base_test_metrics["checkpoint_sha256"] = EXPECTED_CHECKPOINT_SHA256
    write_json(output_dir / "baseline_validation_metrics.json", base_val_metrics)
    write_json(output_dir / "baseline_test_metrics.json", base_test_metrics)
    write_predictions(output_dir / "baseline_validation_predictions.csv", base_val_rows)
    write_predictions(output_dir / "baseline_test_predictions.csv", base_test_rows)

    criteria = read_json(root / "experiments" / EXPERIMENT_NAME / PLAN_NAME)["success_criteria"]
    mel_f1 = validation_metrics["per_class"]["mel"]["f1"]
    macro_f1 = validation_metrics["macro_f1"]
    nv_recall = validation_metrics["per_class"]["nv"]["recall"]
    success = (
        mel_f1 >= criteria["melanoma_f1"]["minimum"]
        and macro_f1 >= criteria["validation_macro_f1"]["minimum"]
        and nv_recall >= criteria["nevus_recall"]["minimum"]
    )
    comparison = {
        "checkpoint_sha256": checkpoint_hash,
        "selected_epoch": selected_epoch,
        "selection_metric": "validation_loss_minimize",
        "validation_success_criteria_met": success,
        "validation_values": {
            "melanoma_f1": mel_f1,
            "validation_macro_f1": macro_f1,
            "nevus_recall": nv_recall,
        },
        "success_thresholds": {
            "melanoma_f1_minimum": criteria["melanoma_f1"]["minimum"],
            "validation_macro_f1_minimum": criteria["validation_macro_f1"]["minimum"],
            "nevus_recall_minimum": criteria["nevus_recall"]["minimum"],
        },
        "baseline": {
            "validation": base_val_metrics,
            "test": base_test_metrics,
        },
        "controlled_experiment": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
    }
    write_json(output_dir / "comparison_metrics.json", comparison)
    manifest = {
        "status": "COMPLETE",
        "started_runtime": preflight_report,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "new_checkpoint_sha256": checkpoint_hash,
        "selected_epoch": selected_epoch,
        "selection_metric": "validation_loss_minimize",
        "selection_value": float(checkpoint["val_loss"]),
        "validation_success_criteria_met": success,
        "class_weights": experiment_config["intervention"]["class_weights"],
        "ph2_accessed": False,
        "test_evaluation_occurred_after_checkpoint_freeze": True,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
        },
    }
    write_json(output_dir / "experiment_manifest.json", manifest)
    return manifest


def metrics_from_predictions_from_matrix(matrix: Tensor) -> dict:
    matrix_array = matrix.detach().cpu().numpy()
    total = int(matrix_array.sum())
    per_class = {}
    f1_values = []
    recalls = []
    for index, name in enumerate(EXPECTED_CLASS_NAMES):
        tp = int(matrix_array[index, index])
        support = int(matrix_array[index, :].sum())
        predicted_count = int(matrix_array[:, index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "support": support,
            "predicted_count": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        recalls.append(recall)
        f1_values.append(f1)
    return {
        "sample_count": total,
        "class_order": list(EXPECTED_CLASS_NAMES),
        "accuracy": float(np.trace(matrix_array) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix_array.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run CPU/read-only integrity checks without requiring CUDA or training.",
    )
    args = parser.parse_args()
    report = preflight(args.project_root, require_t4=False)
    print(json.dumps(report, indent=2))
    if args.preflight_only:
        return 0
    t4_report = preflight(args.project_root, require_t4=True)
    print(json.dumps(t4_report, indent=2))
    result = run_training(args.project_root, workers=args.workers, preflight_report=t4_report)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
