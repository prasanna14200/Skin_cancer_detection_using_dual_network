"""Run or finalize Controlled Experiment #2 using frozen HAM10000 artifacts.

No training occurs unless --train is explicitly supplied. --finalize-existing
uses saved predictions and metrics only; it never constructs or evaluates a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset import CLASS_NAMES, validate_split_integrity  # noqa: E402
from train import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    IMAGE_SIZE,
    LEARNING_RATE,
    SEED,
    build_run_components,
    focal_loss,
    make_optimizer_and_scheduler,
    set_seed,
)

EXPERIMENT_NAME = "efficientnetv2s_controlled_exp2"
SPLIT_NAME = "split_leakage_aware.csv"
CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
BASELINE_SHA256 = "f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848"
SPLIT_SHA256 = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
EXPECTED_TRAIN_COUNTS = {
    "akiec": 257,
    "bcc": 398,
    "bkl": 891,
    "df": 95,
    "mel": 899,
    "nv": 5366,
    "vasc": 109,
}
FROZEN_CRITERIA = {
    "melanoma_f1_minimum": 0.5757731958762886,
    "validation_macro_f1_minimum": 0.6316582381362074,
    "nevus_recall_minimum": 0.9,
}
OUTPUT_NAMES = (
    "config.json",
    "training_history.json",
    "best_checkpoint.pt",
    "validation_metrics.json",
    "validation_predictions.csv",
    "test_metrics.json",
    "test_predictions.csv",
    "comparison_metrics.json",
    "experiment_manifest.json",
    "controlled_exp2_final_report.md",
)
MEL_WEIGHT = math.sqrt(EXPECTED_TRAIN_COUNTS["nv"] / EXPECTED_TRAIN_COUNTS["mel"])


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def json_text(value: dict) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path: Path, content: str, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if not replace and path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
            return
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict, *, replace: bool = True) -> None:
    write_atomic(path, json_text(value), replace=replace)


def class_counts_from_split(split_csv: Path) -> dict[str, dict[str, int]]:
    counts = {part: Counter() for part in ("train", "val", "test")}
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "lesion_id", "dx", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Frozen split is missing required columns: {sorted(required)}")
        for row in reader:
            if row["split"] not in counts or row["dx"] not in CLASS_ORDER:
                raise ValueError(f"Unexpected split or diagnosis in frozen split: {row}")
            counts[row["split"]][row["dx"]] += 1
    return {
        split: {name: int(counts[split][name]) for name in CLASS_ORDER}
        for split in ("train", "val", "test")
    }


def metrics_from_matrix(matrix: Sequence[Sequence[int]]) -> dict:
    matrix = [[int(value) for value in row] for row in matrix]
    if len(matrix) != len(CLASS_ORDER) or any(len(row) != len(CLASS_ORDER) for row in matrix):
        raise ValueError("Confusion matrix shape does not match the frozen seven-class order")
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        raise ValueError("Cannot calculate metrics for an empty prediction set")
    per_class = {}
    f1_values = []
    recalls = []
    correct = 0
    for index, name in enumerate(CLASS_ORDER):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
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
        correct += tp
    return {
        "sample_count": total,
        "class_order": list(CLASS_ORDER),
        "accuracy": correct / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def predictions_from_csv(path: Path) -> tuple[dict, list[dict]]:
    matrix = [[0 for _ in CLASS_ORDER] for _ in CLASS_ORDER]
    rows = []
    seen_ids = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "true_label", "predicted_label", "correct"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Prediction file lacks required columns: {path}")
        for row in reader:
            image_id = row["image_id"]
            if not image_id or image_id in seen_ids:
                raise ValueError(f"Missing or duplicate image_id in {path}: {image_id!r}")
            seen_ids.add(image_id)
            try:
                target = CLASS_ORDER.index(row["true_label"])
                predicted = CLASS_ORDER.index(row["predicted_label"])
            except ValueError as exc:
                raise ValueError(f"Unknown class in {path}: {exc}") from exc
            actual_correct = target == predicted
            if row["correct"].strip().lower() not in {"true", "false"}:
                raise ValueError(f"Invalid correct field in {path}: {row['correct']!r}")
            if (row["correct"].strip().lower() == "true") != actual_correct:
                raise ValueError(f"Correct flag disagrees with labels in {path}, image {image_id}")
            matrix[target][predicted] += 1
            rows.append(row)
    return metrics_from_matrix(matrix), rows


def assert_metric_record(saved: dict, derived: dict, context: str) -> None:
    if saved.get("class_order") not in (None, list(CLASS_ORDER)):
        raise ValueError(f"Class order mismatch in {context}")
    if saved.get("sample_count") is not None and int(saved["sample_count"]) != derived["sample_count"]:
        raise ValueError(f"Sample count mismatch in {context}")
    if saved.get("confusion_matrix") != derived["confusion_matrix"]:
        raise ValueError(f"Confusion matrix mismatch in {context}")
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        if metric in saved and not math.isclose(
            float(saved[metric]), float(derived[metric]), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{context}.{metric} disagrees with saved predictions")
    for label, values in derived["per_class"].items():
        for metric in ("precision", "recall", "f1"):
            saved_value = saved.get("per_class", {}).get(label, {}).get(metric)
            if saved_value is not None and not math.isclose(
                float(saved_value), float(values[metric]), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"{context}.{label}.{metric} disagrees with saved predictions")


def read_split_identity(split_csv: Path, part: str) -> dict[str, str]:
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["image_id"]: row["dx"]
            for row in csv.DictReader(handle)
            if row["split"] == part
        }


def verify_prediction_identity(rows: Sequence[dict], split_rows: dict[str, str], context: str) -> None:
    actual = {row["image_id"]: row["true_label"] for row in rows}
    if actual != split_rows:
        missing = sorted(set(split_rows) - set(actual))[:3]
        extra = sorted(set(actual) - set(split_rows))[:3]
        wrong = sorted(key for key in set(actual) & set(split_rows) if actual[key] != split_rows[key])[:3]
        raise ValueError(
            f"{context} IDs/labels differ from frozen split; missing={missing}, extra={extra}, wrong={wrong}"
        )


def safe_load_checkpoint(path: Path) -> dict:
    version_type = getattr(torch.torch_version, "TorchVersion", None)
    if version_type is not None:
        torch.serialization.add_safe_globals([version_type])
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint does not contain the expected dictionary: {path}")
    return value


def validate_plan(plan: dict) -> None:
    if plan.get("status") != "PLANNED_NOT_RUN":
        raise ValueError("Experiment plan status must remain PLANNED_NOT_RUN")
    if plan.get("baseline_reference", {}).get("checkpoint_sha256") != BASELINE_SHA256:
        raise ValueError("Plan baseline checkpoint hash differs from the verified baseline")
    frozen = plan.get("frozen_inputs", {})
    if frozen.get("split_sha256") != SPLIT_SHA256 or frozen.get("class_order") != list(CLASS_ORDER):
        raise ValueError("Plan split hash or class order differs from frozen values")
    if frozen.get("train_class_counts") != EXPECTED_TRAIN_COUNTS:
        raise ValueError("Plan frozen training counts are incorrect")
    intervention = plan.get("intervention", {})
    if intervention.get("sampler") != "torch.utils.data.WeightedRandomSampler":
        raise ValueError("Plan sampler type changed")
    if intervention.get("sampler_seed") != SEED or intervention.get("replacement") is not True:
        raise ValueError("Sampler seed/replacement differ from the approved protocol")
    if intervention.get("num_samples_per_epoch") != sum(EXPECTED_TRAIN_COUNTS.values()):
        raise ValueError("Sampler num_samples must equal original train partition size")
    if not math.isclose(
        float(intervention.get("melanoma_sampling_weight", -1)), MEL_WEIGHT, rel_tol=1e-14
    ) or intervention.get("non_melanoma_sampling_weight") != 1.0:
        raise ValueError("Sampler weights do not match the frozen formula")
    if intervention.get("weight_formula") != (
        "sqrt(n_train[nv] / n_train[mel]) for mel rows; 1.0 for every non-mel row"
    ):
        raise ValueError("Sampler formula differs from the approved protocol")
    if intervention.get("train_counts_used") != {"nv": 5366, "mel": 899}:
        raise ValueError("Sampler counts must come only from the frozen training split")
    if intervention.get("class_balanced_loss_or_other_loss_weighting") is not False:
        raise ValueError("Experiment #2 must retain baseline loss without class-specific weighting")
    if plan.get("success_criteria") != {
        "all_required": True,
        "melanoma_f1": {"minimum": FROZEN_CRITERIA["melanoma_f1_minimum"]},
        "validation_macro_f1": {"minimum": FROZEN_CRITERIA["validation_macro_f1_minimum"]},
        "nevus_recall": {"minimum": FROZEN_CRITERIA["nevus_recall_minimum"]},
        "evaluation_partition": "HAM10000 validation only",
        "must_not_be_changed_after_results": True,
    }:
        raise ValueError("Frozen success criteria were changed")
    if plan.get("ph2_policy", {}).get("allowed_during_development") is not False:
        raise ValueError("Plan must prohibit PH2 use during development")
    frozen_config = plan.get("frozen_configuration", {})
    expected_config = {
        "model": "EfficientNetV2S",
        "pretrained_weight_source": "torchvision EfficientNet_V2_S_Weights.DEFAULT",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adamax",
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "mixed_precision": True,
        "evaluation_prediction": "argmax",
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 1,
        },
        "training_augmentation": [
            "Resize(384, 384)",
            "RandomHorizontalFlip",
            "RandomVerticalFlip",
            "RandomRotation(15)",
            "ToTensor",
            "ImageNet normalization",
        ],
        "validation_test_preprocessing": [
            "Resize(384, 384)",
            "ToTensor",
            "ImageNet normalization",
        ],
        "checkpoint_selection": {
            "metric": "validation_loss",
            "direction": "minimize",
            "data": "frozen HAM10000 validation split only",
        },
    }
    mismatches = {
        key: (frozen_config.get(key), value)
        for key, value in expected_config.items()
        if frozen_config.get(key) != value
    }
    focal = frozen_config.get("baseline_focal_loss", {})
    if focal != {
        "name": "focal_loss",
        "alpha": FOCAL_ALPHA,
        "gamma": FOCAL_GAMMA,
        "class_specific_loss_weights": False,
    }:
        mismatches["baseline_focal_loss"] = (focal, "original baseline focal loss")
    if mismatches:
        raise ValueError(f"Plan frozen configuration differs from baseline: {mismatches}")


def preflight(
    project_root: Path,
    *,
    require_t4: bool = False,
    require_clean_outputs: bool = False,
) -> dict:
    root = project_root.resolve()
    output_dir = root / "experiments" / EXPERIMENT_NAME
    baseline_dir = root / "experiments" / "efficientnetv2s_leakage_aware"
    split_csv = root / "data" / "splits" / SPLIT_NAME
    report = {
        "project_root": str(root),
        "experiment_directory": str(output_dir),
        "training_requested": require_t4,
        "checks": {},
        "errors": [],
    }

    def record(name: str, passed: bool, details: str) -> None:
        report["checks"][name] = {"status": "PASS" if passed else "FAIL", "details": details}
        if not passed:
            report["errors"].append(f"{name}: {details}")

    record("project_root", root.is_dir(), "project root exists")
    record("experiment_directory", output_dir.is_dir(), "dedicated Experiment #2 directory exists")
    record(
        "runner_location",
        output_dir.resolve() == Path(__file__).resolve().parent,
        "runner is being used from this project's Experiment #2 directory",
    )
    if not root.is_dir() or not output_dir.is_dir():
        report["status"] = "FAIL"
        return report

    try:
        plan = read_json(output_dir / "experiment_plan.json")
        validate_plan(plan)
        record("experiment_plan", True, "plan matches approved frozen values")
    except Exception as exc:
        plan = None
        record("experiment_plan", False, str(exc))

    missing_baseline = [
        name
        for name in ("best_checkpoint.pt", "config.json", "training_history.json")
        if not (baseline_dir / name).is_file()
    ]
    record("baseline_artifacts", not missing_baseline, f"missing={missing_baseline}")
    record("frozen_split", split_csv.is_file(), f"path={split_csv}")
    if split_csv.is_file():
        try:
            split_hash = sha256_file(split_csv)
            integrity = validate_split_integrity(split_csv, require_lesion_isolation=True)
            counts = class_counts_from_split(split_csv)
            ok = (
                split_hash == SPLIT_SHA256
                and integrity == {"rows": 10015, "unique_lesions": 7470, "crossing_lesions": 0}
                and counts["train"] == EXPECTED_TRAIN_COUNTS
            )
            if plan is not None:
                plan_inputs = plan["frozen_inputs"]
                ok = ok and counts["val"] == plan_inputs["validation_class_counts"]
                ok = ok and counts["test"] == plan_inputs["test_class_counts"]
            record(
                "split_identity_class_order_and_counts",
                ok and tuple(CLASS_NAMES) == CLASS_ORDER,
                f"sha256={split_hash}; integrity={integrity}; counts={counts}",
            )
        except Exception as exc:
            counts = None
            record("split_identity_class_order_and_counts", False, str(exc))
    else:
        counts = None
        record("split_identity_class_order_and_counts", False, "split file unavailable")

    baseline_hash = None
    checkpoint_path = baseline_dir / "best_checkpoint.pt"
    if checkpoint_path.is_file():
        baseline_hash = sha256_file(checkpoint_path)
    record(
        "baseline_checkpoint_hash",
        baseline_hash == BASELINE_SHA256,
        f"expected={BASELINE_SHA256}; actual={baseline_hash}",
    )
    if baseline_hash == BASELINE_SHA256:
        try:
            checkpoint = safe_load_checkpoint(checkpoint_path)
            baseline_history = read_json(baseline_dir / "training_history.json").get("epochs", [])
            selected = min(baseline_history, key=lambda item: float(item["val_loss"]))
            ok = int(checkpoint.get("epoch", -1)) == 6 and int(selected["epoch"]) == 6
            record(
                "baseline_selection_evidence",
                ok,
                f"checkpoint_epoch={checkpoint.get('epoch')}; minimum_validation_loss_epoch={selected['epoch']}",
            )
        except Exception as exc:
            record("baseline_selection_evidence", False, str(exc))

    baseline_config_path = baseline_dir / "config.json"
    if baseline_config_path.is_file():
        try:
            baseline_config = read_json(baseline_config_path)
            expected = {
                "model": "EfficientNetV2S",
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "optimizer": "Adamax",
                "learning_rate": LEARNING_RATE,
                "focal_alpha": FOCAL_ALPHA,
                "focal_gamma": FOCAL_GAMMA,
                "seed": SEED,
                "split_csv": SPLIT_NAME,
                "classes": list(CLASS_ORDER),
            }
            mismatches = {
                key: (baseline_config.get(key), value)
                for key, value in expected.items()
                if baseline_config.get(key) != value
            }
            record("baseline_frozen_configuration", not mismatches, f"mismatches={mismatches}")
        except Exception as exc:
            record("baseline_frozen_configuration", False, str(exc))

    sampler_ok = (
        counts is not None
        and counts["train"]["nv"] == 5366
        and counts["train"]["mel"] == 899
        and sum(counts["train"].values()) == 8015
        and math.isclose(MEL_WEIGHT, math.sqrt(5366 / 899), rel_tol=1e-14)
    )
    record(
        "sampler_configuration",
        sampler_ok,
        f"TRAIN-only labels; mel={MEL_WEIGHT:.14f}; non-mel=1.0; seed={SEED}; replacement=True; num_samples=8015",
    )

    try:
        uncertainty_dir = root / "experiments" / "uncertainty"
        summary = read_json(uncertainty_dir / "uncertainty_summary.json")
        baseline_test, baseline_test_rows = predictions_from_csv(
            uncertainty_dir / "predictions.csv"
        )
        test_identity = read_split_identity(split_csv, "test")
        verify_prediction_identity(baseline_test_rows, test_identity, "baseline checkpoint-matched test")
        matched = (
            summary.get("checkpoint_sha256") == BASELINE_SHA256
            and int(summary.get("test_samples", -1)) == baseline_test["sample_count"]
            and summary.get("confusion_matrix") == baseline_test["confusion_matrix"]
        )
        record(
            "checkpoint_matched_baseline_test_evidence",
            matched,
            f"summary_checkpoint={summary.get('checkpoint_sha256')}; samples={baseline_test['sample_count']}; test IDs/labels match split",
        )
    except Exception as exc:
        record("checkpoint_matched_baseline_test_evidence", False, str(exc))

    images_dir = root / "data" / "processed" / "images"
    if split_csv.is_file():
        try:
            with split_csv.open("r", newline="", encoding="utf-8") as handle:
                image_ids = [row["image_id"] for row in csv.DictReader(handle)]
            missing_images = [
                image_id for image_id in image_ids if not (images_dir / f"{image_id}.jpg").is_file()
            ]
            record(
                "processed_ham_images",
                images_dir.is_dir() and not missing_images,
                f"missing_count={len(missing_images)}; examples={missing_images[:5]}",
            )
        except Exception as exc:
            record("processed_ham_images", False, str(exc))
    else:
        record("processed_ham_images", False, "cannot validate images without frozen split")

    if require_clean_outputs:
        collisions = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
        record("output_overwrite_protection", not collisions, f"existing_generated_outputs={collisions}")
    else:
        report["checks"]["output_overwrite_protection"] = {
            "status": "NOT_REQUIRED",
            "details": "training was not requested",
        }

    if require_t4:
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        colab_available = importlib.util.find_spec("google.colab") is not None
        gpu_ok = (
            platform.system() == "Linux"
            and colab_available
            and cuda_available
            and gpu_name is not None
            and "Tesla T4" in gpu_name
        )
        record(
            "approved_colab_tesla_t4",
            gpu_ok,
            f"platform={platform.system()}; google.colab={colab_available}; cuda={cuda_available}; gpu={gpu_name}",
        )
    else:
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        report["checks"]["approved_colab_tesla_t4"] = {
            "status": "NOT_REQUIRED",
            "details": f"preflight-only; CUDA={cuda_available}; device={gpu_name}",
        }

    report["status"] = "PASS" if not report["errors"] else "FAIL"
    report["training_eligible"] = report["status"] == "PASS" and require_t4
    report["ph2_accessed"] = False
    return report


def weight_for_training_label(label: str) -> float:
    if label not in CLASS_ORDER:
        raise ValueError(f"Unexpected training label: {label}")
    return MEL_WEIGHT if label == "mel" else 1.0


def build_training_sampler(dataset) -> WeightedRandomSampler:
    labels = [row["dx"] for row in dataset.rows]
    if len(labels) != 8015:
        raise ValueError(f"Sampler dataset size must remain 8015, found {len(labels)}")
    sampled_counts = Counter(labels)
    if dict(sampled_counts) != EXPECTED_TRAIN_COUNTS:
        raise ValueError(f"Sampler labels/counts differ from frozen TRAIN partition: {sampled_counts}")
    sample_weights = torch.as_tensor(
        [weight_for_training_label(label) for label in labels], dtype=torch.double
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    if sampler.num_samples != len(dataset) or not sampler.replacement:
        raise ValueError("Weighted sampler configuration differs from frozen plan")
    return sampler


def metrics_from_confusion_tensor(matrix: Tensor) -> dict:
    return metrics_from_matrix(matrix.detach().cpu().tolist())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    training: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, dict]:
    model.train(training)
    matrix = torch.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=torch.long)
    loss_sum = 0.0
    item_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            if optimizer is None or scaler is None:
                raise ValueError("Training requires optimizer and gradient scaler")
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = focal_loss(logits, labels, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        predictions = logits.argmax(dim=1)
        for target, predicted in zip(labels.detach().cpu(), predictions.detach().cpu()):
            matrix[int(target), int(predicted)] += 1
        count = labels.size(0)
        loss_sum += float(loss.detach().float().cpu()) * count
        item_count += count
    if item_count == 0:
        raise ValueError("Encountered an empty data loader")
    return loss_sum / item_count, metrics_from_confusion_tensor(matrix)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    matrix = torch.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=torch.long)
    probability_rows: list[list[float]] = []
    target_indices: list[int] = []
    loss_sum = 0.0
    item_count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = focal_loss(logits, labels.to(device), alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            probability_rows.extend(probabilities.tolist())
            target_indices.extend(labels.tolist())
            for target, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[int(target), int(predicted)] += 1
            count = labels.size(0)
            loss_sum += float(loss.detach().float().cpu()) * count
            item_count += count
    if item_count == 0:
        raise ValueError("Cannot evaluate an empty data loader")
    metrics = metrics_from_confusion_tensor(matrix)
    metrics["loss"] = loss_sum / item_count
    rows = []
    for dataset_row, target, probabilities in zip(loader.dataset.rows, target_indices, probability_rows):
        predicted = int(np.argmax(probabilities))
        rows.append(
            {
                "image_id": dataset_row["image_id"],
                "true_label": CLASS_ORDER[target],
                "predicted_label": CLASS_ORDER[predicted],
                "correct": target == predicted,
                "probabilities": probabilities,
            }
        )
    return metrics, rows


def write_predictions(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = ("image_id", "true_label", "predicted_label", "correct", "probabilities")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "probabilities": json.dumps(row["probabilities"], separators=(",", ":")),
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite prediction artifact: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_training(project_root: Path, preflight_report: dict, workers: int = 2) -> None:
    root = project_root.resolve()
    output_dir = root / "experiments" / EXPERIMENT_NAME
    device = torch.device("cuda")
    set_seed(SEED)

    model, train_dataset, val_dataset, test_dataset = build_run_components(root, SPLIT_NAME)
    sampler = build_training_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True
    )
    model.to(device)
    optimizer, scheduler = make_optimizer_and_scheduler(model)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    experiment_config = {
        "experiment": EXPERIMENT_NAME,
        "model": "EfficientNetV2S",
        "pretrained_weight_source": "torchvision EfficientNet_V2_S_Weights.DEFAULT",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adamax",
        "learning_rate": LEARNING_RATE,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "loss": "original baseline focal_loss; no class-specific loss weighting",
        "scheduler": {"name": "ReduceLROnPlateau", "mode": "min", "factor": 0.5, "patience": 1},
        "seed": SEED,
        "sampler": {
            "name": "WeightedRandomSampler",
            "weight_formula": "sqrt(n_train[nv] / n_train[mel]) for mel rows; 1.0 otherwise",
            "train_counts": EXPECTED_TRAIN_COUNTS,
            "mel_weight": MEL_WEIGHT,
            "non_melanoma_weight": 1.0,
            "seed": SEED,
            "replacement": True,
            "num_samples": len(train_dataset),
            "generator": "torch.Generator(device='cpu').manual_seed(42)",
            "label_source": "frozen HAM10000 TRAIN rows only",
        },
        "split_csv": SPLIT_NAME,
        "split_sha256": SPLIT_SHA256,
        "classes": list(CLASS_ORDER),
        "checkpoint_selection_metric": "validation_loss_minimize",
        "success_criteria": FROZEN_CRITERIA,
        "preflight": preflight_report,
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "ph2_accessed": False,
    }
    write_json(output_dir / "config.json", experiment_config, replace=False)

    history = []
    best_validation_loss = math.inf
    checkpoint_path = output_dir / "best_checkpoint.pt"
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            device,
            training=True,
            optimizer=optimizer,
            scaler=scaler,
        )
        validation_loss, validation_metrics = run_epoch(
            model, val_loader, device, training=False
        )
        scheduler.step(validation_loss)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train": train_metrics,
            "val_loss": validation_loss,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "sampled_train_label_counts": {
                label: train_metrics["per_class"][label]["support"] for label in CLASS_ORDER
            },
            "sampler_seed": SEED,
            "sampler_num_samples": len(train_dataset),
        }
        history.append(record)
        write_json(output_dir / "training_history.json", {"epochs": history}, replace=True)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                checkpoint_path,
                {
                    "model_state": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "val_loss": validation_loss,
                    "config": experiment_config,
                    "class_order": list(CLASS_ORDER),
                },
            )
        print(
            f"epoch={epoch:02d}/{EPOCHS} train_loss={train_loss:.6f} "
            f"val_loss={validation_loss:.6f} "
            f"val_mel_f1={validation_metrics['per_class']['mel']['f1']:.6f}"
        )

    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = safe_load_checkpoint(checkpoint_path)
    selected_epoch = int(checkpoint["epoch"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    validation_metrics, validation_rows = evaluate_model(model, val_loader, device)
    test_metrics, test_rows = evaluate_model(model, test_loader, device)
    for metrics in (validation_metrics, test_metrics):
        metrics["selected_epoch"] = selected_epoch
        metrics["checkpoint_sha256"] = checkpoint_hash
        metrics["checkpoint_selection_metric"] = "validation_loss_minimize"
    validation_metrics["loss"] = float(checkpoint["val_loss"])
    validation_metrics["loss_source"] = "minimum validation loss recorded in selected checkpoint"
    test_metrics["evaluation_partition"] = "frozen HAM10000 test; evaluated after checkpoint freeze"

    write_json(output_dir / "validation_metrics.json", validation_metrics, replace=False)
    write_predictions(output_dir / "validation_predictions.csv", validation_rows)
    write_json(output_dir / "test_metrics.json", test_metrics, replace=False)
    write_predictions(output_dir / "test_predictions.csv", test_rows)
    experiment_config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "config.json", experiment_config, replace=True)
    finalize_existing(root)


def finalization_outputs(root: Path) -> dict[str, str]:
    output_dir = root / "experiments" / EXPERIMENT_NAME
    split_csv = root / "data" / "splits" / SPLIT_NAME
    plan = read_json(output_dir / "experiment_plan.json")
    validate_plan(plan)
    config = read_json(output_dir / "config.json")
    history = read_json(output_dir / "training_history.json").get("epochs", [])
    if len(history) != EPOCHS:
        raise ValueError(f"Expected {EPOCHS} saved epochs, found {len(history)}")
    best_epoch = min(history, key=lambda item: float(item["val_loss"]))

    checkpoint_path = output_dir / "best_checkpoint.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    validation_saved = read_json(output_dir / "validation_metrics.json")
    test_saved = read_json(output_dir / "test_metrics.json")
    if checkpoint_hash != validation_saved.get("checkpoint_sha256"):
        raise ValueError("Selected checkpoint hash does not match saved validation metrics")
    if test_saved.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Selected checkpoint hash does not match saved test metrics")
    checkpoint = safe_load_checkpoint(checkpoint_path)
    selected_epoch = int(checkpoint.get("epoch", -1))
    if selected_epoch != int(best_epoch["epoch"]):
        raise ValueError("Checkpoint epoch disagrees with minimum-validation-loss history epoch")
    if not math.isclose(float(checkpoint["val_loss"]), float(best_epoch["val_loss"]), rel_tol=1e-12):
        raise ValueError("Checkpoint validation loss disagrees with its history entry")
    if checkpoint.get("class_order") != list(CLASS_ORDER):
        raise ValueError("Checkpoint class order differs from frozen class order")

    validation_derived, validation_rows = predictions_from_csv(
        output_dir / "validation_predictions.csv"
    )
    test_derived, test_rows = predictions_from_csv(output_dir / "test_predictions.csv")
    assert_metric_record(validation_saved, validation_derived, "Experiment #2 validation")
    assert_metric_record(test_saved, test_derived, "Experiment #2 test")
    if int(validation_saved.get("selected_epoch", -1)) != selected_epoch:
        raise ValueError("Validation metrics refer to the wrong checkpoint epoch")
    if int(test_saved.get("selected_epoch", -1)) != selected_epoch:
        raise ValueError("Test metrics refer to the wrong checkpoint epoch")
    verify_prediction_identity(
        validation_rows, read_split_identity(split_csv, "val"), "Experiment #2 validation"
    )
    verify_prediction_identity(test_rows, read_split_identity(split_csv, "test"), "Experiment #2 test")
    if int(validation_saved.get("sample_count", -1)) != 986 or int(test_saved.get("sample_count", -1)) != 1014:
        raise ValueError("Saved prediction sample counts differ from frozen partitions")
    if best_epoch["validation"].get("confusion_matrix") != validation_derived["confusion_matrix"]:
        raise ValueError("Selected epoch history confusion matrix differs from saved validation predictions")

    baseline_dir = root / "experiments" / "efficientnetv2s_leakage_aware"
    baseline_hash = sha256_file(baseline_dir / "best_checkpoint.pt")
    if baseline_hash != BASELINE_SHA256:
        raise ValueError(f"Baseline checkpoint hash mismatch: {baseline_hash}")
    baseline_history = read_json(baseline_dir / "training_history.json").get("epochs", [])
    baseline_best_epoch = min(baseline_history, key=lambda item: float(item["val_loss"]))
    if int(baseline_best_epoch["epoch"]) != 6:
        raise ValueError("Baseline minimum-validation-loss epoch is no longer epoch 6")
    baseline_validation = metrics_from_matrix(
        baseline_best_epoch["validation"]["confusion_matrix"]
    )
    baseline_validation.update(
        {
            "loss": float(baseline_best_epoch["val_loss"]),
            "selected_epoch": 6,
            "checkpoint_sha256": baseline_hash,
            "source": "epoch-6 validation confusion matrix in frozen baseline training_history.json",
        }
    )

    uncertainty_dir = root / "experiments" / "uncertainty"
    uncertainty_summary = read_json(uncertainty_dir / "uncertainty_summary.json")
    if uncertainty_summary.get("checkpoint_sha256") != baseline_hash:
        raise ValueError("Baseline uncertainty predictions do not match baseline checkpoint hash")
    baseline_test, baseline_test_rows = predictions_from_csv(
        uncertainty_dir / "predictions.csv"
    )
    verify_prediction_identity(
        baseline_test_rows, read_split_identity(split_csv, "test"), "baseline matched test"
    )
    assert_metric_record(uncertainty_summary, baseline_test, "baseline checkpoint-matched test")
    baseline_test.update(
        {
            "selected_epoch": 6,
            "checkpoint_sha256": baseline_hash,
            "source": "checkpoint-hash-verified saved experiments/uncertainty/predictions.csv",
        }
    )

    criteria = plan["success_criteria"]
    values = {
        "melanoma_f1": validation_derived["per_class"]["mel"]["f1"],
        "validation_macro_f1": validation_derived["macro_f1"],
        "nevus_recall": validation_derived["per_class"]["nv"]["recall"],
    }
    thresholds = {
        "melanoma_f1_minimum": criteria["melanoma_f1"]["minimum"],
        "validation_macro_f1_minimum": criteria["validation_macro_f1"]["minimum"],
        "nevus_recall_minimum": criteria["nevus_recall"]["minimum"],
    }
    criterion_results = {
        "melanoma_f1": values["melanoma_f1"] >= thresholds["melanoma_f1_minimum"],
        "validation_macro_f1": values["validation_macro_f1"] >= thresholds["validation_macro_f1_minimum"],
        "nevus_recall": values["nevus_recall"] >= thresholds["nevus_recall_minimum"],
    }
    passed = all(criterion_results.values())
    comparison = {
        "checkpoint_sha256": checkpoint_hash,
        "selected_epoch": selected_epoch,
        "selection_metric": "validation_loss_minimize",
        "validation_success_criteria_met": passed,
        "criterion_results": criterion_results,
        "validation_values": values,
        "success_thresholds": thresholds,
        "baseline": {
            "checkpoint_sha256": baseline_hash,
            "validation": baseline_validation,
            "test": baseline_test,
        },
        "controlled_experiment": {
            "validation": {**validation_saved, "source": "saved Experiment #2 validation artifacts"},
            "test": {**test_saved, "source": "saved Experiment #2 test artifacts"},
        },
        "baseline_test_epoch_alignment": {
            "source": "checkpoint-hash-verified uncertainty predictions",
            "checkpoint_hash_verified": True,
            "split_ids_and_labels_verified": True,
            "baseline_test_metrics_json_used": False,
        },
        "ph2_accessed": False,
    }
    manifest = {
        "status": "COMPLETE",
        "finalization_mode": "saved_artifacts_only",
        "started_at_utc": config.get("started_at_utc"),
        "completed_at_utc": config.get("completed_at_utc"),
        "baseline_checkpoint_sha256": baseline_hash,
        "split_sha256": SPLIT_SHA256,
        "new_checkpoint_sha256": checkpoint_hash,
        "training_epochs": len(history),
        "selected_epoch": selected_epoch,
        "selection_metric": "validation_loss_minimize",
        "selection_value": float(best_epoch["val_loss"]),
        "validation_success_criteria_met": passed,
        "criterion_results": criterion_results,
        "sampler": config["sampler"],
        "ph2_accessed": False,
        "training_rerun_for_finalization": False,
        "inference_rerun_for_finalization": False,
        "baseline_test_source": "checkpoint-hash-verified saved uncertainty predictions",
        "runtime": config.get("runtime"),
    }
    return {
        "comparison_metrics.json": json_text(comparison),
        "experiment_manifest.json": json_text(manifest),
        "controlled_exp2_final_report.md": make_final_report(comparison, manifest),
    }


def make_final_report(comparison: dict, manifest: dict) -> str:
    result = "PASS" if comparison["validation_success_criteria_met"] else "FAIL"
    values = comparison["validation_values"]
    thresholds = comparison["success_thresholds"]
    criteria_rows = (
        ("Melanoma F1", "melanoma_f1", "melanoma_f1_minimum"),
        ("Validation macro-F1", "validation_macro_f1", "validation_macro_f1_minimum"),
        ("Nevus recall", "nevus_recall", "nevus_recall_minimum"),
    )
    table = "\n".join(
        f"| {label} | {values[key]:.12f} | {thresholds[threshold]:.12f} | "
        f"{'PASS' if comparison['criterion_results'][key] else 'FAIL'} |"
        for label, key, threshold in criteria_rows
    )
    baseline_val = comparison["baseline"]["validation"]
    baseline_test = comparison["baseline"]["test"]
    controlled_val = comparison["controlled_experiment"]["validation"]
    controlled_test = comparison["controlled_experiment"]["test"]
    return f"""# Controlled Experiment #2 Final Report

**Outcome: {result}**

## Verified Run

- Model: EfficientNetV2S
- Recorded epochs: {manifest['training_epochs']}
- Selected epoch: {manifest['selected_epoch']}
- Checkpoint selection: minimum frozen HAM10000 validation loss
- Checkpoint SHA256: `{manifest['new_checkpoint_sha256']}`
- Training sampler: deterministic `WeightedRandomSampler`
- Melanoma weight: {manifest['sampler']['mel_weight']:.14f}; non-melanoma weight: 1.0
- Original baseline focal loss retained: alpha 0.25, gamma 2.0; no class-specific loss weights
- PH2 accessed: NO

## Frozen Validation Criteria

| Criterion | Experiment #2 | Required minimum | Result |
|---|---:|---:|---|
{table}

## Checkpoint-Matched Comparison

| Partition / checkpoint | Mel precision | Mel recall | Mel F1 | Nevus recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Baseline validation epoch 6 | {baseline_val['per_class']['mel']['precision']:.6f} | {baseline_val['per_class']['mel']['recall']:.6f} | {baseline_val['per_class']['mel']['f1']:.6f} | {baseline_val['per_class']['nv']['recall']:.6f} | {baseline_val['macro_f1']:.6f} |
| Experiment #2 validation epoch {manifest['selected_epoch']} | {controlled_val['per_class']['mel']['precision']:.6f} | {controlled_val['per_class']['mel']['recall']:.6f} | {controlled_val['per_class']['mel']['f1']:.6f} | {controlled_val['per_class']['nv']['recall']:.6f} | {controlled_val['macro_f1']:.6f} |
| Baseline matched test epoch 6 | {baseline_test['per_class']['mel']['precision']:.6f} | {baseline_test['per_class']['mel']['recall']:.6f} | {baseline_test['per_class']['mel']['f1']:.6f} | {baseline_test['per_class']['nv']['recall']:.6f} | {baseline_test['macro_f1']:.6f} |
| Experiment #2 test epoch {manifest['selected_epoch']} | {controlled_test['per_class']['mel']['precision']:.6f} | {controlled_test['per_class']['mel']['recall']:.6f} | {controlled_test['per_class']['mel']['f1']:.6f} | {controlled_test['per_class']['nv']['recall']:.6f} | {controlled_test['macro_f1']:.6f} |

Baseline test predictions were used only after verifying their checkpoint SHA256 against the frozen baseline checkpoint and their image IDs/labels against the frozen test partition. Baseline `test_metrics.json` was not used as epoch-6 evidence.

This report was assembled from saved metrics and predictions. It does not rerun training or inference. Test metrics are descriptive and were not used for checkpoint or hyperparameter selection.
"""


def finalize_existing(project_root: Path) -> None:
    root = project_root.resolve()
    outputs = finalization_outputs(root)
    output_dir = root / "experiments" / EXPERIMENT_NAME
    for name, content in outputs.items():
        path = output_dir / name
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"Existing finalization artifact differs; refusing overwrite: {path}")
            print(f"VERIFIED existing {name}")
        else:
            write_atomic(path, content, replace=False)
            print(f"CREATED {name}")
    print("Finalization used saved artifacts only; no model or image inference was run.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--train", action="store_true", help="Train only after all preflight checks pass")
    modes.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Rebuild final reports from saved metrics/predictions without inference",
    )
    args = parser.parse_args()
    root = args.project_root.resolve()

    if args.finalize_existing:
        finalize_existing(root)
        return 0

    report = preflight(
        root,
        require_t4=args.train,
        require_clean_outputs=args.train,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        return 2
    if args.train:
        run_training(root, report, workers=args.workers)
    else:
        print("Preflight only: no training or inference was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())