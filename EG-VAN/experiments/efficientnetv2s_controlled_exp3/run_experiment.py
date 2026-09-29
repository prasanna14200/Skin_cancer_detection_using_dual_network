"""Run or finalize Controlled Experiment #3 from the frozen HAM10000 protocol.

Training requires --train and a successful Colab Tesla T4 preflight. The
--finalize-existing path reconstructs reports only from saved artifacts.
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

EXPERIMENT_NAME = "efficientnetv2s_controlled_exp3"
SPLIT_NAME = "split_leakage_aware.csv"
CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
BASELINE_SHA256 = "f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848"
EXPERIMENT_1_SHA256 = "478bb1aa9c48897accceb800a103e7ee76c1381dc6514c7355996babbfaa902f"
EXPERIMENT_2_SHA256 = "9dc38968e91a98693de5e6a8c4ee720a69ef80e275887780a8d714077c6ea93f"
SPLIT_SHA256 = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
TRAIN_COUNTS = {
    "akiec": 257,
    "bcc": 398,
    "bkl": 891,
    "df": 95,
    "mel": 899,
    "nv": 5366,
    "vasc": 109,
}
VAL_COUNTS = {"akiec": 30, "bcc": 58, "bkl": 104, "df": 9, "mel": 107, "nv": 663, "vasc": 15}
TEST_COUNTS = {"akiec": 40, "bcc": 58, "bkl": 104, "df": 11, "mel": 107, "nv": 676, "vasc": 18}
CRITERIA = {
    "melanoma_f1_minimum": 0.5757731958762886,
    "validation_macro_f1_minimum": 0.6316582381362074,
    "nevus_recall_minimum": 0.9,
}
MEL_WEIGHT = math.pow(TRAIN_COUNTS["nv"] / TRAIN_COUNTS["mel"], 0.25)
OUTPUT_NAMES = (
    "best_checkpoint.pt",
    "config.json",
    "training_history.json",
    "validation_metrics.json",
    "validation_predictions.csv",
    "test_metrics.json",
    "test_predictions.csv",
    "comparison_metrics.json",
    "experiment_manifest.json",
    "controlled_exp3_final_report.md",
)
CLASS_INDEX = {label: index for index, label in enumerate(CLASS_ORDER)}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def json_text(value: dict) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_load_checkpoint(path: Path) -> dict:
    version_type = getattr(torch.torch_version, "TorchVersion", None)
    if version_type is not None:
        torch.serialization.add_safe_globals([version_type])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint structure: {path}")
    return checkpoint


def atomic_write(path: Path, content: str, *, allow_replace: bool = False) -> None:
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
        if path.exists() and not allow_replace:
            if path.read_text(encoding="utf-8") == content:
                return
            raise FileExistsError(f"Refusing to overwrite generated artifact: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict, *, allow_replace: bool = False) -> None:
    atomic_write(path, json_text(value), allow_replace=allow_replace)


def split_counts(split_csv: Path) -> dict[str, dict[str, int]]:
    counts = {part: Counter() for part in ("train", "val", "test")}
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "lesion_id", "dx", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Frozen split lacks required columns")
        for row in reader:
            if row["split"] not in counts or row["dx"] not in CLASS_INDEX:
                raise ValueError(f"Unexpected frozen split value: {row}")
            counts[row["split"]][row["dx"]] += 1
    return {
        part: {label: int(counts[part][label]) for label in CLASS_ORDER}
        for part in ("train", "val", "test")
    }


def metrics_from_matrix(matrix: Sequence[Sequence[int]]) -> dict:
    matrix = [[int(value) for value in row] for row in matrix]
    if len(matrix) != len(CLASS_ORDER) or any(len(row) != len(CLASS_ORDER) for row in matrix):
        raise ValueError("Confusion matrix must be 7x7 in frozen class order")
    total = sum(map(sum, matrix))
    if total == 0:
        raise ValueError("Empty confusion matrix")
    per_class, f1_values, recalls = {}, [], []
    correct = 0
    for index, label in enumerate(CLASS_ORDER):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "predicted_count": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
        recalls.append(recall)
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


def metrics_from_csv(path: Path) -> tuple[dict, list[dict]]:
    matrix = [[0 for _ in CLASS_ORDER] for _ in CLASS_ORDER]
    rows, seen = [], set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "true_label", "predicted_label", "correct"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Prediction CSV missing columns: {path}")
        for row in reader:
            image_id = row["image_id"]
            if not image_id or image_id in seen:
                raise ValueError(f"Duplicate or empty image_id in {path}")
            seen.add(image_id)
            try:
                target, prediction = CLASS_INDEX[row["true_label"]], CLASS_INDEX[row["predicted_label"]]
            except KeyError as exc:
                raise ValueError(f"Unknown class in {path}: {exc}") from exc
            if (row["correct"].strip().lower() == "true") != (target == prediction):
                raise ValueError(f"Prediction correct flag mismatch in {path}: {image_id}")
            matrix[target][prediction] += 1
            rows.append(row)
    return metrics_from_matrix(matrix), rows


def assert_metrics(saved: dict, derived: dict, source: str) -> None:
    if saved.get("class_order") not in (None, list(CLASS_ORDER)):
        raise ValueError(f"Class order mismatch: {source}")
    if saved.get("sample_count") is not None and int(saved["sample_count"]) != derived["sample_count"]:
        raise ValueError(f"Sample count mismatch: {source}")
    if saved.get("confusion_matrix") is not None and saved["confusion_matrix"] != derived["confusion_matrix"]:
        raise ValueError(f"Confusion matrix mismatch: {source}")
    for name in CLASS_ORDER:
        for metric in ("precision", "recall", "f1"):
            actual = saved.get("per_class", {}).get(name, {}).get(metric)
            if actual is not None and not math.isclose(
                float(actual), derived["per_class"][name][metric], rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"{source}.{name}.{metric} mismatch")
    for metric in ("accuracy", "macro_f1"):
        if metric in saved and not math.isclose(
            float(saved[metric]), derived[metric], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{source}.{metric} mismatch")


def split_identity(split_csv: Path, part: str) -> dict[str, str]:
    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["image_id"]: row["dx"]
            for row in csv.DictReader(handle)
            if row["split"] == part
        }


def assert_prediction_identity(rows: Sequence[dict], expected: dict[str, str], source: str) -> None:
    actual = {row["image_id"]: row["true_label"] for row in rows}
    if actual != expected:
        raise ValueError(f"Saved prediction IDs or true labels differ from frozen split: {source}")


def validate_plan(plan: dict) -> None:
    if plan.get("status") != "PLANNED_NOT_RUN":
        raise ValueError("Plan must remain PLANNED_NOT_RUN")
    frozen = plan.get("frozen_inputs", {})
    if frozen.get("split_sha256") != SPLIT_SHA256 or frozen.get("class_order") != list(CLASS_ORDER):
        raise ValueError("Plan split hash/class order differ from frozen values")
    if frozen.get("train_class_counts") != TRAIN_COUNTS:
        raise ValueError("Plan training counts differ from frozen split")
    intervention = plan.get("intervention", {})
    if intervention.get("sampler") != "torch.utils.data.WeightedRandomSampler":
        raise ValueError("Sampler type changed")
    if intervention.get("sampler_seed") != SEED or intervention.get("replacement") is not True:
        raise ValueError("Sampler seed or replacement changed")
    if intervention.get("num_samples_per_epoch") != 8015:
        raise ValueError("Sampler sample count must remain 8015")
    if not math.isclose(
        float(intervention.get("melanoma_sampling_weight", -1)), MEL_WEIGHT, rel_tol=1e-14
    ) or intervention.get("non_melanoma_sampling_weight") != 1.0:
        raise ValueError("Experiment #3 sampler weights differ from the approved formula")
    if intervention.get("weight_formula") != (
        "(n_train[nv] / n_train[mel])^(1/4) for mel rows; 1.0 otherwise"
    ):
        raise ValueError("Sampler formula differs from approved protocol")
    if intervention.get("class_balanced_loss_or_other_loss_weighting") is not False:
        raise ValueError("Original focal loss must remain unchanged")
    if plan.get("success_criteria") != {
        "all_required": True,
        "melanoma_f1": {"minimum": CRITERIA["melanoma_f1_minimum"]},
        "validation_macro_f1": {"minimum": CRITERIA["validation_macro_f1_minimum"]},
        "nevus_recall": {"minimum": CRITERIA["nevus_recall_minimum"]},
        "evaluation_partition": "HAM10000 validation only",
        "must_not_be_changed_after_results": True,
    }:
        raise ValueError("Frozen success criteria changed")
    if plan.get("ph2_policy", {}).get("allowed_during_development") is not False:
        raise ValueError("PH2 must remain prohibited during development")
    exp2 = plan.get("experiment_2_motivation_and_provenance", {})
    if exp2.get("checkpoint_sha256") != EXPERIMENT_2_SHA256 or exp2.get("selected_epoch") != 8:
        raise ValueError("Plan Experiment #2 provenance is incorrect")
    frozen_cfg = plan.get("frozen_configuration", {})
    expected_cfg = {
        "model": "EfficientNetV2S",
        "pretrained_weight_source": "torchvision EfficientNet_V2_S_Weights.DEFAULT",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adamax",
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "checkpoint_selection": {
            "metric": "validation_loss",
            "direction": "minimize",
            "data": "frozen HAM10000 validation split only",
        },
    }
    for key, expected in expected_cfg.items():
        if frozen_cfg.get(key) != expected:
            raise ValueError(f"Frozen plan field changed: {key}")
    if frozen_cfg.get("baseline_focal_loss") != {
        "name": "focal_loss",
        "alpha": FOCAL_ALPHA,
        "gamma": FOCAL_GAMMA,
        "class_specific_loss_weights": False,
    }:
        raise ValueError("Baseline focal loss configuration changed")


def verify_exp2(root: Path) -> dict:
    directory = root / "experiments" / "efficientnetv2s_controlled_exp2"
    required = (
        "best_checkpoint.pt", "config.json", "training_history.json", "validation_metrics.json",
        "test_metrics.json", "comparison_metrics.json", "experiment_manifest.json",
        "controlled_exp2_final_report.md", "validation_predictions.csv", "test_predictions.csv",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Experiment #2 provenance files missing: {missing}")
    digest = sha256_file(directory / "best_checkpoint.pt")
    if digest != EXPERIMENT_2_SHA256:
        raise ValueError(f"Experiment #2 checkpoint hash mismatch: {digest}")
    config = read_json(directory / "config.json")
    manifest = read_json(directory / "experiment_manifest.json")
    comparison = read_json(directory / "comparison_metrics.json")
    validation = read_json(directory / "validation_metrics.json")
    test = read_json(directory / "test_metrics.json")
    history = read_json(directory / "training_history.json").get("epochs", [])
    checkpoint = safe_load_checkpoint(directory / "best_checkpoint.pt")
    best = min(history, key=lambda row: float(row["val_loss"]))
    aligned = (
        len(history) == 25
        and int(best["epoch"]) == 8
        and int(checkpoint.get("epoch", -1)) == 8
        and int(validation.get("selected_epoch", -1)) == 8
        and int(test.get("selected_epoch", -1)) == 8
        and int(manifest.get("selected_epoch", -1)) == 8
        and int(comparison.get("selected_epoch", -1)) == 8
        and math.isclose(float(best["val_loss"]), float(validation["loss"]), rel_tol=1e-12)
        and validation.get("checkpoint_sha256") == digest
        and test.get("checkpoint_sha256") == digest
        and manifest.get("new_checkpoint_sha256") == digest
        and comparison.get("checkpoint_sha256") == digest
        and config.get("checkpoint_selection_metric") == "validation_loss_minimize"
        and config.get("loss") == "original baseline focal_loss; no class-specific loss weighting"
        and config.get("sampler", {}).get("mel_weight") == math.sqrt(5366 / 899)
    )
    if not aligned:
        raise ValueError("Experiment #2 checkpoint, history, config, metrics, or manifest disagree")
    derived, val_rows = metrics_from_csv(directory / "validation_predictions.csv")
    assert_metrics(validation, derived, "Experiment #2 validation")
    if best["validation"].get("confusion_matrix") != derived["confusion_matrix"]:
        raise ValueError("Experiment #2 selected history confusion matrix disagrees with saved predictions")
    derived_test, test_rows = metrics_from_csv(directory / "test_predictions.csv")
    assert_metrics(test, derived_test, "Experiment #2 test")
    assert_prediction_identity(val_rows, split_identity(root / "data/splits" / SPLIT_NAME, "val"), "Exp2 validation")
    assert_prediction_identity(test_rows, split_identity(root / "data/splits" / SPLIT_NAME, "test"), "Exp2 test")
    if comparison.get("validation_values", {}).get("melanoma_f1") != validation["per_class"]["mel"]["f1"]:
        raise ValueError("Experiment #2 comparison validation metrics mismatch")
    return {
        "checkpoint_sha256": digest,
        "selected_epoch": 8,
        "validation_loss": float(best["val_loss"]),
        "validation": validation,
        "test": test,
        "config": config,
        "manifest": manifest,
        "comparison": comparison,
    }


def preflight(project_root: Path, *, require_t4: bool, require_clean_outputs: bool) -> dict:
    root = project_root.resolve()
    output = root / "experiments" / EXPERIMENT_NAME
    split = root / "data" / "splits" / SPLIT_NAME
    baseline_dir = root / "experiments" / "efficientnetv2s_leakage_aware"
    report = {"project_root": str(root), "experiment_directory": str(output), "checks": {}, "errors": []}

    def check(name: str, passed: bool, details: str) -> None:
        report["checks"][name] = {"status": "PASS" if passed else "FAIL", "details": details}
        if not passed:
            report["errors"].append(f"{name}: {details}")

    check("project_root", root.is_dir(), "project root exists")
    check("experiment_3_directory", output.is_dir(), "dedicated Experiment #3 directory exists")
    check(
        "runner_location",
        output.resolve() == Path(__file__).resolve().parent,
        "runner is in the requested Experiment #3 directory",
    )
    if not root.is_dir() or not output.is_dir():
        report.update(status="FAIL", training_eligible=False, ph2_accessed=False)
        return report

    try:
        plan = read_json(output / "experiment_plan.json")
        validate_plan(plan)
        check("experiment_plan", True, "sampler formula, frozen configuration, criteria, and provenance validated")
    except Exception as exc:
        plan = None
        check("experiment_plan", False, str(exc))

    baseline_missing = [
        name for name in ("best_checkpoint.pt", "config.json", "training_history.json")
        if not (baseline_dir / name).is_file()
    ]
    check("baseline_artifacts", not baseline_missing, f"missing={baseline_missing}")
    if (baseline_dir / "best_checkpoint.pt").is_file():
        baseline_hash = sha256_file(baseline_dir / "best_checkpoint.pt")
    else:
        baseline_hash = None
    check(
        "baseline_checkpoint_hash",
        baseline_hash == BASELINE_SHA256,
        f"expected={BASELINE_SHA256}; actual={baseline_hash}",
    )
    if baseline_hash == BASELINE_SHA256:
        try:
            baseline_ckpt = safe_load_checkpoint(baseline_dir / "best_checkpoint.pt")
            baseline_history = read_json(baseline_dir / "training_history.json")["epochs"]
            baseline_best = min(baseline_history, key=lambda row: float(row["val_loss"]))
            check(
                "baseline_epoch6_selection",
                int(baseline_ckpt.get("epoch", -1)) == 6 and int(baseline_best["epoch"]) == 6,
                f"checkpoint_epoch={baseline_ckpt.get('epoch')}; history_min_loss_epoch={baseline_best['epoch']}",
            )
            baseline_config = read_json(baseline_dir / "config.json")
            expected = {
                "model": "EfficientNetV2S", "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
                "epochs": EPOCHS, "optimizer": "Adamax", "learning_rate": LEARNING_RATE,
                "focal_alpha": FOCAL_ALPHA, "focal_gamma": FOCAL_GAMMA, "seed": SEED,
                "split_csv": SPLIT_NAME, "classes": list(CLASS_ORDER),
            }
            mismatches = {key: baseline_config.get(key) for key, value in expected.items() if baseline_config.get(key) != value}
            check("baseline_frozen_configuration", not mismatches, f"mismatches={mismatches}")
        except Exception as exc:
            check("baseline_configuration_and_selection", False, str(exc))

    check("frozen_split", split.is_file(), f"path={split}")
    if split.is_file():
        try:
            split_hash = sha256_file(split)
            integrity = validate_split_integrity(split, require_lesion_isolation=True)
            counts = split_counts(split)
            count_match = counts == {"train": TRAIN_COUNTS, "val": VAL_COUNTS, "test": TEST_COUNTS}
            valid = split_hash == SPLIT_SHA256 and integrity == {
                "rows": 10015, "unique_lesions": 7470, "crossing_lesions": 0
            } and count_match and tuple(CLASS_NAMES) == CLASS_ORDER
            check(
                "split_hash_counts_class_order",
                valid,
                f"sha256={split_hash}; integrity={integrity}; counts_match={count_match}; class_order={tuple(CLASS_NAMES)}",
            )
        except Exception as exc:
            check("split_hash_counts_class_order", False, str(exc))
    else:
        check("split_hash_counts_class_order", False, "split unavailable")

    try:
        exp2 = verify_exp2(root)
        exp2_val = exp2["validation"]
        exp2_good = (
            exp2["selected_epoch"] == 8
            and exp2["checkpoint_sha256"] == EXPERIMENT_2_SHA256
            and math.isclose(exp2_val["per_class"]["mel"]["f1"], 0.5258215962441314, abs_tol=1e-12)
            and exp2_val["per_class"]["nv"]["recall"] >= CRITERIA["nevus_recall_minimum"]
            and exp2_val["macro_f1"] >= CRITERIA["validation_macro_f1_minimum"]
            and exp2_val["per_class"]["mel"]["f1"] < CRITERIA["melanoma_f1_minimum"]
        )
        check(
            "experiment_2_provenance_and_result",
            exp2_good,
            f"checkpoint={exp2['checkpoint_sha256']}; selected_epoch={exp2['selected_epoch']}; mel_f1={exp2_val['per_class']['mel']['f1']}; macro_f1={exp2_val['macro_f1']}; nv_recall={exp2_val['per_class']['nv']['recall']}",
        )
    except Exception as exc:
        check("experiment_2_provenance_and_result", False, str(exc))

    sampler_good = math.isclose(MEL_WEIGHT, math.pow(5366 / 899, 0.25), rel_tol=1e-14)
    check(
        "experiment_3_sampler_configuration",
        sampler_good,
        f"formula=(5366/899)^(1/4); mel={MEL_WEIGHT:.16f}; non-mel=1.0; seed=42; replacement=True; num_samples=8015; TRAIN labels only",
    )
    image_dir = root / "data" / "processed" / "images"
    if split.is_file():
        with split.open("r", newline="", encoding="utf-8") as handle:
            image_ids = [row["image_id"] for row in csv.DictReader(handle)]
        missing = [key for key in image_ids if not (image_dir / f"{key}.jpg").is_file()]
        check("processed_ham_images", image_dir.is_dir() and not missing, f"missing_count={len(missing)}")
    else:
        check("processed_ham_images", False, "cannot validate images without split")

    collisions = [name for name in OUTPUT_NAMES if (output / name).exists()]
    check("output_overwrite_protection", not collisions, f"existing_generated_outputs={collisions}")

    if require_t4:
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
        colab = importlib.util.find_spec("google.colab") is not None
        t4_ok = platform.system() == "Linux" and colab and cuda and gpu is not None and "Tesla T4" in gpu
        check("approved_colab_tesla_t4", t4_ok, f"platform={platform.system()}; colab={colab}; cuda={cuda}; gpu={gpu}")
    else:
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
        report["checks"]["approved_colab_tesla_t4"] = {
            "status": "NOT_REQUIRED", "details": f"preflight-only; CUDA={cuda}; device={gpu}"
        }

    report["status"] = "PASS" if not report["errors"] else "FAIL"
    report["training_eligible"] = report["status"] == "PASS" and require_t4
    report["ph2_accessed"] = False
    return report


def label_weight(label: str) -> float:
    if label not in CLASS_INDEX:
        raise ValueError(f"Unexpected TRAIN label: {label}")
    return MEL_WEIGHT if label == "mel" else 1.0


def make_sampler(dataset) -> WeightedRandomSampler:
    labels = [row["dx"] for row in dataset.rows]
    if len(labels) != 8015 or dict(Counter(labels)) != TRAIN_COUNTS:
        raise ValueError("Sampler source must be exactly the frozen 8,015 TRAIN rows")
    weights = torch.as_tensor([label_weight(label) for label in labels], dtype=torch.double)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)


def metrics_from_tensor(matrix: Tensor) -> dict:
    return metrics_from_matrix(matrix.detach().cpu().tolist())


def run_epoch(model, loader, device, *, training, optimizer=None, scaler=None):
    model.train(training)
    matrix = torch.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=torch.long)
    loss_sum = 0.0
    count_sum = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels_device = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = focal_loss(logits, labels_device, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            predictions = logits.argmax(dim=1)
            for target, predicted in zip(labels.cpu(), predictions.detach().cpu()):
                matrix[int(target), int(predicted)] += 1
            batch_count = labels.size(0)
            loss_sum += float(loss.detach().float().cpu()) * batch_count
            count_sum += batch_count
    if count_sum == 0:
        raise ValueError("Empty data loader")
    return loss_sum / count_sum, metrics_from_tensor(matrix)


def evaluate(model, loader, device):
    model.eval()
    rows, matrix = [], torch.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=torch.long)
    loss_sum = 0.0
    count_sum = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = focal_loss(logits, labels.to(device), alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            for target, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[int(target), int(predicted)] += 1
            for source, target, predicted, probability in zip(
                loader.dataset.rows[count_sum:count_sum + len(labels)],
                labels.tolist(), predictions.tolist(), probabilities.tolist(),
            ):
                rows.append({
                    "image_id": source["image_id"],
                    "true_label": CLASS_ORDER[target],
                    "predicted_label": CLASS_ORDER[predicted],
                    "correct": target == predicted,
                    "probabilities": probability,
                })
            batch_count = labels.size(0)
            loss_sum += float(loss.detach().float().cpu()) * batch_count
            count_sum += batch_count
    metrics = metrics_from_tensor(matrix)
    metrics["loss"] = loss_sum / count_sum
    return metrics, rows


def write_predictions(path: Path, rows: Sequence[dict]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(
            handle, fieldnames=("image_id", "true_label", "predicted_label", "correct", "probabilities")
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "probabilities": json.dumps(row["probabilities"], separators=(",", ":"))})
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_experiment(root: Path, preflight_report: dict, workers: int) -> None:
    output = root / "experiments" / EXPERIMENT_NAME
    set_seed(SEED)
    device = torch.device("cuda")
    model, train_set, val_set, test_set = build_run_components(root, SPLIT_NAME)
    sampler = make_sampler(train_set)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False,
        num_workers=workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True
    )
    model.to(device)
    optimizer, scheduler = make_optimizer_and_scheduler(model)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    config = {
        "experiment": EXPERIMENT_NAME,
        "model": "EfficientNetV2S",
        "pretrained_weight_source": "torchvision EfficientNet_V2_S_Weights.DEFAULT",
        "initialization": "pretrained weights; no baseline or Experiment #2 checkpoint loading",
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
            "formula": "(5366 / 899)^(1/4) for melanoma; 1.0 otherwise",
            "mel_weight": MEL_WEIGHT,
            "non_melanoma_weight": 1.0,
            "train_counts": TRAIN_COUNTS,
            "seed": SEED,
            "replacement": True,
            "num_samples": len(train_set),
            "generator": "torch.Generator(device='cpu').manual_seed(42)",
            "labels_from": "frozen TRAIN rows only",
        },
        "split_csv": SPLIT_NAME,
        "split_sha256": SPLIT_SHA256,
        "classes": list(CLASS_ORDER),
        "checkpoint_selection_metric": "validation_loss_minimize",
        "success_criteria": CRITERIA,
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
    write_json(output / "config.json", config)
    history, best_loss = [], math.inf
    checkpoint_path = output / "best_checkpoint.pt"
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, device, training=True, optimizer=optimizer, scaler=scaler
        )
        val_loss, val_metrics = run_epoch(model, val_loader, device, training=False)
        scheduler.step(val_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train": train_metrics,
            "val_loss": val_loss,
            "validation": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "sampler_seed": SEED,
            "sampler_num_samples": len(train_set),
        })
        write_json(output / "training_history.json", {"epochs": history}, allow_replace=True)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(checkpoint_path, {
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "epoch": epoch,
                "val_loss": val_loss,
                "config": config,
                "class_order": list(CLASS_ORDER),
            })
        print(
            f"epoch={epoch:02d}/{EPOCHS} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_mel_f1={val_metrics['per_class']['mel']['f1']:.6f}"
        )

    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = safe_load_checkpoint(checkpoint_path)
    selected_epoch = int(checkpoint["epoch"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    val_metrics, val_rows = evaluate(model, val_loader, device)
    test_metrics, test_rows = evaluate(model, test_loader, device)
    for metrics in (val_metrics, test_metrics):
        metrics.update({
            "selected_epoch": selected_epoch,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_selection_metric": "validation_loss_minimize",
        })
    val_metrics["loss"] = float(checkpoint["val_loss"])
    test_metrics["evaluation_partition"] = "frozen HAM10000 test after checkpoint freeze; evaluation only"
    write_json(output / "validation_metrics.json", val_metrics)
    write_predictions(output / "validation_predictions.csv", val_rows)
    write_json(output / "test_metrics.json", test_metrics)
    write_predictions(output / "test_predictions.csv", test_rows)
    config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "config.json", config, allow_replace=True)
    finalize_existing(root)


def validation_from_history(root: Path, experiment: str, expected_hash: str, selected_epoch: int) -> dict:
    directory = root / "experiments" / experiment
    checkpoint_hash = sha256_file(directory / "best_checkpoint.pt")
    if checkpoint_hash != expected_hash:
        raise ValueError(f"Checkpoint hash mismatch for {experiment}: {checkpoint_hash}")
    history = read_json(directory / "training_history.json").get("epochs", [])
    best = min(history, key=lambda row: float(row["val_loss"]))
    checkpoint = safe_load_checkpoint(directory / "best_checkpoint.pt")
    if int(best["epoch"]) != selected_epoch or int(checkpoint.get("epoch", -1)) != selected_epoch:
        raise ValueError(f"Selected epoch mismatch for {experiment}")
    if experiment == "efficientnetv2s_leakage_aware":
        metrics = metrics_from_matrix(best["validation"]["confusion_matrix"])
        metrics["loss"] = float(best["val_loss"])
    else:
        metrics = read_json(directory / "validation_metrics.json")
        if int(metrics.get("selected_epoch", -1)) != selected_epoch:
            raise ValueError(f"Saved validation metrics epoch mismatch for {experiment}")
        assert_metrics(metrics, metrics_from_csv(directory / "validation_predictions.csv")[0], experiment)
    metrics["selected_epoch"] = selected_epoch
    metrics["checkpoint_sha256"] = checkpoint_hash
    return metrics


def finalization_data(root: Path) -> tuple[dict, dict, str]:
    output = root / "experiments" / EXPERIMENT_NAME
    split = root / "data" / "splits" / SPLIT_NAME
    history = read_json(output / "training_history.json").get("epochs", [])
    if len(history) != EPOCHS:
        raise ValueError(f"Expected {EPOCHS} Experiment #3 epochs; found {len(history)}")
    best = min(history, key=lambda row: float(row["val_loss"]))
    checkpoint_hash = sha256_file(output / "best_checkpoint.pt")
    checkpoint = safe_load_checkpoint(output / "best_checkpoint.pt")
    selected = int(checkpoint.get("epoch", -1))
    if selected != int(best["epoch"]):
        raise ValueError("Saved checkpoint is not the minimum-validation-loss epoch")
    if checkpoint.get("class_order") != list(CLASS_ORDER):
        raise ValueError("Experiment #3 checkpoint class order mismatch")
    config = read_json(output / "config.json")
    if config.get("checkpoint_selection_metric") != "validation_loss_minimize":
        raise ValueError("Experiment #3 config selection rule changed")
    if config.get("split_sha256") != SPLIT_SHA256 or config.get("split_csv") != SPLIT_NAME:
        raise ValueError("Experiment #3 config split identity changed")
    if config.get("sampler", {}).get("mel_weight") != MEL_WEIGHT:
        raise ValueError("Experiment #3 config sampler weight mismatch")
    val_saved, test_saved = read_json(output / "validation_metrics.json"), read_json(output / "test_metrics.json")
    if val_saved.get("checkpoint_sha256") != checkpoint_hash or test_saved.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Experiment #3 metrics checkpoint hash mismatch")
    if int(val_saved.get("selected_epoch", -1)) != selected or int(test_saved.get("selected_epoch", -1)) != selected:
        raise ValueError("Experiment #3 saved metric selected epoch mismatch")
    val_derived, val_rows = metrics_from_csv(output / "validation_predictions.csv")
    test_derived, test_rows = metrics_from_csv(output / "test_predictions.csv")
    assert_metrics(val_saved, val_derived, "Experiment #3 validation")
    assert_metrics(test_saved, test_derived, "Experiment #3 test")
    assert_prediction_identity(val_rows, split_identity(split, "val"), "Experiment #3 validation")
    assert_prediction_identity(test_rows, split_identity(split, "test"), "Experiment #3 test")
    if val_derived["sample_count"] != sum(VAL_COUNTS.values()) or test_derived["sample_count"] != sum(TEST_COUNTS.values()):
        raise ValueError("Experiment #3 prediction counts do not match frozen partitions")

    baseline_val = validation_from_history(root, "efficientnetv2s_leakage_aware", BASELINE_SHA256, 6)
    exp1_val = validation_from_history(root, "efficientnetv2s_controlled_exp1", EXPERIMENT_1_SHA256, 5)
    exp1_dir = root / "experiments" / "efficientnetv2s_controlled_exp1"
    exp1_test = read_json(exp1_dir / "test_metrics.json")
    if exp1_test.get("checkpoint_sha256") != EXPERIMENT_1_SHA256 or int(exp1_test.get("selected_epoch", -1)) != 5:
        raise ValueError("Experiment #1 test metrics are not checkpoint matched")
    exp1_test_derived, exp1_test_rows = metrics_from_csv(exp1_dir / "test_predictions.csv")
    assert_metrics(exp1_test, exp1_test_derived, "Experiment #1 test")
    assert_prediction_identity(exp1_test_rows, split_identity(split, "test"), "Experiment #1 test")

    exp2 = verify_exp2(root)
    exp2_val, exp2_test = exp2["validation"], exp2["test"]
    uncertainty_dir = root / "experiments" / "uncertainty"
    baseline_summary = read_json(uncertainty_dir / "uncertainty_summary.json")
    if baseline_summary.get("checkpoint_sha256") != BASELINE_SHA256:
        raise ValueError("Baseline test evidence checkpoint hash mismatch")
    baseline_test, baseline_test_rows = metrics_from_csv(uncertainty_dir / "predictions.csv")
    assert_metrics(baseline_summary, baseline_test, "baseline checkpoint-matched test")
    assert_prediction_identity(baseline_test_rows, split_identity(split, "test"), "baseline test")
    for metrics, epoch, digest in (
        (exp2_val, 8, EXPERIMENT_2_SHA256),
        (exp2_test, 8, EXPERIMENT_2_SHA256),
        (exp1_val, 5, EXPERIMENT_1_SHA256),
    ):
        metrics["selected_epoch"] = epoch
        metrics["checkpoint_sha256"] = digest
    for metrics in (baseline_val, baseline_test):
        metrics["selected_epoch"] = 6
        metrics["checkpoint_sha256"] = BASELINE_SHA256
    exp1_test["selected_epoch"] = 5
    exp1_test["checkpoint_sha256"] = EXPERIMENT_1_SHA256

    validation_metrics = {"baseline": baseline_val, "experiment_1": exp1_val, "experiment_2": exp2_val, "experiment_3": val_saved}
    test_metrics = {"baseline": baseline_test, "experiment_1": exp1_test, "experiment_2": exp2_test, "experiment_3": test_saved}
    thresholds = {
        "melanoma_f1_minimum": CRITERIA["melanoma_f1_minimum"],
        "validation_macro_f1_minimum": CRITERIA["validation_macro_f1_minimum"],
        "nevus_recall_minimum": CRITERIA["nevus_recall_minimum"],
    }
    values = {
        "melanoma_f1": val_derived["per_class"]["mel"]["f1"],
        "validation_macro_f1": val_derived["macro_f1"],
        "nevus_recall": val_derived["per_class"]["nv"]["recall"],
    }
    results = {
        "melanoma_f1": values["melanoma_f1"] >= thresholds["melanoma_f1_minimum"],
        "validation_macro_f1": values["validation_macro_f1"] >= thresholds["validation_macro_f1_minimum"],
        "nevus_recall": values["nevus_recall"] >= thresholds["nevus_recall_minimum"],
    }
    comparison = {
        "checkpoint_sha256": checkpoint_hash,
        "selected_epoch": selected,
        "selection_metric": "validation_loss_minimize",
        "validation_success_criteria_met": all(results.values()),
        "criterion_results": results,
        "validation_values": values,
        "success_thresholds": thresholds,
        "validation": validation_metrics,
        "test_evaluation_only": test_metrics,
        "test_selection_or_tuning_used": False,
        "ph2_accessed": False,
        "baseline_test_provenance": {
            "source": "experiments/uncertainty saved predictions and summary",
            "checkpoint_hash_verified": True,
            "baseline_test_metrics_json_used": False,
            "split_ids_and_labels_verified": True,
        },
    }
    manifest = {
        "status": "COMPLETE",
        "started_at_utc": config.get("started_at_utc"),
        "completed_at_utc": config.get("completed_at_utc"),
        "baseline_checkpoint_sha256": BASELINE_SHA256,
        "experiment_1_checkpoint_sha256": EXPERIMENT_1_SHA256,
        "experiment_2_checkpoint_sha256": EXPERIMENT_2_SHA256,
        "split_sha256": SPLIT_SHA256,
        "new_checkpoint_sha256": checkpoint_hash,
        "training_epochs": len(history),
        "selected_epoch": selected,
        "selection_metric": "validation_loss_minimize",
        "selection_value": float(best["val_loss"]),
        "validation_success_criteria_met": all(results.values()),
        "criterion_results": results,
        "sampler": config["sampler"],
        "ph2_accessed": False,
        "baseline_test_source": "hash-verified uncertainty predictions",
        "runtime": config.get("runtime"),
    }
    table = make_report(comparison, manifest)
    return comparison, manifest, table


def make_report(comparison: dict, manifest: dict) -> str:
    outcome = "PASS" if comparison["validation_success_criteria_met"] else "FAIL"
    rows = []
    for label, key, threshold in (
        ("Melanoma F1", "melanoma_f1", "melanoma_f1_minimum"),
        ("Validation macro-F1", "validation_macro_f1", "validation_macro_f1_minimum"),
        ("Nevus recall", "nevus_recall", "nevus_recall_minimum"),
    ):
        value = comparison["validation_values"][key]
        minimum = comparison["success_thresholds"][threshold]
        result = "PASS" if comparison["criterion_results"][key] else "FAIL"
        rows.append(f"| {label} | {value:.12f} | {minimum:.12f} | {result} |")
    lines = []
    for name, metrics in comparison["validation"].items():
        mel = metrics["per_class"]["mel"]
        nv = metrics["per_class"]["nv"]
        lines.append(
            f"| {name} epoch {metrics['selected_epoch']} | {mel['precision']:.6f} | {mel['recall']:.6f} | "
            f"{mel['f1']:.6f} | {nv['recall']:.6f} | {metrics['macro_f1']:.6f} |"
        )
    return f"""# Controlled Experiment #3 Final Report

**Outcome: {outcome}**

- Selected checkpoint epoch: {manifest['selected_epoch']} (minimum validation loss)
- Checkpoint SHA256: `{manifest['new_checkpoint_sha256']}`
- Melanoma sampler weight: {manifest['sampler']['mel_weight']:.16f}; non-melanoma weight: 1.0
- PH2 accessed: NO

## Frozen Validation Criteria

| Criterion | Experiment #3 | Required minimum | Result |
|---|---:|---:|---|
{chr(10).join(rows)}

## Checkpoint-Matched Validation Comparison

| Experiment | Mel precision | Mel recall | Mel F1 | Nevus recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Test results are evaluation only. They were not used for checkpoint selection or tuning. Baseline test metrics come from saved uncertainty predictions whose checkpoint hash and frozen test IDs/labels were verified; the final-epoch baseline `test_metrics.json` was not substituted for selected epoch 6.

The comparison and report were reconstructed from saved artifacts only. No inference was rerun during finalization.
"""


def finalize_existing(root: Path) -> None:
    comparison, manifest, report = finalization_data(root)
    output = root / "experiments" / EXPERIMENT_NAME
    results = {
        "comparison_metrics.json": json_text(comparison),
        "experiment_manifest.json": json_text(manifest),
        "controlled_exp3_final_report.md": report,
    }
    collisions = [name for name, content in results.items() if (output / name).exists() and (output / name).read_text(encoding="utf-8") != content]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite differing finalization files: {collisions}")
    for name, content in results.items():
        path = output / name
        if path.exists():
            print(f"VERIFIED existing {name}")
        else:
            atomic_write(path, content)
            print(f"CREATED {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--train", action="store_true")
    modes.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.finalize_existing:
        finalize_existing(root)
        return 0
    report = preflight(root, require_t4=args.train, require_clean_outputs=args.train)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        return 2
    if args.train:
        if not report["training_eligible"]:
            raise RuntimeError("Training is not eligible on this runtime")
        train_experiment(root, report, args.workers)
    else:
        print("Preflight only; no training or inference was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())