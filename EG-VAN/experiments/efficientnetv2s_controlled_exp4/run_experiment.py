"""Run or finalize Controlled Experiment #4 under the frozen HAM10000 protocol.

Training requires --train and a successful Google Colab Tesla T4 preflight.
--finalize-existing reconstructs reports from saved metrics and predictions only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
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
from torch.optim import Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    set_seed,
)

EXPERIMENT_NAME = "efficientnetv2s_controlled_exp4"
SPLIT_NAME = "split_leakage_aware.csv"
CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
BASELINE_SHA256 = "f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848"
EXP1_SHA256 = "478bb1aa9c48897accceb800a103e7ee76c1381dc6514c7355996babbfaa902f"
EXP2_SHA256 = "9dc38968e91a98693de5e6a8c4ee720a69ef80e275887780a8d714077c6ea93f"
EXP3_SHA256 = "8615a9f5afad48c17c21b559aeab01e5e506fa3cbce3f544eed14b47b53377b9"
SPLIT_SHA256 = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
EXP3_WEIGHT = 1.5630495442733532
MEL_WEIGHT = EXP3_WEIGHT
NON_MEL_WEIGHT = 1.0
TRAIN_COUNTS = {"akiec": 257, "bcc": 398, "bkl": 891, "df": 95, "mel": 899, "nv": 5366, "vasc": 109}
VAL_COUNTS = {"akiec": 30, "bcc": 58, "bkl": 104, "df": 9, "mel": 107, "nv": 663, "vasc": 15}
TEST_COUNTS = {"akiec": 40, "bcc": 58, "bkl": 104, "df": 11, "mel": 107, "nv": 676, "vasc": 18}
WEIGHT_DECAY = 0.0001
CRITERIA = {
    "melanoma_f1_minimum": 0.5757731958762886,
    "validation_macro_f1_minimum": 0.6316582381362074,
    "nevus_recall_minimum": 0.9,
}
CLASS_INDEX = {label: index for index, label in enumerate(CLASS_ORDER)}
GENERATED_OUTPUTS = (
    "best_checkpoint.pt",
    "config.json",
    "training_history.json",
    "validation_metrics.json",
    "validation_predictions.csv",
    "test_metrics.json",
    "test_predictions.csv",
    "comparison_metrics.json",
    "experiment_manifest.json",
    "controlled_exp4_final_report.md",
)


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


def atomic_write(path: Path, content: str, *, replace: bool = False) -> None:
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
        if path.exists() and not replace:
            if path.read_text(encoding="utf-8") == content:
                return
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict, *, replace: bool = False) -> None:
    atomic_write(path, json_text(value), replace=replace)


def split_counts(path: Path) -> dict[str, dict[str, int]]:
    counts = {part: Counter() for part in ("train", "val", "test")}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "lesion_id", "dx", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Frozen split lacks required columns")
        for row in reader:
            if row["split"] not in counts or row["dx"] not in CLASS_INDEX:
                raise ValueError(f"Unexpected split or class: {row}")
            counts[row["split"]][row["dx"]] += 1
    return {part: {label: int(counts[part][label]) for label in CLASS_ORDER} for part in counts}


def metrics_from_matrix(matrix: Sequence[Sequence[int]]) -> dict:
    matrix = [[int(value) for value in row] for row in matrix]
    if len(matrix) != 7 or any(len(row) != 7 for row in matrix):
        raise ValueError("Confusion matrix must be 7x7 in frozen class order")
    total = sum(map(sum, matrix))
    if not total:
        raise ValueError("Empty confusion matrix")
    per_class, f1s, recalls = {}, [], []
    correct = 0
    for i, label in enumerate(CLASS_ORDER):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "predicted_count": predicted, "precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
        recalls.append(recall)
        correct += tp
    return {
        "sample_count": total,
        "class_order": list(CLASS_ORDER),
        "accuracy": correct / total,
        "macro_f1": sum(f1s) / len(f1s),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def metrics_from_predictions(path: Path) -> tuple[dict, list[dict]]:
    matrix = [[0] * 7 for _ in range(7)]
    rows, seen = [], set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "true_label", "predicted_label", "correct"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Prediction CSV missing columns: {path}")
        for row in reader:
            image_id = row["image_id"]
            if not image_id or image_id in seen:
                raise ValueError(f"Duplicate/empty image ID in {path}")
            seen.add(image_id)
            try:
                target, prediction = CLASS_INDEX[row["true_label"]], CLASS_INDEX[row["predicted_label"]]
            except KeyError as exc:
                raise ValueError(f"Unknown class in {path}: {exc}") from exc
            if (row["correct"].strip().lower() == "true") != (target == prediction):
                raise ValueError(f"Incorrect correct-flag for {image_id} in {path}")
            matrix[target][prediction] += 1
            rows.append(row)
    return metrics_from_matrix(matrix), rows


def assert_metrics(saved: dict, derived: dict, context: str) -> None:
    if saved.get("class_order") not in (None, list(CLASS_ORDER)):
        raise ValueError(f"Class order mismatch: {context}")
    if saved.get("sample_count") is not None and int(saved["sample_count"]) != derived["sample_count"]:
        raise ValueError(f"Sample count mismatch: {context}")
    if saved.get("confusion_matrix") is not None and saved["confusion_matrix"] != derived["confusion_matrix"]:
        raise ValueError(f"Confusion matrix mismatch: {context}")
    for metric in ("accuracy", "macro_f1"):
        if metric in saved and not math.isclose(float(saved[metric]), derived[metric], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{context}.{metric} mismatch")
    for label in CLASS_ORDER:
        for metric in ("precision", "recall", "f1"):
            actual = saved.get("per_class", {}).get(label, {}).get(metric)
            if actual is not None and not math.isclose(
                float(actual), derived["per_class"][label][metric], rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"{context}.{label}.{metric} mismatch")


def split_identity(split_path: Path, part: str) -> dict[str, str]:
    with split_path.open("r", newline="", encoding="utf-8") as handle:
        return {row["image_id"]: row["dx"] for row in csv.DictReader(handle) if row["split"] == part}


def assert_prediction_identity(rows: Sequence[dict], expected: dict[str, str], context: str) -> None:
    actual = {row["image_id"]: row["true_label"] for row in rows}
    if actual != expected:
        raise ValueError(f"Prediction IDs/labels differ from frozen split: {context}")


def normalize_experiment3_config(config: dict, weight_decay: float) -> dict:
    return {
        "model": config["model"],
        "pretrained_weight_source": config["pretrained_weight_source"],
        "initialization": config["initialization"],
        "image_size": config["image_size"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "optimizer": {
            "name": config["optimizer"],
            "learning_rate": config["learning_rate"],
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": weight_decay,
        },
        "focal_alpha": config["focal_alpha"],
        "focal_gamma": config["focal_gamma"],
        "loss": config["loss"],
        "scheduler": config["scheduler"],
        "seed": config["seed"],
        "sampler": config["sampler"],
        "split_csv": config["split_csv"],
        "split_sha256": config["split_sha256"],
        "classes": config["classes"],
        "checkpoint_selection_metric": config["checkpoint_selection_metric"],
        "success_criteria": config["success_criteria"],
    }


def configuration_differences(exp3_config: dict) -> list[dict]:
    exp3 = normalize_experiment3_config(exp3_config, 0.0)
    exp4 = normalize_experiment3_config(exp3_config, WEIGHT_DECAY)

    def compare(left: dict, right: dict, prefix: str = "") -> list[dict]:
        differences = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            left_value, right_value = left.get(key), right.get(key)
            if isinstance(left_value, dict) and isinstance(right_value, dict):
                differences.extend(compare(left_value, right_value, path))
            elif left_value != right_value:
                differences.append({"path": path, "experiment_3": left_value, "experiment_4": right_value})
        return differences

    return compare(exp3, exp4)


def validate_plan(plan: dict, exp3_config: dict) -> None:
    if plan.get("status") != "PLANNED_NOT_RUN":
        raise ValueError("Plan must remain PLANNED_NOT_RUN")
    frozen = plan.get("frozen_configuration", {})
    if frozen.get("split_sha256") != SPLIT_SHA256 or frozen.get("split_csv") != f"data/splits/{SPLIT_NAME}":
        raise ValueError("Plan split identity differs from frozen split")
    if frozen.get("class_order") != list(CLASS_ORDER):
        raise ValueError("Plan class order differs from source")
    if not math.isclose(float(frozen.get("sampler", {}).get("melanoma_weight", -1)), MEL_WEIGHT, rel_tol=1e-14):
        raise ValueError("Plan melanoma sampler weight changed")
    if frozen.get("sampler", {}).get("non_melanoma_weight") != NON_MEL_WEIGHT:
        raise ValueError("Plan non-melanoma sampler weight changed")
    if frozen.get("sampler", {}).get("seed") != SEED or frozen.get("sampler", {}).get("replacement") is not True or frozen.get("sampler", {}).get("num_samples") != 8015:
        raise ValueError("Plan sampler settings differ from Experiment #3")
    if plan.get("intervention", {}).get("previous_weight_decay") != 0.0 or plan.get("intervention", {}).get("new_weight_decay") != WEIGHT_DECAY:
        raise ValueError("Plan optimizer weight decay differs from audited values")
    audit = configuration_differences(exp3_config)
    if audit != [{"path": "optimizer.weight_decay", "experiment_3": 0.0, "experiment_4": WEIGHT_DECAY}]:
        raise ValueError(f"Configuration audit did not produce exactly the approved change: {audit}")
    if plan.get("configuration_difference_audit") != [
        {"path": "optimizer.weight_decay", "experiment_3": 0.0, "experiment_4": WEIGHT_DECAY, "intentional": True}
    ]:
        raise ValueError("Plan machine-readable configuration audit changed")
    if plan.get("success_criteria") != {
        "all_required": True,
        "melanoma_f1": {"minimum": CRITERIA["melanoma_f1_minimum"]},
        "validation_macro_f1": {"minimum": CRITERIA["validation_macro_f1_minimum"]},
        "nevus_recall": {"minimum": CRITERIA["nevus_recall_minimum"]},
        "evaluation_partition": "HAM10000 validation only",
        "all_required_at_minimum_validation_loss_checkpoint": True,
        "must_not_be_changed_after_results": True,
    }:
        raise ValueError("Frozen success criteria changed")
    if frozen.get("optimizer_weight_decay") != WEIGHT_DECAY:
        raise ValueError("Plan new weight decay must be exactly 0.0001")
    expected = {
        "model": exp3_config["model"],
        "pretrained_weight_source": exp3_config["pretrained_weight_source"],
        "image_size": exp3_config["image_size"],
        "batch_size": exp3_config["batch_size"],
        "epochs": exp3_config["epochs"],
        "optimizer": exp3_config["optimizer"],
        "learning_rate": exp3_config["learning_rate"],
        "scheduler": exp3_config["scheduler"],
        "seed": exp3_config["seed"],
        "focal_loss": {"name": "focal_loss", "alpha": exp3_config["focal_alpha"], "gamma": exp3_config["focal_gamma"], "class_specific_loss_weights": False},
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"Plan frozen setting differs from Experiment #3: {key}")
    if plan.get("ph2_policy", {}).get("allowed_during_development") is not False:
        raise ValueError("PH2 must be prohibited during development")


def verify_exp3(root: Path) -> dict:
    directory = root / "experiments" / "efficientnetv2s_controlled_exp3"
    required = (
        "best_checkpoint.pt", "config.json", "training_history.json", "validation_metrics.json",
        "validation_predictions.csv", "test_metrics.json", "test_predictions.csv",
        "comparison_metrics.json", "experiment_manifest.json", "controlled_exp3_final_report.md",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Experiment #3 verification artifacts missing: {missing}")
    digest = sha256_file(directory / "best_checkpoint.pt")
    if digest != EXP3_SHA256:
        raise ValueError(f"Experiment #3 checkpoint SHA256 mismatch: {digest}")
    config = read_json(directory / "config.json")
    manifest = read_json(directory / "experiment_manifest.json")
    comparison = read_json(directory / "comparison_metrics.json")
    validation = read_json(directory / "validation_metrics.json")
    test = read_json(directory / "test_metrics.json")
    history = read_json(directory / "training_history.json").get("epochs", [])
    checkpoint = safe_load_checkpoint(directory / "best_checkpoint.pt")
    best = min(history, key=lambda row: float(row["val_loss"]))
    if len(history) != EPOCHS or int(best["epoch"]) != 6 or int(checkpoint.get("epoch", -1)) != 6:
        raise ValueError("Experiment #3 selected epoch/history verification failed")
    if not math.isclose(float(best["val_loss"]), 0.07522968407795891, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Experiment #3 minimum validation loss differs from verified result")
    if not math.isclose(float(checkpoint["val_loss"]), float(best["val_loss"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Experiment #3 checkpoint loss differs from history minimum")
    if config.get("checkpoint_selection_metric") != "validation_loss_minimize":
        raise ValueError("Experiment #3 selection metric differs from frozen rule")
    sampler = config.get("sampler", {})
    if (sampler.get("mel_weight") != MEL_WEIGHT or sampler.get("non_melanoma_weight") != NON_MEL_WEIGHT
        or sampler.get("seed") != SEED or sampler.get("replacement") is not True
        or sampler.get("num_samples") != 8015 or sampler.get("labels_from") != "frozen TRAIN rows only"):
        raise ValueError("Experiment #3 sampler settings differ from frozen sampler")
    if manifest.get("new_checkpoint_sha256") != digest or manifest.get("selected_epoch") != 6:
        raise ValueError("Experiment #3 manifest checkpoint metadata mismatch")
    if comparison.get("checkpoint_sha256") != digest or comparison.get("selected_epoch") != 6:
        raise ValueError("Experiment #3 comparison checkpoint metadata mismatch")
    val_derived, val_rows = metrics_from_predictions(directory / "validation_predictions.csv")
    test_derived, test_rows = metrics_from_predictions(directory / "test_predictions.csv")
    assert_metrics(validation, val_derived, "Experiment #3 validation")
    assert_metrics(test, test_derived, "Experiment #3 test")
    if validation.get("checkpoint_sha256") != digest or test.get("checkpoint_sha256") != digest:
        raise ValueError("Experiment #3 metric checkpoint hash mismatch")
    if int(validation.get("selected_epoch", -1)) != 6 or int(test.get("selected_epoch", -1)) != 6:
        raise ValueError("Experiment #3 metric epoch mismatch")
    if comparison.get("validation_values", {}).get("melanoma_f1") != validation["per_class"]["mel"]["f1"]:
        raise ValueError("Experiment #3 comparison metrics disagree with validation metrics")
    if comparison.get("criterion_results", {}).get("melanoma_f1") is not False or comparison.get("validation_success_criteria_met") is not False:
        raise ValueError("Experiment #3 verified outcome has changed")
    split_path = root / "data" / "splits" / SPLIT_NAME
    assert_prediction_identity(val_rows, split_identity(split_path, "val"), "Experiment #3 validation")
    assert_prediction_identity(test_rows, split_identity(split_path, "test"), "Experiment #3 test")
    if val_derived["sample_count"] != 986 or test_derived["sample_count"] != 1014:
        raise ValueError("Experiment #3 prediction counts differ from frozen partition sizes")
    return {"directory": directory, "hash": digest, "config": config, "manifest": manifest,
            "comparison": comparison, "validation": validation, "test": test,
            "history": history, "checkpoint": checkpoint}


def preflight(project_root: Path, *, require_t4: bool) -> dict:
    root = project_root.resolve()
    output = root / "experiments" / EXPERIMENT_NAME
    split = root / "data" / "splits" / SPLIT_NAME
    baseline = root / "experiments" / "efficientnetv2s_leakage_aware"
    report = {"project_root": str(root), "experiment_directory": str(output), "checks": {}, "errors": []}

    def check(name: str, passed: bool, details: str) -> None:
        report["checks"][name] = {"status": "PASS" if passed else "FAIL", "details": details}
        if not passed:
            report["errors"].append(f"{name}: {details}")

    check("project_root", root.is_dir(), "project root exists")
    check("experiment_4_directory", output.is_dir(), "dedicated Experiment #4 directory exists")
    check("runner_location", output.resolve() == Path(__file__).resolve().parent, "runner is in Experiment #4 directory")
    if not root.is_dir() or not output.is_dir():
        report.update(status="FAIL", training_eligible=False, ph2_accessed=False)
        return report

    baseline_missing = [name for name in ("best_checkpoint.pt", "config.json", "training_history.json") if not (baseline / name).is_file()]
    check("baseline_artifacts", not baseline_missing, f"missing={baseline_missing}")
    if (baseline / "best_checkpoint.pt").is_file():
        base_hash = sha256_file(baseline / "best_checkpoint.pt")
    else:
        base_hash = None
    check("baseline_checkpoint_hash", base_hash == BASELINE_SHA256, f"expected={BASELINE_SHA256}; actual={base_hash}")

    check("frozen_split", split.is_file(), f"path={split}")
    if split.is_file():
        try:
            digest = sha256_file(split)
            integrity = validate_split_integrity(split, require_lesion_isolation=True)
            counts = split_counts(split)
            valid = digest == SPLIT_SHA256 and integrity == {"rows": 10015, "unique_lesions": 7470, "crossing_lesions": 0}
            valid = valid and counts == {"train": TRAIN_COUNTS, "val": VAL_COUNTS, "test": TEST_COUNTS}
            valid = valid and tuple(CLASS_NAMES) == CLASS_ORDER
            check("split_hash_counts_class_order", valid, f"sha256={digest}; integrity={integrity}; counts_match={counts == {'train': TRAIN_COUNTS, 'val': VAL_COUNTS, 'test': TEST_COUNTS}}; source_classes={tuple(CLASS_NAMES)}")
        except Exception as exc:
            check("split_hash_counts_class_order", False, str(exc))
    else:
        check("split_hash_counts_class_order", False, "split unavailable")

    exp3 = None
    try:
        exp3 = verify_exp3(root)
        check("experiment_3_checkpoint_and_saved_metrics", True,
              f"sha256={exp3['hash']}; selected_epoch=6; val_loss={exp3['validation']['loss']}; mel_f1={exp3['validation']['per_class']['mel']['f1']}; macro_f1={exp3['validation']['macro_f1']}; nv_recall={exp3['validation']['per_class']['nv']['recall']}")
        audit = configuration_differences(exp3["config"])
        check("configuration_difference_audit", audit == [{"path": "optimizer.weight_decay", "experiment_3": 0.0, "experiment_4": WEIGHT_DECAY}], json.dumps(audit, sort_keys=True))
        plan = read_json(output / "experiment_plan.json")
        validate_plan(plan, exp3["config"])
        check("experiment_4_plan", True, "frozen settings, Experiment #3 provenance, and single-variable diff validated")
    except Exception as exc:
        check("experiment_3_checkpoint_and_saved_metrics", False, str(exc))
        check("configuration_difference_audit", False, "cannot audit without verified Experiment #3 config")
        check("experiment_4_plan", False, "cannot validate without verified Experiment #3 provenance")
        audit = []

    images = root / "data" / "processed" / "images"
    if split.is_file():
        with split.open("r", newline="", encoding="utf-8") as handle:
            image_ids = [row["image_id"] for row in csv.DictReader(handle)]
        missing_images = [image_id for image_id in image_ids if not (images / f"{image_id}.jpg").is_file()]
        check("processed_ham_images", images.is_dir() and not missing_images, f"missing_count={len(missing_images)}; examples={missing_images[:5]}")
    else:
        check("processed_ham_images", False, "cannot validate without frozen split")

    collisions = [name for name in GENERATED_OUTPUTS if (output / name).exists()]
    check("output_overwrite_protection", not collisions, f"existing_generated_outputs={collisions}")
    if require_t4:
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
        colab = importlib.util.find_spec("google.colab") is not None
        t4_ok = platform.system() == "Linux" and colab and cuda and gpu is not None and "Tesla T4" in gpu
        check("colab_cuda_tesla_t4", t4_ok, f"platform={platform.system()}; google.colab={colab}; CUDA={cuda}; GPU={gpu}")
    else:
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
        report["checks"]["colab_cuda_tesla_t4"] = {"status": "NOT_REQUIRED", "details": f"preflight-only; CUDA={cuda}; GPU={gpu}"}

    report["configuration_differences"] = audit
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    report["training_eligible"] = report["status"] == "PASS" and require_t4
    report["ph2_accessed"] = False
    return report


def make_sampler(dataset) -> WeightedRandomSampler:
    labels = [row["dx"] for row in dataset.rows]
    if len(labels) != 8015 or dict(Counter(labels)) != TRAIN_COUNTS:
        raise ValueError("Sampler labels must be exactly the frozen TRAIN partition")
    weights = torch.as_tensor([MEL_WEIGHT if label == "mel" else NON_MEL_WEIGHT for label in labels], dtype=torch.double)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, *, training: bool, optimizer=None, scaler=None):
    model.train(training)
    matrix = torch.zeros((7, 7), dtype=torch.long)
    loss_sum = 0.0
    sample_count = 0
    grad_context = torch.enable_grad() if training else torch.inference_mode()
    with grad_context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
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
            for target, prediction in zip(labels.tolist(), predictions.detach().cpu().tolist()):
                matrix[int(target)][int(prediction)] += 1
            count = labels.size(0)
            loss_sum += float(loss.detach().float().cpu()) * count
            sample_count += count
    if not sample_count:
        raise ValueError("Empty loader")
    return loss_sum / sample_count, metrics_from_matrix(matrix.tolist())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    matrix = torch.zeros((7, 7), dtype=torch.long)
    rows = []
    loss_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = focal_loss(logits, labels.to(device), alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            for target, prediction in zip(labels.tolist(), predictions.tolist()):
                matrix[int(target)][int(prediction)] += 1
            start = sample_count
            for source, target, prediction, probability in zip(
                loader.dataset.rows[start:start + len(labels)], labels.tolist(), predictions.tolist(), probabilities.tolist()
            ):
                rows.append({"image_id": source["image_id"], "true_label": CLASS_ORDER[target],
                             "predicted_label": CLASS_ORDER[prediction], "correct": target == prediction,
                             "probabilities": probability})
            count = labels.size(0)
            loss_sum += float(loss.detach().float().cpu()) * count
            sample_count += count
    metrics = metrics_from_matrix(matrix.tolist())
    metrics["loss"] = loss_sum / sample_count
    return metrics, rows


def write_predictions(path: Path, rows: Sequence[dict]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=("image_id", "true_label", "predicted_label", "correct", "probabilities"))
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


def save_checkpoint(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_experiment(root: Path, preflight_report: dict, workers: int) -> None:
    output = root / "experiments" / EXPERIMENT_NAME
    set_seed(SEED)
    device = torch.device("cuda")
    model, train_set, val_set, test_set = build_run_components(root, SPLIT_NAME)
    train_sampler = make_sampler(train_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=False,
                              num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)
    model.to(device)
    optimizer = Adamax(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    exp3_config = read_json(root / "experiments" / "efficientnetv2s_controlled_exp3" / "config.json")
    differences = configuration_differences(exp3_config)
    if differences != [{"path": "optimizer.weight_decay", "experiment_3": 0.0, "experiment_4": WEIGHT_DECAY}]:
        raise RuntimeError(f"Unexpected experimental differences: {differences}")
    optimizer_defaults = {"name": "Adamax", "learning_rate": LEARNING_RATE, "betas": [0.9, 0.999],
                          "eps": 1e-8, "weight_decay": WEIGHT_DECAY}
    config = {
        "experiment": EXPERIMENT_NAME,
        "model": exp3_config["model"],
        "pretrained_weight_source": exp3_config["pretrained_weight_source"],
        "initialization": "fresh pretrained initialization; no baseline or prior experiment checkpoint loaded",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": optimizer_defaults,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "loss": exp3_config["loss"],
        "scheduler": exp3_config["scheduler"],
        "seed": SEED,
        "sampler": exp3_config["sampler"],
        "split_csv": SPLIT_NAME,
        "split_sha256": SPLIT_SHA256,
        "classes": list(CLASS_ORDER),
        "checkpoint_selection_metric": "validation_loss_minimize",
        "success_criteria": CRITERIA,
        "configuration_differences_from_exp3": differences,
        "preflight": preflight_report,
        "runtime": {"python": sys.version, "torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0), "platform": platform.platform()},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "ph2_accessed": False,
    }
    write_json(output / "config.json", config)
    history, best_loss = [], math.inf
    checkpoint_path = output / "best_checkpoint.pt"
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, device, training=True, optimizer=optimizer, scaler=scaler)
        val_loss, val_metrics = run_epoch(model, val_loader, device, training=False)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "train": train_metrics,
                        "val_loss": val_loss, "validation": val_metrics,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "sampler_seed": SEED, "sampler_num_samples": len(train_set)})
        write_json(output / "training_history.json", {"epochs": history}, replace=True)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(checkpoint_path, {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                              "epoch": epoch, "val_loss": val_loss, "config": config,
                                              "class_order": list(CLASS_ORDER)})
        print(f"epoch={epoch:02d}/{EPOCHS} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"val_mel_f1={val_metrics['per_class']['mel']['f1']:.6f}")

    digest = sha256_file(checkpoint_path)
    checkpoint = safe_load_checkpoint(checkpoint_path)
    selected = int(checkpoint["epoch"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    val_metrics, val_rows = evaluate(model, val_loader, device)
    test_metrics, test_rows = evaluate(model, test_loader, device)
    for metrics in (val_metrics, test_metrics):
        metrics.update({"selected_epoch": selected, "checkpoint_sha256": digest,
                        "checkpoint_selection_metric": "validation_loss_minimize"})
    val_metrics["loss"] = float(checkpoint["val_loss"])
    test_metrics["evaluation_partition"] = "frozen HAM10000 test after checkpoint freeze; evaluation only"
    write_json(output / "validation_metrics.json", val_metrics)
    write_predictions(output / "validation_predictions.csv", val_rows)
    write_json(output / "test_metrics.json", test_metrics)
    write_predictions(output / "test_predictions.csv", test_rows)
    config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "config.json", config, replace=True)
    finalize_existing(root)


def load_run_metrics(root: Path, directory_name: str, expected_hash: str, expected_epoch: int) -> tuple[dict, dict]:
    directory = root / "experiments" / directory_name
    digest = sha256_file(directory / "best_checkpoint.pt")
    if digest != expected_hash:
        raise ValueError(f"Checkpoint hash mismatch for {directory_name}: {digest}")
    history = read_json(directory / "training_history.json").get("epochs", [])
    best = min(history, key=lambda row: float(row["val_loss"]))
    checkpoint = safe_load_checkpoint(directory / "best_checkpoint.pt")
    if int(best["epoch"]) != expected_epoch or int(checkpoint.get("epoch", -1)) != expected_epoch:
        raise ValueError(f"Selected epoch mismatch for {directory_name}")
    if directory_name == "efficientnetv2s_leakage_aware":
        validation = metrics_from_matrix(best["validation"]["confusion_matrix"])
        validation["loss"] = float(best["val_loss"])
    else:
        validation = read_json(directory / "validation_metrics.json")
        pred_metrics, pred_rows = metrics_from_predictions(directory / "validation_predictions.csv")
        assert_metrics(validation, pred_metrics, f"{directory_name} validation")
        if int(validation.get("selected_epoch", -1)) != expected_epoch:
            raise ValueError(f"Validation epoch mismatch for {directory_name}")
        assert_prediction_identity(pred_rows, split_identity(root / "data/splits" / SPLIT_NAME, "val"), directory_name)
    validation.update({"selected_epoch": expected_epoch, "checkpoint_sha256": digest})
    if directory_name == "efficientnetv2s_leakage_aware":
        summary = read_json(root / "experiments/uncertainty/uncertainty_summary.json")
        if summary.get("checkpoint_sha256") != digest:
            raise ValueError("Baseline saved uncertainty predictions have mismatched checkpoint")
        test, test_rows = metrics_from_predictions(root / "experiments/uncertainty/predictions.csv")
        assert_metrics(summary, test, "baseline matched test")
        assert_prediction_identity(test_rows, split_identity(root / "data/splits" / SPLIT_NAME, "test"), "baseline test")
    else:
        test = read_json(directory / "test_metrics.json")
        test_pred, test_rows = metrics_from_predictions(directory / "test_predictions.csv")
        assert_metrics(test, test_pred, f"{directory_name} test")
        if test.get("checkpoint_sha256") != digest or int(test.get("selected_epoch", -1)) != expected_epoch:
            raise ValueError(f"Test checkpoint mismatch for {directory_name}")
        assert_prediction_identity(test_rows, split_identity(root / "data/splits" / SPLIT_NAME, "test"), directory_name)
    test.update({"selected_epoch": expected_epoch, "checkpoint_sha256": digest})
    return validation, test


def finalization_data(root: Path) -> tuple[dict, dict, str]:
    output = root / "experiments" / EXPERIMENT_NAME
    split = root / "data" / "splits" / SPLIT_NAME
    config = read_json(output / "config.json")
    history = read_json(output / "training_history.json").get("epochs", [])
    if len(history) != EPOCHS:
        raise ValueError(f"Expected {EPOCHS} Experiment #4 epochs; found {len(history)}")
    best = min(history, key=lambda row: float(row["val_loss"]))
    checkpoint_hash = sha256_file(output / "best_checkpoint.pt")
    checkpoint = safe_load_checkpoint(output / "best_checkpoint.pt")
    selected = int(checkpoint.get("epoch", -1))
    if selected != int(best["epoch"]):
        raise ValueError("Checkpoint is not the minimum-validation-loss epoch")
    if checkpoint.get("class_order") != list(CLASS_ORDER):
        raise ValueError("Checkpoint class order mismatch")
    if config.get("checkpoint_selection_metric") != "validation_loss_minimize":
        raise ValueError("Checkpoint selection rule changed")
    if config.get("optimizer", {}).get("weight_decay") != WEIGHT_DECAY:
        raise ValueError("Experiment #4 weight decay does not match approved value")
    val, test = read_json(output / "validation_metrics.json"), read_json(output / "test_metrics.json")
    for metrics in (val, test):
        if metrics.get("checkpoint_sha256") != checkpoint_hash or int(metrics.get("selected_epoch", -1)) != selected:
            raise ValueError("Experiment #4 saved metric checkpoint metadata mismatch")
    val_derived, val_rows = metrics_from_predictions(output / "validation_predictions.csv")
    test_derived, test_rows = metrics_from_predictions(output / "test_predictions.csv")
    assert_metrics(val, val_derived, "Experiment #4 validation")
    assert_metrics(test, test_derived, "Experiment #4 test")
    assert_prediction_identity(val_rows, split_identity(split, "val"), "Experiment #4 validation")
    assert_prediction_identity(test_rows, split_identity(split, "test"), "Experiment #4 test")
    if not math.isclose(val["loss"], float(best["val_loss"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Saved validation loss differs from the selected history minimum")

    runs = {}
    for name, folder, digest, epoch in (
        ("baseline", "efficientnetv2s_leakage_aware", BASELINE_SHA256, 6),
        ("experiment_1", "efficientnetv2s_controlled_exp1", EXP1_SHA256, 5),
        ("experiment_2", "efficientnetv2s_controlled_exp2", EXP2_SHA256, 8),
        ("experiment_3", "efficientnetv2s_controlled_exp3", EXP3_SHA256, 6),
    ):
        runs[name] = load_run_metrics(root, folder, digest, epoch)
    runs["experiment_4"] = (val, test)
    validation_runs = {name: pair[0] for name, pair in runs.items()}
    test_runs = {name: pair[1] for name, pair in runs.items()}

    limits = CRITERIA
    values = {
        "melanoma_f1": val["per_class"]["mel"]["f1"],
        "validation_macro_f1": val["macro_f1"],
        "nevus_recall": val["per_class"]["nv"]["recall"],
    }
    flags = {
        "melanoma_f1": values["melanoma_f1"] >= limits["melanoma_f1_minimum"],
        "validation_macro_f1": values["validation_macro_f1"] >= limits["validation_macro_f1_minimum"],
        "nevus_recall": values["nevus_recall"] >= limits["nevus_recall_minimum"],
    }
    comparison = {
        "checkpoint_sha256": checkpoint_hash,
        "selected_epoch": selected,
        "selection_metric": "validation_loss_minimize",
        "validation_success_criteria_met": all(flags.values()),
        "criterion_results": flags,
        "validation_values": values,
        "success_thresholds": limits,
        "configuration_differences_from_exp3": config["configuration_differences_from_exp3"],
        "validation": validation_runs,
        "test_evaluation_only": test_runs,
        "test_selection_or_tuning_used": False,
        "ph2_accessed": False,
        "baseline_test_provenance": {
            "source": "hash-verified experiments/uncertainty predictions",
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
        "experiment_1_checkpoint_sha256": EXP1_SHA256,
        "experiment_2_checkpoint_sha256": EXP2_SHA256,
        "experiment_3_checkpoint_sha256": EXP3_SHA256,
        "split_sha256": SPLIT_SHA256,
        "new_checkpoint_sha256": checkpoint_hash,
        "training_epochs": len(history),
        "selected_epoch": selected,
        "selection_metric": "validation_loss_minimize",
        "selection_value": float(best["val_loss"]),
        "validation_success_criteria_met": all(flags.values()),
        "criterion_results": flags,
        "optimizer": config["optimizer"],
        "sampler": config["sampler"],
        "ph2_accessed": False,
        "test_results_evaluation_only": True,
        "runtime": config.get("runtime"),
    }
    return comparison, manifest, make_report(comparison, manifest)


def make_report(comparison: dict, manifest: dict) -> str:
    outcome = "PASS" if comparison["validation_success_criteria_met"] else "FAIL"
    rows = []
    for title, key, threshold in (
        ("Melanoma F1", "melanoma_f1", "melanoma_f1_minimum"),
        ("Validation macro-F1", "validation_macro_f1", "validation_macro_f1_minimum"),
        ("Nevus recall", "nevus_recall", "nevus_recall_minimum"),
    ):
        value = comparison["validation_values"][key]
        result = "PASS" if comparison["criterion_results"][key] else "FAIL"
        rows.append(f"| {title} | {value:.12f} | {comparison['success_thresholds'][threshold]:.12f} | {result} |")
    table = []
    for name, metrics in comparison["validation"].items():
        mel = metrics["per_class"]["mel"]
        table.append(f"| {name} epoch {metrics['selected_epoch']} | {mel['precision']:.6f} | {mel['recall']:.6f} | {mel['f1']:.6f} | {metrics['per_class']['nv']['recall']:.6f} | {metrics['macro_f1']:.6f} | {metrics['accuracy']:.6f} |")
    return f"""# Controlled Experiment #4 Final Report

**Outcome: {outcome}**

- Selected epoch: {manifest['selected_epoch']} by minimum validation loss
- Checkpoint SHA256: `{manifest['new_checkpoint_sha256']}`
- Adamax weight decay: {manifest['optimizer']['weight_decay']}
- Experiment #3 Adamax weight decay: 0.0
- PH2 accessed: NO

## Frozen Validation Criteria

| Criterion | Experiment #4 | Required minimum | Result |
|---|---:|---:|---|
{chr(10).join(rows)}

## Checkpoint-Matched Validation Comparison

| Run | Epoch | Mel precision | Mel recall | Mel F1 | Nevus recall | Macro-F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Test results are EVALUATION ONLY. They do not inform checkpoint selection or tuning. Baseline test evidence is from checkpoint-hash-verified saved uncertainty predictions; final-epoch baseline test metrics are not substituted for selected-epoch evidence.
"""


def finalize_existing(root: Path) -> None:
    comparison, manifest, report = finalization_data(root)
    output = root / "experiments" / EXPERIMENT_NAME
    artifacts = {
        "comparison_metrics.json": json_text(comparison),
        "experiment_manifest.json": json_text(manifest),
        "controlled_exp4_final_report.md": report,
    }
    differences = [name for name, content in artifacts.items() if (output / name).exists() and (output / name).read_text(encoding="utf-8") != content]
    if differences:
        raise FileExistsError(f"Refusing to overwrite differing final reports: {differences}")
    for name, content in artifacts.items():
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
    result = preflight(root, require_t4=args.train)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        return 2
    if args.train:
        if not result["training_eligible"]:
            raise RuntimeError("Training requires an approved Colab Tesla T4 runtime")
        train_experiment(root, result, args.workers)
    else:
        print("Preflight only; no training or inference was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
