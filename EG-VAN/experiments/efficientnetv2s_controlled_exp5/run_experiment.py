"""Run Controlled Experiment #5 with a validation-constrained checkpoint rule.

Training requires --train and a passing Google Colab Tesla T4 preflight. This
runner never reads PH2. Test evaluation occurs only after eligible checkpoint
selection. If no eligible epoch occurs, no checkpoint is claimed and test
inference is skipped.
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

EXPERIMENT_NAME = "efficientnetv2s_controlled_exp5"
SPLIT_NAME = "split_leakage_aware.csv"
CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
BASELINE_SHA256 = "f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848"
EXP1_SHA256 = "478bb1aa9c48897accceb800a103e7ee76c1381dc6514c7355996babbfaa902f"
EXP2_SHA256 = "9dc38968e91a98693de5e6a8c4ee720a69ef80e275887780a8d714077c6ea93f"
EXP3_SHA256 = "8615a9f5afad48c17c21b559aeab01e5e506fa3cbce3f544eed14b47b53377b9"
EXP4_SHA256 = "c09e2d6b77891a1fcf8b9e7b8aa2f702a4407c84c5bd8798ae189f1f15357b73"
SPLIT_SHA256 = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
TRAIN_COUNTS = {"akiec": 257, "bcc": 398, "bkl": 891, "df": 95, "mel": 899, "nv": 5366, "vasc": 109}
VAL_COUNTS = {"akiec": 30, "bcc": 58, "bkl": 104, "df": 9, "mel": 107, "nv": 663, "vasc": 15}
TEST_COUNTS = {"akiec": 40, "bcc": 58, "bkl": 104, "df": 11, "mel": 107, "nv": 676, "vasc": 18}
MEL_WEIGHT = 1.5630495442733532
NON_MEL_WEIGHT = 1.0
RULE_OLD = "validation_loss_minimize"
RULE_NEW = "validation_threshold_constrained_min_loss"
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
    "eligible_epochs.json",
    "comparison_metrics.json",
    "experiment_manifest.json",
    "controlled_exp5_final_report.md",
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
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"Unexpected checkpoint structure: {path}")
    return value


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
            raise FileExistsError(f"Refusing to overwrite generated output: {path}")
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
                raise ValueError(f"Unexpected frozen split value: {row}")
            counts[row["split"]][row["dx"]] += 1
    return {part: {label: int(counts[part][label]) for label in CLASS_ORDER} for part in counts}


def metrics_from_matrix(matrix: Sequence[Sequence[int]]) -> dict:
    matrix = [[int(value) for value in row] for row in matrix]
    if len(matrix) != 7 or any(len(row) != 7 for row in matrix):
        raise ValueError("Confusion matrix must be 7x7 in frozen class order")
    total = sum(map(sum, matrix))
    if total <= 0:
        raise ValueError("Empty confusion matrix")
    per_class, f1_values, recalls = {}, [], []
    correct = 0
    for index, name in enumerate(CLASS_ORDER):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": support, "predicted_count": predicted, "precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
        recalls.append(recall)
        correct += tp
    return {
        "sample_count": total,
        "class_order": list(CLASS_ORDER),
        "accuracy": correct / total,
        "macro_f1": sum(f1_values) / 7,
        "balanced_accuracy": sum(recalls) / 7,
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
                raise ValueError(f"Incorrect correct flag for {image_id}")
            matrix[target][prediction] += 1
            rows.append(row)
    return metrics_from_matrix(matrix), rows


def assert_metrics(saved: dict, derived: dict, source: str) -> None:
    if saved.get("class_order") not in (None, list(CLASS_ORDER)):
        raise ValueError(f"Class order mismatch: {source}")
    if saved.get("sample_count") is not None and int(saved["sample_count"]) != derived["sample_count"]:
        raise ValueError(f"Sample count mismatch: {source}")
    if "confusion_matrix" in saved and saved["confusion_matrix"] != derived["confusion_matrix"]:
        raise ValueError(f"Confusion matrix mismatch: {source}")
    for metric in ("accuracy", "macro_f1"):
        if metric in saved and not math.isclose(float(saved[metric]), derived[metric], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{source}.{metric} mismatch")
    for label in CLASS_ORDER:
        for metric in ("precision", "recall", "f1"):
            actual = saved.get("per_class", {}).get(label, {}).get(metric)
            if actual is not None and not math.isclose(float(actual), derived["per_class"][label][metric], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{source}.{label}.{metric} mismatch")


def split_identity(path: Path, part: str) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["image_id"]: row["dx"] for row in csv.DictReader(handle) if row["split"] == part}


def assert_prediction_identity(rows: Sequence[dict], expected: dict[str, str], source: str) -> None:
    actual = {row["image_id"]: row["true_label"] for row in rows}
    if actual != expected:
        raise ValueError(f"Prediction IDs/labels differ from frozen split: {source}")


def normalized_training_config(config: dict) -> dict:
    return {
        "model": config["model"],
        "pretrained_weight_source": config["pretrained_weight_source"],
        "initialization": config["initialization"],
        "image_size": config["image_size"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "optimizer": config["optimizer"],
        "focal_alpha": config["focal_alpha"],
        "focal_gamma": config["focal_gamma"],
        "loss": config["loss"],
        "scheduler": config["scheduler"],
        "seed": config["seed"],
        "sampler": config["sampler"],
        "split_csv": config["split_csv"],
        "split_sha256": config["split_sha256"],
        "classes": config["classes"],
        "success_criteria": config["success_criteria"],
        "checkpoint_selection_metric": config["checkpoint_selection_metric"],
    }


def leaf_diff(left: dict, right: dict, prefix: str = "") -> list[dict]:
    differences = []
    for key in sorted(set(left) | set(right)):
        path = f"{prefix}.{key}" if prefix else key
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            differences.extend(leaf_diff(a, b, path))
        elif a != b:
            differences.append({"path": path, "experiment_4": a, "experiment_5": b})
    return differences


def verify_exp4(root: Path) -> dict:
    directory = root / "experiments" / "efficientnetv2s_controlled_exp4"
    required = (
        "best_checkpoint.pt", "config.json", "training_history.json", "validation_metrics.json",
        "validation_predictions.csv", "test_metrics.json", "test_predictions.csv",
        "comparison_metrics.json", "experiment_manifest.json", "controlled_exp4_final_report.md",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Experiment #4 artifacts missing: {missing}")
    digest = sha256_file(directory / "best_checkpoint.pt")
    if digest != EXP4_SHA256:
        raise ValueError(f"Experiment #4 checkpoint SHA256 mismatch: {digest}")
    config = read_json(directory / "config.json")
    manifest = read_json(directory / "experiment_manifest.json")
    comparison = read_json(directory / "comparison_metrics.json")
    validation = read_json(directory / "validation_metrics.json")
    test = read_json(directory / "test_metrics.json")
    history = read_json(directory / "training_history.json").get("epochs", [])
    checkpoint = safe_load_checkpoint(directory / "best_checkpoint.pt")
    best = min(history, key=lambda epoch: float(epoch["val_loss"]))
    if len(history) != EPOCHS or int(best["epoch"]) != 6 or int(checkpoint.get("epoch", -1)) != 6:
        raise ValueError("Experiment #4 selected epoch/history mismatch")
    if not math.isclose(float(best["val_loss"]), 0.06999672452104987, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Experiment #4 minimum validation loss differs from verified value")
    if not math.isclose(float(checkpoint.get("val_loss", math.inf)), float(best["val_loss"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Experiment #4 checkpoint loss differs from history minimum")
    if config.get("checkpoint_selection_metric") != "validation_loss_minimize" or config.get("optimizer", {}).get("weight_decay") != 0.0001:
        raise ValueError("Experiment #4 selector or optimizer config mismatch")
    sampler = config.get("sampler", {})
    if (
        sampler.get("mel_weight") != MEL_WEIGHT
        or sampler.get("non_melanoma_weight") != NON_MEL_WEIGHT
        or sampler.get("seed") != SEED
        or sampler.get("replacement") is not True
        or sampler.get("num_samples") != 8015
        or sampler.get("labels_from") != "frozen TRAIN rows only"
    ):
        raise ValueError("Experiment #4 sampler configuration differs from the frozen values")
    if manifest.get("new_checkpoint_sha256") != digest or int(manifest.get("selected_epoch", -1)) != 6:
        raise ValueError("Experiment #4 manifest mismatch")
    for metrics in (validation, test):
        if metrics.get("checkpoint_sha256") != digest or int(metrics.get("selected_epoch", -1)) != 6:
            raise ValueError("Experiment #4 metrics do not match selected checkpoint")
    val_derived, val_rows = metrics_from_predictions(directory / "validation_predictions.csv")
    test_derived, test_rows = metrics_from_predictions(directory / "test_predictions.csv")
    assert_metrics(validation, val_derived, "Experiment #4 validation")
    assert_metrics(test, test_derived, "Experiment #4 test")
    if not math.isclose(validation["loss"], float(best["val_loss"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Experiment #4 validation loss mismatch")
    if comparison.get("checkpoint_sha256") != digest or int(comparison.get("selected_epoch", -1)) != 6:
        raise ValueError("Experiment #4 comparison selected checkpoint mismatch")
    if comparison.get("validation_success_criteria_met") is not False:
        raise ValueError("Experiment #4 expected failed validation criteria")
    report = (directory / "controlled_exp4_final_report.md").read_text(encoding="utf-8")
    if not all(token in report for token in (digest, "Selected epoch: 6", "Outcome: FAIL", "0.504587155963", "0.663900640131")):
        raise ValueError("Experiment #4 final report disagrees with saved metrics")
    return {"directory": directory, "hash": digest, "config": config, "manifest": manifest,
            "comparison": comparison, "validation": validation, "test": test,
            "history": history, "checkpoint": checkpoint}


def validate_plan(plan: dict, exp4_config: dict) -> list[dict]:
    if plan.get("status") != "PLANNED_NOT_RUN":
        raise ValueError("Experiment #5 plan must remain PLANNED_NOT_RUN")
    exp4_normalized = normalized_training_config(exp4_config)
    exp5_normalized = dict(exp4_normalized)
    exp5_normalized["checkpoint_selection_metric"] = RULE_NEW
    differences = leaf_diff(exp4_normalized, exp5_normalized)
    expected_difference = [{"path": "checkpoint_selection_metric", "experiment_4": RULE_OLD, "experiment_5": RULE_NEW}]
    if differences != expected_difference:
        raise ValueError(f"Only checkpoint selection may differ: {differences}")
    if plan.get("configuration_difference_audit") != [
        {"path": "checkpoint_selection_metric", "experiment_4": RULE_OLD, "experiment_5": RULE_NEW, "intentional": True}
    ]:
        raise ValueError("Plan machine-readable config audit differs from the single approved change")
    frozen = plan.get("frozen_configuration", {})
    sampler = frozen.get("sampler", {})
    if frozen.get("split_sha256") != SPLIT_SHA256 or frozen.get("split_csv") != f"data/splits/{SPLIT_NAME}":
        raise ValueError("Plan frozen split path/hash mismatch")
    if frozen.get("class_order") != list(CLASS_ORDER):
        raise ValueError("Plan class order mismatch")
    if not math.isclose(float(sampler.get("melanoma_weight", -1)), MEL_WEIGHT, rel_tol=1e-14):
        raise ValueError("Plan sampler melanoma weight mismatch")
    if sampler.get("non_melanoma_weight") != NON_MEL_WEIGHT or sampler.get("seed") != SEED or sampler.get("replacement") is not True or sampler.get("num_samples") != 8015:
        raise ValueError("Plan sampler differs from frozen Experiment #4 sampler")
    if plan.get("success_criteria") != {
        "all_required": True,
        "melanoma_f1": {"minimum": CRITERIA["melanoma_f1_minimum"]},
        "validation_macro_f1": {"minimum": CRITERIA["validation_macro_f1_minimum"]},
        "nevus_recall": {"minimum": CRITERIA["nevus_recall_minimum"]},
        "evaluation_partition": "HAM10000 validation only",
        "all_required_at_selected_eligible_checkpoint": True,
        "must_not_be_changed_after_results": True,
    }:
        raise ValueError("Frozen success criteria changed")
    provenance = plan.get("experiment_4_provenance", {})
    if provenance.get("checkpoint_sha256") != EXP4_SHA256 or provenance.get("selected_epoch") != 6:
        raise ValueError("Experiment #4 provenance mismatch in plan")
    if plan.get("ph2_policy", {}).get("allowed_during_development") is not False:
        raise ValueError("PH2 must remain prohibited")
    return differences


def prior_run_metrics(root: Path, name: str, directory_name: str, expected_hash: str, epoch: int) -> tuple[dict, dict]:
    directory = root / "experiments" / directory_name
    digest = sha256_file(directory / "best_checkpoint.pt")
    if digest != expected_hash:
        raise ValueError(f"Prior checkpoint hash mismatch: {directory_name}")
    history = read_json(directory / "training_history.json").get("epochs", [])
    best = min(history, key=lambda row: float(row["val_loss"]))
    checkpoint = safe_load_checkpoint(directory / "best_checkpoint.pt")
    if int(best["epoch"]) != epoch or int(checkpoint.get("epoch", -1)) != epoch:
        raise ValueError(f"Prior selected epoch mismatch: {directory_name}")
    if directory_name == "efficientnetv2s_leakage_aware":
        val = metrics_from_matrix(best["validation"]["confusion_matrix"])
        val["loss"] = float(best["val_loss"])
        unc = root / "experiments" / "uncertainty"
        summary = read_json(unc / "uncertainty_summary.json")
        if summary.get("checkpoint_sha256") != digest:
            raise ValueError("Baseline checkpoint-matched test hash mismatch")
        test, test_rows = metrics_from_predictions(unc / "predictions.csv")
        assert_metrics(summary, test, "baseline checkpoint-matched test")
        assert_prediction_identity(test_rows, split_identity(root / "data/splits" / SPLIT_NAME, "test"), "baseline test")
    else:
        val = read_json(directory / "validation_metrics.json")
        test = read_json(directory / "test_metrics.json")
        val_d, val_rows = metrics_from_predictions(directory / "validation_predictions.csv")
        test_d, test_rows = metrics_from_predictions(directory / "test_predictions.csv")
        assert_metrics(val, val_d, f"{name} validation")
        assert_metrics(test, test_d, f"{name} test")
        assert_prediction_identity(val_rows, split_identity(root / "data/splits" / SPLIT_NAME, "val"), f"{name} validation")
        assert_prediction_identity(test_rows, split_identity(root / "data/splits" / SPLIT_NAME, "test"), f"{name} test")
    val.update({"selected_epoch": epoch, "checkpoint_sha256": digest})
    test.update({"selected_epoch": epoch, "checkpoint_sha256": digest})
    return val, test


def preflight(project_root: Path, *, require_t4: bool) -> dict:
    root = project_root.resolve()
    output = root / "experiments" / EXPERIMENT_NAME
    split = root / "data" / "splits" / SPLIT_NAME
    report = {"project_root": str(root), "experiment_directory": str(output), "checks": {}, "errors": []}

    def record(name: str, passed: bool, details: str) -> None:
        report["checks"][name] = {"status": "PASS" if passed else "FAIL", "details": details}
        if not passed:
            report["errors"].append(f"{name}: {details}")

    record("project_root", root.is_dir(), "project root exists")
    record("split_file", split.is_file(), str(split))
    record("experiment_5_directory", output.is_dir(), "dedicated Experiment #5 directory exists")
    record("runner_location", output.resolve() == Path(__file__).resolve().parent, "runner is in Experiment #5 directory")
    if not root.is_dir() or not output.is_dir() or not split.is_file():
        report.update(status="FAIL", training_eligible=False, ph2_accessed=False)
        return report

    baseline = root / "experiments" / "efficientnetv2s_leakage_aware"
    missing_baseline = [name for name in ("best_checkpoint.pt", "config.json", "training_history.json") if not (baseline / name).is_file()]
    base_hash = sha256_file(baseline / "best_checkpoint.pt") if not missing_baseline else None
    record("baseline_artifacts_and_checkpoint", not missing_baseline and base_hash == BASELINE_SHA256,
           f"missing={missing_baseline}; hash={base_hash}")
    if not missing_baseline and base_hash == BASELINE_SHA256:
        try:
            baseline_checkpoint = safe_load_checkpoint(baseline / "best_checkpoint.pt")
            baseline_history = read_json(baseline / "training_history.json").get("epochs", [])
            baseline_best = min(baseline_history, key=lambda row: float(row["val_loss"]))
            baseline_config = read_json(baseline / "config.json")
            baseline_epoch_ok = int(baseline_checkpoint.get("epoch", -1)) == 6 and int(baseline_best["epoch"]) == 6
            record("baseline_selected_epoch", baseline_epoch_ok,
                   f"checkpoint_epoch={baseline_checkpoint.get('epoch')}; history_min_loss_epoch={baseline_best['epoch']}")
            frozen_baseline = {
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
            mismatches = {key: baseline_config.get(key) for key, value in frozen_baseline.items()
                          if baseline_config.get(key) != value}
            record("baseline_frozen_configuration", not mismatches, f"mismatches={mismatches}")
        except Exception as exc:
            record("baseline_selected_epoch_and_configuration", False, str(exc))
    else:
        record("baseline_selected_epoch", False, "baseline files/hash failed")
        record("baseline_frozen_configuration", False, "baseline files/hash failed")

    try:
        split_hash = sha256_file(split)
        integrity = validate_split_integrity(split, require_lesion_isolation=True)
        counts = split_counts(split)
        valid_split = split_hash == SPLIT_SHA256 and integrity == {"rows": 10015, "unique_lesions": 7470, "crossing_lesions": 0}
        valid_split &= counts == {"train": TRAIN_COUNTS, "val": VAL_COUNTS, "test": TEST_COUNTS}
        valid_split &= tuple(CLASS_NAMES) == CLASS_ORDER
        record("frozen_split_hash_integrity_counts_class_order", valid_split,
               f"sha256={split_hash}; integrity={integrity}; counts_match={counts == {'train': TRAIN_COUNTS, 'val': VAL_COUNTS, 'test': TEST_COUNTS}}; class_order={tuple(CLASS_NAMES)}")
    except Exception as exc:
        record("frozen_split_hash_integrity_counts_class_order", False, str(exc))

    try:
        exp4 = verify_exp4(root)
        plan = read_json(output / "experiment_plan.json")
        differences = validate_plan(plan, exp4["config"])
        record("experiment_4_provenance_and_final_metrics", True,
               f"hash={exp4['hash']}; epoch=6; min_loss={exp4['validation']['loss']}; mel_f1={exp4['validation']['per_class']['mel']['f1']}; report verified")
        record("configuration_difference_audit", len(differences) == 1, json.dumps(differences, sort_keys=True))
        record("experiment_5_plan", True, "rule, criteria, provenance, and frozen settings verified")
    except Exception as exc:
        record("experiment_4_provenance_and_final_metrics", False, str(exc))
        record("configuration_difference_audit", False, "unable to audit without verified Experiment #4 config")
        record("experiment_5_plan", False, "plan/provenance verification failed")

    images = root / "data" / "processed" / "images"
    try:
        with split.open("r", newline="", encoding="utf-8") as handle:
            ids = [row["image_id"] for row in csv.DictReader(handle)]
        missing = [image_id for image_id in ids if not (images / f"{image_id}.jpg").is_file()]
        record("processed_ham_images", images.is_dir() and not missing, f"missing_count={len(missing)}")
    except Exception as exc:
        record("processed_ham_images", False, str(exc))
    collisions = [name for name in GENERATED_OUTPUTS if (output / name).exists()]
    record("exp5_output_overwrite_protection", not collisions, f"existing_generated_outputs={collisions}")

    if require_t4:
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
        colab = importlib.util.find_spec("google.colab") is not None
        allowed = platform.system() == "Linux" and colab and cuda and gpu is not None and "Tesla T4" in gpu
        record("google_colab_cuda_tesla_t4", allowed, f"platform={platform.system()}; colab={colab}; cuda={cuda}; gpu={gpu}")
    else:
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
        report["checks"]["google_colab_cuda_tesla_t4"] = {"status": "NOT_REQUIRED", "details": f"preflight-only; CUDA={cuda}; GPU={gpu}"}

    report["status"] = "PASS" if not report["errors"] else "FAIL"
    report["training_eligible"] = report["status"] == "PASS" and require_t4
    report["configuration_differences"] = [{"path": "checkpoint_selection_metric", "experiment_4": RULE_OLD, "experiment_5": RULE_NEW}]
    report["ph2_accessed"] = False
    return report


def eligibility(metrics: dict) -> dict:
    values = {
        "melanoma_precision": metrics["per_class"]["mel"]["precision"],
        "melanoma_recall": metrics["per_class"]["mel"]["recall"],
        "melanoma_f1": metrics["per_class"]["mel"]["f1"],
        "macro_f1": metrics["macro_f1"],
        "nevus_recall": metrics["per_class"]["nv"]["recall"],
    }
    flags = {
        "melanoma_f1": values["melanoma_f1"] >= CRITERIA["melanoma_f1_minimum"],
        "macro_f1": values["macro_f1"] >= CRITERIA["validation_macro_f1_minimum"],
        "nevus_recall": values["nevus_recall"] >= CRITERIA["nevus_recall_minimum"],
    }
    return {"eligible": all(flags.values()), "criterion_results": flags, **values}


def make_sampler(dataset) -> WeightedRandomSampler:
    labels = [row["dx"] for row in dataset.rows]
    if len(labels) != 8015 or dict(Counter(labels)) != TRAIN_COUNTS:
        raise ValueError("Training sampler must use exactly frozen TRAIN rows")
    weights = torch.as_tensor([MEL_WEIGHT if label == "mel" else NON_MEL_WEIGHT for label in labels], dtype=torch.double)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True, generator=generator)


def run_epoch(model, loader, device, *, training, optimizer=None, scaler=None):
    model.train(training)
    matrix = torch.zeros((7, 7), dtype=torch.long)
    loss_sum = 0.0
    sample_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
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
    if sample_count == 0:
        raise ValueError("Empty data loader")
    return loss_sum / sample_count, metrics_from_matrix(matrix.tolist())


def evaluate(model, loader, device):
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
            raise FileExistsError(f"Refusing to overwrite {path}")
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


def build_comparison(root: Path, exp5_val: dict | None, exp5_test: dict | None, selected_epoch: int | None, selected_hash: str | None, status: str) -> dict:
    runs = {}
    for name, folder, digest, epoch in (
        ("baseline", "efficientnetv2s_leakage_aware", BASELINE_SHA256, 6),
        ("experiment_2", "efficientnetv2s_controlled_exp2", EXP2_SHA256, 8),
        ("experiment_3", "efficientnetv2s_controlled_exp3", EXP3_SHA256, 6),
        ("experiment_4", "efficientnetv2s_controlled_exp4", EXP4_SHA256, 6),
    ):
        runs[name] = prior_run_metrics(root, name, folder, digest, epoch)
    runs["experiment_5"] = (exp5_val, exp5_test) if exp5_val is not None else None
    validation = {name: pair[0] if pair else {"status": status, "selected_epoch": None} for name, pair in runs.items()}
    tests = {name: pair[1] if pair else {"status": "NOT_RUN_NO_ELIGIBLE_CHECKPOINT", "evaluation_only": True} for name, pair in runs.items()}
    flags = eligibility(exp5_val) if exp5_val is not None else {
        "eligible": False,
        "criterion_results": {"melanoma_f1": False, "macro_f1": False, "nevus_recall": False},
    }
    return {
        "experiment_5_checkpoint_sha256": selected_hash,
        "experiment_5_selected_epoch": selected_epoch,
        "experiment_5_selection_metric": RULE_NEW,
        "experiment_5_status": status,
        "experiment_5_validation_success": bool(flags["eligible"]),
        "criterion_results": flags["criterion_results"],
        "success_thresholds": CRITERIA,
        "validation": validation,
        "test_evaluation_only": tests,
        "test_used_for_selection_or_tuning": False,
        "ph2_accessed": False,
        "baseline_test_source": "hash-verified uncertainty predictions; baseline test_metrics.json not substituted",
    }


def write_report(path: Path, comparison: dict, eligible_records: list[dict], result: str) -> None:
    if result == "FAIL_NO_ELIGIBLE_CHECKPOINT":
        content = """# Controlled Experiment #5 Final Report

**Outcome: FAIL_NO_ELIGIBLE_CHECKPOINT**

No epoch satisfied all three frozen validation criteria. No best checkpoint was claimed and test evaluation was not run. The full training history and eligible-epoch record are retained.

Test metrics remain EVALUATION ONLY and were not used for selection or tuning. PH2 was not accessed.
"""
        atomic_write(path, content)
        return
    selected = comparison["validation"]["experiment_5"]
    thresholds = comparison["success_thresholds"]
    rows = []
    for title, key, threshold in (
        ("Melanoma F1", "melanoma_f1", "melanoma_f1_minimum"),
        ("Validation macro-F1", "macro_f1", "validation_macro_f1_minimum"),
        ("Nevus recall", "nevus_recall", "nevus_recall_minimum"),
    ):
        value = selected["per_class"]["mel"]["f1"] if key == "melanoma_f1" else selected["macro_f1"] if key == "macro_f1" else selected["per_class"]["nv"]["recall"]
        rows.append(f"| {title} | {value:.12f} | {thresholds[threshold]:.12f} | {'PASS' if value >= thresholds[threshold] else 'FAIL'} |")
    content = f"""# Controlled Experiment #5 Final Report

**Outcome: {'PASS' if comparison['experiment_5_validation_success'] else 'FAIL'}**

- Checkpoint selection: validation-threshold-constrained minimum validation loss
- Selected epoch: {comparison['experiment_5_selected_epoch']}
- Checkpoint SHA256: `{comparison['experiment_5_checkpoint_sha256']}`
- Eligible epoch count: {len(eligible_records)}
- Test evaluation: after checkpoint selection; evaluation only
- PH2 accessed: NO

## Frozen Validation Criteria

| Criterion | Selected checkpoint | Threshold | Result |
|---|---:|---:|---|
{chr(10).join(rows)}

## Test Policy

Test results are EVALUATION ONLY — NOT USED FOR MODEL SELECTION. They were not used for checkpoint selection or tuning.
"""
    atomic_write(path, content)


def complete_run_outputs(root: Path, config: dict, history: list[dict], eligible_records: list[dict], preflight_report: dict) -> None:
    output = root / "experiments" / EXPERIMENT_NAME
    eligible_records.sort(key=lambda row: (float(row["validation_loss"]), int(row["epoch"])))
    selected_record = eligible_records[0] if eligible_records else None
    eligible_doc = {
        "selection_rule": RULE_NEW,
        "criteria": CRITERIA,
        "eligible_epochs": eligible_records,
        "selected_epoch": selected_record["epoch"] if selected_record else None,
        "selected_validation_loss": selected_record["validation_loss"] if selected_record else None,
        "failure_if_empty": "FAIL_NO_ELIGIBLE_CHECKPOINT; no checkpoint is claimed and test inference is skipped",
    }
    write_json(output / "eligible_epochs.json", eligible_doc, replace=True)
    write_json(output / "training_history.json", {"epochs": history}, replace=True)
    config["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "config.json", config, replace=True)

    if selected_record is None:
        status = "FAIL_NO_ELIGIBLE_CHECKPOINT"
        write_json(output / "validation_metrics.json", {
            "status": status, "selected_epoch": None, "selection_metric": RULE_NEW,
            "eligible_epoch_count": 0, "success_thresholds": CRITERIA,
            "explanation": "No epoch met all three frozen validation thresholds; no checkpoint selected.",
        })
        write_json(output / "test_metrics.json", {
            "status": "NOT_RUN_NO_ELIGIBLE_CHECKPOINT", "evaluation_only": True,
            "reason": "No eligible selected checkpoint; test evaluation was skipped.",
        })
        comparison = build_comparison(root, None, None, None, None, status)
        write_json(output / "comparison_metrics.json", comparison)
        manifest = {
            "status": status, "training_epochs": len(history), "eligible_epoch_count": 0,
            "selected_epoch": None, "selection_metric": RULE_NEW, "checkpoint_sha256": None,
            "criteria": CRITERIA, "ph2_accessed": False, "test_evaluation_run": False,
        }
        write_json(output / "experiment_manifest.json", manifest)
        write_report(output / "controlled_exp5_final_report.md", comparison, [], status)
        return

    checkpoint_path = output / "best_checkpoint.pt"
    selected_checkpoint = safe_load_checkpoint(checkpoint_path)
    selected_epoch = int(selected_checkpoint["epoch"])
    selected_hash = sha256_file(checkpoint_path)
    if selected_epoch != int(selected_record["epoch"]):
        raise ValueError("Saved eligible checkpoint epoch differs from the registered best eligible epoch")
    model, _, val_set, test_set = build_run_components(root, SPLIT_NAME)
    model.load_state_dict(selected_checkpoint["model_state"], strict=True)
    model.to(torch.device("cuda")).eval()
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    val_metrics, val_rows = evaluate(model, val_loader, torch.device("cuda"))
    val_metrics.update({"selected_epoch": selected_epoch, "checkpoint_sha256": selected_hash,
                        "checkpoint_selection_metric": RULE_NEW,
                        "selection_validation_loss": float(selected_checkpoint["val_loss"])})
    test_metrics, test_rows = evaluate(model, test_loader, torch.device("cuda"))
    test_metrics.update({"selected_epoch": selected_epoch, "checkpoint_sha256": selected_hash,
                         "checkpoint_selection_metric": RULE_NEW,
                         "evaluation_partition": "frozen HAM10000 test after selected checkpoint; evaluation only"})
    write_json(output / "validation_metrics.json", val_metrics)
    write_predictions(output / "validation_predictions.csv", val_rows)
    write_json(output / "test_metrics.json", test_metrics)
    write_predictions(output / "test_predictions.csv", test_rows)
    comparison = build_comparison(root, val_metrics, test_metrics, selected_epoch, selected_hash, "COMPLETE")
    comparison["experiment_5_validation_success"] = all(eligibility(val_metrics)["criterion_results"].values())
    write_json(output / "comparison_metrics.json", comparison)
    manifest = {
        "status": "COMPLETE", "training_epochs": len(history), "eligible_epoch_count": len(eligible_records),
        "eligible_epochs": [entry["epoch"] for entry in eligible_records],
        "selected_epoch": selected_epoch, "selected_validation_loss": float(selected_checkpoint["val_loss"]),
        "selection_metric": RULE_NEW, "checkpoint_sha256": selected_hash,
        "criteria": CRITERIA, "criterion_results": eligibility(val_metrics)["criterion_results"],
        "test_evaluation_only": True, "ph2_accessed": False, "runtime": config.get("runtime"),
    }
    write_json(output / "experiment_manifest.json", manifest)
    write_report(output / "controlled_exp5_final_report.md", comparison, eligible_records, "COMPLETE")


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
            raise FileExistsError(f"Refusing to overwrite {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_experiment(root: Path, preflight_report: dict, workers: int) -> None:
    output = root / "experiments" / EXPERIMENT_NAME
    set_seed(SEED)
    device = torch.device("cuda")
    model, train_set, val_set, _ = build_run_components(root, SPLIT_NAME)
    sampler = make_sampler(train_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False,
                              num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers, pin_memory=True)
    model.to(device)
    optimizer = Adamax(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    exp4_config = read_json(root / "experiments" / "efficientnetv2s_controlled_exp4" / "config.json")
    config = dict(exp4_config)
    config.update({
        "experiment": EXPERIMENT_NAME,
        "checkpoint_selection_metric": RULE_NEW,
        "checkpoint_selection_rule": {
            "eligible_if_all": CRITERIA,
            "select": "minimum validation loss among eligible epochs",
            "tie_break": "earlier epoch",
            "no_eligible_result": "FAIL_NO_ELIGIBLE_CHECKPOINT; skip test evaluation",
        },
        "configuration_differences_from_exp4": [{
            "path": "checkpoint_selection_metric", "experiment_4": RULE_OLD,
            "experiment_5": RULE_NEW,
        }],
        "preflight": preflight_report,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "ph2_accessed": False,
    })
    write_json(output / "config.json", config)
    history, eligible_records = [], []
    best_eligible_loss = math.inf
    checkpoint_path = output / "best_checkpoint.pt"
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, device, training=True,
                                              optimizer=optimizer, scaler=scaler)
        val_loss, val_metrics = run_epoch(model, val_loader, device, training=False)
        scheduler.step(val_loss)
        gate = eligibility(val_metrics)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train": train_metrics,
            "val_loss": val_loss,
            "validation": val_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "checkpoint_eligible": gate["eligible"],
            "criterion_results": gate["criterion_results"],
            "sampler_seed": SEED,
            "sampler_num_samples": len(train_set),
        }
        history.append(record)
        if gate["eligible"]:
            eligible = {
                "epoch": epoch,
                "validation_loss": val_loss,
                "melanoma_precision": gate["melanoma_precision"],
                "melanoma_recall": gate["melanoma_recall"],
                "melanoma_f1": gate["melanoma_f1"],
                "macro_f1": gate["macro_f1"],
                "nevus_recall": gate["nevus_recall"],
                "criterion_results": gate["criterion_results"],
            }
            eligible_records.append(eligible)
            if val_loss < best_eligible_loss:
                best_eligible_loss = val_loss
                save_checkpoint(checkpoint_path, {
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": config,
                    "class_order": list(CLASS_ORDER),
                    "selection_rule": RULE_NEW,
                    "eligibility_metrics": eligible,
                })
        write_json(output / "training_history.json", {"epochs": history}, replace=True)
        write_json(output / "eligible_epochs.json", {
            "selection_rule": RULE_NEW,
            "criteria": CRITERIA,
            "eligible_epochs": eligible_records,
            "current_best_epoch": min(eligible_records, key=lambda row: (row["validation_loss"], row["epoch"]))["epoch"] if eligible_records else None,
        }, replace=True)
        print(f"epoch={epoch:02d}/{EPOCHS} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"mel_f1={gate['melanoma_f1']:.6f} macro_f1={gate['macro_f1']:.6f} "
              f"nv_recall={gate['nevus_recall']:.6f} eligible={gate['eligible']}")

    complete_run_outputs(root, config, history, eligible_records, preflight_report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--train", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = preflight(root, require_t4=args.train)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        return 2
    if args.train:
        if not result["training_eligible"]:
            raise RuntimeError("Training requires approved Google Colab CUDA Tesla T4")
        train_experiment(root, result, args.workers)
    else:
        print("Preflight only; no training or inference was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
