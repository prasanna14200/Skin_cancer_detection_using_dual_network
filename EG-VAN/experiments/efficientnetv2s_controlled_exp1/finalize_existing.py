"""Reconstruct Controlled Experiment #1 reports from verified saved artifacts.

This script performs no training, image loading, or model inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = ROOT / "experiments" / "efficientnetv2s_leakage_aware"
EXPERIMENT_CHECKPOINT_SHA256 = (
    "478bb1aa9c48897accceb800a103e7ee76c1381dc6514c7355996babbfaa902f"
)
BASELINE_CHECKPOINT_SHA256 = (
    "f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848"
)
CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
OUTPUT_NAMES = (
    "comparison_metrics.json",
    "experiment_manifest.json",
    "controlled_exp1_final_report.md",
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name} mismatch: saved={actual}, derived={expected}")


def metrics_from_matrix(matrix: list[list[int]]) -> dict:
    if len(matrix) != len(CLASS_NAMES) or any(len(row) != len(CLASS_NAMES) for row in matrix):
        raise ValueError("Confusion matrix shape does not match the frozen class order")
    total = sum(sum(row) for row in matrix)
    if total <= 0:
        raise ValueError("Cannot derive metrics from an empty confusion matrix")
    per_class = {}
    f1_values = []
    recalls = []
    correct = 0
    for index, name in enumerate(CLASS_NAMES):
        tp = int(matrix[index][index])
        support = int(sum(matrix[index]))
        predicted_count = int(sum(row[index] for row in matrix))
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
        f1_values.append(f1)
        recalls.append(recall)
        correct += tp
    return {
        "sample_count": total,
        "class_order": list(CLASS_NAMES),
        "accuracy": correct / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def metrics_from_predictions(path: Path) -> tuple[dict, int]:
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"true_label", "predicted_label", "correct"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Prediction CSV missing required columns: {path}")
        row_count = 0
        for row in reader:
            try:
                target = CLASS_NAMES.index(row["true_label"])
                predicted = CLASS_NAMES.index(row["predicted_label"])
            except ValueError as exc:
                raise ValueError(f"Unknown class in {path}: {exc}") from exc
            expected_correct = target == predicted
            if row["correct"].strip().lower() not in {"true", "false"}:
                raise ValueError(f"Invalid correct flag in {path}: {row['correct']!r}")
            if (row["correct"].strip().lower() == "true") != expected_correct:
                raise ValueError(f"Correct flag disagrees with labels in {path}")
            matrix[target][predicted] += 1
            row_count += 1
    return metrics_from_matrix(matrix), row_count


def validate_metric_record(saved: dict, derived: dict, context: str) -> None:
    if saved.get("class_order") not in (None, list(CLASS_NAMES)):
        raise ValueError(f"Unexpected class order in {context}")
    if saved.get("sample_count") is not None and int(saved["sample_count"]) != derived["sample_count"]:
        raise ValueError(f"Sample count mismatch in {context}")
    if saved.get("confusion_matrix") != derived["confusion_matrix"]:
        raise ValueError(f"Confusion matrix mismatch in {context}")
    for key in ("accuracy", "macro_f1"):
        if key in saved:
            assert_close(saved[key], derived[key], f"{context}.{key}")
    for class_name, actual in derived["per_class"].items():
        saved_per_class = saved.get("per_class", {}).get(class_name)
        if saved_per_class is not None:
            for key in ("support", "predicted_count"):
                if key in saved_per_class and int(saved_per_class[key]) != actual[key]:
                    raise ValueError(f"{context}.{class_name}.{key} mismatch")
            for key in ("precision", "recall", "f1"):
                if key in saved_per_class:
                    assert_close(
                        saved_per_class[key], actual[key], f"{context}.{class_name}.{key}"
                    )
        saved_recall = saved.get("per_class_recall", {}).get(class_name)
        if saved_recall is not None:
            assert_close(saved_recall, actual["recall"], f"{context}.{class_name}.recall")


def validate_history_metrics(saved: dict, derived: dict, context: str) -> None:
    if saved.get("confusion_matrix") != derived["confusion_matrix"]:
        raise ValueError(f"Confusion matrix mismatch in {context}")
    for key in ("accuracy", "macro_f1"):
        if key in saved:
            assert_close(saved[key], derived[key], f"{context}.{key}")
    recalls = saved.get("per_class_recall", {})
    for name, item in derived["per_class"].items():
        if name in recalls:
            assert_close(recalls[name], item["recall"], f"{context}.{name}.recall")


def checkpoint_metadata(path: Path) -> dict:
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint structure: {path}")
    return checkpoint


def markdown_report(comparison: dict, manifest: dict, root_cause: str) -> str:
    validation = comparison["controlled_experiment"]["validation"]
    baseline_validation = comparison["baseline"]["validation"]
    tests = comparison["controlled_experiment"]["test"]
    thresholds = comparison["success_thresholds"]
    values = comparison["validation_values"]
    status = "PASS" if comparison["validation_success_criteria_met"] else "FAIL"
    criterion_rows = (
        ("Melanoma F1", values["melanoma_f1"], thresholds["melanoma_f1_minimum"]),
        ("Validation macro-F1", values["validation_macro_f1"], thresholds["validation_macro_f1_minimum"]),
        ("Nevus recall", values["nevus_recall"], thresholds["nevus_recall_minimum"]),
    )
    table_rows = "\n".join(
        f"| {name} | {value:.12f} | {threshold:.12f} | {'PASS' if value >= threshold else 'FAIL'} |"
        for name, value, threshold in criterion_rows
    )
    return f"""# Controlled Experiment #1 Final Report

**Outcome: {status}**

## Verified Run

- Model: EfficientNetV2S
- Recorded training epochs: {manifest['training_epochs']}
- Selected checkpoint epoch: {manifest['selected_epoch']}
- Selection rule: minimum validation loss
- Checkpoint SHA256: `{manifest['new_checkpoint_sha256']}`
- Validation samples: {validation['sample_count']}
- Test samples: {tests['sample_count']}
- Training rerun: NO
- Inference rerun: NO
- PH2 inference: NO

## Prespecified Validation Criteria

| Criterion | Controlled value | Required minimum | Result |
|---|---:|---:|---|
{table_rows}

Baseline epoch {comparison['baseline']['selected_epoch']} validation metrics recovered from the frozen baseline training history: melanoma F1 {baseline_validation['per_class']['mel']['f1']:.12f}, macro-F1 {baseline_validation['macro_f1']:.12f}, nevus recall {baseline_validation['per_class']['nv']['recall']:.12f}.

Controlled test metrics are included in `comparison_metrics.json`; they were not used for checkpoint selection or success-criterion evaluation. Baseline ROC-AUC values are unavailable from the saved baseline artifacts and are intentionally omitted.

## Finalization Finding

{root_cause}

The reports were reconstructed solely from hash-verified checkpoints, the frozen split, saved metric JSON, training histories, and prediction CSVs. No image data was loaded and no model inference was run.
"""


def write_outputs(outputs: dict[str, str]) -> None:
    temporary_paths = []
    try:
        for name, content in outputs.items():
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=OUTPUT_DIR,
                prefix=f".{name}.", suffix=".tmp", delete=False,
            ) as handle:
                handle.write(content)
                temporary_paths.append((Path(handle.name), OUTPUT_DIR / name))
        for temporary, destination in temporary_paths:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in temporary_paths:
            temporary.unlink(missing_ok=True)


def main() -> None:
    plan = read_json(OUTPUT_DIR / "experiment_plan.json")
    config = read_json(OUTPUT_DIR / "config.json")
    experiment_checkpoint_path = OUTPUT_DIR / "best_checkpoint.pt"
    baseline_checkpoint_path = BASELINE_DIR / "best_checkpoint.pt"
    split_path = ROOT / "data" / "splits" / "split_leakage_aware.csv"

    experiment_hash = sha256_file(experiment_checkpoint_path)
    baseline_hash = sha256_file(baseline_checkpoint_path)
    split_hash = sha256_file(split_path)
    if experiment_hash != EXPERIMENT_CHECKPOINT_SHA256:
        raise ValueError(f"Controlled checkpoint SHA256 mismatch: {experiment_hash}")
    if baseline_hash != BASELINE_CHECKPOINT_SHA256:
        raise ValueError(f"Baseline checkpoint SHA256 mismatch: {baseline_hash}")
    if plan.get("baseline_checkpoint_sha256") != baseline_hash:
        raise ValueError("Baseline checkpoint hash does not match the frozen experiment plan")
    if plan.get("split_reference", {}).get("sha256") != split_hash:
        raise ValueError("Frozen split SHA256 does not match the experiment plan")

    history = read_json(OUTPUT_DIR / "training_history.json").get("epochs", [])
    baseline_history = read_json(BASELINE_DIR / "training_history.json").get("epochs", [])
    if len(history) != 25 or len(baseline_history) != 25:
        raise ValueError("Expected 25 recorded epochs in controlled and baseline histories")
    best_epoch = min(history, key=lambda item: float(item["val_loss"]))
    baseline_best_epoch = min(baseline_history, key=lambda item: float(item["val_loss"]))
    experiment_checkpoint = checkpoint_metadata(experiment_checkpoint_path)
    baseline_checkpoint = checkpoint_metadata(baseline_checkpoint_path)
    selected_epoch = int(best_epoch["epoch"])
    baseline_selected_epoch = int(baseline_best_epoch["epoch"])
    if int(experiment_checkpoint.get("epoch", -1)) != selected_epoch:
        raise ValueError("Controlled checkpoint epoch disagrees with minimum-loss history epoch")
    if int(baseline_checkpoint.get("epoch", -1)) != baseline_selected_epoch:
        raise ValueError("Baseline checkpoint epoch disagrees with minimum-loss history epoch")
    if selected_epoch != 5 or baseline_selected_epoch != 6:
        raise ValueError("Selected epochs differ from the frozen experiment evidence")

    validation_saved = read_json(OUTPUT_DIR / "validation_metrics.json")
    test_saved = read_json(OUTPUT_DIR / "test_metrics.json")
    if int(validation_saved.get("selected_epoch", -1)) != selected_epoch:
        raise ValueError("Saved validation metrics refer to a different selected epoch")
    if int(test_saved.get("selected_epoch", -1)) != selected_epoch:
        raise ValueError("Saved test metrics refer to a different selected epoch")
    if test_saved.get("checkpoint_sha256") != experiment_hash:
        raise ValueError("Saved test metrics checkpoint hash does not match the checkpoint")

    validation_derived, validation_rows = metrics_from_predictions(
        OUTPUT_DIR / "validation_predictions.csv"
    )
    test_derived, test_rows = metrics_from_predictions(OUTPUT_DIR / "test_predictions.csv")
    validate_metric_record(validation_saved, validation_derived, "controlled validation")
    validate_metric_record(test_saved, test_derived, "controlled test")
    if validation_rows != validation_derived["sample_count"] or test_rows != test_derived["sample_count"]:
        raise ValueError("Prediction CSV row count does not match its reconstructed confusion matrix")
    validate_history_metrics(best_epoch["validation"], validation_derived, "selected epoch validation")
    assert_close(best_epoch["val_loss"], validation_saved["loss"], "selected validation loss")

    baseline_validation_derived = metrics_from_matrix(
        baseline_best_epoch["validation"]["confusion_matrix"]
    )
    validate_history_metrics(
        baseline_best_epoch["validation"], baseline_validation_derived, "baseline epoch-6 validation"
    )
    baseline_validation_derived["loss"] = float(baseline_best_epoch["val_loss"])
    baseline_validation_derived["selected_epoch"] = baseline_selected_epoch
    baseline_validation_derived["checkpoint_sha256"] = baseline_hash

    baseline_test_saved = read_json(BASELINE_DIR / "test_metrics.json")
    baseline_test_derived = metrics_from_matrix(baseline_test_saved["confusion_matrix"])
    validate_metric_record(baseline_test_saved, baseline_test_derived, "baseline test")
    baseline_test_derived["loss"] = float(baseline_test_saved["loss"])
    baseline_test_derived["selected_epoch"] = baseline_selected_epoch
    baseline_test_derived["checkpoint_sha256"] = baseline_hash

    class_counts = plan["class_counts"]
    if validation_rows != sum(class_counts["val"].values()):
        raise ValueError("Validation prediction count does not match the frozen split plan")
    if test_rows != sum(class_counts["test"].values()):
        raise ValueError("Test prediction count does not match the frozen split plan")

    criteria = plan["success_criteria"]
    validation_values = {
        "melanoma_f1": validation_derived["per_class"]["mel"]["f1"],
        "validation_macro_f1": validation_derived["macro_f1"],
        "nevus_recall": validation_derived["per_class"]["nv"]["recall"],
    }
    thresholds = {
        "melanoma_f1_minimum": criteria["melanoma_f1"]["minimum"],
        "validation_macro_f1_minimum": criteria["validation_macro_f1"]["minimum"],
        "nevus_recall_minimum": criteria["nevus_recall"]["minimum"],
    }
    individual_results = {
        "melanoma_f1": validation_values["melanoma_f1"] >= thresholds["melanoma_f1_minimum"],
        "validation_macro_f1": validation_values["validation_macro_f1"] >= thresholds["validation_macro_f1_minimum"],
        "nevus_recall": validation_values["nevus_recall"] >= thresholds["nevus_recall_minimum"],
    }
    success = all(individual_results.values())

    root_cause = (
        "The runner writes comparison_metrics.json and experiment_manifest.json only after its "
        "post-training baseline evaluation. Those baseline outputs are absent, so the saved files "
        "show that finalization did not reach its report-writing stage (or those outputs were later "
        "removed). The exact interruption/removal cause cannot be determined: no run log or error "
        "record is present. The runner has no code path that writes controlled_exp1_final_report.md."
    )
    comparison = {
        "checkpoint_sha256": experiment_hash,
        "selected_epoch": selected_epoch,
        "selection_metric": "validation_loss_minimize",
        "validation_success_criteria_met": success,
        "criterion_results": individual_results,
        "validation_values": validation_values,
        "success_thresholds": thresholds,
        "baseline": {
            "checkpoint_sha256": baseline_hash,
            "selected_epoch": baseline_selected_epoch,
            "validation": baseline_validation_derived,
            "test": baseline_test_derived,
        },
        "controlled_experiment": {
            "validation": {**validation_saved, "checkpoint_sha256": experiment_hash},
            "test": test_saved,
        },
        "metric_availability": {
            "baseline_validation_source": "epoch-6 validation confusion matrix in frozen training_history.json",
            "baseline_test_source": "frozen test_metrics.json confusion matrix",
            "baseline_roc_auc": "unavailable in saved baseline artifacts; not reconstructed",
            "controlled_predictions_cross_checked": True,
        },
    }
    manifest = {
        "status": "FINALIZATION_RECOVERED_FROM_SAVED_ARTIFACTS",
        "finalization_mode": "offline_saved_artifacts_only",
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_completion_timestamp_available": False,
        "started_runtime": None,
        "baseline_checkpoint_sha256": baseline_hash,
        "split_sha256": split_hash,
        "new_checkpoint_sha256": experiment_hash,
        "training_epochs": len(history),
        "selected_epoch": selected_epoch,
        "selection_metric": "validation_loss_minimize",
        "selection_value": float(best_epoch["val_loss"]),
        "validation_success_criteria_met": success,
        "criterion_results": individual_results,
        "class_weights": config.get("intervention", {}).get("class_weights"),
        "ph2_accessed": False,
        "training_rerun": False,
        "inference_rerun": False,
        "test_evaluation_occurred_after_checkpoint_freeze": True,
        "runtime": config.get("software"),
        "verified_artifacts": [
            "best_checkpoint.pt SHA256",
            "baseline best_checkpoint.pt SHA256",
            "frozen split_leakage_aware.csv SHA256",
            "25-epoch controlled and baseline training histories",
            "saved validation/test metrics and controlled prediction CSVs",
        ],
        "historical_preflight_report": "unavailable; not fabricated",
    }
    outputs = {
        "comparison_metrics.json": json.dumps(comparison, indent=2, allow_nan=False) + "\n",
        "experiment_manifest.json": json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        "controlled_exp1_final_report.md": markdown_report(comparison, manifest, root_cause),
    }
    write_outputs(outputs)
    print("Finalization complete; no training or inference was run.")
    print(f"Controlled checkpoint SHA256: {experiment_hash}")
    print(f"Selected epoch: {selected_epoch}; epochs recorded: {len(history)}")
    print(f"Validation sample count: {validation_rows}; test sample count: {test_rows}")
    print(f"Validation criteria: {individual_results}")
    for name in OUTPUT_NAMES:
        print(f"CREATED {name}")


if __name__ == "__main__":
    main()