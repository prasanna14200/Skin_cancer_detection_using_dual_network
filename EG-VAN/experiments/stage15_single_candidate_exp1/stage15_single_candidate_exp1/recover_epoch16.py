"""Bounded Stage 15 artifact recovery: replay ONLY epochs 14-16 on Tesla T4.

Never writes to the original run/ directory. --check is CPU/read-only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import io
import json
import math
import platform
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision

import train_epoch13_source as original


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = HERE / "run"
RECOVERY = HERE / "recovery_epoch16"
SOURCE_SHA = "42931bbab2e8aa207ce60c675c46cdcb210618df955617455923f6aaab6422b2"
CHECKPOINT_SHA = "caec0fc13117b31e34b5760a8c7b618be1a39fcb634690eda21cbaf9e7a2c001"
HISTORY_SHA = "2df7edd11a23658e56dc723ee57638476579cb3620d9c5d10039eefa07941fd0"
REFERENCE = {
    14: ("7f2e961cb80bbb3d52626bfea6e3103223d6ad640fa50887f30b730ea04944ef",
         "cbbd37251f3da1968cbf08d521d995abe2999e0a95da125edad744d548afb6a8"),
    15: ("0cd33f9207f3351f226fbefc5880cbe958448d64c81437bf55ee8fc1fc0ab073",
         "56279974986ea9603249aaacd2e998d4af82a4c6ef8648e2cb5cbcc8d154dc2b"),
    16: ("d0510c792d36c9cd3132d9495d70be204fa228c8eb4f2a016bcc4dc2a725e148",
         "630bb7a41c41cd5b125576ca0d14beef7d5b523dba7919680e526501f52c0571"),
}


def reference_paths(epoch: int) -> tuple[Path, Path]:
    return (RUN / "validation_epochs" / f"epoch_{epoch:03d}_predictions.csv",
            RUN / "validation_epochs" / f"epoch_{epoch:03d}_metrics.json")


def recovery_paths(epoch: int) -> tuple[Path, Path, Path]:
    return (RECOVERY / f"epoch_{epoch:03d}_checkpoint.pt",
            RECOVERY / "validation_epochs" / f"epoch_{epoch:03d}_predictions.csv",
            RECOVERY / "validation_epochs" / f"epoch_{epoch:03d}_metrics.json")


def same_structure_with_float_roundoff(left, right) -> bool:
    """Validate archived metrics across Python versions without changing replay criteria."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            same_structure_with_float_roundoff(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            same_structure_with_float_roundoff(a, b) for a, b in zip(left, right))
    if type(left) is bool or type(right) is bool:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-14)
    return left == right


def inspect_reference_epoch(epoch: int, expected_row: dict, validation_loader, rule: dict) -> dict:
    predictions_path, metrics_path = reference_paths(epoch)
    expected_prediction_sha, expected_metrics_sha = REFERENCE[epoch]
    if (original.base.sha256(predictions_path) != expected_prediction_sha or
            original.base.sha256(metrics_path) != expected_metrics_sha):
        raise ValueError(f"Original epoch-{epoch} validation evidence hash changed")
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    metrics = original.base.read_json(metrics_path)
    matrix = [[0] * 7 for _ in range(7)]
    for row in predictions:
        matrix[original.base.CLASSES.index(row["true_label"])][
            original.base.CLASSES.index(row["predicted_label"])] += 1
    rebuilt = original.validation_evidence(
        epoch, float(expected_row["val_loss"]), original.base.metrics(matrix),
        predictions, validation_loader, rule)
    if not same_structure_with_float_roundoff(rebuilt, metrics):
        differing = [key for key in rebuilt if not same_structure_with_float_roundoff(
            rebuilt[key], metrics.get(key))]
        raise ValueError(f"Original epoch-{epoch} validation metrics differ from predictions/rule: {differing}")
    if (metrics["eligible"] != (expected_row["eligible"] == "True") or
            metrics["validation_loss"] != float(expected_row["val_loss"])):
        raise ValueError(f"Original epoch-{epoch} history and validation evidence differ")
    return {"epoch": epoch, "predictions_sha256": expected_prediction_sha,
            "metrics_sha256": expected_metrics_sha, "eligible": metrics["eligible"]}


def preflight(root: Path) -> tuple[dict, dict, dict, list[dict], dict]:
    if root.resolve() != ROOT.resolve():
        raise ValueError("Use the pinned EG-VAN project root")
    if original.base.sha256(HERE / "train_epoch13_source.py") != SOURCE_SHA:
        raise ValueError("Archived epoch-13 runner source changed")
    if original.base.sha256(RUN / "last_checkpoint.pt") != CHECKPOINT_SHA:
        raise ValueError("Original epoch-13 checkpoint hash changed")
    if original.base.sha256(RUN / "training_history.csv") != HISTORY_SHA:
        raise ValueError("Original 1-24 history hash changed")
    if (RUN / "best_checkpoint.pt").exists() or (RUN / "experiment_manifest.json").exists():
        raise ValueError("Original artifact state changed; recovery assumptions must be re-audited")
    cfg, rule, numerical = original.check_preflight(root)
    state = torch.load(RUN / "last_checkpoint.pt", map_location="cpu", weights_only=False, mmap=True)
    if (state["epoch"] != 13 or state["best_epoch"] is not None or
            state["best_validation_loss"] != math.inf or state["runner_sha256"] != SOURCE_SHA):
        raise ValueError("Recovery source is not the audited epoch-13 state")
    inspected = original.validate_resume(state, cfg, rule)
    if inspected["start_epoch"] != 14:
        raise ValueError("Recovery would not begin at epoch 14")
    for epoch, digests in state["validation_artifacts"].items():
        predictions_path, metrics_path = reference_paths(epoch)
        if (original.base.sha256(predictions_path) != digests["predictions_sha256"] or
                original.base.sha256(metrics_path) != digests["metrics_sha256"]):
            raise ValueError(f"Original epoch-{epoch} evidence disagrees with checkpoint")
    with (RUN / "training_history.csv").open(newline="", encoding="utf-8") as handle:
        original_rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in original_rows] != list(range(1, 25)):
        raise ValueError("Original CSV is not exactly epochs 1-24")
    if any(any(original_rows[i][key] != str(value) for key, value in saved.items())
           for i, saved in enumerate(state["history"])):
        raise ValueError("Original CSV 1-13 differs from checkpoint history")
    _, Dataset, _, _, _, _, _ = original.base.imports(root)
    validation_dataset = Dataset(root / "data/processed/images",
                                 root / "data/splits/split_leakage_aware.csv", "val", None)
    validation_loader = SimpleNamespace(dataset=validation_dataset)
    evidence = {epoch: inspect_reference_epoch(epoch, original_rows[epoch - 1],
                                               validation_loader, rule)
                for epoch in (14, 15, 16)}
    if (evidence[14]["eligible"] or evidence[15]["eligible"] or
            not evidence[16]["eligible"] or
            sum(row["eligible"] == "True" for row in original_rows) != 1):
        raise ValueError("Original eligibility is not uniquely epoch 16")
    return state, cfg, rule, original_rows, evidence


def compare_row(actual: dict, reference: dict) -> dict:
    mismatches = {key: {"recovered": str(value), "original": reference.get(key)}
                  for key, value in actual.items() if str(value) != reference.get(key)}
    numeric_diffs = {key: abs(float(value) - float(reference[key]))
                     for key, value in actual.items()
                     if key in reference and key not in ("epoch", "eligible")
                     and isinstance(value, (int, float)) and reference[key] not in ("True", "False")}
    maximum = max(numeric_diffs.values(), default=0.0)
    return {"exact": not mismatches, "mismatches": mismatches,
            "maximum_numeric_difference": maximum,
            "numerically_close": maximum <= 1e-6 and actual["eligible"] == (reference["eligible"] == "True")}


def compare_validation(epoch: int, prediction_bytes: bytes, metrics_bytes: bytes) -> dict:
    original_predictions_path, original_metrics_path = reference_paths(epoch)
    with original_predictions_path.open(newline="", encoding="utf-8") as handle:
        prior_rows = list(csv.DictReader(handle))
    current_rows = list(csv.DictReader(io.StringIO(prediction_bytes.decode("utf-8"))))
    discrete = (len(prior_rows) == len(current_rows) and all(
        all(left[key] == right[key] for key in (
            "image_id", "true_label", "predicted_label", "correct"))
        for left, right in zip(prior_rows, current_rows)))
    probability_gap = 0.0
    if len(prior_rows) == len(current_rows):
        for left, right in zip(prior_rows, current_rows):
            a, b = json.loads(left["probabilities"]), json.loads(right["probabilities"])
            if len(a) != len(b):
                probability_gap = None
                break
            probability_gap = max(probability_gap, *(abs(x - y) for x, y in zip(a, b)))
    else:
        probability_gap = None
    prior_metrics = original.base.read_json(original_metrics_path)
    current_metrics = json.loads(metrics_bytes)
    aggregate = ("validation_loss", "accuracy", "balanced_accuracy", "macro_f1",
                 "mel_recall", "mel_f1", "nv_recall")
    metric_diffs = {name: abs(current_metrics[name] - prior_metrics[name]) for name in aggregate}
    return {"prediction_bytes_exact": original.base.sha256(original_predictions_path) ==
            hashlib.sha256(prediction_bytes).hexdigest(),
            "metrics_bytes_exact": original.base.sha256(original_metrics_path) ==
            hashlib.sha256(metrics_bytes).hexdigest(),
            "ids_true_predictions_exact": discrete,
            "maximum_probability_difference": probability_gap,
            "aggregate_metric_differences": metric_diffs,
            "per_class_exact": current_metrics["per_class"] == prior_metrics["per_class"],
            "per_class_numerically_close": same_structure_with_float_roundoff(
                current_metrics["per_class"], prior_metrics["per_class"]),
            "confusion_matrix_exact": current_metrics["confusion_matrix"] ==
            prior_metrics["confusion_matrix"],
            "eligibility_exact": current_metrics["eligible"] is prior_metrics["eligible"] and
            current_metrics["failed_gates"] == prior_metrics["failed_gates"]}


def recover(root: Path, state: dict, cfg: dict, rule: dict, original_rows: list[dict],
            original_evidence: dict) -> None:
    if RECOVERY.exists():
        raise FileExistsError("Separate recovery directory already exists; refusing to overwrite")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Bounded recovery requires the same Tesla T4 class")
    if len(state["cuda_rng_states"]) != torch.cuda.device_count():
        raise ValueError("Saved CUDA RNG device count differs from recovery runtime")
    RECOVERY.mkdir(parents=True)
    comparisons = []
    created = []
    model = optimizer = scheduler = scaler = train_loader = val_loader = generator = None
    restore_policy = None
    try:
        model, optimizer, scheduler, unused_scaler, train_loader, val_loader, generator = \
            original.guard.build_components(root, "B")
        del unused_scaler
        scaler = torch.amp.GradScaler("cuda", init_scale=original.INITIAL_SCALE)
        restore_policy = original.install_selective_qk_policy(model)
        start_epoch, history, best_epoch, best_loss = original.restore_resume(
            state, model, optimizer, scheduler, scaler, generator)
        if start_epoch != 14 or best_epoch is not None or best_loss != math.inf:
            raise ValueError("Restored recovery trajectory differs from epoch-13 checkpoint")
        original.guard.require_finite_states(model, optimizer, {"recovery": True, "epoch": 13})
        events = copy.deepcopy(state["numerical_events"])
        validation_artifacts = copy.deepcopy(state["validation_artifacts"])
        del state
        gc.collect()
        for epoch in (14, 15, 16):
            train_sum = train_count = steps = 0
            for batch, (images_cpu, labels_cpu) in enumerate(train_loader, start=1):
                images = images_cpu.cuda(non_blocking=True)
                labels = labels_cpu.cuda(non_blocking=True)
                value, stepped, event = original.checked_train_batch(
                    model, optimizer, scaler, images, labels, original.NAME, epoch, batch,
                    cfg["loss"]["mel_multiplier"])
                train_sum += value * len(labels_cpu)
                train_count += len(labels_cpu)
                steps += int(stepped)
                if event is not None:
                    events.append(event)
            if steps == 0:
                raise RuntimeError(f"No optimizer update in recovery epoch {epoch}")
            train_loss = train_sum / train_count
            val_loss, val_metrics, predictions = original.guard.guarded_validation_epoch(
                model, val_loader, original.NAME, epoch, predictions=True)
            original.guard.require_finite_states(model, optimizer, {"recovery": True, "epoch": epoch})
            original.guard.require_finite_epoch_losses(train_loss, val_loss, original.NAME, epoch)
            evidence = original.validation_evidence(epoch, val_loss, val_metrics, predictions, val_loader, rule)
            prediction_bytes, metrics_bytes = original.validation_bytes(predictions, evidence)
            validation_artifacts[epoch] = original.validation_hashes(prediction_bytes, metrics_bytes)
            scheduler.step(val_loss)
            is_eligible = original.eligible(val_metrics, rule)
            if evidence["eligible"] is not is_eligible:
                raise ValueError("Recovery evidence eligibility differs from frozen rule")
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "val_accuracy": val_metrics["accuracy"],
                   "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                   "val_macro_f1": val_metrics["macro_f1"],
                   "val_mel_precision": val_metrics["per_class"]["mel"]["precision"],
                   "val_mel_recall": val_metrics["per_class"]["mel"]["recall"],
                   "val_mel_f1": val_metrics["per_class"]["mel"]["f1"],
                   "val_nv_recall": val_metrics["per_class"]["nv"]["recall"],
                   "eligible": is_eligible, "learning_rate": optimizer.param_groups[0]["lr"],
                   "optimizer_steps": steps,
                   "amp_skips": sum(item["epoch"] == epoch for item in events)}
            history.append(row)
            improved = is_eligible and val_loss < best_loss
            if improved:
                best_epoch, best_loss = epoch, val_loss
            recovered_state = {
                "experiment": original.NAME, "epoch": epoch, "configuration": cfg,
                "config_sha256": original.CONFIG_SHA,
                "selection_rule_sha256": original.RULE_SHA,
                "numerical_protocol_sha256": original.NUMERICAL_SHA,
                "runner_sha256": SOURCE_SHA, "class_order": list(original.base.CLASSES),
                "architecture": cfg["architecture"], "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(), "history": history,
                "best_epoch": best_epoch, "best_validation_loss": best_loss,
                "numerical_events": copy.deepcopy(events),
                "validation_artifacts": copy.deepcopy(validation_artifacts),
                "latest_validation_payload": {"predictions": predictions, "metrics": evidence},
                "sampler_generator_state": generator.get_state(),
                "python_rng_state": original.random.getstate(),
                "numpy_rng_state": original.np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
                "recovery_source_checkpoint_sha256": CHECKPOINT_SHA,
                "recovery_runner_sha256": original.base.sha256(Path(__file__)),
            }
            checkpoint_path, predictions_path, metrics_path = recovery_paths(epoch)
            original.guard.checked_checkpoint(checkpoint_path, recovered_state,
                                             model, optimizer, scaler, original.NAME, epoch)
            original.write_once(predictions_path, prediction_bytes)
            original.write_once(metrics_path, metrics_bytes)
            created += [checkpoint_path, predictions_path, metrics_path]
            row_match = compare_row(row, original_rows[epoch - 1])
            validation_match = compare_validation(epoch, prediction_bytes, metrics_bytes)
            prediction_match = (validation_match["prediction_bytes_exact"] and
                                original.base.sha256(predictions_path) == REFERENCE[epoch][0])
            metrics_match = (validation_match["metrics_bytes_exact"] and
                             original.base.sha256(metrics_path) == REFERENCE[epoch][1])
            exactly_replayed = row_match["exact"] and prediction_match and metrics_match
            numerically_close = (row_match["numerically_close"] and
                                 validation_match["ids_true_predictions_exact"] and
                                 validation_match["maximum_probability_difference"] is not None and
                                 validation_match["maximum_probability_difference"] <= 1e-6 and
                                 validation_match["per_class_numerically_close"] and
                                 validation_match["confusion_matrix_exact"] and
                                 validation_match["eligibility_exact"] and
                                 max(validation_match["aggregate_metric_differences"].values()) <= 1e-6)
            comparison = {"epoch": epoch, "history": row_match,
                          "validation": validation_match,
                          "persisted_prediction_hash_exact": prediction_match,
                          "persisted_metrics_hash_exact": metrics_match,
                          "checkpoint_sha256": original.base.sha256(checkpoint_path),
                          "original_evidence": original_evidence[epoch],
                          "replay_status": "EXACT_OBSERVABLE_MATCH" if exactly_replayed else
                          "NUMERICALLY_CLOSE_ONLY" if numerically_close else "DIVERGED"}
            comparisons.append(comparison)
            original.write_atomic(RECOVERY / "comparison.json",
                (json.dumps(comparisons, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            print(f"recovery epoch={epoch}/16 status={comparison['replay_status']} "
                  f"eligible={is_eligible} scale={scaler.get_scale():.0f}", flush=True)
            if comparison["replay_status"] != "EXACT_OBSERVABLE_MATCH":
                raise RuntimeError(f"Epoch {epoch} did not exactly reproduce saved trajectory")
            if epoch == 16:
                if not improved or best_epoch != 16 or not evidence["eligible"]:
                    raise ValueError("Recovery epoch 16 does not meet unchanged selection rule")
                best_path = RECOVERY / "best_checkpoint.pt"
                original.guard.checked_checkpoint(best_path, recovered_state,
                                                 model, optimizer, scaler, original.NAME, epoch)
                created.append(best_path)
        if len(comparisons) != 3 or any(item["replay_status"] != "EXACT_OBSERVABLE_MATCH"
                                        for item in comparisons):
            raise RuntimeError("Bounded replay did not establish exact observable match")
        manifest = {"status": "RECOVERED_EPOCH16_OBSERVABLE_EXACT_MATCH",
                    "scope": "bounded epochs 14-16 only", "selected_epoch": 16,
                    "source_checkpoint_sha256": CHECKPOINT_SHA,
                    "source_runner_sha256": SOURCE_SHA,
                    "recovery_runner_sha256": original.base.sha256(Path(__file__)),
                    "original_history_sha256": HISTORY_SHA,
                    "original_reference_sha256": {str(e): {"predictions": pair[0], "metrics": pair[1]}
                                                  for e, pair in REFERENCE.items()},
                    "config_sha256": original.CONFIG_SHA,
                    "selection_rule_sha256": original.RULE_SHA,
                    "numerical_protocol_sha256": original.NUMERICAL_SHA,
                    "split_sha256": original.base.SPLIT_SHA,
                    "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                                "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                                "gpu": torch.cuda.get_device_name(0)},
                    "comparisons": comparisons,
                    "artifact_sha256": {str(path.relative_to(RECOVERY)).replace("\\", "/"):
                                        original.base.sha256(path) for path in created},
                    "weight_identity_with_missing_original_checkpoint": "UNPROVABLE",
                    "ham_test_accessed": False, "ph2_accessed": False}
        original.write_atomic(RECOVERY / "recovery_manifest.json",
            (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        print(json.dumps({"status": manifest["status"], "output": str(RECOVERY),
                          "best_checkpoint_sha256": manifest["artifact_sha256"]["best_checkpoint.pt"]},
                         indent=2), flush=True)
    except Exception as exc:
        original.write_atomic(RECOVERY / "recovery_failure.json",
            (json.dumps({"status": "STOPPED", "type": type(exc).__name__, "message": str(exc),
                         "completed_comparisons": comparisons, "source_checkpoint_sha256": CHECKPOINT_SHA},
                        indent=2, allow_nan=False) + "\n").encode("utf-8"))
        raise
    finally:
        if restore_policy is not None:
            restore_policy()
        del model, optimizer, scheduler, scaler, train_loader, val_loader, generator
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="read-only epoch-13/source/evidence preflight")
    mode.add_argument("--recover", action="store_true", help="T4-only bounded 14-16 replay")
    args = parser.parse_args()
    state, cfg, rule, original_rows, evidence = preflight(args.project_root.resolve())
    if args.check:
        print(json.dumps({"status": "RECOVERY_PREFLIGHT_PASS", "source_epoch": state["epoch"],
                          "next_epoch": 14, "stop_after_epoch": 16,
                          "checkpoint_sha256": CHECKPOINT_SHA, "runner_sha256": SOURCE_SHA,
                          "original_evidence": evidence,
                          "exact_replay_established": False,
                          "ham_test_accessed": False, "ph2_accessed": False}, indent=2))
        return
    recover(args.project_root.resolve(), state, cfg, rule, original_rows, evidence)


if __name__ == "__main__":
    main()
