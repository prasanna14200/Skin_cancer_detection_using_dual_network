"""Read-only audit of the downloaded, nested Stage 15 recovery folder."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "experiments/stage15_single_candidate_exp1"
DOWNLOAD = CANONICAL / "stage15_single_candidate_exp1"
RECOVERY = DOWNLOAD / "recovery_epoch16"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bad_tensors(value, prefix=""):
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            result.extend(bad_tensors(item, f"{prefix}/{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.extend(bad_tensors(item, f"{prefix}/{index}"))
    elif isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        if not bool(torch.isfinite(value).all()):
            result.append(prefix)
    return result


def main():
    manifest = json.loads((RECOVERY / "recovery_manifest.json").read_text(encoding="utf-8"))
    comparison = json.loads((RECOVERY / "comparison.json").read_text(encoding="utf-8"))
    files = {}
    for name, digest in manifest["artifact_sha256"].items():
        path = RECOVERY / name
        files[name] = {"exists": path.is_file(),
                       "sha256": sha256(path) if path.is_file() else None,
                       "manifest_match": path.is_file() and sha256(path) == digest}
    with (CANONICAL / "run/training_history.csv").open(newline="", encoding="utf-8") as handle:
        original_rows = list(csv.DictReader(handle))
    def checkpoint_report(path: Path, expected_epoch: int):
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        metric = json.loads((CANONICAL / f"run/validation_epochs/epoch_{expected_epoch:03d}_metrics.json").read_text(encoding="utf-8"))
        with (CANONICAL / f"run/validation_epochs/epoch_{expected_epoch:03d}_predictions.csv").open(
                newline="", encoding="utf-8") as handle:
            original_predictions = list(csv.DictReader(handle))
        latest = state["latest_validation_payload"]
        scheduler = state["scheduler_state"]
        scaler = state["scaler_state"]
        output = {
            "epoch": state["epoch"],
            "expected_epoch_match": state["epoch"] == expected_epoch,
            "history_epochs": [item["epoch"] for item in state["history"]],
            "history_epoch_sequence_valid": [item["epoch"] for item in state["history"]] ==
                list(range(1, expected_epoch + 1)),
            "history_exact_original_prefix": all(
                all(original_rows[index][key] == str(value) for key, value in row.items())
                for index, row in enumerate(state["history"])),
            "best_epoch": state["best_epoch"],
            "best_validation_loss": state["best_validation_loss"] if math.isfinite(
                float(state["best_validation_loss"])) else "inf",
            "model_nonfinite": bad_tensors(state["model_state"]),
            "optimizer_nonfinite": bad_tensors(state["optimizer_state"]),
            "scheduler_last_epoch": scheduler["last_epoch"],
            "scheduler_best": scheduler["best"],
            "scheduler_state_valid": scheduler["last_epoch"] == expected_epoch and
                math.isfinite(float(scheduler["best"])) and
                all(math.isfinite(float(lr)) and float(lr) > 0 for lr in scheduler["_last_lr"]),
            "scheduler_lr_matches_optimizer":
                scheduler["_last_lr"] == [state["optimizer_state"]["param_groups"][0]["lr"]],
            "scaler": scaler,
            "scaler_state_valid": math.isfinite(float(scaler["scale"])) and
                float(scaler["scale"]) > 0 and float(scaler["growth_factor"]) > 1 and
                0 < float(scaler["backoff_factor"]) < 1 and
                int(scaler["growth_interval"]) > 0 and int(scaler["_growth_tracker"]) >= 0,
            "class_order": state["class_order"],
            "configuration_sha256": state["config_sha256"],
            "configuration_matches_manifest": state["config_sha256"] == manifest["config_sha256"],
            "selection_rule_sha256": state["selection_rule_sha256"],
            "selection_rule_matches_manifest": state["selection_rule_sha256"] == manifest["selection_rule_sha256"],
            "numerical_protocol_sha256": state["numerical_protocol_sha256"],
            "numerical_protocol_matches_manifest": state["numerical_protocol_sha256"] == manifest["numerical_protocol_sha256"],
            "source_checkpoint_sha256": state["recovery_source_checkpoint_sha256"],
            "source_checkpoint_matches_manifest": state["recovery_source_checkpoint_sha256"] == manifest["source_checkpoint_sha256"],
            "recovery_runner_sha256": state["recovery_runner_sha256"],
            "recovery_runner_matches_manifest": state["recovery_runner_sha256"] == manifest["recovery_runner_sha256"],
            "latest_metrics_exact_original": latest["metrics"] == metric,
            "latest_predictions_exact_original": latest["predictions"] == original_predictions,
            "latest_eligible": latest["metrics"]["eligible"],
            "latest_failed_gates": latest["metrics"]["failed_gates"],
            "rng_present": all(key in state for key in (
                "python_rng_state", "numpy_rng_state", "torch_rng_state",
                "cuda_rng_states", "sampler_generator_state")),
        }
        return state, output

    epoch14, epoch14_summary = checkpoint_report(RECOVERY / "epoch_014_checkpoint.pt", 14)
    epoch15, epoch15_summary = checkpoint_report(RECOVERY / "epoch_015_checkpoint.pt", 15)
    epoch16, epoch16_summary = checkpoint_report(RECOVERY / "epoch_016_checkpoint.pt", 16)
    best, best_summary = checkpoint_report(RECOVERY / "best_checkpoint.pt", 16)
    scientific_metadata_keys = (
        "experiment", "configuration", "architecture", "class_order",
        "config_sha256", "selection_rule_sha256", "numerical_protocol_sha256",
        "recovery_source_checkpoint_sha256", "recovery_runner_sha256", "runner_sha256",
    )
    metadata_consistency = {
        f"epoch_{number:03d}": all(state[key] == epoch16[key] for key in scientific_metadata_keys)
        for number, state in ((14, epoch14), (15, epoch15))
    }
    model_equal = all(torch.equal(epoch16["model_state"][key], best["model_state"][key])
                      for key in epoch16["model_state"])
    optimizer_equal = (epoch16["optimizer_state"]["param_groups"] ==
                       best["optimizer_state"]["param_groups"] and
                       all(all(torch.equal(value, best["optimizer_state"]["state"][slot][key])
                               if isinstance(value, torch.Tensor) else
                               value == best["optimizer_state"]["state"][slot][key]
                               for key, value in state.items())
                           for slot, state in epoch16["optimizer_state"]["state"].items()))
    output = {
        "download_path": str(DOWNLOAD),
        "manifest_sha256": sha256(RECOVERY / "recovery_manifest.json"),
        "manifest_status": manifest["status"],
        "manifest_selected_epoch": manifest["selected_epoch"],
        "manifest_and_comparison_equal": manifest["comparisons"] == comparison,
        "all_replay_comparisons_exact": len(comparison) == 3 and
            all(item["replay_status"] == "EXACT_OBSERVABLE_MATCH" and
                item["history"]["exact"] and
                item["validation"]["prediction_bytes_exact"] and
                item["validation"]["metrics_bytes_exact"] and
                item["validation"]["per_class_exact"] and
                item["validation"]["confusion_matrix_exact"] for item in comparison),
        "artifact_files": files,
        "all_manifest_artifacts_present_and_hashed": all(
            item["exists"] and item["manifest_match"] for item in files.values()),
        "downloaded_config_file_hash_matches_manifest":
            sha256(DOWNLOAD / "config.json") == manifest["config_sha256"],
        "downloaded_rule_file_hash_matches_manifest":
            sha256(DOWNLOAD / "selection_rule.json") == manifest["selection_rule_sha256"],
        "downloaded_numerical_policy_file_hash_matches_manifest":
            sha256(DOWNLOAD / "numerical_protocol.json") == manifest["numerical_protocol_sha256"],
        "scientific_metadata_matches_epoch16": metadata_consistency,
        "original_run_checkpoint_sha256": sha256(CANONICAL / "run/last_checkpoint.pt"),
        "downloaded_source_checkpoint_sha256": sha256(DOWNLOAD / "run/last_checkpoint.pt"),
        "original_history_sha256": sha256(CANONICAL / "run/training_history.csv"),
        "downloaded_history_sha256": sha256(DOWNLOAD / "run/training_history.csv"),
        "recovery_runner_hash_matches_manifest": sha256(DOWNLOAD / "recover_epoch16.py") ==
            manifest["recovery_runner_sha256"],
        "archived_source_hash_matches_manifest": sha256(DOWNLOAD / "train_epoch13_source.py") ==
            manifest["source_runner_sha256"],
        "epoch14_checkpoint": epoch14_summary,
        "epoch15_checkpoint": epoch15_summary,
        "epoch16_checkpoint": epoch16_summary,
        "best_checkpoint": best_summary,
        "best_and_epoch16_model_tensors_equal": model_equal,
        "best_and_epoch16_optimizer_state_equal": optimizer_equal,
        "epoch16_original_metrics": {key: best["latest_validation_payload"]["metrics"][key] for key in (
            "validation_loss", "accuracy", "macro_f1", "mel_recall", "mel_f1",
            "nv_recall", "eligible", "failed_gates")},
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
