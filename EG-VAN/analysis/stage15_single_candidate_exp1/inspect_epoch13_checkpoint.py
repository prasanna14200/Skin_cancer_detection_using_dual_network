"""Read-only Stage 15 epoch-13 checkpoint and persisted-evidence audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/stage15_single_candidate_exp1"
RUN = EXP / "run"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bad_tensors(value, prefix=""):
    bad = []
    if isinstance(value, dict):
        for key, item in value.items():
            bad.extend(bad_tensors(item, f"{prefix}/{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            bad.extend(bad_tensors(item, f"{prefix}/{index}"))
    elif isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        if not bool(torch.isfinite(value).all()):
            bad.append(prefix)
    return bad


def main():
    state = torch.load(RUN / "last_checkpoint.pt", map_location="cpu", weights_only=False, mmap=True)
    with (RUN / "training_history.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    files = {}
    for epoch, expected in state.get("validation_artifacts", {}).items():
        pred = RUN / "validation_epochs" / f"epoch_{epoch:03d}_predictions.csv"
        metrics = RUN / "validation_epochs" / f"epoch_{epoch:03d}_metrics.json"
        files[epoch] = {"predictions_match": pred.is_file() and sha256(pred) == expected["predictions_sha256"],
                        "metrics_match": metrics.is_file() and sha256(metrics) == expected["metrics_sha256"]}
    scheduler = state["scheduler_state"]
    optimizer = state["optimizer_state"]
    scaler = state["scaler_state"]
    report = {
        "checkpoint_sha256": sha256(RUN / "last_checkpoint.pt"),
        "epoch": state["epoch"], "history_epochs": [row["epoch"] for row in state["history"]],
        "csv_epochs": [int(row["epoch"]) for row in rows],
        "checkpoint_history_matches_csv_prefix": all(
            all(text.get(key) == str(value) for key, value in saved.items())
            for text, saved in zip(rows, state["history"])),
        "best_epoch": state["best_epoch"], "best_validation_loss": state["best_validation_loss"],
        "model_nonfinite": bad_tensors(state["model_state"]),
        "optimizer_nonfinite": bad_tensors(optimizer),
        "model_state_tensors": len(state["model_state"]),
        "bn_buffer_count": sum(key.endswith((".running_mean", ".running_var", ".num_batches_tracked"))
                               for key in state["model_state"]),
        "scheduler": scheduler, "scheduler_valid": (
            scheduler["last_epoch"] == state["epoch"] and scheduler["mode"] == "min" and
            scheduler["factor"] == 0.5 and scheduler["patience"] == 1 and
            math.isfinite(float(scheduler["best"])) and
            scheduler["_last_lr"] == [optimizer["param_groups"][0]["lr"]]),
        "scaler": scaler, "scaler_valid": (
            math.isfinite(float(scaler["scale"])) and float(scaler["scale"]) > 0 and
            scaler["growth_factor"] > 1 and 0 < scaler["backoff_factor"] < 1 and
            scaler["growth_interval"] > 0 and scaler["_growth_tracker"] >= 0),
        "rng_types": {key: type(state.get(key)).__name__ for key in (
            "python_rng_state", "numpy_rng_state", "torch_rng_state", "cuda_rng_states",
            "sampler_generator_state")},
        "cuda_rng_count": len(state["cuda_rng_states"]),
        "rng_tensor_valid": all(isinstance(item, torch.Tensor) and item.dtype == torch.uint8
                                and item.ndim == 1 and item.numel() > 0 for item in (
                                    state["torch_rng_state"], state["sampler_generator_state"],
                                    *state["cuda_rng_states"])),
        "runner_sha256_saved": state["runner_sha256"],
        "runner_sha256_current": sha256(EXP / "train.py"),
        "runner_sha256_archived_original": sha256(EXP / "train_epoch13_source.py"),
        "config_sha256_saved": state["config_sha256"],
        "selection_rule_sha256_saved": state["selection_rule_sha256"],
        "numerical_protocol_sha256_saved": state["numerical_protocol_sha256"],
        "validation_file_matches": files,
        "best_checkpoint_exists": (RUN / "best_checkpoint.pt").exists(),
        "manifest_exists": (RUN / "experiment_manifest.json").exists(),
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
