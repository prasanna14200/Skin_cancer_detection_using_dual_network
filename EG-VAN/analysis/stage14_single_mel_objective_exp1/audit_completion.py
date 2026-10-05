"""Read-only audit of the downloaded Stage 14 validation-only run."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/stage14_single_mel_objective_exp1"
RUN = EXPERIMENT / "run"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonfinite_tensors(value, prefix=""):
    bad = []
    if isinstance(value, dict):
        for key, item in value.items():
            bad.extend(nonfinite_tensors(item, f"{prefix}/{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            bad.extend(nonfinite_tensors(item, f"{prefix}/{index}"))
    elif isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        if not bool(torch.isfinite(value).all()):
            bad.append(prefix)
    return bad


def failures(row, rule):
    floor = rule["eligible_if_all"]
    mel = float(row["val_mel_recall"])
    checks = [
        ("MEL_recall", round(mel * 107) >= 63 and mel + 1e-12 >= floor["melanoma_recall_minimum"]),
        ("MEL_F1", float(row["val_mel_f1"]) + 1e-12 >= floor["melanoma_f1_minimum"]),
        ("macro_F1", float(row["val_macro_f1"]) + 1e-12 >= floor["macro_f1_minimum"]),
        ("accuracy", float(row["val_accuracy"]) + 1e-12 >= floor["accuracy_minimum"]),
        ("NV_recall", float(row["val_nv_recall"]) + 1e-12 >= floor["nevus_recall_minimum"]),
    ]
    return [name for name, passed in checks if not passed]


def main():
    manifest = read_json(RUN / "experiment_manifest.json")
    rule = read_json(EXPERIMENT / "selection_rule.json")
    config = read_json(EXPERIMENT / "config.json")
    numerical = read_json(EXPERIMENT / "numerical_protocol.json")
    state = torch.load(RUN / "last_checkpoint.pt", map_location="cpu", weights_only=False, mmap=True)
    with (RUN / "training_history.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mismatches = []
    for index, (row, saved) in enumerate(zip(rows, state["history"]), 1):
        for key, value in saved.items():
            if row.get(key) != str(value):
                mismatches.append([index, key, row.get(key), str(value)])
    evaluations = [{
        "epoch": int(row["epoch"]),
        "failed": failures(row, rule),
        "mel_correct": round(float(row["val_mel_recall"]) * 107),
        "val_loss": float(row["val_loss"]),
    } for row in rows]

    def extreme(field, minimum=False):
        row = (min if minimum else max)(rows, key=lambda item: float(item[field]))
        return {"epoch": int(row["epoch"]), "value": float(row[field])}

    artifacts = {name: {
        "actual": sha256(RUN / name),
        "manifest": expected,
        "match": sha256(RUN / name) == expected,
    } for name, expected in manifest["artifact_sha256"].items()}
    model_state = state["model_state"]
    buffer_names = [name for name in model_state if name.endswith((
        ".running_mean", ".running_var", ".num_batches_tracked"))]
    optimizer_state = state["optimizer_state"]
    scheduler = state["scheduler_state"]
    scaler = state["scaler_state"]
    group = optimizer_state["param_groups"][0]
    scheduler_valid = (scheduler["mode"] == "min" and scheduler["factor"] == 0.5
                       and scheduler["patience"] == 1 and scheduler["last_epoch"] == 25
                       and math.isclose(scheduler["best"], min(float(row["val_loss"]) for row in rows))
                       and scheduler["_last_lr"] == [group["lr"]]
                       and math.isfinite(scheduler["best"]))
    scaler_valid = (math.isfinite(scaler["scale"]) and scaler["scale"] > 0
                    and scaler["growth_factor"] > 1 and 0 < scaler["backoff_factor"] < 1
                    and isinstance(scaler["_growth_tracker"], int)
                    and scaler["_growth_tracker"] >= 0)
    random.Random().setstate(state["python_rng_state"])
    np.random.RandomState().set_state(state["numpy_rng_state"])
    torch.Generator(device="cpu").set_state(state["torch_rng_state"])
    torch.Generator(device="cpu").set_state(state["sampler_generator_state"])
    rng_tensors_valid = all(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
                            and value.ndim == 1 and value.numel() > 0 for value in (
                                state["torch_rng_state"], state["sampler_generator_state"],
                                *state["cuda_rng_states"]))
    output = {
        "state_keys": sorted(state),
        "checkpoint_epoch": state["epoch"],
        "checkpoint_history_rows": len(state["history"]),
        "csv_rows": len(rows),
        "csv_epoch_sequence": [int(row["epoch"]) for row in rows],
        "csv_state_mismatches": mismatches,
        "train_val_finite": all(math.isfinite(float(row[key])) for row in rows for key in ("train_loss", "val_loss")),
        "metrics_finite": all(math.isfinite(float(row[key])) for row in rows for key in (
            "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "val_mel_precision",
            "val_mel_recall", "val_mel_f1", "val_nv_recall")),
        "model_nonfinite": nonfinite_tensors(state["model_state"]),
        "optimizer_nonfinite": nonfinite_tensors(state["optimizer_state"]),
        "model_state_tensors": len(model_state),
        "batchnorm_buffer_count": len(buffer_names),
        "batchnorm_buffer_nonfinite": nonfinite_tensors({name: model_state[name] for name in buffer_names}),
        "optimizer_slots": len(optimizer_state["state"]),
        "optimizer_group": {key: group[key] for key in ("lr", "betas", "eps", "weight_decay")},
        "scheduler_valid": scheduler_valid,
        "scaler_valid": scaler_valid,
        "scheduler": state["scheduler_state"],
        "scaler": state["scaler_state"],
        "rng": {"python": type(state.get("python_rng_state")).__name__,
                "numpy": type(state.get("numpy_rng_state")).__name__,
                "torch": type(state.get("torch_rng_state")).__name__,
                "cuda_count": len(state.get("cuda_rng_states", [])),
                "sampler": type(state.get("sampler_generator_state")).__name__},
        "rng_structures_restorable": rng_tensors_valid,
        "configuration_equal": state["configuration"] == config == read_json(RUN / "config.json"),
        "rule_snapshot_equal": rule == read_json(RUN / "selection_rule.json"),
        "numerical_snapshot_equal": numerical == read_json(RUN / "numerical_protocol.json"),
        "best_epoch": state["best_epoch"],
        "best_validation_loss": state["best_validation_loss"],
        "artifacts": artifacts,
        "runner_sha_current": sha256(EXPERIMENT / "train.py"),
        "runner_sha_saved": state["runner_sha256"],
        "eligibility": evaluations,
        "eligibility_matches_checkpoint": all((not item["failed"]) is saved["eligible"]
                                              for item, saved in zip(evaluations, state["history"])),
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in rows),
        "amp_skips": sum(int(row["amp_skips"]) for row in rows),
        "max_mel_recall": extreme("val_mel_recall"),
        "max_mel_f1": extreme("val_mel_f1"),
        "min_val_loss": extreme("val_loss", True),
        "max_macro_f1": extreme("val_macro_f1"),
        "events_equal": state["numerical_events"] == read_json(RUN / "numerical_events.json"),
        "best_checkpoint_exists": (RUN / "best_checkpoint.pt").exists(),
        "validation_metrics_exists": (RUN / "validation_metrics.json").exists(),
        "validation_predictions_exists": (RUN / "validation_predictions.csv").exists(),
    }
    print(json.dumps(output, default=str, indent=2))


if __name__ == "__main__":
    main()
