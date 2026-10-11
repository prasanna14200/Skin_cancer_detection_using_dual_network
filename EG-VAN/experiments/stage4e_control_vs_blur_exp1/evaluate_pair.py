"""Validation-only paired Stage 4E analysis from frozen saved predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

import common

_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "stage4e_train_runner", Path(__file__).with_name("train.py")
)
if _RUNNER_SPEC is None or _RUNNER_SPEC.loader is None:
    raise ImportError("Cannot load the Stage 4E training runner")
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = runner
_RUNNER_SPEC.loader.exec_module(runner)


METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "mel_specificity")


def selected_arm(arm: str, expected_rows: list[dict]):
    directory = runner.RUNS / arm
    path = directory / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Complete {arm} run manifest required")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "COMPLETE_INTERNAL_SELECTION_ONLY" or
            manifest.get("arm") != arm or manifest.get("epochs") != 25 or
            manifest.get("config_sha256") != common.CONFIG_SHA or
            manifest.get("stage4d_manifest_sha256") != common.V2_MANIFEST_SHA or
            manifest.get("runner_sha256") != common.sha256(Path(runner.__file__)) or
            manifest.get("common_sha256") != common.sha256(common.HERE / "common.py")):
        raise ValueError(f"Invalid or changed {arm} run provenance")
    if set(manifest["epoch_commit_sha256"]) != {str(i) for i in range(1, 26)}:
        raise ValueError(f"Incomplete {arm} epoch commits")
    for epoch in range(1, 26):
        folder = runner.epoch_folder(directory, epoch)
        if common.sha256(folder / "commit.json") != manifest["epoch_commit_sha256"][str(epoch)]:
            raise ValueError(f"Changed {arm} epoch commit: {epoch}")
    history = common.read_csv(directory / "training_history.csv")
    if len(history) != 25 or [int(row["epoch"]) for row in history] != list(range(1, 26)):
        raise ValueError(f"Incomplete {arm} history")
    winner = min(history, key=lambda row: (float(row["val_loss"]), int(row["epoch"])))
    epoch = int(winner["epoch"])
    if (epoch != manifest["selected_epoch"] or
            float(winner["val_loss"]) != manifest["selected_selection_loss"] or
            manifest["selected_checkpoint"] != f"epochs/epoch_{epoch:03d}/checkpoint.pt"):
        raise ValueError(f"{arm} selected epoch violates minimum-loss rule")
    folder = runner.epoch_folder(directory, epoch)
    runner.verify_epoch(folder, common.read_config(), arm)
    if common.sha256(folder / "checkpoint.pt") != manifest["selected_checkpoint_sha256"]:
        raise ValueError(f"{arm} selected checkpoint hash differs")
    rows = common.read_csv(folder / "predictions.csv")
    metrics = common.validate_predictions(rows, expected_rows)
    saved = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    for key in METRICS + ("confusion_matrix",):
        if saved[key] != metrics[key]:
            raise ValueError(f"{arm} selected saved metric mismatch: {key}")
    return manifest, rows, metrics


def paired_bootstrap(control: list[dict], treatment: list[dict], source: list[dict],
                     *, seed: int = 20261010, repetitions: int = 2000):
    if len(control) != len(treatment) or len(control) != len(source):
        raise ValueError("Paired prediction counts differ")
    by_class = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(source):
        by_class[row["dx"]][row["lesion_id"]].append(index)
        if (control[index]["image_id"] != row["image_id"] or
                treatment[index]["image_id"] != row["image_id"]):
            raise ValueError("Paired prediction IDs/order differ")
    if set(by_class) != set(common.CLASSES):
        raise ValueError("Class missing from paired selection")
    rng = np.random.default_rng(seed)
    names = (*METRICS, "mel_recall")
    differences = {name: [] for name in names}
    for _ in range(repetitions):
        indices = []
        for name in common.CLASSES:
            groups = by_class[name]
            lesion_ids = sorted(groups)
            selected = rng.integers(0, len(lesion_ids), size=len(lesion_ids))
            for i in selected:
                indices.extend(groups[lesion_ids[int(i)]])
        c = common.metrics_from_prediction_rows([control[i] for i in indices])
        t = common.metrics_from_prediction_rows([treatment[i] for i in indices])
        for name in METRICS:
            differences[name].append(t[name] - c[name])
        differences["mel_recall"].append(t["per_class"]["mel"]["recall"] - c["per_class"]["mel"]["recall"])
    return {name: {"lower_95": float(np.quantile(values, 0.025, method="linear")),
                   "upper_95": float(np.quantile(values, 0.975, method="linear"))}
            for name, values in differences.items()}


def candidate_gate(deltas: dict, intervals: dict) -> bool:
    return (deltas["mel_recall"] > 0 and intervals["mel_recall"]["lower_95"] > 0 and
            deltas["macro_f1"] >= 0 and deltas["mel_specificity"] >= 0)


def compare():
    common.preflight(common.ROOT, hash_images=False)
    source = common.read_csv(common.V2 / common.SELECT_NAME)
    control_manifest, control, cm = selected_arm("control", source)
    treatment_manifest, treatment, tm = selected_arm("treatment", source)
    if control_manifest["initial_state_sha256"] != treatment_manifest["initial_state_sha256"]:
        raise ValueError("Fresh arms did not start from the same model state")
    intervals = paired_bootstrap(control, treatment, source)
    mel_c = cm["per_class"]["mel"]
    mel_t = tm["per_class"]["mel"]
    deltas = {name: tm[name] - cm[name] for name in METRICS}
    deltas["mel_recall"] = mel_t["recall"] - mel_c["recall"]
    candidate = candidate_gate(deltas, intervals)
    report = {
        "status": "INTERNAL_TREATMENT_CANDIDATE" if candidate else "NO_INTERNAL_CANDIDATE",
        "scope": "exploratory Stage4D v2 internal selection only; no Stage23 comparison",
        "control_selected_epoch": control_manifest["selected_epoch"],
        "treatment_selected_epoch": treatment_manifest["selected_epoch"],
        "common_initial_state_sha256": control_manifest["initial_state_sha256"],
        "stage4d_manifest_sha256": common.V2_MANIFEST_SHA,
        "selection_images": len(source),
        "selection_mel_images": mel_c["support"],
        "selection_non_mel_images": cm["mel_specificity_denominator"],
        "control": cm, "treatment": tm, "treatment_minus_control": deltas,
        "paired_lesion_bootstrap_percentile_95": intervals,
        "bootstrap_replicates": 2000, "bootstrap_seed": 20261010,
        "human_adjudicated_or_patient_disjoint": False,
        "ham_test_accessed": False, "ph2_accessed": False,
    }
    path = runner.RUNS / "paired_internal_selection_result.json"
    runner.write_once(path, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=common.ROOT)
    parser.add_argument("--analyze", action="store_true", required=True)
    args = parser.parse_args()
    if args.project_root.resolve() != common.ROOT.resolve():
        raise ValueError("Use pinned repository root")
    print(json.dumps(compare(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
