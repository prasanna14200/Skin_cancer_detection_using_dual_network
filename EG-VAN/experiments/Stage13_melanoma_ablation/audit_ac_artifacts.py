"""Read-only Stage 13B artifact audit for completed A and C Colab folders."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import torch

import train_ablation as stage13


def finite_tensors(value, prefix="") -> dict:
    result = {"tensors": 0, "nonfinite_tensors": 0, "nan_elements": 0, "inf_elements": 0, "examples": []}

    def walk(obj, path):
        if isinstance(obj, torch.Tensor):
            result["tensors"] += 1
            if obj.is_floating_point() or obj.is_complex():
                nans = int(torch.isnan(obj).sum().item())
                infs = int(torch.isinf(obj).sum().item())
                if nans or infs:
                    result["nonfinite_tensors"] += 1
                    result["nan_elements"] += nans
                    result["inf_elements"] += infs
                    if len(result["examples"]) < 5:
                        result["examples"].append({"path": path, "nan": nans, "inf": infs})
        elif isinstance(obj, dict):
            for key, item in obj.items():
                walk(item, path + "/" + str(key))
        elif isinstance(obj, (list, tuple)):
            for index, item in enumerate(obj):
                walk(item, path + "/" + str(index))

    walk(value, prefix)
    result["all_finite"] = result["nonfinite_tensors"] == 0
    return result


def checkpoint(path: Path, variant: str, expected_epoch: int | None = None) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model = finite_tensors(state["model_state"], "model_state")
    optimizer = finite_tensors(state["optimizer_state"], "optimizer_state")
    scaler = state["scaler_state"]
    scale = float(scaler.get("scale", float("nan")))
    scheduler = state["scheduler_state"]
    scheduler_valid = (math.isfinite(float(scheduler.get("_last_lr", [float("nan")])[0]))
                       and math.isfinite(float(scheduler.get("best", float("nan")))))
    result = {"sha256": stage13.sha256(path), "epoch": state["epoch"],
              "variant_matches": state["variant"] == variant,
              "configuration_matches": state["configuration"] == stage13.variant_config(variant),
              "class_order_matches": state["class_order"] == list(stage13.CLASSES),
              "model": model, "optimizer": optimizer,
              "scheduler_valid": scheduler_valid, "scheduler_best": scheduler.get("best"),
              "amp_scale": scale, "amp_scaler_valid": math.isfinite(scale) and scale > 0,
              "epoch_matches": expected_epoch is None or state["epoch"] == expected_epoch}
    del state
    return result


def audit_variant(root: Path, variant: str, val_ids: dict[str, str]) -> dict:
    out = root / "experiments/egvan_melanoma_ablation_exp" / variant
    config = stage13.read_json(out / "config.json")
    manifest = stage13.read_json(out / "experiment_manifest.json")
    with (out / "training_history.csv").open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in history]
    losses_finite = all(math.isfinite(float(row[key])) for row in history for key in ("train_loss", "val_loss"))
    lr_finite = all(math.isfinite(float(row["learning_rate"])) for row in history)
    eligible_epochs = [int(row["epoch"]) for row in history if row["eligible"] == "True"]
    result = {"variant": variant, "artifact_directory": str(out),
              "configuration_matches_preregistered": config == stage13.variant_config(variant),
              "history_epochs": epochs, "history_complete": epochs == list(range(1, 26)),
              "train_and_val_losses_all_finite": losses_finite, "learning_rates_all_finite": lr_finite,
              "eligible_epochs": eligible_epochs,
              "manifest_status": manifest["status"],
              "manifest_boundary_flags_clear": all(manifest.get(key) is False for key in
                  ("ham_test_accessed", "ph2_accessed", "test_inference_performed", "ph2_inference_performed")),
              "split_sha256_matches": manifest["split_sha256"] == stage13.SPLIT_SHA,
              "last_checkpoint": checkpoint(out / "last_checkpoint.pt", variant, 25)}
    result["last_checkpoint_hash_matches_manifest"] = (
        result["last_checkpoint"]["sha256"] == manifest["last_checkpoint_sha256"])

    best_path = out / "best_checkpoint.pt"
    if best_path.exists():
        best = checkpoint(best_path, variant, manifest["selected_epoch"])
        result["best_checkpoint"] = best
        result["best_checkpoint_hash_matches_manifest"] = best["sha256"] == manifest["best_checkpoint_sha256"]
        chosen = min((row for row in history if row["eligible"] == "True"),
                     key=lambda row: (float(row["val_loss"]), int(row["epoch"])))
        result["selected_epoch_matches_rule"] = int(chosen["epoch"]) == manifest["selected_epoch"]
        selected = stage13.read_json(out / "validation_metrics.json")
        with (out / "validation_predictions.csv").open(newline="", encoding="utf-8") as handle:
            predictions = list(csv.DictReader(handle))
        ids_match = (len(predictions) == len(val_ids) and
                     {row["image_id"]: row["true_label"] for row in predictions} == val_ids)
        matrix = [[0] * 7 for _ in range(7)]
        probs_valid = correct_valid = True
        for row in predictions:
            true = stage13.CLASSES.index(row["true_label"])
            pred = stage13.CLASSES.index(row["predicted_label"])
            matrix[true][pred] += 1
            probs = json.loads(row["probabilities"])
            probs_valid &= (len(probs) == 7 and all(isinstance(p, (int, float)) and
                math.isfinite(p) and 0 <= p <= 1 for p in probs) and
                math.isclose(sum(probs), 1, abs_tol=1e-4) and
                max(range(7), key=lambda i: probs[i]) == pred)
            correct_valid &= row["correct"] == str(true == pred)
        derived = stage13.metrics(matrix)
        metric_matches = all(math.isclose(float(selected[key]), derived[key], abs_tol=1e-12)
                             for key in ("accuracy", "balanced_accuracy", "macro_f1"))
        metric_matches &= all(math.isclose(float(selected["per_class"][name][key]),
                             derived["per_class"][name][key], abs_tol=1e-12)
                             for name in stage13.CLASSES for key in ("support", "precision", "recall", "f1"))
        result["selected_validation"] = {"ids_match_frozen_val": ids_match,
            "probabilities_valid": bool(probs_valid), "correct_flags_valid": bool(correct_valid),
            "metrics_recomputed_and_match": bool(metric_matches),
            "selected_epoch_matches_manifest": selected["epoch"] == manifest["selected_epoch"],
            "selected_loss_matches_history": math.isclose(float(selected["val_loss"]),
                float(chosen["val_loss"]), rel_tol=1e-3, abs_tol=1e-4)}
    else:
        result["best_checkpoint"] = None
        result["no_best_consistent_with_no_eligible_epoch"] = (
            not eligible_epochs and manifest["selected_epoch"] is None and
            manifest["status"] == "FAIL_NO_ELIGIBLE_CHECKPOINT" and
            not (out / "validation_metrics.json").exists() and
            not (out / "validation_predictions.csv").exists())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=stage13.ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    root = args.project_root.resolve()
    if stage13.sha256(root / "data/splits/split_leakage_aware.csv") != stage13.SPLIT_SHA:
        raise ValueError("Frozen split changed")
    with (root / "data/splits/split_leakage_aware.csv").open(newline="", encoding="utf-8") as handle:
        val_ids = {row["image_id"]: row["dx"] for row in csv.DictReader(handle) if row["split"] == "val"}
    if len(val_ids) != 986 or Counter(val_ids.values()) != Counter(stage13.VAL_COUNTS):
        raise ValueError("Frozen validation partition changed")
    results = {variant: audit_variant(root, variant, val_ids) for variant in ("A", "C")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
