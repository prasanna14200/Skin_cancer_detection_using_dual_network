"""Build evaluator tables/figures from validated saved Stage 23 validation records only."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
PILOT = SOURCE.parent / "phase_b"
sys.path.insert(0, str(SOURCE))
from analyze_cuda_full_validation import validate_final_table  # noqa: E402
from run_cuda_full_validation import (CLASSES, FROZEN, frozen_inputs, sha)  # noqa: E402


def verified_manifest():
    manifest = json.loads((SOURCE / "cuda_execution_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "COMPLETE_FULL_VALIDATION" or manifest["predictions"] != 6902:
        raise ValueError("CUDA execution manifest is incomplete")
    for name, expected in manifest["output_sha256"].items():
        if sha(SOURCE / name) != expected:
            raise ValueError(f"Downloaded CUDA artifact hash mismatch: {name}")
    for name, key in (("run_cuda_full_validation.py", "runner_sha256"),
                      ("analyze_cuda_full_validation.py", "analysis_sha256")):
        if sha(SOURCE / name) != manifest[key]:
            raise ValueError(f"CUDA source hash mismatch: {name}")
    pilot_manifest = json.loads((PILOT / "phase_b_manifest.json").read_text(encoding="utf-8"))
    for name in ("phase_b_predictions.csv", "phase_b_summary.csv", "phase_b_statistical_results.json"):
        if sha(PILOT / name) != pilot_manifest["output_sha256"][name]:
            raise ValueError(f"Pilot artifact hash mismatch: {name}")
    return manifest, pilot_manifest


def main():
    manifest, pilot_manifest = verified_manifest()
    protocol, validation, reference, hashes = frozen_inputs()
    if hashes["checkpoint"] != manifest["checkpoint_sha256"] or sha(FROZEN) != manifest["protocol_sha256"]:
        raise ValueError("Frozen input/protocol provenance mismatch")
    table, gate = validate_final_table(protocol, validation, reference, sha(FROZEN))
    if gate["max_abs_probability_delta"] != 0 or gate["class_disagreements"]:
        raise ValueError("Baseline gate is not an exact reference replay")
    statistical = json.loads((SOURCE / "cuda_statistical_results.json").read_text(encoding="utf-8"))
    reported = {x["condition"]: x for x in statistical["conditions"]}
    intervals = statistical["paired_lesion_cluster_bootstrap_95"]
    names = [x["name"] for x in protocol["degradation_conditions"]]
    baseline = table.loc[table.condition.eq("baseline")].set_index("image_id")
    comparison, melanoma, uncertainty = [], [], []
    for name in names:
        sub = table.loc[table.condition.eq(name)].set_index("image_id").loc[baseline.index]
        y, pred = sub.true_class.to_numpy(), sub.predicted_class.to_numpy()
        mel = y == "mel"
        correct = pred == y
        base_correct = baseline.predicted_class.to_numpy() == y
        base_mel_correct = mel & (baseline.predicted_class.to_numpy() == "mel")
        mel_correct = mel & (pred == "mel")
        values = reported[name]
        actual = {
            "accuracy": float(correct.mean()),
            "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
            "mel_recall": float(mel_correct.sum() / mel.sum()),
            "prediction_flips": int((pred != baseline.predicted_class.to_numpy()).sum()),
            "mean_entropy_nats": float(sub.entropy_nats.mean()),
            "mel_to_nv": int((mel & (pred == "nv")).sum()),
        }
        for key, value in actual.items():
            if not np.isclose(value, values[key], rtol=0, atol=1e-12):
                raise ValueError(f"Saved summary disagrees with predictions: {name}/{key}")
        row = {"condition": name, "images": len(sub), "lesions": sub.lesion_id.nunique(),
               "accuracy": values["accuracy"], "accuracy_delta": values["accuracy_delta"],
               "macro_f1": values["macro_f1"], "macro_f1_delta": values["macro_f1_delta"],
               "mel_correct": int(mel_correct.sum()), "mel_support": int(mel.sum()),
               "mel_recall": values["mel_recall"], "mel_recall_delta": values["mel_recall_delta"],
               "prediction_flips": values["prediction_flips"],
               "prediction_flip_rate": values["prediction_flip_rate"],
               "mean_entropy_nats": values["mean_entropy_nats"],
               "median_entropy_nats": values["median_entropy_nats"],
               "mean_entropy_delta": values["mean_entropy_delta"],
               "correct_to_incorrect": values["correct_to_incorrect"],
               "incorrect_to_correct": values["incorrect_to_correct"]}
        for key in ("accuracy_delta", "macro_f1_delta", "mel_recall_delta",
                    "prediction_flip_rate", "mean_entropy_delta"):
            lo, hi = intervals[name][key]["percentile_95"]
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = lo, hi
        comparison.append(row)
        melanoma.append({"condition": name, "mel_support": int(mel.sum()),
                         "mel_correct": int(mel_correct.sum()),
                         "mel_false_negatives": int((mel & ~mel_correct).sum()),
                         "mel_to_nv": int((mel & (pred == "nv")).sum()),
                         "baseline_mel_correct_to_wrong": int((base_mel_correct & ~mel_correct).sum()),
                         "baseline_mel_wrong_to_correct": int((mel & ~base_mel_correct & mel_correct).sum())})
        error = (~correct).astype(int)
        uncertainty.append({"condition": name, "error_count": int(error.sum()),
                            "error_prevalence": float(error.mean()),
                            "entropy_error_auroc": float(roc_auc_score(error, sub.entropy_nats)),
                            "entropy_error_average_precision": float(average_precision_score(error, sub.entropy_nats)),
                            "median_entropy_correct": float(sub.entropy_nats[correct].median()),
                            "median_entropy_incorrect": float(sub.entropy_nats[~correct].median())})
    comp = pd.DataFrame(comparison)
    mel_table = pd.DataFrame(melanoma)
    uncert = pd.DataFrame(uncertainty)
    pilot = pd.read_csv(PILOT / "phase_b_predictions.csv")
    pilot_rows = []
    for name in names:
        small = pilot.loc[pilot.condition.eq(name)]
        full_subset = table.loc[table.condition.eq(name) & table.image_id.isin(small.image_id)]
        paired = small.merge(full_subset, on="image_id", suffixes=("_pilot", "_cuda"), validate="one_to_one")
        if len(paired) != 21 or not paired.true_class_pilot.eq(paired.true_class_cuda).all():
            raise ValueError(f"Pilot/full subset cannot be paired: {name}")
        pilot_rows.append({"condition": name, "pilot_images": len(small),
                           "pilot_accuracy": float(small.correct.mean()),
                           "full_accuracy": float(comp.loc[comp.condition.eq(name), "accuracy"].iloc[0]),
                           "pilot_mel_correct": int(((small.true_class == "mel") & (small.predicted_class == "mel")).sum()),
                           "full_mel_correct": int(comp.loc[comp.condition.eq(name), "mel_correct"].iloc[0]),
                           "same_21_case_prediction_disagreements": int((paired.predicted_class_pilot != paired.predicted_class_cuda).sum())})
    pilot_table = pd.DataFrame(pilot_rows)
    if pilot_table.same_21_case_prediction_disagreements.sum() != 0:
        raise ValueError("Pilot and CUDA classifications differ on matched cases")
    HERE.mkdir(exist_ok=True)
    original_files = sorted(path for path in SOURCE.rglob("*") if path.is_file()
                            and HERE not in path.parents and "__pycache__" not in path.parts)
    inventory = pd.DataFrame([{"relative_path": path.relative_to(SOURCE).as_posix(),
                               "bytes": path.stat().st_size, "sha256": sha(path)}
                              for path in original_files])
    inventory.to_csv(HERE / "original_artifact_inventory.csv", index=False)
    outputs = {"condition_comparison.csv": comp, "melanoma_transitions.csv": mel_table,
               "entropy_error_discrimination.csv": uncert, "pilot_full_comparison.csv": pilot_table}
    for name, frame in outputs.items():
        frame.to_csv(HERE / name, index=False, float_format="%.17g")
    labels = [n.replace("_", " ") for n in names[1:]]
    x = np.arange(1, len(names))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    for ax, key, label in zip(axes, ("accuracy_delta", "mel_recall_delta", "mean_entropy_delta"),
                              ("Accuracy change", "Melanoma recall change", "Mean entropy change (nats)")):
        vals = comp[key].to_numpy()[1:]
        low = comp[f"{key}_ci_low"].to_numpy()[1:]
        high = comp[f"{key}_ci_high"].to_numpy()[1:]
        ax.errorbar(x, vals, yerr=np.vstack((vals-low, high-vals)), fmt="o", capsize=4)
        ax.axhline(0, color="black", linewidth=1)
        ax.set(xticks=x, xticklabels=labels, ylabel=label)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Stage 23 validation: paired synthetic degradation effects (95% lesion-cluster bootstrap intervals)")
    fig.savefig(HERE / "paired_effects_with_cluster_intervals.png", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.plot(x, uncert.entropy_error_auroc.to_numpy()[1:], "o-", label="Entropy AUROC for error detection")
    ax.axhline(float(uncert.entropy_error_auroc.iloc[0]), linestyle="--", color="gray", label="Baseline AUROC")
    ax.set(xticks=x, xticklabels=labels, ylim=(.5, 1), ylabel="AUROC", title="Entropy ranks classification errors within each validation condition")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(HERE / "entropy_error_discrimination.png", dpi=300)
    plt.close(fig)
    provenance = {"scope": "Stage 23 validation-only saved-record analysis", "status": "SOURCE_INTEGRITY_PASS",
                  "cuda_manifest_sha256": sha(SOURCE / "cuda_execution_manifest.json"),
                  "pilot_manifest_sha256": sha(PILOT / "phase_b_manifest.json"),
                  "frozen_checkpoint_sha256": hashes["checkpoint"],
                  "frozen_split_sha256": hashes["split"],
                  "frozen_protocol_sha256": sha(FROZEN),
                  "source_prediction_sha256": sha(SOURCE / "cuda_full_validation_predictions.csv"),
                  "results_builder_sha256": sha(Path(__file__)),
                  "batch_directories_validated": 434, "prediction_records_validated": len(table),
                  "original_files_inventoried": len(inventory),
                  "output_sha256": {name: sha(HERE / name) for name in [*outputs, "original_artifact_inventory.csv",
                      "paired_effects_with_cluster_intervals.png", "entropy_error_discrimination.png"]},
                  "training_performed": False, "inference_performed": False,
                  "ham_test_or_ph2_accessed": False}
    (HERE / "derived_results_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DERIVED_RESULTS_COMPLETE", "records": len(table),
                      "conditions": names, "output": str(HERE)}, indent=2))


if __name__ == "__main__":
    main()
