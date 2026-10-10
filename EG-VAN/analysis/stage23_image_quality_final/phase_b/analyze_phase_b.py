"""Analyze completed Phase B saved validation predictions; no model inference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRED = HERE / "phase_b_predictions.csv"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pct(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def read_complete():
    protocol = json.loads((HERE / "phase_b_protocol.json").read_text())
    table = pd.read_csv(PRED)
    cases = pd.DataFrame(protocol["cases"])
    conditions = [c["name"] for c in protocol["conditions"]]
    if len(table) != 147 or len(cases) != 21 or table.image_id.nunique() != 21 or set(table.condition) != set(conditions):
        raise ValueError("Incomplete Phase B inference; analysis stopped")
    if table.duplicated(["image_id", "condition"]).any() or table.groupby("image_id").condition.nunique().ne(7).any():
        raise ValueError("Duplicate or missing image/condition")
    chk = table.merge(cases, on=["image_id", "lesion_id", "true_class"], validate="many_to_one")
    if len(chk) != 147 or table.lesion_id.nunique() != 21:
        raise ValueError("Wrong frozen cases or lesion IDs")
    probs = np.asarray([json.loads(s) for s in table.probabilities], dtype=float)
    if probs.shape != (147,7) or not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid probability vectors")
    if (np.asarray(CLASSES)[probs.argmax(1)] != table.predicted_class.to_numpy()).any():
        raise ValueError("Predicted labels disagree with probabilities")
    baseline = table.loc[table.condition.eq("baseline")].set_index("image_id")
    if len(baseline) != 21 or (baseline.predicted_class != baseline.saved_stage23_predicted_class).any() or baseline.baseline_max_abs_probability_delta.max() > .005:
        raise ValueError("Baseline reproduction gate failed")
    table = table.copy()
    table["baseline_predicted_class"] = table.image_id.map(baseline.predicted_class)
    table["baseline_correct"] = table.image_id.map(baseline.correct).astype(int)
    table["baseline_entropy_nats"] = table.image_id.map(baseline.entropy_nats)
    table["prediction_changed"] = (table.predicted_class != table.baseline_predicted_class).astype(int)
    table["correct_to_incorrect"] = ((table.baseline_correct == 1) & (table.correct == 0)).astype(int)
    table["incorrect_to_correct"] = ((table.baseline_correct == 0) & (table.correct == 1)).astype(int)
    table["mel_false_negative"] = ((table.true_class == "mel") & (table.predicted_class != "mel")).astype(int)
    table["mel_to_nv"] = ((table.true_class == "mel") & (table.predicted_class == "nv")).astype(int)
    table["entropy_delta"] = table.entropy_nats - table.baseline_entropy_nats
    return protocol, table, baseline


def bootstrap_pair(part, mel=False, seed=2310, reps=1000):
    groups = part.groupby("lesion_id").indices
    ids = np.array(list(groups))
    rng = np.random.default_rng(seed)
    output = {"accuracy_delta": [], "flip_rate": [], "mean_entropy_delta": []}
    if mel:
        output["mel_recall_delta"] = []
    for _ in range(reps):
        take = np.concatenate([groups[k] for k in rng.choice(ids, len(ids), replace=True)])
        b = part.iloc[take]
        output["accuracy_delta"].append(float((b.correct - b.baseline_correct).mean()))
        output["flip_rate"].append(float(b.prediction_changed.mean()))
        output["mean_entropy_delta"].append(float(b.entropy_delta.mean()))
        if mel:
            output["mel_recall_delta"].append(float((b.correct-b.baseline_correct).mean()))
    return {k: pct(v) for k,v in output.items()}


def analyze():
    protocol, table, baseline = read_complete()
    # Enrich the saved inference table with explicit paired transition flags.
    table.to_csv(PRED, index=False, float_format="%.17g")
    names = [c["name"] for c in protocol["conditions"]]
    summaries = []
    intervals = {}
    per_class = {}
    for i, name in enumerate(names):
        sub = table.loc[table.condition.eq(name)].sort_values("image_id")
        mel = sub.loc[sub.true_class.eq("mel")]
        summaries.append({"condition": name, "images": len(sub), "lesions": sub.lesion_id.nunique(),
            "accuracy": float(sub.correct.mean()), "correct": int(sub.correct.sum()),
            "accuracy_delta_from_baseline": float((sub.correct-sub.baseline_correct).mean()),
            "mel_support": len(mel), "mel_recall": float(mel.correct.mean()), "mel_correct": int(mel.correct.sum()),
            "mel_false_negatives": int(mel.mel_false_negative.sum()), "mel_to_nv": int(mel.mel_to_nv.sum()),
            "prediction_flips": int(sub.prediction_changed.sum()), "prediction_flip_rate": float(sub.prediction_changed.mean()),
            "correct_to_incorrect": int(sub.correct_to_incorrect.sum()),
            "incorrect_to_correct": int(sub.incorrect_to_correct.sum()),
            "mean_entropy_nats": float(sub.entropy_nats.mean()), "median_entropy_nats": float(sub.entropy_nats.median()),
            "mean_entropy_delta": float(sub.entropy_delta.mean()), "median_entropy_delta": float(sub.entropy_delta.median())})
        intervals[name] = {"all_lesions": bootstrap_pair(sub, seed=2310+i),
                           "melanoma_lesions": bootstrap_pair(mel, mel=True, seed=2410+i)}
        per_class[name] = {cls: {"n": int(len(part)), "correct": int(part.correct.sum()),
                                   "accuracy": float(part.correct.mean()), "flips": int(part.prediction_changed.sum())}
                           for cls, part in sub.groupby("true_class")}
    summary = pd.DataFrame(summaries)
    summary.to_csv(HERE / "phase_b_summary.csv", index=False, float_format="%.17g")
    table.loc[table.true_class.eq("mel")].to_csv(HERE / "phase_b_melanoma_analysis.csv", index=False, float_format="%.17g")
    stats = {"status": "COMPLETE_BOUNDED_SAMPLE", "scope": "21 class-balanced validation images, 21 lesions; raw pre-preprocessing synthetic degradations",
             "baseline": {"argmax_matches_saved": int((baseline.predicted_class == baseline.saved_stage23_predicted_class).sum()),
                          "max_abs_probability_delta": float(baseline.baseline_max_abs_probability_delta.max()),
                          "processed_pixel_identity": "required and passed for all 21 by inference runner"},
             "conditions": summaries, "paired_lesion_bootstrap_95": intervals, "per_class": per_class,
             "bootstrap_seed": 2310, "bootstrap_repetitions": 1000,
             "interpretation": "Exploratory small class-balanced sample; not an unbiased 986-image validation estimate, not clinical image-quality validation",
             "training_performed": False, "ham_test_outcomes_accessed": False, "ph2_accessed": False}
    (HERE / "phase_b_statistical_results.json").write_text(json.dumps(stats, indent=2, allow_nan=False)+"\n")
    plot(summary)
    hashes = {"checkpoint": sha(ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt"),
              "frozen_split": sha(ROOT / "data/splits/split_leakage_aware.csv"),
              "validation_predictions": sha(ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_predictions.csv"),
              "model_registry": sha(ROOT / "models/frozen_stage23/model_registry.json"),
              "preprocessing_source": sha(ROOT / "src/preprocessing.py"),
              "frozen_loader_source": sha(ROOT / "models/frozen_stage23/load_frozen.py"),
              "protocol": sha(HERE / "phase_b_protocol.json"),
              "runner": sha(HERE / "run_phase_b.py"), "analysis": sha(Path(__file__))}
    output_files = ["phase_b_predictions.csv", "phase_b_summary.csv", "phase_b_melanoma_analysis.csv",
                    "phase_b_statistical_results.json", "phase_b_robustness_curves.png",
                    "phase_b_prediction_transitions.png", "phase_b_melanoma_robustness.png"]
    manifest = {"status": "COMPLETE_BOUNDED_SAMPLE", "input_and_source_sha256": hashes,
                "output_sha256": {f: sha(HERE/f) for f in output_files}, "sample_size": 21,
                "distinct_lesions": 21, "conditions": names, "predictions": 147,
                "training_performed": False, "test_or_ph2_accessed": False}
    (HERE / "phase_b_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"status": stats["status"], "baseline": stats["baseline"], "conditions": summaries}, indent=2))


def plot(summary):
    x = np.arange(len(summary))
    names = summary.condition.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(x, summary.accuracy, "o-", label="Accuracy, n=21")
    axes[0].plot(x, summary.mel_recall, "s-", label="MEL recall, n=3")
    axes[0].set(ylabel="Fraction correct", ylim=(0,1), title="Raw-image synthetic degradation")
    axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(x, summary.mean_entropy_delta, "o-")
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set(ylabel="Mean entropy change (nats)", title="Paired change from unchanged baseline")
    axes[1].grid(alpha=.2)
    for ax in axes: ax.set_xticks(x, names, rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(HERE / "phase_b_robustness_curves.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x-.2, summary.correct_to_incorrect, width=.4, label="Correct → incorrect")
    ax.bar(x+.2, summary.incorrect_to_correct, width=.4, label="Incorrect → correct")
    ax.set(xticks=x, xticklabels=names, ylabel="Cases of 21", title="Paired prediction transitions")
    ax.tick_params(axis="x", rotation=45); ax.legend(); fig.tight_layout()
    fig.savefig(HERE / "phase_b_prediction_transitions.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, summary.mel_correct, "o-", label="Correct MEL")
    ax.plot(x, summary.mel_to_nv, "s-", label="MEL → NV")
    ax.plot(x, summary.mel_false_negatives, "^-", label="MEL false negatives")
    ax.set(xticks=x, xticklabels=names, ylabel="Cases of 3", ylim=(0,3), title="Sampled melanoma cases; exploratory")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(HERE / "phase_b_melanoma_robustness.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    analyze()
