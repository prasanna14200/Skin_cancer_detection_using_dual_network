"""Read-only Stage 23 validation-probability uncertainty analysis.

Never loads a model or images. No HAM test or PH2 data path is referenced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/stage23_uncertainty_final"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CHECKPOINT_SHA = "42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab"
SPLIT_SHA = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
PREDICTIONS = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_predictions.csv"
SPLIT = ROOT / "data/splits/split_leakage_aware.csv"
METRICS = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_metrics.json"
REGISTRY = ROOT / "models/frozen_stage23/model_registry.json"
MANIFEST = ROOT / "models/frozen_stage23/freeze_manifest.json"
CHECKPOINT = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt"
COVERAGES = (1.0, .9, .8, .7, .6, .5)
RANDOM_SEED = 42
RANDOM_REPETITIONS = 500


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audit_inputs() -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Fail closed before producing any analysis artifact."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = registry["source_artifact_sha256"]
    checks = {
        "checkpoint": (CHECKPOINT, CHECKPOINT_SHA),
        "validation_predictions": (PREDICTIONS, expected[str(PREDICTIONS.relative_to(ROOT)).replace("\\", "/")]),
        "validation_metrics": (METRICS, expected[str(METRICS.relative_to(ROOT)).replace("\\", "/")]),
        "frozen_split": (SPLIT, SPLIT_SHA),
        "model_registry": (REGISTRY, manifest["model_registry_sha256"]),
    }
    hashes = {}
    for name, (path, expected_hash) in checks.items():
        actual = sha256(path)
        if actual != expected_hash:
            raise ValueError(f"{name} SHA256 mismatch: {actual} != {expected_hash}")
        hashes[name] = actual
    if (registry["model_identifier"] != "EGVAN_STAGE23_FINAL" or
            registry["selected_epoch"] != 14 or
            tuple(registry["class_order"]) != CLASSES or
            manifest["frozen_checkpoint_sha256"] != CHECKPOINT_SHA or
            registry["checkpoint"]["sha256"] != CHECKPOINT_SHA):
        raise ValueError("Frozen Stage 23 identity mismatch")

    saved = json.loads(METRICS.read_text(encoding="utf-8"))
    frame = pd.read_csv(PREDICTIONS)
    required = {"image_id", "true_label", "predicted_label", "correct", "probabilities"}
    if not required.issubset(frame.columns) or len(frame) != 986 or frame.image_id.isna().any() or frame.image_id.duplicated().any():
        raise ValueError("Validation prediction schema/count/IDs invalid")
    split = pd.read_csv(SPLIT)
    val = split.loc[split.split == "val", ["image_id", "dx"]]
    if len(val) != 986 or val.image_id.duplicated().any() or set(frame.image_id) != set(val.image_id):
        raise ValueError("Saved prediction IDs do not equal frozen validation IDs")
    merged = frame[["image_id", "true_label"]].merge(val, on="image_id", validate="one_to_one")
    if not (merged.true_label == merged.dx).all():
        raise ValueError("True labels disagree with frozen split")
    if (set(frame.true_label) - set(CLASSES) or set(frame.predicted_label) - set(CLASSES) or
            tuple(saved["class_order"]) != CLASSES or saved["sample_count"] != 986):
        raise ValueError("Class order or labels invalid")
    try:
        probs = np.asarray([json.loads(item) for item in frame.probabilities], dtype=np.float64)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid probability JSON") from exc
    if (probs.shape != (986, 7) or not np.isfinite(probs).all() or
            ((probs < 0) | (probs > 1)).any() or
            not np.allclose(probs.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Probability matrix is incomplete or invalid")
    predicted = np.asarray(CLASSES)[probs.argmax(axis=1)]
    correct = frame.true_label.to_numpy() == frame.predicted_label.to_numpy()
    saved_correct = frame.correct.astype(str).str.lower().to_numpy()
    if not np.array_equal(predicted, frame.predicted_label.to_numpy()) or not np.array_equal(np.where(correct, "true", "false"), saved_correct):
        raise ValueError("Saved predicted labels/correctness disagree with probabilities")
    accuracy = float(correct.mean())
    mel_tp = int(((frame.true_label == "mel") & (frame.predicted_label == "mel")).sum())
    mel_n = int((frame.true_label == "mel").sum())
    if (not math.isclose(accuracy, saved["accuracy"], abs_tol=1e-12) or
            mel_tp != 70 or mel_n != 107 or
            not math.isclose(saved["accuracy"], registry["selected_validation_metrics"]["accuracy"], abs_tol=1e-12)):
        raise ValueError("Saved validation metrics do not match predictions/frozen registry")
    frame = frame.copy()
    frame["correct_bool"] = correct
    return frame, probs, {"sha256": hashes, "manifest_sha256": sha256(MANIFEST),
                           "sample_count": len(frame), "unique_ids": frame.image_id.nunique(),
                           "split_match": True, "true_labels_match_split": True,
                           "class_order": CLASSES, "accuracy_recomputed": accuracy,
                           "mel_tp": mel_tp, "mel_support": mel_n,
                           "probability_checks": "all finite, in [0,1], row sums within 1e-5"}


def summaries(values: np.ndarray) -> dict:
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {"n": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def calibration(frame: pd.DataFrame, probs: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    true = frame.true_label.to_numpy()
    correct = frame.correct_bool.to_numpy()
    true_index = np.asarray([CLASSES.index(label) for label in true])
    confidence = probs.max(axis=1)
    positive = probs > 0
    entropy = -np.sum(np.where(positive, probs * np.log(np.maximum(probs, np.finfo(float).tiny)), 0), axis=1)
    target = np.eye(7)[true_index]
    nll = float(-np.log(np.maximum(probs[np.arange(len(probs)), true_index], np.finfo(float).tiny)).mean())
    bins = []
    ece = 0.0
    for i in range(10):
        low, high = i / 10, (i + 1) / 10
        mask = (confidence >= low) & ((confidence < high) if i < 9 else (confidence <= high))
        count = int(mask.sum())
        mean_conf = float(confidence[mask].mean()) if count else None
        acc = float(correct[mask].mean()) if count else None
        if count:
            ece += count / len(frame) * abs(mean_conf - acc)
        bins.append({"lower": low, "upper": high, "upper_inclusive": i == 9,
                     "count": count, "mean_confidence": mean_conf, "empirical_accuracy": acc})
    error = ~correct
    detection = {}
    for name, score in (("entropy", entropy), ("one_minus_msp", 1 - confidence)):
        detection[name] = {
            "auroc": float(roc_auc_score(error, score)) if error.any() and (~error).any() else None,
            "average_precision": float(average_precision_score(error, score)) if error.any() else None,
        }
    result = {
        "scope": "Stage 23 selected epoch-14 HAM validation only; exploratory",
        "classification_accuracy": float(correct.mean()),
        "errors": int(error.sum()), "error_prevalence": float(error.mean()),
        "multiclass_brier_sum": float(np.mean(np.sum((probs - target) ** 2, axis=1))),
        "brier_convention": "Mean over images of sum over seven classes (p_k - 1[y=k])^2; unnormalized seven-class sum",
        "negative_log_likelihood_nats": nll,
        "top_label_ece_10_equal_width": float(ece),
        "ece_convention": "10 bins [i/10,(i+1)/10), last includes 1; sum n_bin/n * abs(mean max-softmax - observed top-label accuracy); empty bins contribute 0",
        "calibration_bins": bins,
        "error_detection": {"positive_class": "misclassification", "average_precision_baseline": float(error.mean()), **detection},
        "confidence_by_correctness": {"correct": summaries(confidence[correct]), "incorrect": summaries(confidence[error])},
        "entropy_by_correctness": {"correct": summaries(entropy[correct]), "incorrect": summaries(entropy[error])},
        "normalized_entropy_definition": "Shannon entropy in nats divided by ln(7)",
        "normalized_entropy_by_correctness": {"correct": summaries((entropy / math.log(7))[correct]),
                                              "incorrect": summaries((entropy / math.log(7))[error])},
        "interpretation": "Softmax scores are uncalibrated research scores, not clinical probabilities; all results reuse the selection validation set",
    }
    return result, confidence, entropy


def ranked_indices(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Lowest uncertainty first; lexicographically smaller ID wins exact score ties."""
    return np.lexsort((ids.astype(str), scores))


def operating_point(frame: pd.DataFrame, keep: np.ndarray, method: str, target: float,
                    repetition: int | None = None) -> dict:
    n = len(frame)
    retained = np.zeros(n, dtype=bool)
    retained[keep] = True
    reviewed = ~retained
    true = frame.true_label.to_numpy()
    pred = frame.predicted_label.to_numpy()
    correct = frame.correct_bool.to_numpy()
    errors = ~correct
    mel = true == "mel"
    mel_tp = mel & (pred == "mel")
    mel_fn = mel & (pred != "mel")
    mel_retained = int((mel & retained).sum())
    row = {
        "method": method, "target_coverage": target, "repetition": repetition,
        "retained": int(retained.sum()), "reviewed": int(reviewed.sum()),
        "actual_coverage": float(retained.mean()),
        "retained_accuracy": float(correct[retained].mean()) if retained.any() else None,
        "retained_risk": float(errors[retained].mean()) if retained.any() else None,
        "retained_macro_f1": float(f1_score(true[retained], pred[retained], labels=CLASSES, average="macro", zero_division=0)) if retained.any() else None,
        "retained_mel_true_count": mel_retained,
        "retained_mel_true_positives": int((mel_tp & retained).sum()),
        "retained_mel_recall": float((mel_tp & retained).sum() / mel_retained) if mel_retained else None,
        "total_mel_true_count": int(mel.sum()),
        "total_mel_false_negatives": int(mel_fn.sum()),
        "mel_false_negatives_reviewed": int((mel_fn & reviewed).sum()),
        "mel_false_negatives_retained": int((mel_fn & retained).sum()),
        "total_errors": int(errors.sum()),
        "errors_reviewed": int((errors & reviewed).sum()),
        "fraction_total_errors_reviewed": float((errors & reviewed).sum() / errors.sum()) if errors.any() else None,
    }
    return row


def risk_curve(errors: np.ndarray, order: np.ndarray) -> np.ndarray:
    return np.cumsum(errors[order]) / np.arange(1, len(order) + 1)


def selective_analysis(frame: pd.DataFrame, confidence: np.ndarray, entropy: np.ndarray) -> tuple[pd.DataFrame, dict, dict]:
    n = len(frame)
    ids = frame.image_id.to_numpy()
    errors = ~frame.correct_bool.to_numpy()
    orders = {"entropy": ranked_indices(entropy, ids),
              "maximum_softmax_confidence": ranked_indices(1 - confidence, ids)}
    curves = {name: risk_curve(errors, order) for name, order in orders.items()}
    rows = []
    for name, order in orders.items():
        for coverage in COVERAGES:
            count = math.ceil(coverage * n)
            rows.append(operating_point(frame, order[:count], name, coverage))
    rng = np.random.default_rng(RANDOM_SEED)
    random_curves = np.zeros(n, dtype=np.float64)
    random_rows: dict[float, list[dict]] = {c: [] for c in COVERAGES}
    for repeat in range(RANDOM_REPETITIONS):
        order = rng.permutation(n)
        random_curves += risk_curve(errors, order)
        for coverage in COVERAGES:
            count = math.ceil(coverage * n)
            random_rows[coverage].append(operating_point(frame, order[:count], "random_review", coverage, repeat))
    curves["random_review_mean"] = random_curves / RANDOM_REPETITIONS
    numeric_fields = ["retained_accuracy", "retained_risk", "retained_macro_f1", "retained_mel_true_count",
                      "retained_mel_true_positives", "retained_mel_recall", "mel_false_negatives_reviewed",
                      "mel_false_negatives_retained", "errors_reviewed", "fraction_total_errors_reviewed"]
    for coverage in COVERAGES:
        replicates = random_rows[coverage]
        row = {key: replicates[0][key] for key in replicates[0] if key not in numeric_fields}
        row["method"] = "random_review_mean"
        row["repetition"] = f"mean_of_{RANDOM_REPETITIONS}_seed_{RANDOM_SEED}"
        for key in numeric_fields:
            values = np.asarray([r[key] for r in replicates if r[key] is not None], dtype=float)
            row[key] = float(values.mean()) if len(values) else None
        rows.append(row)
    summary = {
        "scope": "Exploratory Stage 23 validation selective prediction; no cutoff promoted",
        "coverage_targets": COVERAGES,
        "retained_count_rule": "ceil(target_coverage * 986), therefore actual coverage is at least target",
        "score_and_order": "Retain lowest entropy or lowest 1-max-softmax; exact ties broken by ascending image_id",
        "retained_macro_f1_convention": "Fixed seven-class macro F1; absent classes have F1=0 via zero_division=0",
        "retained_mel_recall_denominator": "Number of true MEL cases retained, shown in retained_mel_true_count; null if zero. Not 107 unless coverage=100%.",
        "aurc_convention": "Discrete mean of retained error rate for every prefix k=1..N, i.e. sum_{k=1}^N risk(k)/N; no k=0 point or trapezoidal interpolation",
        "aurc": {name: float(np.mean(values)) for name, values in curves.items()},
        "random_baseline": {"seed": RANDOM_SEED, "repetitions": RANDOM_REPETITIONS,
                            "order": "Uniform permutation of fixed saved-prediction row order per repetition; averages reported",
                            "expected_risk_at_each_coverage": float(errors.mean())},
    }
    return pd.DataFrame(rows), summary, curves


def melanoma_analysis(frame: pd.DataFrame, probs: np.ndarray, confidence: np.ndarray, entropy: np.ndarray) -> tuple[pd.DataFrame, dict]:
    mel = frame.true_label.to_numpy() == "mel"
    correct = frame.correct_bool.to_numpy()
    pred = frame.predicted_label.to_numpy()
    ids = frame.image_id.to_numpy()
    subset = pd.DataFrame({
        "image_id": ids[mel], "true_class": "mel", "predicted_class": pred[mel],
        "correct": correct[mel], "error_type": np.where(correct[mel], "correct_mel", np.char.add("mel_to_", pred[mel].astype(str))),
        "confidence": confidence[mel], "predictive_entropy_nats": entropy[mel],
        "normalized_entropy": entropy[mel] / math.log(7), "p_mel": probs[mel, CLASSES.index("mel")],
    }).sort_values("image_id")
    mel_fn = mel & ~correct
    mel_correct = mel & correct
    mel_to_nv = mel & (pred == "nv")
    fn_conf = confidence[mel_fn]
    fn_entropy = entropy[mel_fn]
    # Descriptive overlap with correctly classified MEL, not a deployment cutoff.
    median_correct_entropy = float(np.median(entropy[mel_correct]))
    confident_wrong = int((fn_entropy <= median_correct_entropy).sum())
    detection = {
        "entropy_auroc_for_mel_false_negative_vs_correct_mel": float(roc_auc_score(mel_fn[mel], entropy[mel])) if mel_fn.any() and mel_correct.any() else None,
        "entropy_average_precision_for_mel_false_negative_vs_correct_mel": float(average_precision_score(mel_fn[mel], entropy[mel])) if mel_fn.any() else None,
        "comparison_scope": "Among 107 true MEL only; positive means MEL false negative",
    }
    summary = {
        "mel_total": int(mel.sum()), "mel_correct": int(mel_correct.sum()),
        "mel_false_negatives": int(mel_fn.sum()), "mel_to_nv": int(mel_to_nv.sum()),
        "correct_mel_confidence": summaries(confidence[mel_correct]),
        "false_negative_confidence": summaries(fn_conf),
        "correct_mel_entropy": summaries(entropy[mel_correct]),
        "false_negative_entropy": summaries(fn_entropy),
        "mel_error_detection": detection,
        "low_entropy_false_negatives_vs_correct_mel_median": confident_wrong,
        "median_entropy_correct_mel": median_correct_entropy,
        "low_entropy_definition": "FN entropy <= median entropy of correctly classified true MEL; descriptive overlap, not review cutoff",
    }
    return subset, summary


def make_figures(metrics: dict, frame: pd.DataFrame, confidence: np.ndarray,
                 entropy: np.ndarray, curves: dict) -> None:
    bins = [b for b in metrics["calibration_bins"] if b["count"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot([b["mean_confidence"] for b in bins], [b["empirical_accuracy"] for b in bins],
            "o-", color="#1b6c8e", label="Stage 23 validation")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean maximum softmax score", ylabel="Observed top-label accuracy",
           title="Validation reliability (exploratory)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "reliability_diagram.png", dpi=350); plt.close(fig)

    x = np.arange(1, len(frame) + 1) / len(frame)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, color in (("entropy", "#20688f"), ("maximum_softmax_confidence", "#bc6039"),
                        ("random_review_mean", "#777777")):
        label = {"entropy": "Entropy", "maximum_softmax_confidence": "1 − max-softmax",
                 "random_review_mean": "Random review mean"}[name]
        ax.plot(x, curves[name], label=label, color=color,
                linewidth=1.8 if name != "random_review_mean" else 1.3)
    ax.set(xlim=(0, 1), ylim=(0, .25), xlabel="Retained coverage", ylabel="Retained error rate",
           title="Stage 23 validation risk–coverage (exploratory)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "risk_coverage.png", dpi=350); plt.close(fig)

    correct = frame.correct_bool.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, title, edges in ((axes[0], confidence, "Maximum softmax score", np.linspace(0, 1, 21)),
                                     (axes[1], entropy, "Predictive entropy (nats)", np.linspace(0, math.log(7), 21))):
        ax.hist(values[correct], bins=edges, alpha=.6, label="Correct", color="#26734d")
        ax.hist(values[~correct], bins=edges, alpha=.6, label="Incorrect", color="#a63838")
        ax.set(xlabel=title, ylabel="Validation images")
        ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "uncertainty_distributions.png", dpi=350); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.project_root.resolve() != ROOT.resolve():
        raise ValueError("Only the pinned EG-VAN repository root is accepted")
    frame, probs, integrity = audit_inputs()
    metrics, confidence, entropy = calibration(frame, probs)
    rows, selective, curves = selective_analysis(frame, confidence, entropy)
    mel_rows, mel_summary = melanoma_analysis(frame, probs, confidence, entropy)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "uncertainty_metrics.json", {"integrity": integrity, "metrics": metrics,
                                                   "selective_prediction": selective, "melanoma": mel_summary,
                                                   "training_performed": False, "inference_performed": False,
                                                   "ham_test_outcomes_accessed": False, "ph2_outcomes_accessed": False})
    rows.to_csv(OUT / "risk_coverage.csv", index=False, float_format="%.17g")
    mel_rows.to_csv(OUT / "melanoma_error_analysis.csv", index=False, float_format="%.17g")
    pd.DataFrame({"image_id": frame.image_id, "true_label": frame.true_label,
                  "predicted_label": frame.predicted_label, "correct": frame.correct_bool,
                  "maximum_softmax_probability": confidence,
                  "predictive_entropy_nats": entropy,
                  "normalized_predictive_entropy": entropy / math.log(7)}).to_csv(
                      OUT / "validation_uncertainty.csv", index=False, float_format="%.17g")
    make_figures(metrics, frame, confidence, entropy, curves)
    generated = ("uncertainty_metrics.json", "risk_coverage.csv", "melanoma_error_analysis.csv",
                 "validation_uncertainty.csv", "reliability_diagram.png", "risk_coverage.png",
                 "uncertainty_distributions.png")
    write_json(OUT / "analysis_manifest.json", {
        "scope": "Stage 23 selected epoch-14 validation saved probabilities only",
        "analysis_source_sha256": sha256(Path(__file__).resolve()),
        "input_sha256": integrity["sha256"],
        "output_sha256": {name: sha256(OUT / name) for name in generated},
        "training_performed": False, "inference_performed": False,
        "ham_test_outcomes_accessed": False, "ph2_outcomes_accessed": False,
    })
    print(json.dumps({"status": "COMPLETE_VALIDATION_ONLY", "n": len(frame),
                      "accuracy": metrics["classification_accuracy"],
                      "errors": metrics["errors"], "mel_false_negatives": mel_summary["mel_false_negatives"],
                      "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
