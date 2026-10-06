"""Stage 20 analysis of immutable saved probabilities. No model is loaded."""

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
from sklearn.metrics import roc_auc_score

CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_predictions(path: Path, kind: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path)
    if frame.empty or frame.image_id.isna().any() or frame.image_id.duplicated().any():
        raise ValueError(f"Missing/duplicate IDs: {path}")
    if kind == "validation":
        probs = np.asarray([json.loads(value) for value in frame.probabilities], dtype=float)
        true = frame.true_label.to_numpy()
        pred = frame.predicted_label.to_numpy()
    else:
        probs = frame[[f"p_{c}" for c in CLASSES]].to_numpy(dtype=float)
        true = frame.true_class.to_numpy() if kind == "ham" else frame.true_ham_label.to_numpy()
        pred = frame.predicted_class.to_numpy()
    if probs.shape != (len(frame), len(CLASSES)):
        raise ValueError(f"Wrong probability shape: {path} {probs.shape}")
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError(f"Invalid probabilities: {path}")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError(f"Probabilities do not sum to 1: {path}")
    if not set(true).issubset(CLASSES) or not set(pred).issubset(CLASSES):
        raise ValueError(f"Unknown label: {path}")
    if not np.array_equal(np.asarray(CLASSES)[probs.argmax(axis=1)], pred):
        raise ValueError(f"Saved prediction disagrees with argmax: {path}")
    correct = true == pred
    if "correct" in frame.columns:
        raw = frame.correct.astype(str).str.lower().to_numpy()
        if not np.array_equal(np.where(correct, "true", "false"), raw):
            raise ValueError(f"Saved correctness disagrees: {path}")
    frame["_true"] = true
    frame["_pred"] = pred
    frame["_correct"] = correct
    frame["_confidence"] = probs.max(axis=1)
    frame["_entropy"] = -(np.where(probs > 0, probs * np.log(np.clip(probs, 1e-300, 1)), 0)).sum(axis=1)
    return frame, probs


def calibration(frame: pd.DataFrame, probs: np.ndarray) -> dict:
    true_idx = np.asarray([CLASSES.index(label) for label in frame._true])
    conf = frame._confidence.to_numpy()
    correct = frame._correct.to_numpy().astype(float)
    bins = []
    ece = 0.0
    for b in range(10):
        low, high = b / 10, (b + 1) / 10
        mask = (conf >= low) & ((conf < high) if b < 9 else (conf <= high))
        n = int(mask.sum())
        mean_conf = float(conf[mask].mean()) if n else None
        acc = float(correct[mask].mean()) if n else None
        if n:
            ece += n / len(frame) * abs(mean_conf - acc)
        bins.append({"lower": low, "upper": high, "count": n, "mean_confidence": mean_conf, "accuracy": acc})
    target = np.eye(len(CLASSES))[true_idx]
    error = ~frame._correct.to_numpy()
    msp_uncertainty = 1 - conf
    entropy = frame._entropy.to_numpy()
    return {
        "n": len(frame), "correct": int((~error).sum()), "accuracy": float(correct.mean()),
        "ece_10_bins": float(ece), "calibration_bins": bins,
        "brier_7class_sum": float(np.mean(np.sum((probs - target) ** 2, axis=1))),
        "nll": float(-np.log(np.clip(probs[np.arange(len(frame)), true_idx], 1e-12, 1)).mean()),
        "error_detection_auroc": {
            "one_minus_msp": float(roc_auc_score(error, msp_uncertainty)) if error.any() and (~error).any() else None,
            "entropy": float(roc_auc_score(error, entropy)) if error.any() and (~error).any() else None,
        },
        "mean_confidence_correct": float(conf[~error].mean()) if (~error).any() else None,
        "mean_confidence_incorrect": float(conf[error].mean()) if error.any() else None,
    }


def plot_calibration(result: dict, path: Path, title: str) -> None:
    bins = [b for b in result["calibration_bins"] if b["count"]]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.scatter([b["mean_confidence"] for b in bins], [b["accuracy"] for b in bins],
               s=[max(25, b["count"] * 0.45) for b in bins], color="#226a92", label="confidence bins")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed accuracy", title=title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=350)
    plt.close(fig)


def plot_confidence(frame: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for flag, label, color in ((True, "Correct", "#26734d"), (False, "Incorrect", "#a63838")):
        ax.hist(frame.loc[frame._correct == flag, "_confidence"], bins=np.linspace(0, 1, 21),
                alpha=.55, label=label, color=color)
    ax.set(xlabel="Maximum softmax probability", ylabel="Images", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=350)
    plt.close(fig)


def score(frame: pd.DataFrame, metric: str) -> np.ndarray:
    return 1 - frame._confidence.to_numpy() if metric == "one_minus_msp" else frame._entropy.to_numpy()


def operating_point(frame: pd.DataFrame, metric: str, threshold: float) -> dict:
    reviewed = score(frame, metric) > threshold
    accepted = ~reviewed
    errors = ~frame._correct.to_numpy()
    mel = frame._true.to_numpy() == "mel"
    mel_fn = mel & errors
    return {
        "n": len(frame), "accepted": int(accepted.sum()), "reviewed": int(reviewed.sum()),
        "coverage": float(accepted.mean()), "review_rate": float(reviewed.mean()),
        "retained_accuracy": float(frame._correct.to_numpy()[accepted].mean()) if accepted.any() else None,
        "retained_error_rate": float(errors[accepted].mean()) if accepted.any() else None,
        "all_errors": int(errors.sum()), "errors_reviewed": int((reviewed & errors).sum()),
        "fraction_errors_reviewed": float((reviewed & errors).sum() / errors.sum()) if errors.any() else None,
        "mel_cases": int(mel.sum()), "mel_cases_reviewed": int((reviewed & mel).sum()),
        "mel_false_negatives": int(mel_fn.sum()),
        "mel_false_negatives_reviewed": int((reviewed & mel_fn).sum()),
    }


def plot_risk(frame: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    order = np.argsort(score(frame, metric), kind="stable")
    errors = (~frame._correct.to_numpy())[order]
    n = len(errors)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, n + 1) / n, np.cumsum(errors) / np.arange(1, n + 1), color="#226a92")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Coverage", ylabel="Retained error rate", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=350)
    plt.close(fig)


def quality_analysis(root: Path, val: pd.DataFrame) -> dict:
    quality_path = root / "experiments/image_quality/image_quality.csv"
    q = pd.read_csv(quality_path)
    if q.image_id.duplicated().any():
        raise ValueError("Duplicate quality IDs")
    merged = val.merge(q, on="image_id", how="left", validate="one_to_one")
    if merged.brightness.isna().any() or not (merged.split == "val").all():
        raise ValueError("Validation quality join/split failed")
    out = {"source_sha256": digest(quality_path), "joined_validation_rows": len(merged),
           "scope": "Processed HAM image technical proxies; descriptive association only; no binary quality gate",
           "features": {}}
    for feature in ("brightness", "contrast", "sharpness", "saturation", "dark_pixel_ratio",
                    "bright_pixel_ratio", "entropy", "illumination_variation"):
        x = merged[feature]
        item = {
            "mean_correct": float(x[merged._correct].mean()),
            "mean_incorrect": float(x[~merged._correct].mean()),
            "spearman_vs_confidence": float(x.corr(merged._confidence, method="spearman")),
            "spearman_vs_predictive_entropy": float(x.corr(merged._entropy, method="spearman")),
        }
        if feature in {"brightness", "contrast", "sharpness"}:
            tercile = pd.qcut(x.rank(method="first"), 3, labels=["low", "middle", "high"])
            item["validation_terciles"] = {}
            for label in ("low", "middle", "high"):
                g = merged[tercile == label]
                mel_fn = (g._true == "mel") & (~g._correct)
                item["validation_terciles"][label] = {
                    "n": len(g), "error_rate": float((~g._correct).mean()),
                    "mean_confidence": float(g._confidence.mean()),
                    "mean_predictive_entropy": float(g._entropy.mean()),
                    "mel_false_negative_count": int(mel_fn.sum()),
                }
        out["features"][feature] = item
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("validation", "evaluation"), required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = root / "analysis/stage20_reliability_completion"
    out.mkdir(parents=True, exist_ok=True)
    val_path = root / "experiments/stage15_single_candidate_exp1/stage15_single_candidate_exp1/run/validation_epochs/epoch_016_predictions.csv"
    protocol_path = out / "uncertainty_protocol.json"
    if args.phase == "validation":
        if protocol_path.exists():
            raise FileExistsError("Validation rule already frozen; refusing to overwrite")
        frame, probs = load_predictions(val_path, "validation")
        if len(frame) != 986:
            raise ValueError("Unexpected validation count")
        result = calibration(frame, probs)
        auc = result["error_detection_auroc"]
        metric = "entropy" if auc["entropy"] > auc["one_minus_msp"] else "one_minus_msp"
        values = np.sort(score(frame, metric))
        threshold = float(values[math.ceil(.8 * len(values)) - 1])
        protocol = {"checkpoint_sha256": "85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5",
                    "class_order": CLASSES, "validation_predictions": str(val_path.relative_to(root)),
                    "validation_predictions_sha256": digest(val_path), "metric": metric,
                    "selection": "higher validation error-detection AUROC of 1-MSP vs entropy; exact tie=1-MSP",
                    "coverage_target": .8, "threshold": threshold,
                    "accept_if": "uncertainty_score <= threshold; ties accepted",
                    "validation_auroc": auc, "validation_operating_point": operating_point(frame, metric, threshold),
                    "test_or_ph2_used_for_selection": False}
        write_json(out / "validation_reliability.json", result)
        write_json(out / "quality_analysis.json", quality_analysis(root, frame))
        plot_calibration(result, out / "validation_reliability_diagram.png", "Stage 15 selected validation")
        plot_confidence(frame, out / "validation_confidence_distribution.png", "Validation confidence by correctness")
        plot_risk(frame, metric, out / "validation_risk_coverage.png", "Validation risk–coverage (descriptive)")
        write_json(protocol_path, protocol)
        print(f"VALIDATION_RULE_FROZEN metric={metric} threshold={threshold:.17g} sha256={digest(protocol_path)}")
        return
    if not protocol_path.exists():
        raise FileNotFoundError("Freeze validation rule before evaluation")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if digest(val_path) != protocol["validation_predictions_sha256"]:
        raise ValueError("Validation data changed after rule freeze")
    output = {"protocol_sha256": digest(protocol_path), "datasets": {}}
    for kind, name in (("ham", "final_ham_test_predictions.csv"), ("ph2", "final_ph2_predictions.csv")):
        path = root / "analysis/stage16_final_evaluation" / name
        frame, probs = load_predictions(path, kind)
        expected = 1014 if kind == "ham" else 120
        if len(frame) != expected:
            raise ValueError(f"Unexpected {kind} count")
        cal = calibration(frame, probs)
        op = operating_point(frame, protocol["metric"], protocol["threshold"])
        output["datasets"][kind] = {"predictions_sha256": digest(path), "calibration": cal,
                                    "frozen_review_rule": op,
                                    "interpretation": "PH2 mapped NV/MEL cohort; other HAM predictions count incorrect; exploratory external follow-up" if kind == "ph2" else "Held-out HAM; post-Stage-16 exploratory reliability analysis"}
        plot_calibration(cal, out / f"{kind}_reliability_diagram.png", f"{kind.upper()} reliability (descriptive)")
        plot_confidence(frame, out / f"{kind}_confidence_distribution.png", f"{kind.upper()} confidence by correctness")
        plot_risk(frame, protocol["metric"], out / f"{kind}_risk_coverage.png", f"{kind.upper()} risk–coverage (descriptive)")
    write_json(out / "reliability_analysis.json", output)
    print(f"EVALUATION_COMPLETE protocol_sha256={output['protocol_sha256']}")


if __name__ == "__main__":
    main()
