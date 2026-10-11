"""Stage 3B saved-probability audit. No model/image loading or inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
REPLICATES = 1000
SEED = 20261010


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_verified() -> dict:
    sys.path.insert(0, str(ROOT / "analysis/stage23_uncertainty_final"))
    sys.path.insert(0, str(ROOT / "analysis/stage23_external_validation"))
    from analyze_stage23_uncertainty import audit_inputs
    from audit_saved_followup import audit

    ham, hp, ham_evidence = audit_inputs()
    ph2_evidence = audit()
    if ph2_evidence["status"] != "PASS":
        raise ValueError("PH2 saved-run audit did not pass")
    old = ROOT / "analysis/stage23_uncertainty_final"
    old_manifest = json.loads((old / "analysis_manifest.json").read_text(encoding="utf-8"))
    for name in ("uncertainty_metrics.json", "risk_coverage.csv", "validation_uncertainty.csv", "reliability_diagram.png"):
        if sha(old / name) != old_manifest["output_sha256"][name]:
            raise ValueError(f"Historical uncertainty artifact changed: {name}")

    split = pd.read_csv(ROOT / "data/splits/split_leakage_aware.csv")
    val = split.loc[split.split == "val", ["image_id", "lesion_id"]]
    joined = ham[["image_id"]].merge(val, on="image_id", how="left", validate="one_to_one")
    if joined.lesion_id.isna().any() or len(joined) != len(ham):
        raise ValueError("HAM lesion linkage missing")
    ph2_path = ROOT / "analysis/stage23_external_validation/run/predictions.csv"
    ph2 = pd.read_csv(ph2_path)
    pp = ph2[[f"p_{name}" for name in CLASSES]].to_numpy(dtype=float)
    true_ham = ham.true_label.to_numpy(str)
    pred_ham = ham.predicted_label.to_numpy(str)
    true_ph2 = ph2.true_ham_label.to_numpy(str)
    pred_ph2 = ph2.predicted_class.to_numpy(str)
    if (hp.shape != (986, 7) or pp.shape != (120, 7) or
            not np.array_equal(np.asarray(CLASSES)[hp.argmax(axis=1)], pred_ham) or
            not np.array_equal(np.asarray(CLASSES)[pp.argmax(axis=1)], pred_ph2) or
            set(true_ph2) != {"nv", "mel"}):
        raise ValueError("Verified predictions no longer match seven-class argmax")
    return {
        "HAM_validation": {"ids": ham.image_id.to_numpy(str), "groups": joined.lesion_id.to_numpy(str),
                           "group_unit": "lesion_id", "truth": true_ham, "pred": pred_ham, "probs": hp},
        "PH2_followup": {"ids": ph2.image_id.to_numpy(str), "groups": ph2.image_id.to_numpy(str),
                         "group_unit": "image_id; patient IDs unavailable", "truth": true_ph2,
                         "pred": pred_ph2, "probs": pp},
        "evidence": {"ham": ham_evidence, "ph2_status": ph2_evidence["status"],
                     "ph2_predictions_sha256": sha(ph2_path),
                     "historical_uncertainty_manifest_sha256": sha(old / "analysis_manifest.json")},
    }


def measurements(d: dict) -> dict:
    p = d["probs"]
    n = len(p)
    truth = d["truth"]
    pred = d["pred"]
    true_idx = np.fromiter((CLASSES.index(x) for x in truth), dtype=int, count=n)
    correct = truth == pred
    confidence = p.max(axis=1)
    entropy = -(p * np.log(np.maximum(p, np.finfo(float).tiny))).sum(axis=1)
    sorted_p = np.sort(p, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    return {"correct": correct, "confidence": confidence, "entropy": entropy,
            "margin": margin, "true_idx": true_idx,
            "scores": {"predictive_entropy": entropy, "one_minus_max_softmax": 1-confidence,
                       "negative_top_two_margin": -margin}}


def calibration(p: np.ndarray, m: dict, idx: np.ndarray) -> dict:
    q = p[idx]
    correct = m["correct"][idx]
    conf = m["confidence"][idx]
    truth = m["true_idx"][idx]
    n = len(idx)
    target = np.eye(7)[truth]
    bins = []
    ece = 0.0
    for b in range(10):
        low, high = b / 10, (b+1) / 10
        mask = (conf >= low) & ((conf < high) if b < 9 else (conf <= high))
        count = int(mask.sum())
        mean_conf = float(conf[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count/n * abs(mean_conf-accuracy)
        bins.append({"low": low, "high": high, "n": count, "mean_confidence": mean_conf,
                     "top_label_accuracy": accuracy})
    return {"top_label_ece": float(ece),
            "multiclass_brier_sum": float(np.sum((q-target)**2, axis=1).mean()),
            "negative_log_likelihood_nats": float(-np.log(np.maximum(q[np.arange(n), truth],
                                                     np.finfo(float).tiny)).mean()),
            "bins": bins}


def error_detection(m: dict, idx: np.ndarray, score_name: str) -> tuple[float, float]:
    y = (~m["correct"][idx]).astype(int)
    if y.min() == y.max():
        return math.nan, math.nan
    score = m["scores"][score_name][idx]
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def bootstrap_indices(groups: np.ndarray, rng: np.random.Generator):
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == j) for j in range(len(unique))]
    for _ in range(REPLICATES):
        sampled = rng.integers(0, len(unique), len(unique))
        yield np.concatenate([members[j] for j in sampled])


def interval(values: list[float]) -> tuple[float | None, float | None, int]:
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not len(finite):
        return None, None, 0
    return float(np.quantile(finite, .025)), float(np.quantile(finite, .975)), len(finite)


def analyze_one(name: str, d: dict) -> tuple[list[dict], list[dict], dict]:
    m = measurements(d)
    p = d["probs"]
    full = np.arange(len(p))
    mel = np.flatnonzero(d["truth"] == "mel")
    correct_mel = int(m["correct"][mel].sum())
    errors_mel = len(mel)-correct_mel
    rng = np.random.default_rng(SEED + (0 if name == "HAM_validation" else 1))
    cal = calibration(p, m, full)
    boot_cal = {k: [] for k in ("top_label_ece", "multiclass_brier_sum", "negative_log_likelihood_nats")}
    for idx in bootstrap_indices(d["groups"], rng):
        x = calibration(p, m, idx)
        for key in boot_cal:
            boot_cal[key].append(x[key])
    rows_cal = []
    for key, values in boot_cal.items():
        lo, hi, valid = interval(values)
        rows_cal.append({"dataset": name, "metric": key, "n": len(p), "estimate": cal[key],
                         "ci95_low": lo, "ci95_high": hi, "valid_bootstrap_replicates": valid,
                         "resample_unit": d["group_unit"]})
    rows_detect = []
    for scope, subset in (("all_cases", full), ("true_melanoma_only", mel)):
        groups = d["groups"][subset]
        for score in m["scores"]:
            auc, ap = error_detection(m, subset, score)
            samples_auc, samples_ap = [], []
            rng_sub = np.random.default_rng(SEED + (0 if name == "HAM_validation" else 1)
                                            + (10 if scope == "all_cases" else 20))
            for sub_idx in bootstrap_indices(groups, rng_sub):
                a, b = error_detection(m, subset[sub_idx], score)
                samples_auc.append(a)
                samples_ap.append(b)
            for metric, estimate, vals in (("AUROC", auc, samples_auc), ("average_precision", ap, samples_ap)):
                lo, hi, valid = interval(vals)
                rows_detect.append({"dataset": name, "scope": scope, "score": score, "metric": metric,
                                    "n": len(subset), "errors": int((~m["correct"][subset]).sum()),
                                    "error_prevalence": float((~m["correct"][subset]).mean()),
                                    "estimate": estimate, "ci95_low": lo, "ci95_high": hi,
                                    "valid_bootstrap_replicates": valid,
                                    "resample_unit": d["group_unit"]})
    wrong = ~m["correct"]
    detail = {"n": len(p), "groups": int(len(np.unique(d["groups"]))),
              "errors": int(wrong.sum()), "mel_total": len(mel), "mel_correct": correct_mel,
              "mel_false_negatives": errors_mel,
              "confidence_ge_0_8_errors": int((wrong & (m["confidence"] >= .8)).sum()),
              "confidence_ge_0_9_errors": int((wrong & (m["confidence"] >= .9)).sum()),
              "confidence_ge_0_8_mel_false_negatives": int(((~m["correct"][mel]) & (m["confidence"][mel] >= .8)).sum()),
              "confidence_ge_0_9_mel_false_negatives": int(((~m["correct"][mel]) & (m["confidence"][mel] >= .9)).sum()),
              "mel_correct_median_entropy": float(np.median(m["entropy"][mel][m["correct"][mel]])),
              "mel_miss_median_entropy": float(np.median(m["entropy"][mel][~m["correct"][mel]])),
              "mel_correct_median_confidence": float(np.median(m["confidence"][mel][m["correct"][mel]])),
              "mel_miss_median_confidence": float(np.median(m["confidence"][mel][~m["correct"][mel]])),
              "mel_miss_above_correct_median_entropy": int((m["entropy"][mel][~m["correct"][mel]] >
                                                      np.median(m["entropy"][mel][m["correct"][mel]])).sum()),
              "calibration_bins": cal["bins"]}
    return rows_cal, rows_detect, detail


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_reliability(name: str, bins: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect top-label calibration")
    nonempty = [b for b in bins if b["n"]]
    ax.scatter([b["mean_confidence"] for b in nonempty],
               [b["top_label_accuracy"] for b in nonempty],
               s=[max(30, 3*b["n"]) for b in nonempty], alpha=.65, label="Fixed 0.1-width bins")
    for b in nonempty:
        ax.annotate(str(b["n"]), (b["mean_confidence"], b["top_label_accuracy"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set(xlim=(-.02, 1.07), ylim=(-.02, 1.07), xlabel="Mean maximum softmax score",
           ylabel="Observed top-label accuracy", title=f"{name.replace('_', ' ')}: descriptive reliability")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Exclusively create derived Stage 3B outputs")
    args = parser.parse_args()
    datasets = load_verified()
    cal_rows, det_rows, details = [], [], {}
    for name in ("HAM_validation", "PH2_followup"):
        c, e, detail = analyze_one(name, datasets[name])
        cal_rows.extend(c)
        det_rows.extend(e)
        details[name] = detail
    if args.write:
        names = ("CALIBRATION_METRICS.csv", "ERROR_DETECTION_METRICS.csv", "stage3b_details.json",
                 "ham_reliability_diagram.png", "ph2_reliability_diagram.png")
        if any((OUT / n).exists() for n in names):
            raise FileExistsError("Refusing to overwrite an existing Stage 3B derived output")
        write_csv(OUT / names[0], cal_rows)
        write_csv(OUT / names[1], det_rows)
        with (OUT / names[2]).open("x", encoding="utf-8") as f:
            json.dump({"status": "VERIFIED_SAVED_PREDICTION_AUDIT", "seed": SEED,
                       "bootstrap_replicates": REPLICATES, "source_evidence": datasets["evidence"],
                       "details": details}, f, indent=2, allow_nan=False)
            f.write("\n")
        for name, file in (("HAM_validation", names[3]), ("PH2_followup", names[4])):
            plot_reliability(name, details[name]["calibration_bins"], OUT / file)
    print(json.dumps({"status": "VERIFIED_SAVED_PREDICTION_AUDIT", "details": {k: {j:v for j,v in d.items()
                       if j != "calibration_bins"} for k,d in details.items()}}, indent=2))


if __name__ == "__main__":
    main()
