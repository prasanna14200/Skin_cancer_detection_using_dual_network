"""Recompute Stage 23 PH2 follow-up evidence from immutable saved CSV/JSON only.

No model loading, image processing, training, or inference. Writes derived audit
files only to final_report/ with exclusive creation when --write-derived is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
RUN = BASE / "run"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
SHA_EXPECTED = {
    "checkpoint_sha256": "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt",
    "split_sha256": "data/splits/split_leakage_aware.csv",
    "protocol_sha256": "analysis/stage23_external_validation/phase23_ph2_protocol.json",
    "ph2_manifest_sha256": "analysis/ph2_external_validation_exp5/ph2_manifest.csv",
    "metadata_sha256": "data/external/ph2/metadata/PH2_dataset.txt",
    "runner_sha256": "analysis/stage23_external_validation/evaluate_stage23_ph2.py",
    "preprocessing_sha256": "src/preprocessing.py",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual: float, saved: float, name: str, tol: float = 1e-12) -> None:
    if not math.isclose(actual, saved, rel_tol=0, abs_tol=tol):
        raise ValueError(f"{name} mismatch: recomputed={actual}, saved={saved}")


def audit() -> dict:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    required = ("predictions.csv", "metrics.json", "confusion_matrix.csv", "manifest.json")
    if not RUN.is_dir() or any(not (RUN/name).is_file() for name in required):
        raise FileNotFoundError("Incomplete Stage 23 PH2 downloaded run")
    manifest = json.loads((RUN/"manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((RUN/"metrics.json").read_text(encoding="utf-8"))
    protocol = json.loads((BASE/"phase23_ph2_protocol.json").read_text(encoding="utf-8"))
    if manifest["status"] != "COMPLETE_QUALIFIED_EXTERNAL_FOLLOWUP":
        raise ValueError("Run status is not complete")
    if (protocol["status"] != "FROZEN_BEFORE_STAGE23_PH2_INFERENCE" or
            tuple(protocol["class_order"]) != CLASSES or
            manifest["checkpoint_sha256"] != protocol["checkpoint_sha256"] or
            tuple(manifest["class_order"]) != CLASSES or
            manifest["batch_size"] != 16 or "T4" not in manifest["gpu"] or
            manifest["prior_ph2_project_use"] is not True or
            manifest["training_performed"] is not False or
            manifest["threshold_changed"] is not False or
            manifest["checkpoint_changed"] is not False):
        raise ValueError("Provenance, class order, execution or boundary mismatch")
    for name, relative in SHA_EXPECTED.items():
        if sha(ROOT/relative) != manifest[name]:
            raise ValueError(f"Source hash mismatch: {name}")
    for name in required[:-1]:
        if sha(RUN/name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Downloaded result hash mismatch: {name}")
    source_hashes = manifest["source_image_sha256"]
    source_rows = read_csv(ROOT/"analysis/ph2_external_validation_exp5/ph2_manifest.csv")
    if len(source_rows) != 200 or len(source_hashes) != 200:
        raise ValueError("Incomplete PH2 source audit")
    for source in source_rows:
        image_id = source["image_id"]
        if sha(ROOT/source["relative_image_path"]) != source_hashes[image_id]:
            raise ValueError(f"Source image hash mismatch: {image_id}")
    included = [r for r in source_rows if r["included"] == "True"]
    excluded = [r for r in source_rows if r["included"] == "False"]
    if (len(included) != 120 or len(excluded) != 80 or
            Counter(r["true_ham_label"] for r in included) != {"nv": 80, "mel": 40} or
            any(r["true_external_label"] != "atypical nevus" for r in excluded)):
        raise ValueError("Frozen PH2 mapping/inclusion mismatch")
    rows = read_csv(RUN/"predictions.csv")
    if len(rows) != 120 or len({r["image_id"] for r in rows}) != 120:
        raise ValueError("Prediction count/uniqueness failure")
    if [r["image_id"] for r in rows] != [r["image_id"] for r in included]:
        raise ValueError("Prediction IDs/order differ from fixed cohort")
    cm = np.zeros((2,3), dtype=int)
    for row, source in zip(rows, included):
        if (row["true_ham_label"] != source["true_ham_label"] or
                row["true_external_label"] != source["true_external_label"]):
            raise ValueError(f"Ground-truth label mismatch: {row['image_id']}")
        p = np.array([float(row[f"p_{class_name}"]) for class_name in CLASSES])
        if (p.shape != (7,) or not np.isfinite(p).all() or (p < 0).any() or
                (p > 1).any() or abs(float(p.sum())-1) > 1e-5):
            raise ValueError(f"Invalid seven-class probabilities: {row['image_id']}")
        predicted = CLASSES[int(np.argmax(p))]
        if row["predicted_class"] != predicted:
            raise ValueError(f"Prediction differs from probability argmax: {row['image_id']}")
        mapped = predicted if predicted in ("nv", "mel") else "other"
        if row["mapped_prediction"] != mapped or int(row["binary_mel_prediction"]) != (predicted == "mel"):
            raise ValueError(f"Mapped/binary prediction mismatch: {row['image_id']}")
        close(float(p.max()), float(row["confidence"]), "confidence")
        entropy = float(-(p[p>0]*np.log(p[p>0])).sum())
        close(entropy, float(row["entropy_nats"]), "entropy")
        cm[0 if row["true_ham_label"] == "nv" else 1][{"nv":0,"mel":1,"other":2}[mapped]] += 1
    if cm.tolist() != [[68,5,7],[13,13,14]]:
        # No expected performance was needed to run inference. This audit freezes
        # the observed downloaded confusion after recomputing all rows.
        raise ValueError("Downloaded confusion differs from audited observed table")
    if cm.tolist() != metrics["confusion_matrix"]:
        raise ValueError("Saved metrics confusion differs from predictions")
    matrix_csv = read_csv(RUN/"confusion_matrix.csv")
    if [r["true_class"] for r in matrix_csv] != ["nv","mel"] or [
        [int(r[k]) for k in ("nv","mel","other")] for r in matrix_csv] != cm.tolist():
        raise ValueError("Saved confusion CSV differs from predictions")
    nv_tp, nv_to_mel, nv_other = map(int, cm[0])
    mel_to_nv, mel_tp, mel_other = map(int, cm[1])
    mel_recall = mel_tp/40
    nv_recall = nv_tp/80
    specificity = (80-nv_to_mel)/80
    precision = mel_tp/(mel_tp+nv_to_mel)
    derived = {"mapped_accuracy": (nv_tp+mel_tp)/120,
               "mel_recall": mel_recall, "mel_precision": precision,
               "mel_f1": 2*precision*mel_recall/(precision+mel_recall),
               "nv_exact_label_recall": nv_recall,
               "mapped_balanced_accuracy": (nv_recall+mel_recall)/2,
               "binary_mel_specificity": specificity}
    truth = np.array([r["true_ham_label"] == "mel" for r in rows], dtype=int)
    scores = np.array([float(r["p_mel"]) for r in rows])
    confidence = np.array([float(r["confidence"]) for r in rows])
    entropy = np.array([float(r["entropy_nats"]) for r in rows])
    correct = np.array([r["predicted_class"] == r["true_ham_label"] for r in rows])
    derived.update({"mel_p_roc_auc": float(roc_auc_score(truth,scores)),
                    "mel_p_average_precision": float(average_precision_score(truth,scores)),
                    "mean_confidence": float(confidence.mean()),
                    "mean_entropy_nats": float(entropy.mean()),
                    "entropy_error_detection_auc": float(roc_auc_score(~correct, entropy))})
    for name, value in derived.items():
        close(value, float(metrics[name]), name)
    # Binomial intervals use the same registered Wilson convention, computed
    # here independently from the confusion counts.
    z = 1.959963984540054
    wilson = {}
    for name, hits, count in (("mapped_accuracy",nv_tp+mel_tp,120),
                              ("mel_recall",mel_tp,40),
                              ("nv_exact_label_recall",nv_tp,80),
                              ("binary_mel_specificity",80-nv_to_mel,80)):
        p = hits/count
        center = (p+z*z/(2*count))/(1+z*z/count)
        radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
        wilson[name] = [center-radius, center+radius]
        for index in (0,1):
            close(wilson[name][index], metrics["wilson_95"][name][index], f"Wilson {name}")
    # Independently replay the fixed-seed class-stratified image-ID bootstrap.
    # No patient cluster variable exists in the available PH2 metadata.
    rng = np.random.default_rng(42)
    by_class = {name: [r for r in rows if r["true_ham_label"] == name]
                for name in ("nv","mel")}
    replicates = {name: [] for name in ("mapped_accuracy", "mel_f1", "mel_p_roc_auc",
                                        "mean_entropy_nats", "mapped_balanced_accuracy")}
    for _ in range(2000):
        nv_rows = [by_class["nv"][int(i)] for i in rng.integers(0,80,80)]
        mel_rows = [by_class["mel"][int(i)] for i in rng.integers(0,40,40)]
        nv_hits = sum(r["predicted_class"] == "nv" for r in nv_rows)
        mel_hits = sum(r["predicted_class"] == "mel" for r in mel_rows)
        mel_fp_sample = sum(r["predicted_class"] == "mel" for r in nv_rows)
        precision_sample = mel_hits/(mel_hits+mel_fp_sample) if mel_hits+mel_fp_sample else 0.0
        recall_sample = mel_hits/40
        f1_sample = (2*precision_sample*recall_sample/(precision_sample+recall_sample)
                     if precision_sample+recall_sample else 0.0)
        sample_rows = nv_rows + mel_rows
        replicates["mapped_accuracy"].append((nv_hits+mel_hits)/120)
        replicates["mel_f1"].append(f1_sample)
        replicates["mel_p_roc_auc"].append(float(roc_auc_score(
            [r["true_ham_label"] == "mel" for r in sample_rows],
            [float(r["p_mel"]) for r in sample_rows])))
        replicates["mean_entropy_nats"].append(float(np.mean(
            [float(r["entropy_nats"]) for r in sample_rows])))
        replicates["mapped_balanced_accuracy"].append((nv_hits/80+mel_hits/40)/2)
    bootstrap_ci = {name: [float(x) for x in np.percentile(samples,[2.5,97.5])]
                    for name,samples in replicates.items()}
    for name in ("mapped_accuracy", "mel_f1", "mel_p_roc_auc", "mean_entropy_nats"):
        saved_interval = metrics["bootstrap_95"]["percentile_95"][name]
        for index in (0,1):
            close(bootstrap_ci[name][index], saved_interval[index], f"bootstrap {name}")
        if metrics["bootstrap_95"]["undefined_replicates"][name] != 0:
            raise ValueError(f"Unexpected undefined bootstrap replicates: {name}")
    balanced_ci = bootstrap_ci["mapped_balanced_accuracy"]
    patterns = {"mel_false_negatives": 40-mel_tp, "mel_to_nv": mel_to_nv,
                "mel_to_other": mel_other, "nv_to_mel": nv_to_mel,
                "mel_predicted_breakdown": dict(Counter(r["predicted_class"] for r in rows if r["true_ham_label"] == "mel")),
                "nv_predicted_breakdown": dict(Counter(r["predicted_class"] for r in rows if r["true_ham_label"] == "nv"))}
    return {"status": "PASS", "source": "saved predictions only; no inference",
            "artifact_sha256": {name: sha(RUN/name) for name in required},
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "protocol_sha256": manifest["protocol_sha256"],
            "sample_count": 120, "unique_ids": 120,
            "class_counts": {"nv":80,"mel":40}, "confusion_matrix": cm.tolist(),
            "recomputed_metrics": derived, "verified_wilson_95": wilson,
            "balanced_accuracy_bootstrap_95": balanced_ci,
            "balanced_accuracy_bootstrap_note": "descriptive post-run interval; 2000 class-stratified image-ID draws, seed 42; patient grouping unavailable",
            "recomputed_bootstrap_95": bootstrap_ci,
            "saved_bootstrap_95": metrics["bootstrap_95"], "patterns": patterns,
            "prior_ph2_project_use": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-derived", action="store_true")
    args = parser.parse_args()
    result = audit()
    if args.write_derived:
        target = BASE/"final_report"
        target.mkdir(exist_ok=True)
        with (target/"recomputed_audit.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False);stream.write("\n")
        names = ("mapped_accuracy", "mel_recall", "mel_precision", "mel_f1",
                 "nv_exact_label_recall", "mapped_balanced_accuracy", "binary_mel_specificity",
                 "mel_p_roc_auc", "mel_p_average_precision", "mean_entropy_nats",
                 "mean_confidence", "entropy_error_detection_auc")
        with (target/"METRICS_TABLE.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["metric", "estimate", "interval_95_low", "interval_95_high", "interval_method"])
            for name in names:
                ci = result["verified_wilson_95"].get(name)
                method = "Wilson" if ci else ""
                if name == "mapped_balanced_accuracy":
                    ci = result["balanced_accuracy_bootstrap_95"]
                    method = "descriptive stratified image-ID bootstrap"
                elif name in result["recomputed_bootstrap_95"] and ci is None:
                    ci = result["recomputed_bootstrap_95"][name]
                    method = "independently recomputed 2000-replicate stratified image-ID bootstrap"
                writer.writerow([name, result["recomputed_metrics"][name],
                                 ci[0] if ci else "", ci[1] if ci else "", method])
    print(json.dumps({"status": result["status"], "confusion_matrix": result["confusion_matrix"],
                      "metrics": result["recomputed_metrics"],
                      "balanced_accuracy_bootstrap_95": result["balanced_accuracy_bootstrap_95"]}, indent=2))


if __name__ == "__main__":
    main()
