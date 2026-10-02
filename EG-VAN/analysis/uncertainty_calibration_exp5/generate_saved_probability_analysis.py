from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
OUT = OUTPUT_DIR
SCRIPT_PATH = Path(__file__).resolve()
CHECKPOINT = ROOT / "experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt"
SPLIT = ROOT / "data/splits/split_leakage_aware.csv"
HAM_PREDICTIONS = ROOT / "experiments/efficientnetv2s_controlled_exp5/test_predictions.csv"
HAM_METRICS = ROOT / "experiments/efficientnetv2s_controlled_exp5/test_metrics.json"
HAM_CONFIG = ROOT / "experiments/efficientnetv2s_controlled_exp5/config.json"
PH2_PREDICTIONS = ROOT / "analysis/ph2_external_validation_exp5/ph2_predictions.csv"
PH2_METRICS = ROOT / "analysis/ph2_external_validation_exp5/ph2_metrics.json"
PH2_CONFIG = ROOT / "analysis/ph2_external_validation_exp5/external_validation_config.json"
GRADCAM_REVIEW = ROOT / "analysis/domain_shift_exp5/stage3_synthesis/stage3_case_comparison_reviewed.csv"
EXPECTED_CHECKPOINT_SHA256 = "81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a"
EXPECTED_SPLIT_SHA256 = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
PROBABILITY_COLUMNS = tuple(f"p_{name}" for name in CLASS_ORDER)
FIXED_PH2_IDS = (
    "IMD003", "IMD009", "IMD010", "IMD020", "IMD035", "IMD045",
    "IMD058", "IMD061", "IMD063", "IMD065", "IMD085", "IMD168",
)
EPSILON = 1e-12
EXPECTED_OUTPUTS = {
    "ham_uncertainty_predictions.csv",
    "ph2_uncertainty_predictions.csv",
    "ham_calibration_bins.csv",
    "ph2_calibration_bins.csv",
    "calibration_metrics.json",
    "confidence_summary.csv",
    "high_confidence_errors.csv",
    "class_uncertainty_summary.csv",
    "error_transitions.csv",
    "confidence_band_error_summary.csv",
    "ph2_gradcam_uncertainty_join.csv",
    "ham_reliability_diagram.png",
    "ph2_reliability_diagram.png",
    "ham_vs_ph2_reliability.png",
    "confidence_correct_vs_incorrect_ham.png",
    "confidence_correct_vs_incorrect_ph2.png",
    "entropy_correct_vs_incorrect_ham.png",
    "entropy_correct_vs_incorrect_ph2.png",
    "uncertainty_calibration_report.md",
    "experiment_manifest.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log(max(value, EPSILON)) for value in probabilities)


def confidence_band(confidence: float) -> str:
    if 0.0 <= confidence < 0.50:
        return "LOW"
    if confidence < 0.75:
        return "MODERATE"
    if confidence < 0.90:
        return "HIGH"
    if confidence <= 1.0:
        return "VERY_HIGH"
    raise ValueError(f"Confidence outside [0, 1]: {confidence}")


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def make_records(rows: list[dict[str, str]], dataset: str) -> list[dict]:
    records = []
    for row in rows:
        if dataset == "HAM10000":
            image_id = row["image_id"]
            true_class = row["true_label"]
            predicted_class = row["predicted_label"]
            probabilities = [float(value) for value in json.loads(row["probabilities"])]
            if (row["correct"].lower() == "true") != (true_class == predicted_class):
                raise ValueError(f"Incorrect saved correctness field for {image_id}")
        else:
            image_id = row["image_id"]
            true_class = row["true_ham_label"]
            predicted_class = row["predicted_class"]
            probabilities = [float(row[column]) for column in PROBABILITY_COLUMNS]

        if len(probabilities) != len(CLASS_ORDER):
            raise ValueError(f"Expected seven probabilities for {dataset}/{image_id}")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError(f"Invalid probability for {dataset}/{image_id}")
        if abs(sum(probabilities) - 1.0) > 1e-5:
            raise ValueError(f"Probability vector does not sum to 1 for {dataset}/{image_id}")
        argmax_class = CLASS_ORDER[max(range(7), key=probabilities.__getitem__)]
        if argmax_class != predicted_class:
            raise ValueError(f"Saved predicted class is not probability argmax for {dataset}/{image_id}")
        if true_class not in CLASS_ORDER:
            raise ValueError(f"Unknown true class for {dataset}/{image_id}: {true_class}")

        ordered = sorted(probabilities, reverse=True)
        confidence = max(probabilities)
        record = {
            "dataset": dataset,
            "image_id": image_id,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "correct": predicted_class == true_class,
            "confidence": confidence,
            "true_class_probability": probabilities[CLASS_ORDER.index(true_class)],
            "entropy": entropy(probabilities),
            "top1_probability": ordered[0],
            "top2_probability": ordered[1],
            "top1_top2_margin": ordered[0] - ordered[1],
            "confidence_band": confidence_band(confidence),
            "probabilities": probabilities,
        }
        for name in ("confidence", "true_class_probability", "entropy", "top1_probability", "top2_probability", "top1_top2_margin"):
            if not math.isfinite(record[name]):
                raise ValueError(f"Non-finite {name} for {dataset}/{image_id}")
        if record["top1_top2_margin"] < 0.0:
            raise ValueError(f"Negative top-two margin for {dataset}/{image_id}")
        records.append(record)
    return records


def calibration(records: list[dict]) -> dict:
    bins = [[] for _ in range(10)]
    for record in records:
        bins[min(int(record["confidence"] * 10), 9)].append(record)
    bin_rows = []
    ece = 0.0
    for index, group in enumerate(bins):
        if group:
            mean_confidence = sum(item["confidence"] for item in group) / len(group)
            accuracy = sum(item["correct"] for item in group) / len(group)
            gap = abs(accuracy - mean_confidence)
            fraction = len(group) / len(records)
            ece += fraction * gap
        else:
            mean_confidence = accuracy = gap = None
            fraction = 0.0
        bin_rows.append({
            "lower_bound": index / 10,
            "upper_bound": (index + 1) / 10,
            "sample_count": len(group),
            "sample_fraction": fraction,
            "mean_confidence": mean_confidence,
            "empirical_accuracy": accuracy,
            "absolute_calibration_gap": gap,
        })
    brier = sum(
        sum((record["probabilities"][index] - (1.0 if CLASS_ORDER[index] == record["true_class"] else 0.0)) ** 2 for index in range(7))
        for record in records
    ) / len(records)
    nll = -sum(math.log(max(record["true_class_probability"], EPSILON)) for record in records) / len(records)
    return {"bins": bin_rows, "ece": ece, "brier_score": brier, "nll": nll}


def summarize_dataset(records: list[dict], cal: dict) -> dict:
    correct = [record for record in records if record["correct"]]
    incorrect = [record for record in records if not record["correct"]]
    confidence = distribution([record["confidence"] for record in records])
    entropy_values = distribution([record["entropy"] for record in records])
    return {
        "sample_count": len(records),
        "correct_count": len(correct),
        "incorrect_count": len(incorrect),
        "accuracy": len(correct) / len(records),
        "mean_confidence": confidence["mean"],
        "median_confidence": confidence["median"],
        "confidence_q25": confidence["q25"],
        "confidence_q75": confidence["q75"],
        "mean_confidence_correct": distribution([record["confidence"] for record in correct])["mean"],
        "mean_confidence_incorrect": distribution([record["confidence"] for record in incorrect])["mean"],
        "median_confidence_correct": distribution([record["confidence"] for record in correct])["median"],
        "median_confidence_incorrect": distribution([record["confidence"] for record in incorrect])["median"],
        "mean_entropy": entropy_values["mean"],
        "median_entropy": entropy_values["median"],
        "entropy_q25": entropy_values["q25"],
        "entropy_q75": entropy_values["q75"],
        "mean_entropy_correct": distribution([record["entropy"] for record in correct])["mean"],
        "mean_entropy_incorrect": distribution([record["entropy"] for record in incorrect])["mean"],
        "mean_top1_top2_margin_correct": distribution([record["top1_top2_margin"] for record in correct])["mean"],
        "mean_top1_top2_margin_incorrect": distribution([record["top1_top2_margin"] for record in incorrect])["mean"],
        "ece": cal["ece"],
        "brier_score": cal["brier_score"],
        "nll": cal["nll"],
        "high_confidence_error_count": sum(record["confidence_band"] in ("HIGH", "VERY_HIGH") for record in incorrect),
    }


def reliability_plot(path: Path, series: list[tuple[str, dict, str]], title: str) -> None:
    fig, axis = plt.subplots(figsize=(5.7, 5.7))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal y=x")
    for label, cal, color in series:
        x = [row["mean_confidence"] for row in cal["bins"] if row["sample_count"] > 0]
        y = [row["empirical_accuracy"] for row in cal["bins"] if row["sample_count"] > 0]
        axis.plot(x, y, marker="o", color=color, label=label)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Empirical accuracy", title=title)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_outputs():
    if OUT.exists():
        unexpected = [path.name for path in OUT.iterdir() if path.resolve() != SCRIPT_PATH]
        if unexpected:
            raise FileExistsError(f"Refusing to overwrite existing analysis outputs: {unexpected}")
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Experiment #5 checkpoint hash mismatch")
    if sha256_file(SPLIT) != EXPECTED_SPLIT_SHA256:
        raise ValueError("Frozen split hash mismatch")

    ham_fields, ham_source = read_csv(HAM_PREDICTIONS)
    ph2_fields, ph2_source = read_csv(PH2_PREDICTIONS)
    if ham_fields != ["image_id", "true_label", "predicted_label", "correct", "probabilities"]:
        raise ValueError(f"Unexpected HAM prediction schema: {ham_fields}")
    if ph2_fields != ["image_id", "true_external_label", "true_ham_label", "predicted_class", *PROBABILITY_COLUMNS]:
        raise ValueError(f"Unexpected PH2 prediction schema: {ph2_fields}")
    if len(ham_source) != 1014 or len({row["image_id"] for row in ham_source}) != 1014:
        raise ValueError("HAM saved test predictions must contain 1,014 unique rows")
    if len(ph2_source) != 120 or len({row["image_id"] for row in ph2_source}) != 120:
        raise ValueError("PH2 saved predictions must contain 120 unique rows")

    ham = make_records(ham_source, "HAM10000")
    ph2 = make_records(ph2_source, "PH2")
    if Counter(row["true_class"] for row in ham) != Counter({"nv":676,"mel":107,"bkl":104,"bcc":58,"akiec":40,"vasc":18,"df":11}):
        raise ValueError("HAM true-class counts do not match frozen test split")
    if Counter(row["true_class"] for row in ph2) != Counter({"nv":80,"mel":40}):
        raise ValueError("PH2 direct-overlap counts do not match frozen cohort")

    ham_cal = calibration(ham)
    ph2_cal = calibration(ph2)
    ham_summary = summarize_dataset(ham, ham_cal)
    ph2_summary = summarize_dataset(ph2, ph2_cal)

    confidence_band_rows = []
    for dataset, records in (("HAM10000", ham), ("PH2", ph2)):
        errors = [record for record in records if not record["correct"]]
        for confidence_band_name in ("LOW", "MODERATE", "HIGH", "VERY_HIGH"):
            group = [record for record in records if record["confidence_band"] == confidence_band_name]
            error_count = sum(not record["correct"] for record in group)
            confidence_band_rows.append({
                "dataset": dataset,
                "confidence_band": confidence_band_name,
                "sample_count": len(group),
                "fraction_all_samples": len(group) / len(records),
                "correct_count": len(group) - error_count,
                "incorrect_count": error_count,
                "percentage_of_all_errors": error_count / len(errors) if errors else 0.0,
            })

    class_rows = []
    transition_rows = []
    for dataset, records in (("HAM10000", ham), ("PH2", ph2)):
        by_true_class = defaultdict(list)
        for record in records:
            by_true_class[record["true_class"]].append(record)
        for true_class, group in sorted(by_true_class.items()):
            correct = [record for record in group if record["correct"]]
            class_rows.append({
                "dataset": dataset,
                "true_class": true_class,
                "N": len(group),
                "correct": len(correct),
                "accuracy": len(correct) / len(group),
                "mean_confidence": distribution([record["confidence"] for record in group])["mean"],
                "mean_entropy": distribution([record["entropy"] for record in group])["mean"],
                "mean_true_class_probability": distribution([record["true_class_probability"] for record in group])["mean"],
                "subgroup_note": "Descriptive; PH2 class N is small." if dataset == "PH2" and len(group) < 50 else "",
            })
        errors_by_transition = defaultdict(list)
        for record in records:
            if not record["correct"]:
                errors_by_transition[(record["true_class"], record["predicted_class"])].append(record["confidence"])
        for (true_class, predicted_class), confidences in sorted(errors_by_transition.items()):
            transition_rows.append({
                "dataset": dataset,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "error_count": len(confidences),
                "mean_error_confidence": sum(confidences) / len(confidences),
            })

    high_confidence_errors = [
        {key: record[key] for key in ("dataset", "image_id", "true_class", "predicted_class", "confidence", "confidence_band", "true_class_probability", "entropy", "top1_top2_margin")}
        for record in ham + ph2 if not record["correct"]
    ]
    high_confidence_errors.sort(key=lambda row: (-row["confidence"], row["dataset"], row["image_id"]))

    gradcam_fields, gradcam_records = read_csv(GRADCAM_REVIEW)
    if len(gradcam_records) != 12 or {row["image_id"] for row in gradcam_records} != set(FIXED_PH2_IDS):
        raise ValueError("Existing reviewed Grad-CAM join does not contain exactly the fixed 12 PH2 IDs")
    ph2_by_id = {record["image_id"]: record for record in ph2}
    existing_review_columns = [column for column in gradcam_records[0] if column not in ("image_id", "true_class", "predicted_class", "correct")]
    joined = []
    for row in gradcam_records:
        uncertainty = ph2_by_id[row["image_id"]]
        if row["true_class"] != uncertainty["true_class"] or row["predicted_class"] != uncertainty["predicted_class"] or row["correct"].lower() != str(uncertainty["correct"]).lower():
            raise ValueError(f"Saved Stage 3 labels disagree with PH2 prediction for {row['image_id']}")
        joined.append({
            "image_id": row["image_id"],
            "case_category": row["case_category"],
            "true_class": uncertainty["true_class"],
            "predicted_class": uncertainty["predicted_class"],
            "correct": uncertainty["correct"],
            "confidence": uncertainty["confidence"],
            "confidence_band": uncertainty["confidence_band"],
            "true_class_probability": uncertainty["true_class_probability"],
            "entropy": uncertainty["entropy"],
            "top1_top2_margin": uncertainty["top1_top2_margin"],
            **{column: row[column] for column in existing_review_columns},
        })
    if len(joined) != 12 or len({row["image_id"] for row in joined}) != 12:
        raise ValueError("Grad-CAM uncertainty join is not exactly 12 unique cases")

    source_paths = [CHECKPOINT, SPLIT, HAM_PREDICTIONS, HAM_METRICS, HAM_CONFIG, PH2_PREDICTIONS, PH2_METRICS, PH2_CONFIG, GRADCAM_REVIEW]
    source_hashes = {str(path): sha256_file(path) for path in source_paths}
    stage = Path(tempfile.mkdtemp(prefix=".uncertainty_calibration_exp5.", dir=ROOT / "analysis"))
    try:
        prediction_fields = ["image_id", "true_class", "predicted_class", "correct", "confidence", "true_class_probability", "entropy", "top1_probability", "top2_probability", "top1_top2_margin", "confidence_band", *PROBABILITY_COLUMNS]
        def export_predictions(records):
            return [{**{key: record[key] for key in prediction_fields if key not in PROBABILITY_COLUMNS}, **{PROBABILITY_COLUMNS[i]: record["probabilities"][i] for i in range(7)}} for record in records]
        write_csv(stage / "ham_uncertainty_predictions.csv", prediction_fields, export_predictions(ham))
        write_csv(stage / "ph2_uncertainty_predictions.csv", prediction_fields, export_predictions(ph2))

        bin_fields = ["lower_bound", "upper_bound", "sample_count", "sample_fraction", "mean_confidence", "empirical_accuracy", "absolute_calibration_gap"]
        write_csv(stage / "ham_calibration_bins.csv", bin_fields, ham_cal["bins"])
        write_csv(stage / "ph2_calibration_bins.csv", bin_fields, ph2_cal["bins"])

        summary_fields = ["dataset", "sample_count", "correct_count", "incorrect_count", "accuracy", "mean_confidence", "median_confidence", "confidence_q25", "confidence_q75", "mean_confidence_correct", "mean_confidence_incorrect", "median_confidence_correct", "median_confidence_incorrect", "mean_entropy", "median_entropy", "entropy_q25", "entropy_q75", "mean_entropy_correct", "mean_entropy_incorrect", "mean_top1_top2_margin_correct", "mean_top1_top2_margin_incorrect", "ece", "brier_score", "nll", "high_confidence_error_count"]
        write_csv(stage / "confidence_summary.csv", summary_fields, [{"dataset":"HAM10000",**ham_summary},{"dataset":"PH2",**ph2_summary}])
        write_csv(stage / "high_confidence_errors.csv", ["dataset", "image_id", "true_class", "predicted_class", "confidence", "confidence_band", "true_class_probability", "entropy", "top1_top2_margin"], high_confidence_errors)
        write_csv(stage / "class_uncertainty_summary.csv", ["dataset", "true_class", "N", "correct", "accuracy", "mean_confidence", "mean_entropy", "mean_true_class_probability", "subgroup_note"], class_rows)
        write_csv(stage / "error_transitions.csv", ["dataset", "true_class", "predicted_class", "error_count", "mean_error_confidence"], transition_rows)
        write_csv(stage / "confidence_band_error_summary.csv", ["dataset", "confidence_band", "sample_count", "fraction_all_samples", "correct_count", "incorrect_count", "percentage_of_all_errors"], confidence_band_rows)
        write_csv(stage / "ph2_gradcam_uncertainty_join.csv", list(joined[0]), joined)

        calibration_metrics = {
            "ece_definition": "10 fixed equal-width confidence bins over [0,1]; ECE=sum_b(n_b/N)*abs(acc_b-conf_b); empty bins retained and contribute zero.",
            "ece_bin_count": 10,
            "brier_definition": "Mean over samples of sum over all seven classes (p_k-y_k)^2; not divided by class count.",
            "nll_definition": "Negative mean ln true-class probability, lower clipped at 1e-12 only for numerical log stability.",
            "entropy_definition": "Natural-log entropy -sum(p_k*ln(max(p_k,1e-12))); saved probabilities unchanged.",
            "confidence_bands": {"LOW":"0<=confidence<0.50", "MODERATE":"0.50<=confidence<0.75", "HIGH":"0.75<=confidence<0.90", "VERY_HIGH":"0.90<=confidence<=1.00"},
            "HAM10000": ham_summary,
            "PH2": ph2_summary,
            "confidence_band_error_summary": confidence_band_rows,
        }
        (stage / "calibration_metrics.json").write_text(json.dumps(calibration_metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")

        analysis_config = {
            "analysis_name": "Experiment #5 uncertainty and calibration analysis",
            "checkpoint_path": "experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "split_path": "data/splits/split_leakage_aware.csv",
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "selected_epoch": 15,
            "class_order": list(CLASS_ORDER),
            "ham_prediction_source": "experiments/efficientnetv2s_controlled_exp5/test_predictions.csv",
            "ph2_prediction_source": "analysis/ph2_external_validation_exp5/ph2_predictions.csv",
            "ham_sample_count": len(ham),
            "ph2_sample_count": len(ph2),
            "ece_definition": "10 fixed equal-width confidence bins over [0,1]; sample-weighted absolute accuracy-confidence gap; empty bins retained.",
            "ece_bin_count": 10,
            "brier_definition": "Mean per sample of sum across seven classes (p_k-y_k)^2; not divided by class count.",
            "nll_definition": "Negative mean natural log true-class probability; clip lower tail at 1e-12 only for numerical stability.",
            "entropy_definition": "Natural-log entropy -sum(p_k*ln(max(p_k,1e-12))); saved probabilities unchanged.",
            "confidence_bands": {
                "LOW": "0.00 <= confidence < 0.50",
                "MODERATE": "0.50 <= confidence < 0.75",
                "HIGH": "0.75 <= confidence < 0.90",
                "VERY_HIGH": "0.90 <= confidence <= 1.00",
            },
            "gradcam_join_source": "analysis/domain_shift_exp5/stage3_synthesis/stage3_case_comparison_reviewed.csv",
            "gradcam_join_ids": FIXED_PH2_IDS,
            "inference_rerun": False,
            "training_or_finetuning": False,
            "threshold_tuning": False,
            "temperature_scaling": False,
            "calibration_fitting": False,
            "ph2_used_for_model_selection": False,
            "ph2_used_for_calibration_fitting": False,
        }
        (stage / "uncertainty_calibration_config.json").write_text(
            json.dumps(analysis_config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

        def plot_reliability(axis, cal, label, color):
            x = [row["mean_confidence"] for row in cal["bins"] if row["sample_count"] > 0]
            y = [row["empirical_accuracy"] for row in cal["bins"] if row["sample_count"] > 0]
            axis.plot([0,1],[0,1],"k--",label="Ideal y=x")
            axis.plot(x,y,"o-",color=color,label=label)
            axis.set(xlim=(0,1),ylim=(0,1),xlabel="Mean confidence",ylabel="Empirical accuracy")
            axis.grid(alpha=.25)
            axis.legend()
        for cal, label, color, filename in ((ham_cal,"HAM10000","#2463a6","ham_reliability_diagram.png"),(ph2_cal,"PH2","#bc4b32","ph2_reliability_diagram.png")):
            fig, axis = plt.subplots(figsize=(5.7,5.7))
            plot_reliability(axis,cal,label,color)
            axis.set_title(label+" reliability")
            fig.tight_layout();fig.savefig(stage/filename,dpi=160);plt.close(fig)
        fig, axis = plt.subplots(figsize=(6,6))
        axis.plot([0,1],[0,1],"k--",label="Ideal y=x")
        for cal,label,color in ((ham_cal,"HAM10000","#2463a6"),(ph2_cal,"PH2","#bc4b32")):
            x=[row["mean_confidence"] for row in cal["bins"] if row["sample_count"]>0]
            y=[row["empirical_accuracy"] for row in cal["bins"] if row["sample_count"]>0]
            axis.plot(x,y,"o-",label=label,color=color)
        axis.set(xlim=(0,1),ylim=(0,1),xlabel="Mean confidence",ylabel="Empirical accuracy",title="HAM10000 vs PH2 reliability")
        axis.grid(alpha=.25);axis.legend();fig.tight_layout();fig.savefig(stage/"ham_vs_ph2_reliability.png",dpi=160);plt.close(fig)
        for dataset,records,suffix in (("HAM10000",ham,"ham"),("PH2",ph2,"ph2")):
            correct=[record for record in records if record["correct"]]
            incorrect=[record for record in records if not record["correct"]]
            fig,axis=plt.subplots(figsize=(6,4.5))
            axis.boxplot([[record["confidence"] for record in correct],[record["confidence"] for record in incorrect]],tick_labels=[f"Correct (n={len(correct)})",f"Incorrect (n={len(incorrect)})"],showfliers=False)
            axis.set_ylim(0,1);axis.set_ylabel("Maximum predicted probability");axis.set_title(dataset+": confidence by correctness");axis.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(stage/f"confidence_correct_vs_incorrect_{suffix}.png",dpi=160);plt.close(fig)
            fig,axis=plt.subplots(figsize=(6,4.5));edges=np.linspace(0,math.log(7),21)
            axis.hist([record["entropy"] for record in correct],bins=edges,density=True,alpha=.55,label=f"Correct (n={len(correct)})")
            axis.hist([record["entropy"] for record in incorrect],bins=edges,density=True,alpha=.55,label=f"Incorrect (n={len(incorrect)})")
            axis.set_xlim(0,math.log(7));axis.set_xlabel("Predictive entropy (nats)");axis.set_ylabel("Density");axis.set_title(dataset+": entropy by correctness");axis.grid(axis="y",alpha=.25);axis.legend();fig.tight_layout();fig.savefig(stage/f"entropy_correct_vs_incorrect_{suffix}.png",dpi=160);plt.close(fig)

        report_lines = [
            "# Experiment #5 Uncertainty and Calibration Analysis", "",
            "## 1. Objective", "",
            "Evaluate confidence and probabilistic reliability of frozen Experiment #5 saved predictions on the internal HAM10000 TEST set and PH2 external follow-up set. No model forward pass or calibration fitting was performed.", "",
            "## 2. Frozen Model Provenance", "",
            f"- Checkpoint: `experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt` (SHA256 `{EXPECTED_CHECKPOINT_SHA256}`).",
            f"- Frozen split SHA256: `{EXPECTED_SPLIT_SHA256}`; selected epoch: 15.",
            "- Class order: `akiec, bcc, bkl, df, mel, nv, vasc`.", "",
            "## 3. Data Sources", "",
            f"- HAM: saved Experiment #5 TEST probabilities, {len(ham)} samples across seven classes.",
            f"- PH2: saved external follow-up probabilities, {len(ph2)} samples (80 nv, 40 mel); atypical nevi excluded.",
            "- PH2 was previously used in project analyses and is treated as external follow-up, not an untouched independent validation.", "",
            "## 4. Methods", "",
            "- Confidence is the maximum of the seven saved class probabilities; true-class probability is the saved probability for the true label.",
            "- Entropy is `-sum(p_k * ln(max(p_k, 1e-12)))` in natural-log units; saved probabilities are unchanged.",
            "- Top-1/top-2 margin is the largest probability minus the second largest.",
            "- ECE uses 10 fixed equal-width bins over [0,1]: `sum_b (n_b/N) * abs(empirical_accuracy_b - mean_confidence_b)`. Empty bins are retained and contribute zero.",
            "- Brier is the per-sample sum of squared errors across all seven classes, averaged over samples; it is not divided by seven.",
            "- NLL is negative mean natural log of true-class probability, clipped below at `1e-12` only for log stability.",
            "- Confidence bands are descriptive: LOW <0.50; MODERATE [0.50,0.75); HIGH [0.75,0.90); VERY_HIGH [0.90,1.00]. No threshold is optimized.", "",
            "## 5. HAM10000 Internal Results", "",
            f"- N={ham_summary['sample_count']}; correct={ham_summary['correct_count']}; incorrect={ham_summary['incorrect_count']}; accuracy={ham_summary['accuracy']:.4f}.",
            f"- ECE={ham_summary['ece']:.4f}; Brier={ham_summary['brier_score']:.4f}; NLL={ham_summary['nll']:.4f}.",
            f"- Mean/median confidence={ham_summary['mean_confidence']:.4f}/{ham_summary['median_confidence']:.4f}; Q25/Q75={ham_summary['confidence_q25']:.4f}/{ham_summary['confidence_q75']:.4f}.",
            f"- Mean confidence correct/incorrect={ham_summary['mean_confidence_correct']:.4f}/{ham_summary['mean_confidence_incorrect']:.4f}; medians={ham_summary['median_confidence_correct']:.4f}/{ham_summary['median_confidence_incorrect']:.4f}.",
            f"- Mean entropy correct/incorrect={ham_summary['mean_entropy_correct']:.4f}/{ham_summary['mean_entropy_incorrect']:.4f}; mean margin correct/incorrect={ham_summary['mean_top1_top2_margin_correct']:.4f}/{ham_summary['mean_top1_top2_margin_incorrect']:.4f}.",
            f"- HIGH+VERY_HIGH confidence errors: {ham_summary['high_confidence_error_count']}.", "",
            "## 6. PH² External Follow-Up Results", "",
            f"- N={ph2_summary['sample_count']}; correct={ph2_summary['correct_count']}; incorrect={ph2_summary['incorrect_count']}; accuracy={ph2_summary['accuracy']:.4f}.",
            f"- ECE={ph2_summary['ece']:.4f}; Brier={ph2_summary['brier_score']:.4f}; NLL={ph2_summary['nll']:.4f}.",
            f"- Mean/median confidence={ph2_summary['mean_confidence']:.4f}/{ph2_summary['median_confidence']:.4f}; Q25/Q75={ph2_summary['confidence_q25']:.4f}/{ph2_summary['confidence_q75']:.4f}.",
            f"- Mean confidence correct/incorrect={ph2_summary['mean_confidence_correct']:.4f}/{ph2_summary['mean_confidence_incorrect']:.4f}; medians={ph2_summary['median_confidence_correct']:.4f}/{ph2_summary['median_confidence_incorrect']:.4f}.",
            f"- Mean entropy correct/incorrect={ph2_summary['mean_entropy_correct']:.4f}/{ph2_summary['mean_entropy_incorrect']:.4f}; mean margin correct/incorrect={ph2_summary['mean_top1_top2_margin_correct']:.4f}/{ph2_summary['mean_top1_top2_margin_incorrect']:.4f}.",
            f"- HIGH+VERY_HIGH confidence errors: {ph2_summary['high_confidence_error_count']}.", "",
            "## 7. Internal vs External Calibration Comparison", "",
            f"PH2 ECE is {'higher' if ph2_summary['ece'] > ham_summary['ece'] else 'lower'} than HAM ({ph2_summary['ece']:.4f} vs {ham_summary['ece']:.4f}); Brier is {ph2_summary['brier_score']:.4f} vs {ham_summary['brier_score']:.4f}; NLL is {ph2_summary['nll']:.4f} vs {ham_summary['nll']:.4f}. These cross-dataset contrasts are descriptive and not causal.", "",
            "## 8. Correct vs Incorrect Prediction Confidence", "",
            f"HAM mean confidence correct/incorrect is {ham_summary['mean_confidence_correct']:.4f}/{ham_summary['mean_confidence_incorrect']:.4f}; mean entropy correct/incorrect is {ham_summary['mean_entropy_correct']:.4f}/{ham_summary['mean_entropy_incorrect']:.4f}.",
            f"PH2 mean confidence correct/incorrect is {ph2_summary['mean_confidence_correct']:.4f}/{ph2_summary['mean_confidence_incorrect']:.4f}; mean entropy correct/incorrect is {ph2_summary['mean_entropy_correct']:.4f}/{ph2_summary['mean_entropy_incorrect']:.4f}. These describe saved samples; confidence is not a guarantee of correctness.", "",
            "## 9. High-Confidence Misclassifications", "",
            f"HIGH+VERY_HIGH error counts are {ham_summary['high_confidence_error_count']} for HAM and {ph2_summary['high_confidence_error_count']} for PH2. Band counts and error percentages are in `confidence_band_error_summary.csv`; all errors are in `high_confidence_errors.csv`, sorted by descending confidence. These are high-confidence misclassifications, not clinical-risk labels.", "",
            "## 10. Class-Level Observations", "",
            "`class_uncertainty_summary.csv` reports class N, accuracy, mean confidence, entropy, and true-class probability. `error_transitions.csv` reports error counts and mean error confidence. PH2 has 40 melanoma cases; smaller PH2 error groups are descriptive and uncertain.", "",
            "## 11. Relationship With the 12 PH² Grad-CAM Cases", "",
            f"`ph2_gradcam_uncertainty_join.csv` joins exactly {len(joined)} fixed IDs to saved uncertainty features and preserves the prior human-review fields. The visual judgments were not reinterpreted. No Grad-CAM was regenerated, and no causal relationship is inferred.", "",
            "## 12. Interpretation", "",
            "ECE, Brier, and NLL describe different aspects of saved probability behavior. Results are computed separately for HAM and PH2. Differences may be consistent with changed reliability on the external follow-up distribution but are limited by cohort size, dataset differences, and ECE binning.", "",
            "## 13. Limitations", "",
            "- PH2 is an external follow-up dataset previously used in project analyses, not a completely untouched independent validation.",
            "- PH2 contains 120 included cases and only 40 melanoma cases; only true nv and mel classes are represented.",
            "- Calibration estimates depend on sample size; ECE is bin-dependent.",
            "- HAM and PH2 are different populations and acquisition domains.",
            "- Grad-CAM is exploratory and does not establish causal reasoning or clinical validity.",
            "- No temperature scaling, calibration fitting, or confidence-threshold tuning was performed.", "",
            "## 14. Conclusion", "",
            "These values describe reliability metrics of the saved Experiment #5 probabilities. No calibration correction or model adjustment was fitted from PH2."
        ]
        (stage/'uncertainty_calibration_report.md').write_text('\n'.join(report_lines)+'\n',encoding='utf-8')

        generated_names=sorted([p.name for p in stage.iterdir()]+['experiment_manifest.json'])
        manifest={
          'analysis_name':'Experiment #5 uncertainty and calibration analysis',
          'generated_at_utc':datetime.now(timezone.utc).isoformat(),
          'checkpoint_path':'experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt','checkpoint_sha256':EXPECTED_CHECKPOINT_SHA256,
          'split_path':'data/splits/split_leakage_aware.csv','split_sha256':EXPECTED_SPLIT_SHA256,'selected_epoch':15,'class_order':CLASS_ORDER,
          'ham_prediction_source':'experiments/efficientnetv2s_controlled_exp5/test_predictions.csv',
          'ph2_prediction_source':'analysis/ph2_external_validation_exp5/ph2_predictions.csv',
          'ham_sample_count':len(ham),'ph2_sample_count':len(ph2),
          'ece_definition':'10 equal-width confidence bins over [0,1]; ECE=sum_b(n_b/N)*abs(acc_b-conf_b); empty bins retained.',
          'ece_bin_count':10,'brier_definition':'Mean over samples of sum over all seven classes (p_k-y_k)^2; not divided by class count.',
          'nll_definition':'Negative mean ln true-class probability; lower clipped at 1e-12 only for numerical stability.',
          'entropy_definition':'Natural-log entropy -sum(p_k*ln(max(p_k,1e-12))); original probabilities unchanged.',
          'confidence_bands':{'LOW':'0<=confidence<0.50','MODERATE':'0.50<=confidence<0.75','HIGH':'0.75<=confidence<0.90','VERY_HIGH':'0.90<=confidence<=1.00'},
          'gradcam_join_source':'analysis/domain_shift_exp5/stage3_synthesis/stage3_case_comparison_reviewed.csv','gradcam_join_ids':FIXED_PH2_IDS,
          'inference_rerun':False,'training_or_finetuning':False,'threshold_tuning':False,'temperature_scaling':False,'calibration_fitting':False,'ph2_used_for_model_selection':False,'ph2_used_for_calibration_fitting':False,
          'generated_files':generated_names,
        }
        (stage/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')

        actual_names={p.name for p in stage.iterdir()}
        if actual_names!=set(generated_names):raise ValueError(f'Output manifest mismatch: {actual_names ^ set(generated_names)}')
        for path in stage.iterdir():
            if not path.is_file() or path.stat().st_size==0:raise ValueError(f'Empty/missing staged output: {path}')
        for bins,n in ((ham_cal['bins'],1014),(ph2_cal['bins'],120)):
            if len(bins)!=10 or sum(row['sample_count'] for row in bins)!=n:raise ValueError('Calibration bins do not account for every sample')
        if not (0<=ham_cal['ece']<=1 and 0<=ph2_cal['ece']<=1):raise ValueError('ECE outside [0,1]')
        for path in source_paths:
            if sha256_file(path)!=source_hashes[str(path)]:raise RuntimeError(f'Source artifact changed: {path}')
        unexpected = [path.name for path in OUT.iterdir() if path.resolve() != SCRIPT_PATH]
        if unexpected:
            raise FileExistsError(f'Refusing to overwrite existing analysis outputs: {unexpected}')
        for staged_file in list(stage.iterdir()):
            destination = OUT / staged_file.name
            if destination.exists():
                raise FileExistsError(f'Refusing to overwrite {destination}')
            os.replace(staged_file, destination)
    finally:
        if stage.exists():shutil.rmtree(stage)

    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256 or sha256_file(SPLIT) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("Frozen checkpoint or split hash changed after analysis")
    print("UNCERTAINTY_CALIBRATION_STATUS: PASS")
    for name, result in (("HAM", ham_summary), ("PH2", ph2_summary)):
        print(
            f"{name}: N={result['sample_count']} accuracy={result['accuracy']:.4f} "
            f"ECE={result['ece']:.4f} Brier={result['brier_score']:.4f} "
            f"NLL={result['nll']:.4f} mean_conf={result['mean_confidence']:.4f} "
            f"mean_entropy={result['mean_entropy']:.4f} "
            f"mean_conf_correct={result['mean_confidence_correct']:.4f} "
            f"mean_conf_incorrect={result['mean_confidence_incorrect']:.4f} "
            f"high+very_high_errors={result['high_confidence_error_count']}"
        )
    print(f"Grad-CAM cases joined: {len(joined)}/12")
    print("Inference rerun: NO; Training: NO; Fine-tuning: NO; Temperature scaling: NO; Threshold tuning: NO; Calibration fitting: NO")
    print("Checkpoint modified: NO; Split modified: NO")
    print("NEW_FILES=", json.dumps(sorted(str(p.relative_to(OUT)).replace("\\", "/") for p in OUT.rglob("*") if p.is_file())))


if __name__ == "__main__":
    build_outputs()
