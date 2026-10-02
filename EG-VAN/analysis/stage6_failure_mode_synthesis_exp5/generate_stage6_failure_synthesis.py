"""Deterministic Stage 6 synthesis from frozen, saved artifacts only."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FIXED = "IMD003 IMD009 IMD010 IMD020 IMD035 IMD045 IMD058 IMD061 IMD063 IMD065 IMD085 IMD168".split()
QUALITY = ["brightness", "contrast", "sharpness", "saturation", "dark_pixel_ratio", "bright_pixel_ratio", "entropy", "illumination_variation"]
DOMAIN = ["native_width", "native_height", "native_aspect_ratio", "rgb_mean_r", "rgb_mean_g", "rgb_mean_b", "rgb_std_r", "rgb_std_g", "rgb_std_b"]
CP_HASH = "81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a"
SPLIT_HASH = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
SOURCES = {
    "checkpoint": "experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt",
    "split": "data/splits/split_leakage_aware.csv",
    "ham_config": "experiments/efficientnetv2s_controlled_exp5/config.json",
    "ham_manifest": "experiments/efficientnetv2s_controlled_exp5/experiment_manifest.json",
    "ham_metrics": "experiments/efficientnetv2s_controlled_exp5/test_metrics.json",
    "ham_predictions": "experiments/efficientnetv2s_controlled_exp5/test_predictions.csv",
    "ph2_manifest": "analysis/ph2_external_validation_exp5/ph2_manifest.csv",
    "ph2_predictions": "analysis/ph2_external_validation_exp5/ph2_predictions.csv",
    "ph2_metrics": "analysis/ph2_external_validation_exp5/ph2_metrics.json",
    "ph2_config": "analysis/ph2_external_validation_exp5/external_validation_config.json",
    "domain_metrics": "analysis/domain_shift_exp5/domain_shift_metrics.json",
    "domain_prediction_shift": "analysis/domain_shift_exp5/prediction_shift_summary.json",
    "domain_report": "analysis/domain_shift_exp5/domain_shift_stage1_report.md",
    "domain_features": "analysis/domain_shift_exp5/domain_features.csv",
    "gradcam_manifest": "analysis/domain_shift_exp5/gradcam_comparison/ph2_gradcam_manifest.csv",
    "gradcam_summary": "analysis/domain_shift_exp5/gradcam_comparison/gradcam_comparison_summary.json",
    "gradcam_review": "analysis/domain_shift_exp5/gradcam_comparison/gradcam_review.csv",
    "human_review": "analysis/domain_shift_exp5/stage3_synthesis/stage3_case_comparison_reviewed.csv",
    "human_summary": "analysis/domain_shift_exp5/stage3_synthesis/human_review_summary.json",
    "human_manifest": "analysis/domain_shift_exp5/stage3_synthesis/stage3_artifact_manifest.json",
    "quality_table": "experiments/image_quality/image_quality.csv",
    "quality_report": "docs/PHASE_6_IMAGE_QUALITY_ASSESSMENT.md",
    "quality_aggregate": "experiments/image_quality/quality_stats_by_split.csv",
    "ham_uncertainty": "analysis/uncertainty_calibration_exp5/ham_uncertainty_predictions.csv",
    "ph2_uncertainty": "analysis/uncertainty_calibration_exp5/ph2_uncertainty_predictions.csv",
    "ham_bins": "analysis/uncertainty_calibration_exp5/ham_calibration_bins.csv",
    "ph2_bins": "analysis/uncertainty_calibration_exp5/ph2_calibration_bins.csv",
    "calibration_metrics": "analysis/uncertainty_calibration_exp5/calibration_metrics.json",
    "confidence_summary": "analysis/uncertainty_calibration_exp5/confidence_summary.csv",
    "high_errors": "analysis/uncertainty_calibration_exp5/high_confidence_errors.csv",
    "class_uncertainty": "analysis/uncertainty_calibration_exp5/class_uncertainty_summary.csv",
    "error_transitions": "analysis/uncertainty_calibration_exp5/error_transitions.csv",
    "confidence_bands": "analysis/uncertainty_calibration_exp5/confidence_band_error_summary.csv",
    "uncertainty_join": "analysis/uncertainty_calibration_exp5/ph2_gradcam_uncertainty_join.csv",
    "uncertainty_config": "analysis/uncertainty_calibration_exp5/uncertainty_calibration_config.json",
    "uncertainty_report": "analysis/uncertainty_calibration_exp5/uncertainty_calibration_report.md",
    "uncertainty_manifest": "analysis/uncertainty_calibration_exp5/experiment_manifest.json",
    "uncertainty_script": "analysis/uncertainty_calibration_exp5/generate_saved_probability_analysis.py",
}

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_csv(key):
    return pd.read_csv(ROOT / SOURCES[key], dtype={"image_id": str}, keep_default_na=False)

def read_json(key):
    return json.loads((ROOT / SOURCES[key]).read_text(encoding="utf-8"))

def require_columns(df, columns, key):
    missing = set(columns) - set(df.columns)
    assert not missing, f"{key} missing columns: {sorted(missing)}"

def exact_ids(df, expected, key):
    ids = df.image_id.tolist()
    assert len(ids) == len(set(ids)), f"{key}: duplicate IDs"
    assert set(ids) == set(expected), f"{key}: missing={sorted(set(expected)-set(ids))}; extra={sorted(set(ids)-set(expected))}"

def check_prob(df, key, true_col, pred_col):
    require_columns(df, ["image_id", true_col, pred_col] + ["p_" + c for c in CLASSES], key)
    p = df[["p_" + c for c in CLASSES]].apply(pd.to_numeric, errors="raise").to_numpy()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), f"{key}: invalid probabilities"
    assert np.allclose(p.sum(axis=1), 1, atol=1e-5), f"{key}: probability sums"
    assert (np.array(CLASSES)[p.argmax(axis=1)] == df[pred_col].to_numpy()).all(), f"{key}: argmax mismatch"
    assert df[true_col].isin(CLASSES).all(), f"{key}: true class outside frozen order"
    return p

def metrics(df, p):
    truth = df.true_class.to_numpy()
    pred = df.predicted_class.to_numpy()
    correct = truth == pred
    confidence = p.max(axis=1)
    tc = p[np.arange(len(p)), np.array([CLASSES.index(c) for c in truth])]
    entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
    top = np.sort(p, axis=1)
    margin = top[:, -1] - top[:, -2]
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(np.sum(bins == b) / len(p) * abs(correct[bins == b].mean() - confidence[bins == b].mean()) for b in range(10) if np.any(bins == b))
    onehot = np.zeros_like(p)
    onehot[np.arange(len(p)), [CLASSES.index(c) for c in truth]] = 1
    return dict(n=len(p), correct=int(correct.sum()), errors=int((~correct).sum()), accuracy=float(correct.mean()), mean_confidence=float(confidence.mean()), mean_entropy=float(entropy.mean()), ece=float(ece), brier=float(np.mean(np.sum((p-onehot)**2, axis=1))), nll=float(-np.mean(np.log(np.maximum(tc, 1e-12))))), dict(correct=correct, confidence=confidence, true_class_probability=tc, entropy=entropy, prediction_margin=margin)

def band(c):
    return "LOW" if c < .5 else "MODERATE" if c < .75 else "HIGH" if c < .9 else "VERY_HIGH"

def preflight():
    missing = [v for v in SOURCES.values() if not (ROOT / v).is_file()]
    assert not missing, f"Required source files missing: {missing}"
    hashes = {v: sha(ROOT / v) for v in SOURCES.values()}
    assert hashes[SOURCES["checkpoint"]] == CP_HASH, "Frozen checkpoint hash mismatch"
    assert hashes[SOURCES["split"]] == SPLIT_HASH, "Frozen split hash mismatch"
    for key, field in [("ham_config", "classes"), ("ph2_config", "class_order"), ("uncertainty_config", "class_order"), ("uncertainty_manifest", "class_order")]:
        assert read_json(key)[field] == CLASSES, f"{key}: class order mismatch"
    for key in ("ham_manifest", "ph2_config", "domain_metrics", "uncertainty_config", "uncertainty_manifest"):
        obj = read_json(key)
        assert obj["checkpoint_sha256"] == CP_HASH and obj["split_sha256"] == SPLIT_HASH, f"{key}: frozen provenance mismatch"
    policy = read_json("uncertainty_config")
    for flag in ("inference_rerun", "training_or_finetuning", "threshold_tuning", "temperature_scaling", "calibration_fitting", "ph2_used_for_model_selection", "ph2_used_for_calibration_fitting"):
        assert policy[flag] is False, f"Stage 5 policy violation: {flag}"
        assert read_json("uncertainty_manifest")[flag] is False, f"Stage 5 manifest violation: {flag}"
    assert read_json("ham_manifest")["ph2_accessed"] is False
    assert read_json("ph2_config")["training_performed"] is False
    ham_raw, ph_raw = read_csv("ham_predictions"), read_csv("ph2_predictions")
    assert len(ham_raw) == 1014 and len(ph_raw) == 120
    assert ph_raw.true_ham_label.value_counts().to_dict() == {"nv": 80, "mel": 40}
    assert not ham_raw.image_id.duplicated().any() and not ph_raw.image_id.duplicated().any()
    ph_p = check_prob(ph_raw, "PH2", "true_ham_label", "predicted_class")
    assert (ph_raw.true_ham_label == ph_raw.predicted_class).sum() == 75
    manifest = read_csv("ph2_manifest")
    included = manifest[manifest.included.astype(str).str.lower() == "true"]
    exact_ids(included, ph_raw.image_id, "PH2 included manifest")
    assert (included.set_index("image_id").loc[ph_raw.image_id, "true_ham_label"].to_numpy() == ph_raw.true_ham_label.to_numpy()).all()
    ham_unc, ph_unc = read_csv("ham_uncertainty"), read_csv("ph2_uncertainty")
    exact_ids(ham_unc, ham_raw.image_id, "HAM uncertainty")
    exact_ids(ph_unc, ph_raw.image_id, "PH2 uncertainty")
    ham_unc = ham_unc.set_index("image_id").loc[ham_raw.image_id].reset_index()
    ph_unc = ph_unc.set_index("image_id").loc[ph_raw.image_id].reset_index()
    ham_p = check_prob(ham_unc, "HAM uncertainty", "true_class", "predicted_class")
    assert (ham_unc.true_class.to_numpy() == ham_raw.true_label.to_numpy()).all()
    assert (ham_unc.predicted_class.to_numpy() == ham_raw.predicted_label.to_numpy()).all()
    for i, row in ham_raw.iterrows():
        assert np.allclose(json.loads(row.probabilities), ham_p[i], atol=1e-7)
    for c in CLASSES:
        assert np.allclose(ph_unc["p_"+c], ph_raw["p_"+c], atol=1e-8)
    assert (ph_unc.true_class.to_numpy() == ph_raw.true_ham_label.to_numpy()).all()
    hm, hd = metrics(ham_unc, ham_p)
    pm, pdct = metrics(ph_unc, ph_p)
    for frame, derived, key in [(ham_unc, hd, "HAM"), (ph_unc, pdct, "PH2")]:
        for col, vals in derived.items():
            saved = frame["top1_top2_margin" if col == "prediction_margin" else col]
            if col == "correct":
                assert (saved.astype(str).str.lower().to_numpy() == np.where(vals, "true", "false")).all()
            else:
                assert np.allclose(saved.astype(float), vals, atol=1e-9), f"{key}: {col} mismatch"
        assert (frame.confidence_band.to_numpy() == np.array([band(x) for x in derived["confidence"]])).all()
    saved_metrics = read_json("calibration_metrics")
    for key, calculated in [("HAM10000", hm), ("PH2", pm)]:
        saved = saved_metrics[key]
        for a, b in [("sample_count", "n"), ("correct_count", "correct"), ("accuracy", "accuracy"), ("mean_confidence", "mean_confidence"), ("mean_entropy", "mean_entropy"), ("ece", "ece"), ("brier_score", "brier"), ("nll", "nll")]:
            assert math.isclose(saved[a], calculated[b], abs_tol=1e-9), f"{key}: {a} mismatch"
    for key in ("gradcam_manifest", "gradcam_review", "human_review", "uncertainty_join"):
        exact_ids(read_csv(key), FIXED, key)
    exact_ids(ph_unc[ph_unc.image_id.isin(FIXED)], FIXED, "fixed uncertainty")
    dom = read_csv("domain_features")
    require_columns(dom, ["domain", "image_id", "true_label", "predicted_label", "correct"] + QUALITY + DOMAIN, "domain_features")
    ph_dom = dom[dom.domain == "PH2"]
    exact_ids(ph_dom, ph_raw.image_id, "PH2 domain features")
    for col1, col2 in [("true_label", "true_ham_label"), ("predicted_label", "predicted_class")]:
        assert (ph_dom.set_index("image_id").loc[ph_raw.image_id, col1].to_numpy() == ph_raw[col2].to_numpy()).all()
    quality = read_csv("quality_table")
    require_columns(quality, ["image_id", "split", "dx"] + QUALITY, "image_quality")
    assert len(quality) == 10015 and not quality.image_id.duplicated().any()
    assert set(quality.image_id).isdisjoint(ph_raw.image_id)
    print("PREFLIGHT: PASS")
    print(f"Checkpoint {CP_HASH}; split {SPLIT_HASH}; classes {CLASSES}")
    print(f"HAM {hm['n']} rows; PH2 {pm['n']} rows (NV 80, MEL 40); valid probabilities and saved uncertainty")
    print(f"Calibration recalculated: HAM ECE {hm['ece']:.6f}, Brier {hm['brier']:.6f}, NLL {hm['nll']:.6f}; PH2 ECE {pm['ece']:.6f}, Brier {pm['brier']:.6f}, NLL {pm['nll']:.6f}")
    print("Fixed Grad-CAM, human review, uncertainty join: 12/12 each")
    print("Image quality: HAM image_quality.csv; PH2 domain_features.csv joins 120/120")
    print("Stage 5 no inference/training/fitting/tuning/PH2 selection flags: verified")
    return hashes, ham_unc, ph_raw, ph_unc, dom, quality, hm, pm

def write_csv(name, frame):
    frame.to_csv(OUT / name, index=False, na_rep="")

def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

def describe(values):
    s = pd.to_numeric(pd.Series(values), errors="raise").dropna()
    return {"n": len(s), "mean": s.mean() if len(s) else np.nan, "median": s.median() if len(s) else np.nan, "standard_deviation": s.std(ddof=1) if len(s) > 1 else np.nan, "iqr": s.quantile(.75)-s.quantile(.25) if len(s) else np.nan}

def fig_save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=250, bbox_inches="tight")
    plt.close()

def boxplot(master, variable, ylabel, filename):
    a = master[master.correct][variable].astype(float)
    b = master[~master.correct][variable].astype(float)
    plt.figure(figsize=(6.5, 4.5))
    plt.boxplot([a, b], tick_labels=[f"Correct (n={len(a)})", f"Incorrect (n={len(b)})"], showfliers=True)
    plt.ylabel(ylabel)
    plt.title("PH² saved predictions: " + ylabel.lower())
    plt.ylim(bottom=0)
    fig_save(filename)

def main():
    hashes, ham, ph_raw, ph_unc, dom, ham_quality, hm, pm = preflight()
    OUT.mkdir(exist_ok=True)
    source_hashes = hashes.copy()
    review = read_csv("human_review").set_index("image_id")
    grad = read_csv("gradcam_manifest").set_index("image_id")
    grad_review = read_csv("gradcam_review").set_index("image_id")
    join = read_csv("uncertainty_join").set_index("image_id")
    ph_dom = dom[dom.domain == "PH2"].set_index("image_id")
    master = ph_unc.rename(columns={"top1_top2_margin": "prediction_margin"}).copy()
    master["correct"] = master.correct.astype(str).str.lower().eq("true")
    master["high_confidence_error"] = ~master.correct & master.confidence_band.isin(["HIGH", "VERY_HIGH"])
    master["is_fixed_gradcam_case"] = master.image_id.isin(FIXED)
    for col in QUALITY + DOMAIN:
        master["quality_" + col if col in QUALITY else "domain_" + col] = master.image_id.map(pd.to_numeric(ph_dom[col], errors="raise"))
    for col in review.columns:
        if col not in ("true_class", "predicted_class", "correct", "saved_confidence", "p_mel"):
            master["review_" + col] = master.image_id.map(review[col])
    for col in ("target_layer", "predicted_target_class", "true_target_class", "true_equals_predicted", "max_abs_probability_difference"):
        master["gradcam_" + col] = master.image_id.map(grad[col])
    for col in ("lesion_centered", "border_attention", "background_attention", "artifact_attention", "corner_or_frame_attention", "diffuse_attention", "ambiguous", "review_notes"):
        master["initial_gradcam_review_" + col] = master.image_id.map(grad_review[col])
    assert len(master) == 120 and master.image_id.nunique() == 120
    for image_id in FIXED:
        row = master.set_index("image_id").loc[image_id]
        assert row.true_class == review.loc[image_id, "true_class"] == join.loc[image_id, "true_class"]
        assert row.predicted_class == review.loc[image_id, "predicted_class"] == grad.loc[image_id, "predicted_class"]
        assert math.isclose(row.confidence, float(join.loc[image_id, "confidence"]), abs_tol=1e-9)
    write_csv("ph2_master_evidence.csv", master)

    errors = master[~master.correct]
    transitions = []
    for (true, pred), group in errors.groupby(["true_class", "predicted_class"], sort=True):
        record = {"true_class": true, "predicted_class": pred, "count": len(group), "fraction_all_ph2_errors": len(group)/len(errors)}
        for col in ("confidence", "entropy", "true_class_probability", "prediction_margin"):
            record["mean_"+col] = group[col].mean()
            if col in ("confidence", "entropy"):
                record["median_"+col] = group[col].median()
        for col in QUALITY:
            record["mean_quality_"+col] = group["quality_"+col].mean()
            record["median_quality_"+col] = group["quality_"+col].median()
        transitions.append(record)
    transitions = pd.DataFrame(transitions).sort_values(["count", "true_class", "predicted_class"], ascending=[False, True, True])
    write_csv("ph2_error_transition_summary.csv", transitions)

    groups = {
        "A_correct": master[master.correct], "B_incorrect": errors,
        "C_incorrect_high_very_high": errors[errors.confidence_band.isin(["HIGH", "VERY_HIGH"])],
        "D_incorrect_lower": errors[~errors.confidence_band.isin(["HIGH", "VERY_HIGH"])],
        "E_correct_high_very_high": master[master.correct & master.confidence_band.isin(["HIGH", "VERY_HIGH"])],
    }
    quality_rows = []
    uncertainty_rows = []
    for name, group in groups.items():
        for col in QUALITY:
            quality_rows.append({"group": name, "variable": col, "source": SOURCES["domain_features"], **describe(group["quality_"+col])})
        for col in ("confidence", "entropy", "true_class_probability", "prediction_margin"):
            uncertainty_rows.append({"group": name, "variable": col, **describe(group[col])})
    write_csv("image_quality_failure_summary.csv", pd.DataFrame(quality_rows))
    write_csv("uncertainty_failure_summary.csv", pd.DataFrame(uncertainty_rows))
    write_csv("high_confidence_external_failures.csv", master[master.high_confidence_error])
    fixed = master.set_index("image_id").loc[FIXED].reset_index()
    write_csv("gradcam_uncertainty_human_synthesis.csv", fixed)

    tags = []
    def tag(row, name, source, value, interpretation, confidence):
        tags.append(dict(image_id=row.image_id, failure_tag=name, evidence_source=source, evidence_value=str(value), interpretation=interpretation, confidence_of_interpretation=confidence))
    for row in master.itertuples(index=False):
        if not row.correct:
            tag(row, "class-confusion failure", SOURCES["ph2_predictions"], row.true_class+" -> "+row.predicted_class, "Observed incorrect seven-class prediction.", "DIRECT")
        if row.high_confidence_error:
            tag(row, "high-confidence misclassification", SOURCES["ph2_uncertainty"], f"{row.confidence:.4f} ({row.confidence_band})", "Incorrect prediction in the prespecified HIGH or VERY_HIGH band.", "DIRECT")
        if row.confidence_band == "LOW":
            tag(row, "low-confidence correct prediction" if row.correct else "low-confidence misclassification", SOURCES["ph2_uncertainty"], f"{row.confidence:.4f}", "Observed LOW confidence band; this is not a clinical uncertainty threshold.", "DIRECT")
        if row.image_id in FIXED:
            r = review.loc[row.image_id]
            tag(row, "reviewed Grad-CAM case category", SOURCES["human_review"], r.case_category, "Human-reviewed category preserved verbatim.", "DIRECT")
            if not row.correct and any(word in str(r.qualitative_background_attention).lower()+" "+str(r.qualitative_border_attention).lower() for word in ("moderate", "substantial", "noticeable", "strong", "prominent")):
                tag(row, "attention-localization concern", SOURCES["human_review"], str(r.qualitative_border_attention)+" | "+str(r.qualitative_background_attention), "Prediction error co-occurs with reviewed border/background attention; no localization metric or causal claim.", "TENTATIVE")
    taxonomy = pd.DataFrame(tags)
    write_csv("failure_mode_taxonomy.csv", taxonomy)
    tag_counts = taxonomy.groupby("failure_tag").image_id.nunique().sort_values(ascending=False).to_dict()

    context = []
    for dataset, m, frame in [("HAM10000_internal_test", hm, ham), ("PH2_external_followup", pm, master)]:
        high = frame.confidence_band.isin(["HIGH", "VERY_HIGH"])
        wrong = frame.correct.astype(str).str.lower().eq("false")
        domain_subset = dom[dom.domain == ("HAM10000" if dataset.startswith("HAM") else "PH2")]
        record = dict(dataset=dataset, n=m["n"], accuracy=m["accuracy"], mean_confidence=m["mean_confidence"], mean_entropy=m["mean_entropy"], ece=m["ece"], brier=m["brier"], nll=m["nll"], error_count=m["errors"], high_very_high_confidence_error_count=int((high & wrong).sum()), high_very_high_confidence_error_rate_all_samples=float((high & wrong).mean()), high_very_high_confidence_error_fraction_errors=float((high & wrong).sum()/wrong.sum()), cohort_note="Internal held-out seven-class test" if dataset.startswith("HAM") else "External follow-up, mapped NV/MEL truth only", domain_descriptor_subset_n=len(domain_subset), domain_descriptor_subset_note="NV/MEL truth only in both domain-feature cohorts")
        for col in QUALITY + DOMAIN:
            record["domain_subset_mean_" + col] = pd.to_numeric(domain_subset[col], errors="raise").mean()
        context.append(record)
    write_csv("ham_vs_ph2_failure_context.csv", pd.DataFrame(context))

    boxplot(master, "confidence", "Confidence", "ph2_correct_vs_incorrect_confidence.png")
    boxplot(master, "entropy", "Predictive entropy (nats)", "ph2_correct_vs_incorrect_entropy.png")
    plt.figure(figsize=(8, 4.5))
    labels = [r.true_class.upper()+" → "+r.predicted_class.upper() for r in transitions.itertuples()]
    plt.bar(labels, transitions["count"], color="#446c91")
    plt.ylabel("PH² error count")
    plt.title("All observed PH² true-to-predicted errors (n=45)")
    plt.ylim(0, max(transitions["count"])*1.17)
    plt.xticks(rotation=35, ha="right")
    fig_save("ph2_error_transition_counts.png")
    bands = ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    good = [len(master[master.correct & (master.confidence_band == b)]) for b in bands]
    bad = [len(master[~master.correct & (master.confidence_band == b)]) for b in bands]
    plt.figure(figsize=(7, 4.5))
    plt.bar(bands, good, label="Correct", color="#377f77")
    plt.bar(bands, bad, bottom=good, label="Incorrect", color="#ba665e")
    plt.ylabel("PH² image count")
    plt.title("Outcomes by saved confidence band (n=120)")
    plt.legend()
    fig_save("ph2_confidence_error_distribution.png")
    plt.figure(figsize=(7, 4.5))
    x = np.arange(2)
    plt.bar(x-.2, [hm["ece"], pm["ece"]], .4, label="ECE")
    plt.bar(x+.2, [hm["brier"], pm["brier"]], .4, label="Seven-class Brier")
    plt.xticks(x, ["HAM internal (n=1,014)", "PH² NV/MEL (n=120)"])
    plt.ylabel("Metric value (lower is better)")
    plt.title("Saved-prediction calibration context; cohorts differ")
    plt.legend()
    fig_save("ham_vs_ph2_calibration_context.png")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.flat, ["brightness", "contrast", "sharpness", "saturation"]):
        ax.boxplot([master.loc[master.correct, "quality_"+col], master.loc[~master.correct, "quality_"+col]], tick_labels=["Correct (75)", "Incorrect (45)"])
        ax.set_title(col.replace("_", " ").title())
        ax.set_ylim(bottom=0)
    fig.suptitle("PH² saved image descriptors by outcome; descriptive only")
    fig_save("image_quality_correct_vs_incorrect.png")
    plt.figure(figsize=(9, 4.7))
    colors = ["#377f77" if x else "#ba665e" for x in fixed.correct]
    plt.bar(fixed.image_id, fixed.confidence, color=colors)
    plt.axhline(.75, color="gray", linestyle="--", label="Saved HIGH band starts at 0.75")
    plt.ylim(0, 1)
    plt.ylabel("Saved top-class probability")
    plt.title("Frozen 12 PH² Grad-CAM cases; green correct, red incorrect")
    plt.xticks(rotation=45)
    plt.legend()
    fig_save("fixed_gradcam_case_confidence.png")
    plt.figure(figsize=(9, 5))
    plt.barh(list(reversed(tag_counts)), list(reversed(tag_counts.values())), color="#446c91")
    plt.xlabel("Tagged PH² cases; tags can overlap")
    plt.title("Evidence-backed descriptive tags")
    fig_save("failure_mode_counts.png")

    card_lines = ["# Fixed PH² case evidence cards", "", "All 12 cases were fixed before Stage 6. Attention text is copied from saved human review; no new map interpretation was performed.", ""]
    for row in fixed.itertuples(index=False):
        r = review.loc[row.image_id]
        mytags = taxonomy[taxonomy.image_id == row.image_id].failure_tag.tolist()
        q = ", ".join(f"{v}={getattr(row, 'quality_'+v):.3f}" for v in QUALITY)
        card_lines += [f"## {row.image_id}", "", f"Ground truth: {row.true_class}; prediction: {row.predicted_class}; {'correct' if row.correct else 'incorrect'}.", "", f"Confidence: {row.confidence:.4f}; entropy: {row.entropy:.4f} nats; margin: {row.prediction_margin:.4f}; band: {row.confidence_band}.", "", f"Image-quality evidence (saved PH² domain descriptors): {q}.", "", f"Grad-CAM evidence: {r.qualitative_lesion_attention} Border: {r.qualitative_border_attention} Background: {r.qualitative_background_attention} Predicted versus true map: {r.predicted_vs_true_map_difference}", "", f"Human review: category **{r.case_category}**. {r.qualitative_interpretation} Limitation: {r.limitation}", "", f"Failure tags: {', '.join(mytags)}.", "", "Interpretation: These saved findings co-occur in this selected case; the map does not establish why the prediction was made.", ""]
    (OUT / "case_failure_cards.md").write_text("\n".join(card_lines), encoding="utf-8")

    dominant = [{"transition": r.true_class+" -> "+r.predicted_class, "count": int(r.count)} for r in transitions.itertuples()]
    quality_differences = {v: float(master.loc[~master.correct, "quality_"+v].mean() - master.loc[master.correct, "quality_"+v].mean()) for v in QUALITY}
    high_errors = master[master.high_confidence_error]
    summary = {
        "experiment_identity": "efficientnetv2s_controlled_exp5", "checkpoint_sha256": CP_HASH, "split_sha256": SPLIT_HASH, "class_order": CLASSES,
        "ham_n": hm["n"], "ph2_n": pm["n"], "ham_accuracy": hm["accuracy"], "ph2_accuracy": pm["accuracy"],
        "ham_ece": hm["ece"], "ph2_ece": pm["ece"], "ham_brier": hm["brier"], "ph2_brier": pm["brier"], "ham_nll": hm["nll"], "ph2_nll": pm["nll"],
        "ph2_error_count": pm["errors"], "ph2_high_confidence_error_count": len(high_errors), "dominant_error_transitions": dominant,
        "fixed_gradcam_case_count": len(FIXED), "fixed_gradcam_join_count": len(fixed), "image_quality_evidence_available": True,
        "per_image_quality_join_available": True, "failure_mode_counts": tag_counts,
        "major_observations": [f"HAM accuracy {hm['accuracy']:.3f}; PH² accuracy {pm['accuracy']:.3f}; cohorts differ in class composition.", f"PH² has {pm['errors']} errors; {len(high_errors)} are HIGH or VERY_HIGH confidence.", "PH² image descriptors exist for all 120 cases in saved domain_features.csv."],
        "supported_inferences": ["Saved PH² predictions have lower accuracy and higher ECE/Brier/NLL than the HAM held-out test.", "Human-reviewed attention concerns co-occur with some fixed-case errors, without establishing cause."],
        "unsupported_hypotheses": ["Domain shift alone caused the accuracy difference.", "Image quality caused individual errors.", "Grad-CAM attention caused misclassification or tracks clinical reasoning.", "These findings establish clinical readiness or generalization to other external datasets."],
        "limitations": ["PH² was previously used in project preprocessing diagnostics and an earlier evaluation.", "PH² includes mapped NV/MEL truth only; HAM test covers seven classes.", "Grad-CAM cases are a frozen, deliberately stratified set of 12 and cannot estimate population prevalence.", "Image descriptors are proxies, not clinical quality scores; no lesion masks or localization accuracy metric exist."],
        "recommended_next_step": "Evaluate a separately prespecified, untouched external cohort with matched NV/MEL composition and independently assessed image quality and lesion localization.",
        "skipped_optional_outputs": [], "quality_mean_difference_incorrect_minus_correct": quality_differences,
    }
    write_json("stage6_summary.json", summary)
    config = {"stage_name": "Stage 6 unified external-validation and failure-mode synthesis", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "source_directories": sorted(set(str(Path(v).parent) for v in SOURCES.values())), "source_files": list(SOURCES.values()), "checkpoint_sha256": CP_HASH, "split_sha256": SPLIT_HASH, "class_order": CLASSES, "fixed_gradcam_ids": FIXED,
              "definitions": {"correct": "true_class equals predicted_class", "entropy": "-sum(p ln(max(p,1e-12)))", "prediction_margin": "largest minus second-largest class probability", "ECE": "10 fixed equal-width bins", "Brier": "seven-class sum of squared probability error per case", "NLL": "negative log true-class probability"},
              "confidence_band_source": SOURCES["uncertainty_config"], "failure_group_definitions": {k: v for k,v in zip(groups, ["Correct PH² predictions", "Incorrect PH² predictions", "Incorrect HIGH/VERY_HIGH", "Incorrect LOW/MODERATE", "Correct HIGH/VERY_HIGH"])},
              "missing_value_policy": "Blank CSV cells for unavailable case-level reviewed evidence; no imputation.", "image_quality_variables_used": QUALITY,
              "image_quality_source_note": "HAM source image_quality.csv; PH² per-image descriptors from saved domain_features.csv.",
              "no_inference_rerun": True, "no_training": True, "no_calibration_fitting": True, "no_threshold_tuning": True, "ph2_used_for_model_selection": False}
    write_json("stage6_config.json", config)

    human_summary = read_json("human_summary")
    lines = ["# Stage 6: Unified external-validation and failure-mode synthesis", "", "## Executive summary", "", f"Frozen Experiment #5 achieved {hm['accuracy']:.1%} on the 1,014-image HAM internal test and {pm['accuracy']:.1%} on the 120-image PH² external follow-up. PH² produced {pm['errors']} errors, including {len(high_errors)} in the saved HIGH/VERY_HIGH bands. These observations describe different cohorts and do not isolate domain shift as the cause.", "", "## Research motivation and provenance", "", f"This is saved-artifact analysis. Checkpoint SHA256 `{CP_HASH}` and frozen split SHA256 `{SPLIT_HASH}` matched before writing outputs. Frozen class order: {', '.join(CLASSES)}. No training, inference, map regeneration, calibration fitting, or threshold selection occurred.", "", "## Evidence sources and cohort definitions", "", "Sources include Experiment #5 test probabilities, PH² external probabilities, domain descriptors, HAM image-quality assessment, saved uncertainty/calibration, and the frozen 12-case human Grad-CAM review. HAM is an internal held-out seven-class test. PH² is external follow-up evidence restricted to 80 mapped NV and 40 mapped MEL images; 80 atypical nevi were excluded by the existing mapping. PH² had prior project use in diagnostics and earlier evaluation.", "", "Groups were fixed before analysis: A correct, B incorrect, C incorrect HIGH/VERY_HIGH, D incorrect LOW/MODERATE, and E correct HIGH/VERY_HIGH. HIGH means 0.75–<0.90 and VERY_HIGH 0.90–1.00, as specified by Stage 5. The 12 Grad-CAM cases retain their recorded human-review categories.", "", "## Internal HAM and PH² context (RQ1, RQ2)", "", "| Metric | HAM internal | PH² follow-up |", "|---|---:|---:|", f"| N | {hm['n']} | {pm['n']} |", f"| Accuracy | {hm['accuracy']:.4f} | {pm['accuracy']:.4f} |", f"| Mean confidence | {hm['mean_confidence']:.4f} | {pm['mean_confidence']:.4f} |", f"| Mean entropy, nats | {hm['mean_entropy']:.4f} | {pm['mean_entropy']:.4f} |", f"| ECE, 10 bins | {hm['ece']:.4f} | {pm['ece']:.4f} |", f"| Seven-class Brier | {hm['brier']:.4f} | {pm['brier']:.4f} |", f"| NLL | {hm['nll']:.4f} | {pm['nll']:.4f} |", "", f"Observed accuracy difference is {(pm['accuracy']-hm['accuracy'])*100:.2f} percentage points. PH² has higher ECE, Brier, NLL and entropy, and lower mean confidence in these saved predictions. Accuracy is not directly attributable to image domain: class composition and cohort selection differ. The pre-existing NV/MEL-only HAM comparison (n=783) reports accuracy {read_json('domain_metrics')['overlap_metrics']['HAM10000_internal_test_nv_mel_only']['accuracy']:.4f}; this is a more comparable label subset, but still differs in prevalence and acquisition.", "", "## Domain-shift and image-quality evidence (RQ4)", "", "The saved domain analysis provides per-image brightness, contrast, sharpness, saturation, dark/bright pixel ratios, image entropy, and illumination variation for all PH² cases. The earlier `image_quality.csv` itself covers HAM only. Stage 6 joins PH² descriptors by exact image ID from `domain_features.csv` and reports descriptive outcome groups in `image_quality_failure_summary.csv`. These are image statistics, not clinical quality ratings. No broad hypothesis-testing search or causal attribution was performed.", "", f"In the saved NV/MEL domain-feature cohorts, HAM has {context[0]['domain_descriptor_subset_n']} images and PH² has {context[1]['domain_descriptor_subset_n']}; their mean brightness is {context[0]['domain_subset_mean_brightness']:.2f} versus {context[1]['domain_subset_mean_brightness']:.2f}, and mean contrast is {context[0]['domain_subset_mean_contrast']:.2f} versus {context[1]['domain_subset_mean_contrast']:.2f}. All saved descriptor means are in `ham_vs_ph2_failure_context.csv`.", "", "Incorrect minus correct PH² mean descriptor differences: " + "; ".join(f"{k} {v:+.3f}" for k,v in quality_differences.items()) + ".", "", "## External error transitions (RQ3)", "", "| True → predicted | Count | Fraction of errors | Mean confidence |", "|---|---:|---:|---:|"]
    for r in transitions.itertuples():
        lines.append(f"| {r.true_class.upper()} → {r.predicted_class.upper()} | {r.count} | {r.fraction_all_ph2_errors:.1%} | {r.mean_confidence:.3f} |")
    lines += ["", "All observed transitions are included. MEL→NV and NV→MEL are interpreted descriptively alongside out-of-subset predictions.", "", "## High-confidence external failures (RQ6)", "", f"There are {len(high_errors)} incorrect PH² predictions in HIGH/VERY_HIGH bands: " + ", ".join(high_errors.image_id) + ". This is a saved-band count, not a threshold selected on PH². Full probabilities and case descriptors appear in `high_confidence_external_failures.csv`.", "", "## Fixed 12-case Grad-CAM, uncertainty and human review (RQ5)", "", f"All 12 frozen cases join exactly. The human-reviewed set contains {human_summary['correct_cases']} correct and {human_summary['incorrect_cases']} incorrect cases. The saved human summary records {human_summary['noticeable_border_or_background_activation_count']} cases with noticeable border/background activation and {human_summary['misclassified_cases_with_visibly_different_predicted_and_true_maps']} misclassified cases with visibly different predicted and true maps. These counts are confined to this stratified set. IMD085 remains an uncertain reviewed case. `gradcam_review.csv` contains pending `UNCERTAIN` placeholders; Stage 6 uses the later `stage3_case_comparison_reviewed.csv` for qualitative interpretation.", "", "## Failure-mode taxonomy (RQ7)", "", "Tags are nonexclusive. Direct tags encode observed errors, confidence bands, and recorded human categories. Attention-localization concern tags are tentative textual associations, not localization measurements or explanations of model behavior.", "", "| Tag | Unique cases |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k,v in tag_counts.items()]
    lines += ["", "## Integrated interpretation: observation, inference, hypothesis", "", "**Observation:** PH² accuracy and calibration metrics are worse than HAM metrics in the saved predictions; PH² has 45 errors and 11 HIGH/VERY_HIGH errors. Some selected errors co-occur with reviewed attention concerns and measured image descriptors.", "", "**Supported inference:** This external follow-up exposes prediction and calibration weaknesses for the mapped NV/MEL subset; the combined table allows exact case-level audit.", "", "**Unsupported hypotheses (RQ8):** Domain shift alone caused the difference; image quality caused particular errors; Grad-CAM localization caused class confusion; the model is clinically ready or generalizes to other external datasets. These require controlled evidence.", "", "## Limitations", "", "PH² previously contributed to project diagnostics and an earlier evaluation. Cohort composition differs from HAM. The 12 Grad-CAM cases were fixed and stratified, not a random sample. Human visual lesion boundaries have no masks or localization accuracy measure. Image statistics are imperfect proxies. No uncertainty intervals or significance claims are made.", "", "## Recommended next experiment", "", summary["recommended_next_step"], "", "## Reproducibility and figures", "", "Run `python analysis/stage6_failure_mode_synthesis_exp5/generate_stage6_failure_synthesis.py` from any working directory. It verifies source hashes and schemas before writing. Figures show saved distributions, transitions, confidence bands, calibration context, quality descriptors, fixed-case confidence, and overlapping tag counts. No requested optional figure was skipped. `experiment_manifest.json` records source and generated SHA256 values; it excludes its own hash to avoid a recursive digest.", ""]
    (OUT / "stage6_failure_mode_synthesis_report.md").write_text("\n".join(lines), encoding="utf-8")
    assert {v: sha(ROOT / v) for v in SOURCES.values()} == source_hashes, "Source files changed during Stage 6"
    generated = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "experiment_manifest.json"}
    write_json("experiment_manifest.json", {"stage": "Stage 6", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "source_file_sha256": source_hashes, "generated_file_sha256": generated, "manifest_self_hash_policy": "Excluded to avoid recursive digest", "training_or_finetuning": False, "inference_rerun": False, "gradcam_regeneration": False, "calibration_fitting": False, "temperature_scaling": False, "threshold_tuning": False, "ph2_used_for_model_selection": False, "ph2_used_for_calibration_fitting": False})
    print("STAGE 6 STATUS: PASS")
    print(f"Frozen checkpoint: {CP_HASH}\nFrozen split: {SPLIT_HASH}\nHAM samples: {hm['n']}\nPH2 samples: {pm['n']}\nPH2 correct: {pm['correct']}\nPH2 incorrect: {pm['errors']}\nPH2 high-confidence errors: {len(high_errors)}\nFixed Grad-CAM cases: 12/12\nImage-quality evidence: available\nPer-image quality join: available (120/120)")
    print("Dominant external error transitions:", dominant)
    print("Failure modes supported:", tag_counts)
    print("Training performed: NO\nInference rerun: NO\nGrad-CAM regenerated: NO\nCalibration fitted: NO\nThreshold tuned: NO\nPrevious artifacts modified: NO")
    print("Stage 6 output directory:", OUT)
    print("Generated files:", sorted(p.name for p in OUT.iterdir() if p.is_file()))
    print("Important observations:", summary["major_observations"])
    print("Unsupported conclusions:", summary["unsupported_hypotheses"])
    print("Recommended next experiment:", summary["recommended_next_step"])

if __name__ == "__main__":
    main()
