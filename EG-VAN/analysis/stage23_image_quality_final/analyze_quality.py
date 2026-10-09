"""Stage 23 validation-only technical image-quality and error-ranking study.

Reads only frozen validation IDs and their raw/processed HAM images. Never
loads a model, HAM test outcome, PH2 image, or quality ground-truth label.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "analysis/stage23_uncertainty_final/analyze_stage23_uncertainty.py"
sys.path.insert(0, str(ROOT / "src"))
from image_quality import QUALITY_FEATURES, compute_image_quality  # noqa: E402

spec = importlib.util.spec_from_file_location("stage23_uncertainty_source", SOURCE)
assert spec and spec.loader
day1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day1)

CLASSES = day1.CLASSES
PRIMARY_FEATURES = tuple(f"raw_{name}" for name in QUALITY_FEATURES)
BAND_FEATURES = ("raw_sharpness", "raw_brightness", "raw_contrast", "raw_dark_pixel_ratio",
                 "raw_bright_pixel_ratio", "raw_saturation")
COVERAGES = (1.0, .9, .8, .7, .6, .5)
CV_SEED = 42
BOOTSTRAP_SEED = 142
BOOTSTRAP_REPETITIONS = 1000


def sha256(path: Path) -> str:
    return day1.sha256(path)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def decode_image(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        if image.format != "JPEG" or image.mode not in ("RGB", "L"):
            raise ValueError(f"Unexpected validation image format/mode: {path}")
        dimensions = image.size
        image.verify()
    if dimensions[0] < 16 or dimensions[1] < 16:
        raise ValueError(f"Validation image too small: {path}")
    return dimensions


def joined_measurements() -> tuple[pd.DataFrame, dict]:
    predictions, probabilities, audit = day1.audit_inputs()
    split = pd.read_csv(day1.SPLIT)
    val = split.loc[split.split == "val", ["image_id", "lesion_id", "dx"]]
    metadata = pd.read_csv(ROOT / "data/processed/metadata_clean.csv", usecols=["image_id", "lesion_id", "dx"])
    if metadata.image_id.duplicated().any() or val.image_id.duplicated().any():
        raise ValueError("Duplicate validation or metadata IDs")
    joined = predictions.merge(val, on="image_id", validate="one_to_one")
    joined = joined.merge(metadata, on="image_id", suffixes=("_split", "_metadata"), validate="one_to_one")
    if (len(joined) != 986 or not (joined.true_label == joined.dx_split).all() or
            not (joined.dx_split == joined.dx_metadata).all() or
            not (joined.lesion_id_split == joined.lesion_id_metadata).all()):
        raise ValueError("Validation prediction/split/metadata join mismatch")
    quality_table = pd.read_csv(ROOT / "experiments/image_quality/image_quality.csv")
    if quality_table.image_id.duplicated().any():
        raise ValueError("Historical quality table has duplicate IDs")
    historical = quality_table.set_index("image_id")
    probabilities_by_id = dict(zip(predictions.image_id, probabilities))
    rows = []
    failures = []
    for record in joined.sort_values("image_id").itertuples(index=False):
        image_id = record.image_id
        raw = ROOT / "data/raw/images" / f"{image_id}.jpg"
        processed = ROOT / "data/processed/images" / f"{image_id}.jpg"
        try:
            raw_dims = decode_image(raw)
            proc_dims = decode_image(processed)
            raw_metrics = compute_image_quality(raw)
            proc_metrics = compute_image_quality(processed)
            with Image.open(processed) as pil:
                rgb = pil.convert("RGB")
                resized = np.asarray(rgb.resize((384, 384), Image.Resampling.BILINEAR), dtype=np.uint8)
            view_metrics = compute_image_quality(resized)
            if image_id not in historical.index or historical.loc[image_id, "split"] != "val" or historical.loc[image_id, "dx"] != record.true_label:
                raise ValueError("Historical processed-quality join mismatch")
            for feature in QUALITY_FEATURES:
                if not math.isclose(proc_metrics[feature], float(historical.loc[image_id, feature]), rel_tol=0, abs_tol=1e-8):
                    raise ValueError(f"Historical processed quality differs: {feature}")
            values = probabilities_by_id[image_id]
            positive = values[values > 0]
            entropy = float(-np.sum(positive * np.log(positive)))
            rows.append({
                "image_id": image_id, "lesion_id": record.lesion_id_split,
                "true_class": record.true_label, "predicted_class": record.predicted_label,
                "correct": bool(record.correct_bool), "error": int(not record.correct_bool),
                "mel_false_negative": int(record.true_label == "mel" and record.predicted_label != "mel"),
                "mel_to_nv": int(record.true_label == "mel" and record.predicted_label == "nv"),
                "confidence": float(np.max(values)), "predictive_entropy_nats": entropy,
                "normalized_entropy": entropy / math.log(7),
                "raw_width": raw_dims[0], "raw_height": raw_dims[1],
                "processed_width": proc_dims[0], "processed_height": proc_dims[1],
                "raw_sha256": sha256(raw), "processed_sha256": sha256(processed),
                **{f"raw_{k}": v for k, v in raw_metrics.items()},
                **{f"processed_{k}": v for k, v in proc_metrics.items()},
                **{f"model_view_{k}": v for k, v in view_metrics.items()},
            })
        except Exception as exc:
            failures.append({"image_id": image_id, "reason": f"{type(exc).__name__}: {exc}"})
    if failures:
        raise RuntimeError(f"{len(failures)} validation image failures; first: {failures[:3]}")
    table = pd.DataFrame(rows)
    if len(table) != 986 or table.image_id.nunique() != 986 or not np.isfinite(table[list(PRIMARY_FEATURES)].to_numpy()).all():
        raise ValueError("Incomplete/non-finite image quality table")
    provenance = {"source_sha256": audit["sha256"],
                  "historical_processed_quality_table_sha256": sha256(ROOT / "experiments/image_quality/image_quality.csv"),
                  "metadata_sha256": sha256(ROOT / "data/processed/metadata_clean.csv"),
                  "validation_images": len(table), "raw_images_readable": len(table), "processed_images_readable": len(table),
                  "raw_image_hashes_recorded": True, "processed_image_hashes_recorded": True,
                  "processed_measurements_match_historical_table": True,
                  "missing_or_duplicate_joins": 0,
                  "measurement_views": {
                      "raw": "Original 600x450 JPEG as saved in data/raw/images; RGB decoding via OpenCV quality function",
                      "processed": "EG-VAN paper-informed output JPEG before Stage 23 resize; matches historical table",
                      "model_view": "Processed JPEG decoded RGB and PIL bilinear resized to 384x384 before tensor/normalization",
                  }}
    return table, provenance


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(center - radius), float(center + radius)]


def distribution(values: pd.Series) -> dict:
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(values.median()),
            "q25": float(values.quantile(.25)), "q75": float(values.quantile(.75))}


def quality_error_analysis(table: pd.DataFrame, provenance: dict) -> dict:
    correct = table.correct.to_numpy(dtype=bool)
    metrics = {}
    for view in ("raw", "processed", "model_view"):
        metrics[view] = {}
        for feature in QUALITY_FEATURES:
            key = f"{view}_{feature}"
            values = table[key]
            metrics[view][feature] = {
                "correct": distribution(values[correct]), "incorrect": distribution(values[~correct]),
                "spearman_with_confidence": float(spearmanr(values, table.confidence).statistic),
                "spearman_with_predictive_entropy": float(spearmanr(values, table.predictive_entropy_nats).statistic),
            }
    by_class = {}
    for column in ("true_class", "predicted_class"):
        by_class[column] = {}
        for cls in CLASSES:
            part = table[table[column] == cls]
            by_class[column][cls] = {
                "n": int(len(part)), "error_count": int(part.error.sum()),
                "error_rate": float(part.error.mean()) if len(part) else None,
                "raw_feature_medians": {f: float(part[f"raw_{f}"].median()) for f in QUALITY_FEATURES} if len(part) else {},
            }
    bands = {}
    sorted_table = table.sort_values("image_id")
    for feature in BAND_FEATURES:
        # Equal-count exploratory bands. Stable ascending image_id breaks value ties.
        ranks = sorted_table[feature].rank(method="first")
        groups = pd.qcut(ranks, 3, labels=("low", "middle", "high"))
        bands[feature] = {}
        for label in ("low", "middle", "high"):
            part = sorted_table[groups == label]
            errors = int(part.error.sum())
            bands[feature][label] = {
                "n": len(part), "feature_min": float(part[feature].min()),
                "feature_max": float(part[feature].max()), "errors": errors,
                "error_rate": errors / len(part), "error_rate_wilson_95": wilson(errors, len(part)),
                "mel_cases": int((part.true_class == "mel").sum()),
                "mel_false_negatives": int(part.mel_false_negative.sum()),
                "mel_to_nv": int(part.mel_to_nv.sum()),
                "mean_entropy": float(part.predictive_entropy_nats.mean()),
                "mean_confidence": float(part.confidence.mean()),
            }
    mel = table[table.true_class == "mel"]
    mel_groups = {}
    for name, part in (("correct_mel", mel[mel.correct]),
                       ("mel_false_negative", mel[~mel.correct]),
                       ("mel_to_nv", mel[mel.predicted_class == "nv"])):
        mel_groups[name] = {"n": len(part),
                            "raw_feature_medians": {f: float(part[f"raw_{f}"].median()) for f in QUALITY_FEATURES},
                            "mean_entropy": float(part.predictive_entropy_nats.mean()),
                            "mean_confidence": float(part.confidence.mean())}
    return {
        "scope": "Stage 23 selected HAM validation only; observational and exploratory",
        "provenance": provenance,
        "definitions": {
            "sharpness": "Variance of cv2.Laplacian(grayscale, CV_64F); higher usually means more edges, not a validated blur label",
            "brightness": "Mean grayscale intensity on 0..255 scale",
            "contrast": "Standard deviation of grayscale intensity",
            "underexposure_indicator": "Fraction of grayscale pixels <30 (raw_dark_pixel_ratio)",
            "overexposure_indicator": "Fraction of grayscale pixels >225 (raw_bright_pixel_ratio)",
            "saturation": "Mean HSV saturation /255",
            "other": "8-bit grayscale histogram entropy; 16x16-grid spatial illumination variation",
            "artifact_detection": "Not implemented: no validated hair/ruler/occlusion ground truth or defensible detector",
            "band_rule": "Exploratory equal-count terciles from validation feature ranks; image_id breaks ties; no clinical cutoffs",
            "interval_rule": "Wilson 95% binomial intervals for descriptive error rates; no multiplicity correction",
        },
        "n": len(table), "errors": int(table.error.sum()),
        "quality_by_correctness_and_view": metrics,
        "classwise_distributions": by_class,
        "exploratory_quality_bands": bands,
        "melanoma_subgroups": mel_groups,
        "interpretation_limit": "Association cannot establish acquisition-quality causation; class and lesion difficulty may confound",
    }


def rank_metrics(error: np.ndarray, score: np.ndarray, ids: np.ndarray) -> dict:
    # Scores increase with predicted error; retain the lowest-risk images first.
    order = day1.ranked_indices(score, ids)
    risks = day1.risk_curve(error.astype(bool), order)
    result = {"auroc": float(roc_auc_score(error, score)),
              "average_precision": float(average_precision_score(error, score)),
              "aurc": float(np.mean(risks)), "risk_coverage": {}}
    n = len(error)
    for coverage in COVERAGES:
        k = math.ceil(coverage * n)
        keep = order[:k]
        reviewed = order[k:]
        result["risk_coverage"][str(coverage)] = {
            "retained": k, "reviewed": n - k, "retained_error_rate": float(error[keep].mean()),
            "retained_accuracy": float(1 - error[keep].mean()),
            "errors_reviewed": int(error[reviewed].sum()),
            "fraction_errors_reviewed": float(error[reviewed].sum() / error.sum()),
        }
    return result


def oof_scores(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    y = table.error.to_numpy(dtype=int)
    groups = table.lesion_id.astype(str).to_numpy()
    entropy = table.predictive_entropy_nats.to_numpy(dtype=float)
    raw = table[list(PRIMARY_FEATURES)].to_numpy(dtype=float)
    designs = {"quality_only_oof": raw, "entropy_plus_quality_oof": np.column_stack((entropy, raw))}
    scores = {name: np.zeros(len(table), dtype=float) for name in designs}
    fold_ids = np.full(len(table), -1, dtype=int)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    folds = []
    for fold, (train, held) in enumerate(cv.split(raw, y, groups)):
        if set(groups[train]) & set(groups[held]) or len(set(y[train])) != 2 or len(set(y[held])) != 2:
            raise ValueError("Group leakage or undefined CV fold")
        fold_ids[held] = fold
        folds.append({"fold": fold, "train_n": len(train), "held_n": len(held),
                      "train_errors": int(y[train].sum()), "held_errors": int(y[held].sum()),
                      "group_overlap": 0})
        for name, design in designs.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs"))
            model.fit(design[train], y[train])
            scores[name][held] = model.predict_proba(design[held])[:, 1]
    if (fold_ids < 0).any() or not all(np.isfinite(value).all() for value in scores.values()):
        raise ValueError("Incomplete OOF predictions")
    ids = table.image_id.to_numpy()
    scores = {"entropy_direct": entropy, **scores}
    metrics = {name: rank_metrics(y, value, ids) for name, value in scores.items()}
    # Cluster bootstrap conditions on fixed OOF scores; it does not refit models.
    unique = np.unique(groups)
    by_group = {name: np.flatnonzero(groups == name) for name in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = []
    auc_values = {name: [] for name in scores}
    for _ in range(BOOTSTRAP_REPETITIONS):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([by_group[name] for name in chosen])
        if len(np.unique(y[positions])) < 2:
            continue
        sample_aucs = {name: roc_auc_score(y[positions], values[positions]) for name, values in scores.items()}
        for name, value in sample_aucs.items():
            auc_values[name].append(value)
        deltas.append(sample_aucs["entropy_plus_quality_oof"] - sample_aucs["entropy_direct"])
    bootstrap = {"seed": BOOTSTRAP_SEED, "requested_repetitions": BOOTSTRAP_REPETITIONS,
                 "valid_repetitions": len(deltas),
                 "convention": "Percentile 95% lesion-cluster bootstrap of fixed OOF scores; no model refit or independent cohort",
                 "auroc_95": {name: list(map(float, np.quantile(values, [.025, .975]))) for name, values in auc_values.items()},
                 "combined_minus_entropy_auroc_delta": float(metrics["entropy_plus_quality_oof"]["auroc"] - metrics["entropy_direct"]["auroc"]),
                 "combined_minus_entropy_auroc_delta_95": list(map(float, np.quantile(deltas, [.025, .975])))}
    out = table[["image_id", "lesion_id", "true_class", "predicted_class", "error"]].copy()
    out["fold"] = fold_ids
    for name, values in scores.items():
        out[name] = values
    return out, {"scope": "Exploratory Stage 23 validation error ranking, lesion-grouped out-of-fold",
                 "positive_class": "classification error", "error_prevalence": float(y.mean()),
                 "quality_features": PRIMARY_FEATURES, "folds": folds,
                 "cv": "StratifiedGroupKFold(5, shuffle=True, random_state=42), grouped by lesion_id",
                 "estimator": "StandardScaler fitted on each training fold + LogisticRegression(C=1, class_weight='balanced', solver='lbfgs', max_iter=2000); no tuning",
                 "entropy_direct": "Original saved predictive entropy, no fit",
                 "risk_convention": "Retain lowest predicted error score, exact ties by ascending image_id; ceil(coverage*n); AURC is mean risk at every k=1..n",
                 "methods": metrics, "bootstrap": bootstrap,
                 "interpretation_limit": "OOF is development-only and not an untouched external estimate; no fitted quality warning threshold is deployed"}


def figures(table: pd.DataFrame, analysis: dict, comparison: dict) -> None:
    correct = table.correct.to_numpy(dtype=bool)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, feature, title in zip(axes, ("raw_sharpness", "raw_brightness", "raw_contrast"),
                                  ("Raw sharpness (log1p)", "Raw brightness", "Raw contrast")):
        for flag, color, label in ((True, "#26734d", "Correct"), (False, "#a63838", "Incorrect")):
            values = table.loc[correct == flag, feature].to_numpy()
            if feature == "raw_sharpness": values = np.log1p(values)
            ax.hist(values, bins=25, alpha=.5, density=True, color=color, label=label)
        ax.set(xlabel=title, ylabel="Density")
        ax.legend()
    fig.suptitle("Stage 23 validation: raw image proxies by classification outcome")
    fig.tight_layout(); fig.savefig(OUT / "quality_error_distributions.png", dpi=350); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, feature, title in zip(axes, ("raw_sharpness", "raw_brightness"),
                                  ("Raw sharpness (log1p)", "Raw brightness")):
        x = table[feature].to_numpy()
        if feature == "raw_sharpness": x = np.log1p(x)
        ax.scatter(x[correct], table.predictive_entropy_nats.to_numpy()[correct], s=10, alpha=.25, label="Correct", color="#26734d")
        ax.scatter(x[~correct], table.predictive_entropy_nats.to_numpy()[~correct], s=12, alpha=.55, label="Incorrect", color="#a63838")
        ax.set(xlabel=title, ylabel="Predictive entropy (nats)")
        ax.legend()
    fig.suptitle("Stage 23 validation: quality versus model uncertainty")
    fig.tight_layout(); fig.savefig(OUT / "quality_vs_entropy.png", dpi=350); plt.close(fig)

    selected = ("raw_sharpness", "raw_brightness", "raw_contrast")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    for ax, feature in zip(axes, selected):
        band = analysis["exploratory_quality_bands"][feature]
        names = ("low", "middle", "high")
        rates = [band[name]["error_rate"] for name in names]
        lows = [band[name]["error_rate_wilson_95"][0] for name in names]
        highs = [band[name]["error_rate_wilson_95"][1] for name in names]
        ax.bar(names, rates, color="#437c9a")
        ax.errorbar(range(3), rates, yerr=[np.asarray(rates) - lows, np.asarray(highs) - rates], fmt="none", color="black", capsize=4)
        ax.set(title=feature.replace("raw_", "Raw ").replace("_", " "), ylim=(0, .35), ylabel="Validation error rate")
    fig.suptitle("Exploratory equal-count quality bands (Wilson 95% intervals)")
    fig.tight_layout(); fig.savefig(OUT / "quality_error_rate_analysis.png", dpi=350); plt.close(fig)


def main() -> None:
    table, provenance = joined_measurements()
    analysis = quality_error_analysis(table, provenance)
    oof, comparison = oof_scores(table)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "image_quality_metrics.csv", index=False, float_format="%.17g")
    oof.to_csv(OUT / "quality_ranking_oof.csv", index=False, float_format="%.17g")
    write_json(OUT / "quality_error_analysis.json", analysis)
    write_json(OUT / "quality_uncertainty_comparison.json", comparison)
    figures(table, analysis, comparison)
    generated = ("image_quality_metrics.csv", "quality_ranking_oof.csv", "quality_error_analysis.json",
                 "quality_uncertainty_comparison.json", "quality_error_distributions.png",
                 "quality_vs_entropy.png", "quality_error_rate_analysis.png")
    write_json(OUT / "quality_analysis_manifest.json", {
        "scope": "Stage 23 selected validation images and saved predictions only",
        "analysis_source_sha256": sha256(Path(__file__).resolve()),
        "input_sha256": provenance["source_sha256"],
        "historical_processed_quality_table_sha256": provenance["historical_processed_quality_table_sha256"],
        "output_sha256": {name: sha256(OUT / name) for name in generated},
        "training_performed": False, "inference_performed": False,
        "ham_test_outcomes_accessed": False, "ph2_accessed": False,
    })
    print(json.dumps({"status": "COMPLETE_VALIDATION_QUALITY", "images": len(table),
                      "errors": int(table.error.sum()),
                      "auroc": {name: value["auroc"] for name, value in comparison["methods"].items()}}, indent=2))


if __name__ == "__main__":
    main()
