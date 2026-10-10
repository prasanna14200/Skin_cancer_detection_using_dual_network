"""Stage 23 validation-only Phase A analysis; no model loading or inference."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "analysis/stage23_image_quality_final"
OUT = Path(__file__).resolve().parent
FEATURES = ["raw_" + x for x in ("brightness", "contrast", "sharpness", "saturation",
    "dark_pixel_ratio", "bright_pixel_ratio", "entropy", "illumination_variation")]
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
REPS = 400
SEED = 2405


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources():
    manifest = json.loads((BASE / "quality_analysis_manifest.json").read_text())
    verified = {}
    for name in ("image_quality_metrics.csv", "quality_ranking_oof.csv"):
        actual = sha(BASE / name)
        assert actual == manifest["output_sha256"][name], name
        verified[name] = actual
    split = ROOT / "data/splits/split_leakage_aware.csv"
    assert sha(split) == manifest["input_sha256"]["frozen_split"]
    verified["frozen_split"] = sha(split)
    checkpoint = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt"
    assert sha(checkpoint) == manifest["input_sha256"]["checkpoint"]
    verified["checkpoint"] = sha(checkpoint)
    predictions = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_predictions.csv"
    assert sha(predictions) == manifest["input_sha256"]["validation_predictions"]
    verified["validation_predictions"] = sha(predictions)
    return verified


def load_data():
    q = pd.read_csv(BASE / "image_quality_metrics.csv")
    o = pd.read_csv(BASE / "quality_ranking_oof.csv")
    assert len(q) == len(o) == 986
    assert q.image_id.is_unique and o.image_id.is_unique
    assert q.lesion_id.notna().all()
    assert all(np.isfinite(q[f]) .all() for f in FEATURES)
    assert all(np.isfinite(o[f]).all() for f in ("entropy_direct", "quality_only_oof", "entropy_plus_quality_oof"))
    joined = q.merge(o, on=["image_id", "lesion_id", "true_class", "predicted_class", "error"],
                     validate="one_to_one")
    assert len(joined) == 986
    split = pd.read_csv(ROOT / "data/splits/split_leakage_aware.csv")
    val = split.loc[split.split == "val", ["image_id", "lesion_id", "dx"]]
    assert len(val) == 986 and val.image_id.is_unique
    chk = joined.merge(val, on=["image_id", "lesion_id"], validate="one_to_one")
    assert len(chk) == 986 and (chk.true_class == chk.dx).all()
    assert (joined.groupby("lesion_id").fold.nunique() == 1).all()
    assert joined.error.sum() == 168
    assert ((joined.true_class == "mel") & (joined.error == 1)).sum() == 37
    return joined.sort_values("image_id").reset_index(drop=True)


def cluster_samples(frame, reps=REPS, seed=SEED):
    groups = frame.groupby("lesion_id").indices
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        draw = rng.choice(keys, len(keys), replace=True)
        yield np.concatenate([groups[k] for k in draw])


def percentile(vals):
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals else None


def rank_effect(x, y):
    """2*AUC-1: positive means higher feature values among errors."""
    return float(2 * roc_auc_score(y, x) - 1) if len(np.unique(y)) == 2 else None


def class_adjusted_or(frame, feature, center, scale):
    y = frame.error.to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return None
    x = ((frame[feature].to_numpy(float) - center) / scale).reshape(-1, 1)
    dummies = pd.get_dummies(pd.Categorical(frame.true_class, categories=CLASSES),
                             drop_first=True).to_numpy(dtype=float)
    design = np.column_stack([x, dummies])
    # Eight separate one-feature models; fixed mild ridge avoids separation in small classes.
    fit = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300).fit(design, y)
    return float(np.exp(fit.coef_[0, 0]))


def quality_associations(df):
    rows = []
    adjusted = {}
    for fi, feat in enumerate(FEATURES):
        center = float(df[feat].mean())
        scale = float(df[feat].std(ddof=0))
        assert scale > 0
        whole = class_adjusted_or(df, feat, center, scale)
        boot_adj = []
        # Each draw samples lesions, preserving all of each lesion's images.
        for idx in cluster_samples(df, seed=SEED + fi):
            z = class_adjusted_or(df.iloc[idx], feat, center, scale)
            if z is not None and math.isfinite(z):
                boot_adj.append(z)
        adjusted[feat] = {"odds_ratio_per_full_cohort_SD": whole,
                          "cluster_bootstrap_95": percentile(boot_adj),
                          "valid_replicates": len(boot_adj), "full_cohort_SD": scale}
        for cls in ["ALL"] + CLASSES:
            sub = df if cls == "ALL" else df.loc[df.true_class == cls]
            x, y = sub[feat].to_numpy(float), sub.error.to_numpy(int)
            effect = rank_effect(x, y)
            bs = []
            for idx in cluster_samples(sub, seed=SEED + fi + 100 * (CLASSES.index(cls) + 1 if cls != "ALL" else 0)):
                z = rank_effect(x[idx], y[idx])
                if z is not None:
                    bs.append(z)
            rows.append({"feature": feat, "true_class": cls, "images": len(sub),
                         "lesions": sub.lesion_id.nunique(), "errors": int(y.sum()),
                         "error_rate": float(y.mean()), "feature_median_all": float(np.median(x)),
                         "feature_q1": float(np.percentile(x, 25)),
                         "feature_q3": float(np.percentile(x, 75)),
                         "median_correct": float(np.median(x[y == 0])) if (y == 0).any() else None,
                         "median_error": float(np.median(x[y == 1])) if (y == 1).any() else None,
                         "median_error_minus_correct": float(np.median(x[y == 1]) - np.median(x[y == 0])) if len(np.unique(y)) == 2 else None,
                         "rank_biserial_error_higher": effect,
                         "rank_biserial_ci_low": percentile(bs)[0] if bs else None,
                         "rank_biserial_ci_high": percentile(bs)[1] if bs else None,
                         "bootstrap_valid": len(bs),
                         "adjusted_error_OR_per_SD": whole if cls == "ALL" else None,
                         "adjusted_OR_ci_low": adjusted[feat]["cluster_bootstrap_95"][0] if cls == "ALL" else None,
                         "adjusted_OR_ci_high": adjusted[feat]["cluster_bootstrap_95"][1] if cls == "ALL" else None})
    return pd.DataFrame(rows), adjusted


def melanoma_analysis(df):
    mel = df.loc[df.true_class == "mel"].copy()
    assert len(mel) == 107 and mel.error.sum() == 37 and mel.mel_to_nv.sum() == 26
    rows = []
    for fi, feat in enumerate(FEATURES):
        x, y = mel[feat].to_numpy(float), mel.error.to_numpy(int)
        bs = []
        for idx in cluster_samples(mel, seed=SEED + 500 + fi):
            z = rank_effect(x[idx], y[idx])
            if z is not None:
                bs.append(z)
        rows.append({"feature": feat, "mel_total": 107, "mel_correct": 70,
                     "mel_false_negatives": 37, "mel_to_nv": 26,
                     "correct_median": float(np.median(x[y == 0])),
                     "false_negative_median": float(np.median(x[y == 1])),
                     "mel_to_nv_median": float(mel.loc[mel.mel_to_nv == 1, feat].median()),
                     "fn_minus_correct_median": float(np.median(x[y == 1]) - np.median(x[y == 0])),
                     "fn_rank_auc_higher_feature": float(roc_auc_score(y, x)),
                     "rank_biserial_fn_higher": rank_effect(x, y),
                     "rank_biserial_ci_low": percentile(bs)[0],
                     "rank_biserial_ci_high": percentile(bs)[1],
                     "bootstrap_valid": len(bs)})
    high_conf = mel.loc[(mel.error == 1) & (mel.confidence >= 0.9)]
    details = {"mel_total": 107, "mel_correct": 70, "mel_false_negatives": 37,
               "mel_to_nv": 26, "high_confidence_fn_definition": "saved maximum softmax >= 0.9; descriptive, not review cutoff",
               "high_confidence_fn_count": len(high_conf),
               "high_confidence_fn_ids": high_conf.image_id.tolist(),
               "correct_confidence_median": float(mel.loc[mel.error == 0, "confidence"].median()),
               "fn_confidence_median": float(mel.loc[mel.error == 1, "confidence"].median()),
               "correct_entropy_median": float(mel.loc[mel.error == 0, "predictive_entropy_nats"].median()),
               "fn_entropy_median": float(mel.loc[mel.error == 1, "predictive_entropy_nats"].median())}
    return pd.DataFrame(rows), details


def ranking(y, score, ids):
    order = np.lexsort((ids, score))
    ordered_y = y[order]
    k = math.ceil(.8 * len(y))
    reviewed = order[k:]
    curve = {}
    for coverage in (1.0, .9, .8, .7, .6, .5):
        kk = math.ceil(coverage * len(y))
        curve[str(coverage)] = {"retained": kk, "reviewed": len(y)-kk,
            "retained_error_rate": float(ordered_y[:kk].mean()),
            "errors_captured": int(ordered_y[kk:].sum())}
    return {"auroc": float(roc_auc_score(y, score)),
            "average_precision": float(average_precision_score(y, score)),
            "aurc": float(np.mean(np.cumsum(ordered_y) / np.arange(1, len(y) + 1))),
            "risk_coverage": curve,
            "at_80pct_coverage": {"retained": k, "reviewed": len(y) - k,
                "errors_captured": int(y[reviewed].sum()),
                "error_capture_fraction": float(y[reviewed].sum() / y.sum())}}


def ranking_comparison(df):
    y = df.error.to_numpy(int)
    ids = df.image_id.to_numpy(str)
    methods = ["entropy_direct", "quality_only_oof", "entropy_plus_quality_oof"]
    result = {"methods": {m: ranking(y, df[m].to_numpy(float), ids) for m in methods}}
    mel_fn = (df.true_class.eq("mel") & df.error.eq(1)).to_numpy()
    for m in methods:
        order = np.lexsort((ids, df[m].to_numpy(float)))
        result["methods"][m]["at_80pct_coverage"]["mel_false_negatives_captured"] = int(mel_fn[order[math.ceil(.8 * len(df)):]].sum())
        for coverage in (1.0, .9, .8, .7, .6, .5):
            kk = math.ceil(coverage * len(df))
            result["methods"][m]["risk_coverage"][str(coverage)]["mel_false_negatives_captured"] = int(mel_fn[order[kk:]].sum())
    delta_keys = ["auroc", "average_precision", "aurc", "error_capture_fraction", "mel_fn_capture_fraction"]
    deltas = {key: [] for key in delta_keys}
    for idx in cluster_samples(df, reps=1000, seed=142):
        b = df.iloc[idx]
        yy = b.error.to_numpy(int)
        if len(np.unique(yy)) < 2:
            continue
        iid = b.image_id.to_numpy(str)
        e = ranking(yy, b.entropy_direct.to_numpy(float), iid)
        c = ranking(yy, b.entropy_plus_quality_oof.to_numpy(float), iid)
        for key in ("auroc", "average_precision", "aurc"):
            deltas[key].append(c[key] - e[key])
        deltas["error_capture_fraction"].append(c["at_80pct_coverage"]["error_capture_fraction"] - e["at_80pct_coverage"]["error_capture_fraction"])
        mf = (b.true_class.eq("mel") & b.error.eq(1)).to_numpy()
        if mf.sum():
            n = len(b); k = math.ceil(.8*n)
            oe = np.lexsort((iid, b.entropy_direct.to_numpy(float)))
            oc = np.lexsort((iid, b.entropy_plus_quality_oof.to_numpy(float)))
            deltas["mel_fn_capture_fraction"].append(float((mf[oc[k:]].sum() - mf[oe[k:]].sum()) / mf.sum()))
    points = {key: (result["methods"]["entropy_plus_quality_oof"][key] - result["methods"]["entropy_direct"][key]) for key in ("auroc", "average_precision", "aurc")}
    a = result["methods"]["entropy_plus_quality_oof"]["at_80pct_coverage"]
    b = result["methods"]["entropy_direct"]["at_80pct_coverage"]
    points["error_capture_fraction"] = a["error_capture_fraction"] - b["error_capture_fraction"]
    points["mel_fn_capture_fraction"] = (a["mel_false_negatives_captured"] - b["mel_false_negatives_captured"]) / 37
    result["combined_minus_entropy"] = {k: {"point": points[k], "cluster_bootstrap_95": percentile(v)} for k,v in deltas.items()}
    result["method"] = "Existing fixed 5-fold lesion-grouped OOF scores; paired lesion-cluster bootstrap of fixed scores, 1000 draws, seed 142; no re-selection or refitting"
    result["risk_convention"] = "Retain lowest error score; image ID breaks ties; ceil(0.8*n); AURC = mean prefix error rate k=1..n"
    result["limitation"] = "Same development validation set; intervals condition on fitted OOF scores and do not include model refit uncertainty; no independent performance claim"
    return result


def make_figures(rows, comparison):
    all_rows = rows.loc[rows.true_class == "ALL"]
    names = [x.removeprefix("raw_").replace("_", " ") for x in all_rows.feature]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(all_rows.adjusted_error_OR_per_SD, y,
                xerr=[all_rows.adjusted_error_OR_per_SD - all_rows.adjusted_OR_ci_low,
                      all_rows.adjusted_OR_ci_high - all_rows.adjusted_error_OR_per_SD],
                fmt="o", capsize=3)
    ax.axvline(1, color="gray", linestyle="--"); ax.set_yticks(y, names)
    ax.set_xscale("log"); ax.set_xlabel("Class-adjusted error odds ratio per 1 SD (lesion bootstrap 95% CI)")
    ax.set_title("Exploratory raw-image proxy associations"); fig.tight_layout()
    fig.savefig(OUT / "class_adjusted_quality_or.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for m, label in [("entropy_direct", "Entropy"), ("entropy_plus_quality_oof", "Entropy + raw proxies"), ("quality_only_oof", "Raw proxies")]:
        curve = comparison["methods"][m]["risk_coverage"]
        xx = sorted(float(k) for k in curve)
        ax.plot(xx, [curve[str(k)]["retained_error_rate"] for k in xx], marker="o", label=label)
    ax.set(xlabel="Coverage", ylabel="Retained error rate", title="Validation OOF risk–coverage (exploratory)")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(OUT / "quality_entropy_risk_coverage.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for m, label in [("entropy_direct", "Entropy"), ("entropy_plus_quality_oof", "Entropy + raw proxies"), ("quality_only_oof", "Raw proxies")]:
        v = comparison["methods"][m]["at_80pct_coverage"]
        ax.scatter(v["error_capture_fraction"], v["mel_false_negatives_captured"] / 37, label=label, s=75)
    ax.set(xlabel="Fraction of all errors referred at 80% coverage", ylabel="Fraction of MEL false negatives referred", xlim=(0,1), ylim=(0,1))
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(OUT / "quality_entropy_80pct_comparison.png", dpi=300); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    hashes = check_sources()
    df = load_data()
    rows, adjusted = quality_associations(df)
    mel_rows, mel_details = melanoma_analysis(df)
    comp = ranking_comparison(df)
    rows.to_csv(OUT / "class_adjusted_quality_analysis.csv", index=False)
    mel_rows.to_csv(OUT / "melanoma_quality_analysis.csv", index=False)
    summary = {"scope": "Stage 23 frozen validation only", "input_sha256": hashes,
               "images": len(df), "lesions": df.lesion_id.nunique(), "errors": int(df.error.sum()),
               "class_counts": df.true_class.value_counts().to_dict(),
               "class_errors": df.groupby("true_class").error.sum().astype(int).to_dict(),
               "features": adjusted, "melanoma": mel_details,
               "method": "Raw proxy, rank-biserial=2*AUROC(feature,error)-1; lesion-cluster percentile bootstrap 400 draws; separate one-feature L2 logistic models adjusted for true class (C=1); per full-cohort SD; exploratory, no causal interpretation",
               "seed": SEED, "bootstrap_repetitions": REPS}
    (OUT / "class_adjusted_results.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (OUT / "quality_entropy_comparison.json").write_text(json.dumps(comp, indent=2, allow_nan=False)+"\n")
    make_figures(rows, comp)
    print(json.dumps({"images": len(df), "lesions": df.lesion_id.nunique(), "errors": int(df.error.sum()),
                      "mel": mel_details, "delta": comp["combined_minus_entropy"],
                      "adjusted_OR": adjusted}, indent=2))


if __name__ == "__main__":
    main()
