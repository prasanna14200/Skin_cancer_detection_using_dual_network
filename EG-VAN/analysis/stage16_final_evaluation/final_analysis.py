"""Read-only final-artifact audit and figures from saved Stage 16 CSV/JSON only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CHECKPOINT = ROOT / ("experiments/stage15_single_candidate_exp1/"
                     "stage15_single_candidate_exp1/recovery_epoch16/best_checkpoint.pt")
CHECKPOINT_SHA = "85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5"
SPLIT_SHA = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
PH2_SHA = "3c130b3b44dcc1052ae29656510a95a4e91df4b8f65c121838941efb375d3842"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_csv(name: str) -> list[dict]:
    with (HERE / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_json(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def close(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12)


def check_manifest(prefix: str, expected_names: tuple[str, ...]) -> dict:
    manifest_name = "final_ham_test_manifest.json" if prefix == "ham" else "final_ph2_manifest.json"
    manifest = load_json(manifest_name)
    if manifest["status"] != "COMPLETE" or manifest["checkpoint_sha256"] != CHECKPOINT_SHA or manifest["checkpoint_epoch"] != 16:
        raise ValueError(f"{prefix}: checkpoint/manifest mismatch")
    if set(manifest["artifact_sha256"]) != set(expected_names):
        raise ValueError(f"{prefix}: artifact list mismatch")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256(HERE / name) != digest:
            raise ValueError(f"{prefix}: artifact hash mismatch: {name}")
    for relative, digest in manifest["source_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"{prefix}: source provenance hash mismatch: {relative}")
    if manifest["evaluation_script_sha256"] != manifest["source_sha256"][
        f"experiments/stage16_final_evaluation/evaluate_{prefix}.py"]:
        raise ValueError(f"{prefix}: evaluator source hash mismatch")
    if manifest["training_performed"] is not False:
        raise ValueError(f"{prefix}: manifest indicates training")
    return manifest


def check_ham() -> tuple[dict, list[dict], list[list[int]], dict]:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    manifest = check_manifest("ham", (
        "final_ham_test_predictions.csv", "final_ham_test_metrics.json",
        "final_ham_test_confusion_matrix.csv", "final_ham_test_classwise_metrics.csv"))
    if manifest["split_sha256"] != SPLIT_SHA or manifest["ph2_accessed"] is not False:
        raise ValueError("HAM split/scope mismatch")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Frozen checkpoint hash mismatch")
    split_path = ROOT / "data/splits/split_leakage_aware.csv"
    if sha256(split_path) != SPLIT_SHA:
        raise ValueError("HAM split hash mismatch")
    with split_path.open(newline="", encoding="utf-8") as f:
        test = {r["image_id"]: r["dx"] for r in csv.DictReader(f) if r["split"] == "test"}
    rows = load_csv("final_ham_test_predictions.csv")
    saved = load_json("final_ham_test_metrics.json")
    if len(rows) != len(test) != 1014 or {r["image_id"] for r in rows} != set(test):
        raise ValueError("HAM prediction IDs/count mismatch")
    matrix = [[0] * 7 for _ in CLASSES]
    probabilities = []
    truth = []
    for r in rows:
        if r["true_class"] != test[r["image_id"]] or r["predicted_class"] not in CLASSES:
            raise ValueError("HAM truth/predicted label mismatch")
        p = [float(r[f"p_{name}"]) for name in CLASSES]
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in p) or abs(sum(p) - 1) > 1e-4:
            raise ValueError("HAM probability invalid")
        if CLASSES[max(range(7), key=lambda i: p[i])] != r["predicted_class"]:
            raise ValueError("HAM argmax mismatch")
        if (r["correct"] == "True") != (r["true_class"] == r["predicted_class"]):
            raise ValueError("HAM correctness flag mismatch")
        i, j = CLASSES.index(r["true_class"]), CLASSES.index(r["predicted_class"])
        matrix[i][j] += 1
        truth.append(i)
        probabilities.append(p)
    if matrix != saved["confusion_matrix"]:
        raise ValueError("HAM confusion matrix mismatch")
    per = {}
    for i, name in enumerate(CLASSES):
        tp, support = matrix[i][i], sum(matrix[i])
        fp = sum(matrix[j][i] for j in range(7) if j != i)
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        expected = saved["per_class"][name]
        for key, value in {"support": support, "tp": tp, "fp": fp, "fn": fn,
                           "precision": precision, "recall": recall, "f1": f1}.items():
            if not close(expected[key], value):
                raise ValueError(f"HAM {name} {key} mismatch")
        per[name] = {"support": support, "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1}
    for key, value in {
        "accuracy": sum(matrix[i][i] for i in range(7)) / 1014,
        "balanced_accuracy": sum(x["recall"] for x in per.values()) / 7,
        "macro_precision": sum(x["precision"] for x in per.values()) / 7,
        "macro_recall": sum(x["recall"] for x in per.values()) / 7,
        "macro_f1": sum(x["f1"] for x in per.values()) / 7,
        "weighted_f1": sum(x["f1"] * x["support"] for x in per.values()) / 1014,
    }.items():
        if not close(saved[key], value):
            raise ValueError(f"HAM {key} mismatch")
    auc_macro = roc_auc_score(np.asarray(truth), np.asarray(probabilities), multi_class="ovr", average="macro")
    auc_weighted = roc_auc_score(np.asarray(truth), np.asarray(probabilities), multi_class="ovr", average="weighted")
    if not close(saved["multiclass_roc_auc"]["ovr_macro"], auc_macro) or not close(
        saved["multiclass_roc_auc"]["ovr_weighted"], auc_weighted):
        raise ValueError("HAM AUC mismatch")
    for name, row in zip(CLASSES, load_csv("final_ham_test_confusion_matrix.csv")):
        if row["true_class"] != name or [int(row[c]) for c in CLASSES] != matrix[CLASSES.index(name)]:
            raise ValueError("HAM confusion CSV mismatch")
    for name, row in zip(CLASSES, load_csv("final_ham_test_classwise_metrics.csv")):
        if row["class"] != name or any(not close(row[k], per[name][k]) for k in
                                        ("support", "tp", "fp", "fn", "precision", "recall", "f1")):
            raise ValueError("HAM classwise CSV mismatch")
    if dict(Counter(r["true_class"] for r in rows)) != {c: per[c]["support"] for c in CLASSES}:
        raise ValueError("HAM class distribution mismatch")
    return saved, rows, matrix, manifest


def check_ph2() -> tuple[dict, list[dict], list[list[int]], dict]:
    from sklearn.metrics import roc_auc_score
    manifest = check_manifest("ph2", (
        "final_ph2_predictions.csv", "final_ph2_metrics.json", "final_ph2_confusion_matrix.csv"))
    if manifest["ph2_manifest_sha256"] != PH2_SHA or manifest["mapping"] != {
        "common nevus": "nv", "melanoma": "mel", "atypical nevus": None}:
        raise ValueError("PH2 cohort/mapping mismatch")
    cohort_path = ROOT / "analysis/ph2_external_validation_exp5/ph2_manifest.csv"
    if sha256(cohort_path) != PH2_SHA:
        raise ValueError("PH2 manifest hash mismatch")
    with cohort_path.open(newline="", encoding="utf-8-sig") as f:
        cohort = {r["image_id"]: r for r in csv.DictReader(f) if r["included"] == "True"}
    rows = load_csv("final_ph2_predictions.csv")
    saved = load_json("final_ph2_metrics.json")
    if len(rows) != len(cohort) != 120 or {r["image_id"] for r in rows} != set(cohort):
        raise ValueError("PH2 ID/count mismatch")
    matrix = [[0, 0, 0], [0, 0, 0]]
    for r in rows:
        src = cohort[r["image_id"]]
        if (r["true_external_label"] != src["true_external_label"] or
            r["true_ham_label"] != src["true_ham_label"] or r["predicted_class"] not in CLASSES):
            raise ValueError("PH2 mapped truth mismatch")
        p = [float(r[f"p_{name}"]) for name in CLASSES]
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in p) or abs(sum(p) - 1) > 1e-4:
            raise ValueError("PH2 probability invalid")
        if CLASSES[max(range(7), key=lambda i: p[i])] != r["predicted_class"]:
            raise ValueError("PH2 argmax mismatch")
        i = 0 if r["true_ham_label"] == "nv" else 1
        j = 0 if r["predicted_class"] == "nv" else 1 if r["predicted_class"] == "mel" else 2
        matrix[i][j] += 1
    if matrix != saved["confusion_matrix"] or list(map(sum, matrix)) != [80, 40]:
        raise ValueError("PH2 confusion mismatch")
    per = {}
    for i, name in enumerate(("nv", "mel")):
        tp, support = matrix[i][i], sum(matrix[i])
        fp, fn = matrix[1-i][i], support - tp
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per[name] = {"support": support, "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1}
        if any(not close(saved["per_class"][name][k], v) for k, v in per[name].items()):
            raise ValueError(f"PH2 {name} metric mismatch")
    for key, value in {"accuracy": (matrix[0][0] + matrix[1][1]) / 120,
                       "balanced_accuracy": (per["nv"]["recall"] + per["mel"]["recall"]) / 2,
                       "mel_precision": per["mel"]["precision"], "mel_recall": per["mel"]["recall"],
                       "mel_f1": per["mel"]["f1"], "nv_recall": per["nv"]["recall"]}.items():
        if not close(saved[key], value):
            raise ValueError(f"PH2 {key} mismatch")
    auc = roc_auc_score([int(r["true_ham_label"] == "mel") for r in rows],
                        [float(r["p_mel"]) for r in rows])
    if not close(saved["melanoma_probability_roc_auc"]["value"], auc):
        raise ValueError("PH2 MEL AUC mismatch")
    for i, row in enumerate(load_csv("final_ph2_confusion_matrix.csv")):
        if row["true_class"] != ("nv", "mel")[i] or [int(row[c]) for c in ("nv", "mel", "other")] != matrix[i]:
            raise ValueError("PH2 confusion CSV mismatch")
    return saved, rows, matrix, manifest


def make_figures(ham: dict, ham_rows: list[dict], ham_matrix: list[list[int]],
                 ph2: dict, ph2_matrix: list[list[int]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    created = []
    def save(name):
        plt.savefig(folder / name, dpi=350, bbox_inches="tight")
        plt.close()
        created.append(name)
    def confusion(matrix, rows, cols, title, name, normalized=False):
        data = np.asarray(matrix, dtype=float)
        if normalized:
            data = data / data.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(8.4, 6.7))
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        ax.set_xticks(range(len(cols)), [c.upper() for c in cols], rotation=35, ha="right")
        ax.set_yticks(range(len(rows)), [c.upper() for c in rows])
        ax.set(xlabel="Predicted class", ylabel="True class", title=title)
        for i in range(len(rows)):
            for j in range(len(cols)):
                value = f"{data[i,j]:.2f}" if normalized else str(int(data[i,j]))
                ax.text(j, i, value, ha="center", va="center",
                        color="white" if data[i,j] > data.max() * 0.55 else "black", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        save(name)
    confusion(ham_matrix, CLASSES, CLASSES, "HAM10000 test confusion matrix",
              "ham_confusion_matrix.png")
    confusion(ham_matrix, CLASSES, CLASSES, "HAM10000 test confusion matrix (row normalized)",
              "ham_confusion_matrix_normalized.png", True)
    for key, title, name in (("f1", "HAM10000 test F1 by class", "ham_classwise_f1.png"),
                             ("recall", "HAM10000 test recall by class", "ham_classwise_recall.png")):
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        values = [ham["per_class"][c][key] for c in CLASSES]
        bars = ax.bar([c.upper() for c in CLASSES], values, color="#2c638a")
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3)
        ax.set(ylim=(0, 1.08), ylabel=key.upper(), title=title)
        ax.grid(axis="y", alpha=0.2)
        save(name)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for c in CLASSES:
        y = np.asarray([int(r["true_class"] == c) for r in ham_rows])
        score = np.asarray([float(r[f"p_{c}"]) for r in ham_rows])
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, lw=1.8, label=f"{c.upper()} (AUC {auc(fpr,tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title=f"HAM10000 test OVR ROC (macro AUC {ham['multiclass_roc_auc']['ovr_macro']:.3f})",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)
    save("ham_roc_curves.png")
    confusion(ph2_matrix, ("nv", "mel"), ("nv", "mel", "other"),
              "PH² external follow-up confusion matrix", "ph2_confusion_matrix.png")
    confusion(ph2_matrix, ("nv", "mel"), ("nv", "mel", "other"),
              "PH² external follow-up confusion matrix (row normalized)",
              "ph2_confusion_matrix_normalized.png", True)
    validation = json.loads((ROOT / ("experiments/stage15_single_candidate_exp1/"
        "stage15_single_candidate_exp1/recovery_epoch16/validation_epochs/epoch_016_metrics.json")
        ).read_text(encoding="utf-8"))
    labels = ["Stage 15\nvalidation", "HAM10000\ntest", "PH²\nfollow-up"]
    values = [validation["mel_recall"], ham["mel_recall"], ph2["mel_recall"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, values, color=("#648eaa", "#32779e", "#ab704a"))
    ax.bar_label(bars, labels=[f"{v:.2%}" for v in values], padding=4)
    ax.set(ylim=(0, 0.72), ylabel="Melanoma recall", title="Melanoma recall across distinct evaluation datasets")
    ax.text(0.5, -0.19, "Descriptive comparison only; cohorts and label spaces differ.",
            transform=ax.transAxes, ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    save("internal_external_melanoma_comparison.png")
    return created


def main() -> None:
    ham, ham_rows, ham_matrix, _ = check_ham()
    ph2, ph2_rows, ph2_matrix, _ = check_ph2()
    errors = {f"{a}->{b}": sum(r["true_class"] == a and r["predicted_class"] == b for r in ham_rows)
              for a, b in (("mel", "nv"), ("mel", "bkl"), ("bkl", "mel"),
                           ("nv", "mel"), ("akiec", "bkl"), ("bcc", "nv"))}
    ph2_mel = Counter(r["predicted_class"] if r["predicted_class"] in ("nv", "mel") else "other"
                      for r in ph2_rows if r["true_ham_label"] == "mel")
    print(json.dumps({"audit": "PASS", "ham_samples": len(ham_rows), "ph2_samples": len(ph2_rows),
                      "ham_errors": errors, "ph2_mel": dict(ph2_mel),
                      "figures": make_figures(ham, ham_rows, ham_matrix, ph2, ph2_matrix)}, indent=2))


if __name__ == "__main__":
    main()
