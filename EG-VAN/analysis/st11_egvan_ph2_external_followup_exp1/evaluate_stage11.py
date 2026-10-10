"""Frozen EG-VAN PH2 external follow-up; --preflight performs no inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_V2_S_Weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
EGVAN_SHA = "60f34ea4cfa6cbf3dce69e8d8bcc82292d8d8813af30f33b2f666a4f06340de0"
SPLIT_SHA = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
EXP5_SHA = "81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a"
OUTPUTS = ("ph2_predictions.csv", "ph2_metrics.json", "per_class_metrics.csv", "confusion_matrix.csv",
           "confusion_matrix.png", "normalized_confusion_matrix.png", "egvan_vs_exp5_ph2_comparison.csv",
           "internal_external_shift_comparison.csv", "stage11_external_followup_report.md")
PREDICTION_FIELDS = ("image_id", "true_external_label", "true_ham_label", "predicted_class",
                     *(f"p_{name}" for name in CLASSES))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def imports(root: Path):
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from models.egvan import EGVAN
    from preprocessing import preprocess_image, HAIR_KERNEL_SIZE, HAIR_THRESHOLD, INPAINT_RADIUS, RETINEX_SIGMAS
    from train import make_transforms
    from external_eval import read_clinical_labels
    return EGVAN, preprocess_image, make_transforms, read_clinical_labels, (
        HAIR_KERNEL_SIZE, HAIR_THRESHOLD, INPAINT_RADIUS, tuple(RETINEX_SIGMAS))


def preflight(root: Path) -> dict:
    root = root.resolve()
    egvan = root / "experiments/egvan_reconstruction_controlled_exp1/best_checkpoint.pt"
    split = root / "data/splits/split_leakage_aware.csv"
    baseline = root / "experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt"
    s10dir = root / "analysis/egvan_internal_test_exp1"
    s10metrics = s10dir / "test_metrics.json"
    s10manifest = json.loads((s10dir / "stage10_manifest.json").read_text(encoding="utf-8"))
    exp5dir = root / "analysis/ph2_external_validation_exp5"
    manifest_path = exp5dir / "ph2_manifest.csv"
    baseline_predictions_path = exp5dir / "ph2_predictions.csv"
    baseline_metrics_path = exp5dir / "ph2_metrics.json"
    baseline_config_path = exp5dir / "external_validation_config.json"
    hashes = {"egvan_best_checkpoint": sha256(egvan), "frozen_split": sha256(split),
              "experiment5_checkpoint": sha256(baseline), "ph2_cohort_manifest": sha256(manifest_path),
              "experiment5_ph2_predictions": sha256(baseline_predictions_path),
              "stage10_test_metrics": sha256(s10metrics)}
    if (hashes["egvan_best_checkpoint"] != EGVAN_SHA or hashes["frozen_split"] != SPLIT_SHA
            or hashes["experiment5_checkpoint"] != EXP5_SHA):
        raise ValueError("Protected checkpoint/split SHA256 mismatch")
    if (s10manifest.get("status") != "COMPLETE" or s10manifest.get("selected_epoch") != 15
            or s10manifest.get("egvan_checkpoint_sha256") != EGVAN_SHA
            or s10manifest.get("frozen_split_sha256") != SPLIT_SHA
            or s10manifest.get("prediction_integrity", {}).get("status") != "PASS"):
        raise ValueError("Stage 10 is not complete for epoch 15")
    if s10manifest["generated_artifact_hashes"]["test_metrics.json"] != hashes["stage10_test_metrics"]:
        raise ValueError("Stage 10 saved metric hash differs from its manifest")
    s10 = json.loads(s10metrics.read_text(encoding="utf-8"))
    if (s10["sample_count"] != 1014 or tuple(s10["class_order"]) != CLASSES
            or not math.isclose(s10["accuracy"], 0.8244575936883629, abs_tol=1e-12)
            or not math.isclose(s10["balanced_accuracy"], 0.6815754537927868, abs_tol=1e-12)):
        raise ValueError("Stage 10 HAM metric scope mismatch")
    state = torch.load(egvan, map_location="cpu", weights_only=False, mmap=True)
    if state["epoch"] != 15 or state["best_epoch"] != 15 or tuple(state["class_order"]) != CLASSES:
        raise ValueError("EG-VAN best checkpoint epoch/class order mismatch")
    del state
    base_cfg = json.loads(baseline_config_path.read_text(encoding="utf-8"))
    if (base_cfg["checkpoint_sha256"] != EXP5_SHA or base_cfg["split_sha256"] != SPLIT_SHA
            or tuple(base_cfg["class_order"]) != CLASSES or
            base_cfg["cohort"]["mapping"] != {"common nevus": "nv", "melanoma": "mel", "atypical nevus": None} or
            base_cfg["preprocessing"].get("status") != "PASS" or
            not base_cfg["preprocessing"].get("matches_diagnostic_config")):
        raise ValueError("Experiment #5 PH2 configuration provenance mismatch")
    operations = base_cfg["preprocessing"]["operation_contract"]["operations"]
    if (base_cfg["preprocessing"]["implementation"] != "src/preprocessing.py:preprocess_image"
            or "in-memory JPEG encode/decode with OpenCV default JPEG settings to match saved HAM JPG representation" not in operations
            or base_cfg["preprocessing"]["operation_contract"]["augmentation"] != "none"
            or base_cfg["preprocessing"]["operation_contract"]["masks_or_roi"] != "not used"):
        raise ValueError("Experiment #5 PH2 preprocessing contract mismatch")
    EGVAN, preprocess_image, make_transforms, read_clinical_labels, constants = imports(root)
    if constants != (17, 10, 1.0, (15.0, 80.0, 250.0)):
        raise ValueError("Frozen Phase 4B preprocessing constants drifted")
    _, transform = make_transforms(EfficientNet_V2_S_Weights.DEFAULT)
    names = [type(x).__name__ for x in transform.transforms]
    if names != ["Resize", "ToTensor", "Normalize"] or tuple(transform.transforms[0].size) != (384, 384):
        raise ValueError("HAM evaluation transform drifted")
    preprocess_weights = EfficientNet_V2_S_Weights.DEFAULT.transforms()
    if (list(transform.transforms[2].mean) != base_cfg["preprocessing"]["normalization_mean"]
            or list(transform.transforms[2].std) != base_cfg["preprocessing"]["normalization_std"]
            or tuple(transform.transforms[2].mean) != tuple(preprocess_weights.mean)
            or tuple(transform.transforms[2].std) != tuple(preprocess_weights.std)):
        raise ValueError("ImageNet normalization drifted")
    rows = read_csv(manifest_path)
    if len(rows) != 200 or len({x["image_id"] for x in rows}) != 200:
        raise ValueError("Experiment #5 PH2 manifest must contain 200 unique cases")
    clinical = read_clinical_labels(root / "data/external/ph2/metadata/PH2_dataset.txt")
    if len(clinical) != 200:
        raise ValueError("PH2 clinical metadata count mismatch")
    mapped = {"common nevus": ("0", "nv"), "melanoma": ("2", "mel"),
              "atypical nevus": ("1", "")}
    included, excluded = [], []
    for row in rows:
        image_id, label = row["image_id"], row["true_external_label"]
        if label not in mapped or clinical.get(image_id) != label:
            raise ValueError(f"PH2 manifest/clinical label mismatch: {image_id}")
        code, ham = mapped[label]
        expected_path = f"data/external/ph2/images/{image_id}.bmp"
        if (row["clinical_code"] != code or row["true_ham_label"] != ham
                or row["relative_image_path"].replace("\\", "/") != expected_path):
            raise ValueError(f"PH2 mapping/path mismatch: {image_id}")
        actual_included = row["included"].strip().lower() == "true"
        if actual_included != bool(ham):
            raise ValueError(f"PH2 inclusion mismatch: {image_id}")
        if actual_included:
            if not (root / expected_path).is_file():
                raise FileNotFoundError(f"Missing PH2 BMP: {expected_path}")
            included.append(row)
        else:
            excluded.append(row)
    if (len(included) != 120 or len(excluded) != 80 or
            Counter(x["true_ham_label"] for x in included) != {"nv": 80, "mel": 40} or
            any(x["true_external_label"] != "atypical nevus" for x in excluded)):
        raise ValueError("PH2 mapped cohort counts mismatch")
    baseline_rows = read_csv(baseline_predictions_path)
    baseline_by_id = {x["image_id"]: x for x in baseline_rows}
    if len(baseline_rows) != 120 or len(baseline_by_id) != 120 or set(baseline_by_id) != {x["image_id"] for x in included}:
        raise ValueError("Experiment #5 PH2 prediction IDs do not match fixed cohort")
    for row in included:
        old = baseline_by_id[row["image_id"]]
        if old["true_external_label"] != row["true_external_label"] or old["true_ham_label"] != row["true_ham_label"]:
            raise ValueError(f"Experiment #5 PH2 prediction truth mismatch: {row['image_id']}")
    base_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    if base_metrics["total_samples"] != 120 or base_metrics["confusion_matrix"]["counts"] != confusion(baseline_rows):
        raise ValueError("Experiment #5 PH2 saved metrics and predictions differ")
    return {"status": "PASS", "checkpoint_epoch": 15, "hashes": hashes,
            "class_order": list(CLASSES), "ph2_total": 200, "included": 120,
            "included_class_counts": {"nv": 80, "mel": 40}, "excluded_atypical": 80,
            "baseline_id_match": "120/120", "missing_source_images": 0,
            "preprocessing": "original BMP RGB -> BGR -> src/preprocessing.py:preprocess_image -> in-memory OpenCV JPEG encode/decode -> RGB -> Resize((384,384)) -> ToTensor -> ImageNet mean/std; no augmentation/TTA/masks/ROI",
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}


def confusion(rows: list[dict]) -> list[list[int]]:
    result = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        actual = row["true_ham_label"]
        predicted = row["predicted_class"]
        if actual not in ("nv", "mel") or predicted not in CLASSES:
            raise ValueError("Unknown PH2 true/predicted class")
        result[0 if actual == "nv" else 1][0 if predicted == "nv" else 1 if predicted == "mel" else 2] += 1
    return result


def metrics_from_rows(rows: list[dict]) -> dict:
    m = confusion(rows)
    n = sum(map(sum, m))
    if n != 120 or sum(m[0]) != 80 or sum(m[1]) != 40:
        raise ValueError("PH2 mapped cohort counts differ")
    per = {}
    for i, label in enumerate(("nv", "mel")):
        tp = m[i][i]
        predicted = m[0][i] + m[1][i]
        support = sum(m[i])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {"total_samples": 120, "accuracy": (m[0][0] + m[1][1]) / n,
            "balanced_accuracy": (per["nv"]["recall"] + per["mel"]["recall"]) / 2,
            "class_metrics": per, "confusion_matrix": {"true_rows": ["nv", "mel"],
                "predicted_columns": ["nv", "mel", "other"], "counts": m,
                "other_policy": "Other HAM predictions remain separate and count as incorrect."},
            "prediction_distribution": dict(Counter(x["predicted_class"] for x in rows))}


def verify_rows(rows: list[dict], cohort: list[dict], saved_metrics: dict) -> dict:
    included = {x["image_id"]: x for x in cohort if x["included"].strip().lower() == "true"}
    ids = [x["image_id"] for x in rows]
    if len(rows) != 120 or len(set(ids)) != 120 or set(ids) != set(included):
        raise ValueError("PH2 prediction IDs differ from fixed mapped cohort")
    for row in rows:
        source = included[row["image_id"]]
        if (row["true_external_label"] != source["true_external_label"] or
                row["true_ham_label"] != source["true_ham_label"] or
                row["predicted_class"] not in CLASSES):
            raise ValueError("PH2 saved label mismatch")
        probs = [float(row[f"p_{name}"]) for name in CLASSES]
        if any(not math.isfinite(p) or p < 0 or p > 1 for p in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-4):
            raise ValueError("PH2 probability invalid")
        if CLASSES[max(range(7), key=lambda i: probs[i])] != row["predicted_class"]:
            raise ValueError("PH2 predicted class not probability argmax")
    if metrics_from_rows(rows) != saved_metrics:
        raise ValueError("Saved PH2 metrics do not reproduce from predictions")
    return {"status": "PASS", "rows": 120, "unique_ids": 120, "cohort_match": "120/120",
            "seven_probabilities_valid": True, "metrics_recomputed_equal_saved": True}


class PH2Dataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], transform, preprocess_image):
        self.root, self.rows, self.transform, self.preprocess_image = root, rows, transform, preprocess_image

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / "data/external/ph2/images" / f"{row['image_id']}.bmp"
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            processed_bgr = self.preprocess_image(bgr)
            ok, encoded = cv2.imencode(".jpg", processed_bgr)
            if not ok:
                raise RuntimeError(f"JPEG encode failed: {row['image_id']}")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError(f"JPEG decode failed: {row['image_id']}")
            processed_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(processed_rgb, mode="RGB")
            return self.transform(image)


def write_csv_new(path: Path, rows: list[dict], fields: list[str]):
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path: Path, value: dict):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def plot_matrix(path: Path, matrix: list[list[float]], title: str, normalized: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(3), ["nv", "mel", "other"])
    ax.set_yticks(range(2), ["nv", "mel"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True mapped label")
    ax.set_title(title)
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{matrix[i][j]:.2f}" if normalized else str(matrix[i][j]),
                    ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def comparison_rows(eg: dict, baseline: dict) -> list[dict]:
    values = [("accuracy", baseline["accuracy"], eg["accuracy"]),
              ("balanced_accuracy", baseline["balanced_accuracy"], eg["balanced_accuracy"])]
    for label in ("nv", "mel"):
        for key in ("precision", "recall", "f1"):
            values.append((f"{label}_{key}", baseline["class_metrics"][label][key], eg["class_metrics"][label][key]))
    return [{"metric": k, "experiment5": a, "egvan": b, "egvan_minus_experiment5": b-a}
            for k, a, b in values]


def paired_correctness(rows: list[dict], baseline_rows: list[dict]) -> dict:
    old = {x["image_id"]: x for x in baseline_rows}
    new = {x["image_id"]: x for x in rows}
    if len(old) != 120 or len(new) != 120 or set(old) != set(new):
        raise ValueError("Paired PH2 join is not 120/120")
    result = {"both_correct": 0, "egvan_only_correct": 0, "experiment5_only_correct": 0, "both_wrong": 0}
    for image_id, row in new.items():
        previous = old[image_id]
        if previous["true_ham_label"] != row["true_ham_label"]:
            raise ValueError("Paired PH2 truth differs")
        eg_good = row["predicted_class"] == row["true_ham_label"]
        base_good = previous["predicted_class"] == previous["true_ham_label"]
        result["both_correct" if eg_good and base_good else "egvan_only_correct" if eg_good else
               "experiment5_only_correct" if base_good else "both_wrong"] += 1
    if sum(result.values()) != 120:
        raise ValueError("Paired PH2 counts do not sum to 120")
    return result


def shift_rows(root: Path, eg: dict, baseline: dict) -> list[dict]:
    ham_base = json.loads((root / "experiments/efficientnetv2s_controlled_exp5/test_metrics.json").read_text(encoding="utf-8"))
    ham_eg = json.loads((root / "analysis/egvan_internal_test_exp1/test_metrics.json").read_text(encoding="utf-8"))
    result = []
    for model, ham, ph2 in (("EfficientNetV2S Experiment #5", ham_base, baseline),
                            ("Reconstructed EG-VAN", ham_eg, eg)):
        if ham["sample_count"] != 1014 or tuple(ham["class_order"]) != CLASSES:
            raise ValueError("Saved HAM internal metrics not on frozen test")
        hm, pm = ham["per_class"]["mel"]["recall"], ph2["class_metrics"]["mel"]["recall"]
        hn, pn = ham["per_class"]["nv"]["recall"], ph2["class_metrics"]["nv"]["recall"]
        result.append({"model": model, "ham_accuracy": ham["accuracy"], "ph2_accuracy": ph2["accuracy"],
                       "accuracy_difference": ph2["accuracy"] - ham["accuracy"],
                       "ham_melanoma_recall": hm, "ph2_melanoma_recall": pm,
                       "melanoma_recall_difference": pm - hm, "ham_nevus_recall": hn,
                       "ph2_nevus_recall": pn, "nevus_recall_difference": pn - hn})
    return result


def run(root: Path, batch_size: int, workers: int):
    report = preflight(root)
    print(json.dumps(report, indent=2), flush=True)
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Stage 11 PH2 inference requires CUDA Tesla T4; no CPU substitute")
    if batch_size < 1 or workers < 0:
        raise ValueError("Invalid batch size/workers")
    output = root / "analysis/egvan_ph2_external_followup_exp1"
    if any((output / name).exists() for name in OUTPUTS):
        raise FileExistsError("Stage 11 output exists; refusing repeat inference/overwrite")
    cohort = read_csv(root / "analysis/ph2_external_validation_exp5/ph2_manifest.csv")
    included = [x for x in cohort if x["included"].strip().lower() == "true"]
    EGVAN, preprocess_image, make_transforms, _, _ = imports(root)
    _, transform = make_transforms(EfficientNet_V2_S_Weights.DEFAULT)
    ds = PH2Dataset(root, included, transform, preprocess_image)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    ckpt = torch.load(root / "experiments/egvan_reconstruction_controlled_exp1/best_checkpoint.pt",
                      map_location="cpu", weights_only=False, mmap=True)
    model = EGVAN(num_classes=7, pretrained_efficient=False, pretrained_resnet=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    del ckpt
    model.cuda().eval()
    rows = []
    with torch.inference_mode():
        for images in loader:
            with torch.autocast("cuda"):
                logits = model(images.cuda(non_blocking=True))
            probabilities = torch.softmax(logits.float(), dim=1).cpu().tolist()
            for probs in probabilities:
                source = included[len(rows)]
                row = {"image_id": source["image_id"], "true_external_label": source["true_external_label"],
                       "true_ham_label": source["true_ham_label"],
                       "predicted_class": CLASSES[max(range(7), key=lambda i: probs[i])]}
                row.update({f"p_{name}": probs[i] for i, name in enumerate(CLASSES)})
                rows.append(row)
    metrics = metrics_from_rows(rows)
    integrity = verify_rows(rows, cohort, metrics)
    baseline_dir = root / "analysis/ph2_external_validation_exp5"
    baseline = json.loads((baseline_dir / "ph2_metrics.json").read_text(encoding="utf-8"))
    baseline_rows = read_csv(baseline_dir / "ph2_predictions.csv")
    paired = paired_correctness(rows, baseline_rows)
    try:
        from scipy.stats import binomtest
        b, c = paired["egvan_only_correct"], paired["experiment5_only_correct"]
        p_value = binomtest(b, b+c, 0.5).pvalue if b+c else 1.0
        mcnemar = {"performed": True, "b": b, "c": c, "exact_p_value": p_value}
    except ImportError:
        mcnemar = {"performed": False, "b": paired["egvan_only_correct"],
                   "c": paired["experiment5_only_correct"], "exact_p_value": None}
    comparison = comparison_rows(metrics, baseline)
    shifts = shift_rows(root, metrics, baseline)
    write_csv_new(output / "ph2_predictions.csv", rows, list(PREDICTION_FIELDS))
    write_json_new(output / "ph2_metrics.json", metrics)
    verify_rows(read_csv(output / "ph2_predictions.csv"), cohort,
                json.loads((output / "ph2_metrics.json").read_text(encoding="utf-8")))
    write_csv_new(output / "per_class_metrics.csv",
                  [{"class": name, **metrics["class_metrics"][name]} for name in ("nv", "mel")],
                  ["class", "support", "precision", "recall", "f1"])
    m = metrics["confusion_matrix"]["counts"]
    write_csv_new(output / "confusion_matrix.csv",
                  [{"true_label": name, "nv": m[i][0], "mel": m[i][1], "other": m[i][2]}
                   for i, name in enumerate(("nv", "mel"))], ["true_label", "nv", "mel", "other"])
    plot_matrix(output / "confusion_matrix.png", m, "EG-VAN PH2 mapped confusion matrix", False)
    normalized = [[x/sum(row) if sum(row) else 0 for x in row] for row in m]
    plot_matrix(output / "normalized_confusion_matrix.png", normalized,
                "EG-VAN PH2 mapped normalized confusion matrix", True)
    write_csv_new(output / "egvan_vs_exp5_ph2_comparison.csv", comparison,
                  ["metric", "experiment5", "egvan", "egvan_minus_experiment5"])
    write_csv_new(output / "internal_external_shift_comparison.csv", shifts, list(shifts[0]))
    report_text = ("# Stage 11 frozen EG-VAN PH2 external follow-up\n\n"
        "PH² is an external follow-up cohort, not untouched independent external validation: it was previously used "
        "in project diagnostics/evaluation. The frozen epoch-15 EG-VAN checkpoint and exact Experiment #5 120-image mapped cohort were used. "
        "No training, fine-tuning, threshold tuning, model selection, or Grad-CAM occurred.\n\n"
        f"EG-VAN PH² accuracy {metrics['accuracy']:.6f}; mapped balanced accuracy {metrics['balanced_accuracy']:.6f}. "
        f"MEL precision/recall/F1 {metrics['class_metrics']['mel']['precision']:.6f}/"
        f"{metrics['class_metrics']['mel']['recall']:.6f}/{metrics['class_metrics']['mel']['f1']:.6f}; "
        f"NV precision/recall/F1 {metrics['class_metrics']['nv']['precision']:.6f}/"
        f"{metrics['class_metrics']['nv']['recall']:.6f}/{metrics['class_metrics']['nv']['f1']:.6f}.\n\n"
        f"MEL outcomes mel/nv/other: {m[1][1]}/{m[1][0]}/{m[1][2]}; "
        f"NV outcomes nv/mel/other: {m[0][0]}/{m[0][1]}/{m[0][2]}. "
        f"Experiment #5 outcomes MEL mel/nv/other: {baseline['confusion_matrix']['counts'][1][1]}/"
        f"{baseline['confusion_matrix']['counts'][1][0]}/{baseline['confusion_matrix']['counts'][1][2]}; "
        f"NV nv/mel/other: {baseline['confusion_matrix']['counts'][0][0]}/"
        f"{baseline['confusion_matrix']['counts'][0][1]}/{baseline['confusion_matrix']['counts'][0][2]}.\n\n"
        f"Paired correctness: {paired}. McNemar exact test: {mcnemar}. "
        "A p-value, if present, is not a clinical superiority claim.\n\n"
        "Internal-to-external differences are descriptive cross-dataset changes, not causal domain-shift magnitudes. "
        "HAM is a seven-class population whereas the mapped PH² cohort contains only NV/MEL truth, so overall accuracies have different class composition.\n\n"
        f"Internal/external metrics and PH² minus HAM differences: {shifts}.\n")
    with (output / "stage11_external_followup_report.md").open("x", encoding="utf-8") as f:
        f.write(report_text)
    source = {"egvan_best_checkpoint": root / "experiments/egvan_reconstruction_controlled_exp1/best_checkpoint.pt",
              "experiment5_checkpoint": root / "experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt",
              "frozen_split": root / "data/splits/split_leakage_aware.csv",
              "ph2_cohort_manifest": baseline_dir / "ph2_manifest.csv",
              "baseline_ph2_predictions": baseline_dir / "ph2_predictions.csv",
              "baseline_ph2_metrics": baseline_dir / "ph2_metrics.json",
              "baseline_ph2_config": baseline_dir / "external_validation_config.json",
              "stage10_metrics": root / "analysis/egvan_internal_test_exp1/test_metrics.json",
              "stage10_manifest": root / "analysis/egvan_internal_test_exp1/stage10_manifest.json",
              "experiment5_ham_metrics": root / "experiments/efficientnetv2s_controlled_exp5/test_metrics.json",
              "preprocessing_source": root / "src/preprocessing.py", "evaluator_source": Path(__file__)}
    if (sha256(source["egvan_best_checkpoint"]) != EGVAN_SHA or
            sha256(source["frozen_split"]) != SPLIT_SHA or
            sha256(source["ph2_cohort_manifest"]) != report["hashes"]["ph2_cohort_manifest"] or
            sha256(source["baseline_ph2_predictions"]) != report["hashes"]["experiment5_ph2_predictions"] or
            sha256(source["stage10_metrics"]) != report["hashes"]["stage10_test_metrics"]):
        raise ValueError("Protected source hash changed during PH2 evaluation")
    manifest = {"status": "COMPLETE", "egvan_checkpoint_sha256": EGVAN_SHA, "selected_epoch": 15,
                "frozen_split_sha256": SPLIT_SHA, "ph2_cohort_manifest_sha256": report["hashes"]["ph2_cohort_manifest"],
                "baseline_ph2_prediction_sha256": report["hashes"]["experiment5_ph2_predictions"],
                "stage10_metric_sha256": report["hashes"]["stage10_test_metrics"],
                "class_order": list(CLASSES), "preprocessing_contract": report["preprocessing"],
                "runtime": report["runtime"], "gpu": report["runtime"]["gpu"],
                "source_artifact_hashes": {k: sha256(v) for k, v in source.items()},
                "generated_artifact_hashes": {name: sha256(output / name) for name in OUTPUTS},
                "prediction_integrity": integrity, "paired_correctness": paired, "mcnemar_exact": mcnemar,
                "training_performed": False, "fine_tuning_performed": False, "threshold_tuning": False,
                "checkpoint_selection_using_ph2": False, "ph2_used_for_training": False,
                "ph2_used_for_model_selection": False,
                "limitation": "PH2 was previously used in project diagnostics/evaluation; this is external follow-up, not untouched independent validation."}
    manifest_path = output / "stage11_manifest.json"
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("status") != "READY_FOR_COLAB_EVALUATION":
            raise FileExistsError("Stage 11 manifest is not a preflight handoff")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        write_json_new(manifest_path, manifest)
    print(json.dumps({"status": "COMPLETE", "metrics": metrics, "paired": paired,
                      "mcnemar": mcnemar, "shifts": shifts, "integrity": integrity}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.preflight:
        print(json.dumps(preflight(args.project_root), indent=2))
    else:
        run(args.project_root.resolve(), args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
