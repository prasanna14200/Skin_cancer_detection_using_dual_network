"""Run the frozen PH2 preprocessing sensitivity diagnostic on Colab/T4.

This script performs strict read-only preflight checks before inference. It
refuses to overwrite diagnostic outputs or the already verified PH2 artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path("/content/drive/MyDrive/EG-VAN")
EXPECTED_CHECKPOINT_SHA256 = (
    "f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848"
)
CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
PH2_MAPPING = {
    "common nevus": "nv",
    "melanoma": "mel",
    "atypical nevus": None,
}
EXPECTED_FILES = (
    "preprocessing_comparison.csv",
    "summary_metrics.json",
    "vasc_case_analysis.csv",
    "diagnostic_config.json",
    "README.md",
    "confusion_matrix_raw.csv",
    "confusion_matrix_ham_preprocessed.csv",
)
OUTPUT_DIR = ROOT / "experiments" / "ph2_preprocessing_diagnostic"
CHECKPOINT = ROOT / "experiments" / "efficientnetv2s_leakage_aware" / "best_checkpoint.pt"
MANIFEST = ROOT / "data" / "external" / "ph2" / "metadata" / "ph2_manifest.csv"
EVAL_DIR = ROOT / "experiments" / "ph2_external_eval"
PREVIOUS_PREDICTIONS = EVAL_DIR / "predictions.csv"
EVAL_ARTIFACTS = (
    "metrics.json",
    "evaluation_config.json",
    "confusion_matrix.csv",
    "predictions.csv",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_unique_image_ids(rows: list[dict[str, str]]) -> None:
    image_ids = [row.get("image_id", "").strip() for row in rows]
    duplicates = sorted(
        image_id for image_id, count in Counter(image_ids).items()
        if not image_id or count > 1
    )
    if duplicates:
        raise ValueError(f"Manifest contains blank or duplicate image IDs: {duplicates[:10]}")


def validate_manifest_image_path(
    image_path: str,
    image_id: str,
    local_image_path: Path,
) -> None:
    """Validate manifest basename independent of the host path convention."""
    normalized_path = image_path.strip().replace("\\", "/")
    if not normalized_path:
        raise ValueError(f"Manifest image_path is empty for {image_id}")

    manifest_name = PurePosixPath(normalized_path).name
    if PurePosixPath(manifest_name).suffix.lower() != ".bmp":
        raise ValueError(f"Unexpected manifest image extension for {image_id}: {manifest_name}")
    if manifest_name != f"{image_id}.bmp":
        raise ValueError(
            f"Manifest path/name mismatch for {image_id}: {manifest_name}"
        )
    if local_image_path.name != f"{image_id}.bmp":
        raise ValueError(f"Unexpected local image path for {image_id}: {local_image_path}")
    if not local_image_path.is_file():
        raise FileNotFoundError(f"PH2 image missing for {image_id}: {local_image_path}")


def preflight() -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict]:
    failures: list[str] = []
    checks: dict[str, object] = {}
    checks["root_exists"] = ROOT.is_dir()
    if not checks["root_exists"]:
        failures.append(f"Project root does not exist: {ROOT}")

    checks["checkpoint_exists"] = CHECKPOINT.is_file()
    if not checks["checkpoint_exists"]:
        failures.append(f"Checkpoint does not exist: {CHECKPOINT}")
        checkpoint_hash = None
    else:
        checkpoint_hash = sha256_file(CHECKPOINT)
        checks["checkpoint_sha256"] = checkpoint_hash
        checks["checkpoint_hash_matches"] = checkpoint_hash == EXPECTED_CHECKPOINT_SHA256
        if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
            failures.append(f"Checkpoint SHA256 mismatch: {checkpoint_hash}")

    checks["cuda_available"] = torch.cuda.is_available()
    if not checks["cuda_available"]:
        failures.append("CUDA is unavailable")
        gpu_name = None
    else:
        gpu_name = torch.cuda.get_device_name(0)
        checks["gpu"] = gpu_name
        checks["tesla_t4_detected"] = "t4" in gpu_name.lower()
        if not checks["tesla_t4_detected"]:
            failures.append(f"Expected Tesla T4, detected {gpu_name}")

    checks["manifest_exists"] = MANIFEST.is_file()
    if not checks["manifest_exists"]:
        failures.append(f"PH2 manifest does not exist: {MANIFEST}")

    missing_eval = [name for name in EVAL_ARTIFACTS if not (EVAL_DIR / name).is_file()]
    checks["existing_evaluation_artifacts_exist"] = not missing_eval
    checks["missing_evaluation_artifacts"] = missing_eval
    if missing_eval:
        failures.append(f"Existing verified evaluation artifacts missing: {missing_eval}")

    existing_outputs = [name for name in EXPECTED_FILES if (OUTPUT_DIR / name).exists()]
    checks["diagnostic_outputs_will_not_be_overwritten"] = not existing_outputs
    checks["existing_diagnostic_outputs"] = existing_outputs
    if existing_outputs:
        failures.append(f"Refusing to overwrite existing diagnostic outputs: {existing_outputs}")

    if failures:
        print(json.dumps({"preflight": "FAIL", "checks": checks, "failures": failures}, indent=2))
        raise SystemExit("Preflight failed; inference was not run.")

    manifest_rows = read_csv(MANIFEST)
    previous_rows = read_csv(PREVIOUS_PREDICTIONS)
    if not manifest_rows or not previous_rows:
        failures.append("PH2 manifest or existing predictions CSV is empty")
    if len(manifest_rows) != 200:
        failures.append(f"Expected 200 manifest rows; found {len(manifest_rows)}")
    try:
        validate_unique_image_ids(manifest_rows)
    except ValueError as exc:
        failures.append(str(exc))

    required_manifest = {"image_id", "image_path", "external_label", "ham_label", "included"}
    if not manifest_rows or not required_manifest.issubset(manifest_rows[0]):
        failures.append(f"Manifest lacks required columns: {sorted(required_manifest)}")
    required_predictions = {
        "image_id",
        "true_label",
        "predicted_ham10000_label",
        "binary_prediction",
    }
    if not previous_rows or not required_predictions.issubset(previous_rows[0]):
        failures.append("Existing PH2 predictions lack required columns")

    included = [
        row for row in manifest_rows
        if row.get("included", "").strip().lower() == "true"
    ]
    excluded = [
        row for row in manifest_rows
        if row.get("included", "").strip().lower() != "true"
    ]
    included_ids = [row.get("image_id", "").strip() for row in included]
    previous_ids = [row.get("image_id", "").strip() for row in previous_rows]
    if len(included) != 120 or len(set(included_ids)) != 120:
        failures.append(f"Expected 120 unique included manifest rows; found {len(included)}")
    if len(excluded) != 80:
        failures.append(f"Expected 80 excluded manifest rows; found {len(excluded)}")
    if any(
        row.get("external_label", "").strip().lower() != "atypical nevus"
        or row.get("ham_label", "").strip()
        for row in excluded
    ):
        failures.append("Excluded rows must be atypical nevi with no HAM label")
    previous_by_id = {row["image_id"]: row for row in previous_rows}
    if len(previous_by_id) != len(previous_rows):
        failures.append("Existing PH2 predictions contain duplicate image IDs")
    if set(included_ids) != set(previous_ids):
        failures.append("Included manifest IDs do not match the existing PH2 prediction IDs")

    label_counts = Counter(row.get("external_label", "").strip().lower() for row in included)
    checks["included_count"] = len(included)
    checks["included_class_counts"] = dict(label_counts)
    if label_counts != Counter({"common nevus": 80, "melanoma": 40}):
        failures.append(f"Unexpected included PH2 class counts: {dict(label_counts)}")

    images_dir = ROOT / "data" / "external" / "ph2" / "images"
    for row in manifest_rows:
        image_id = row.get("image_id", "").strip()
        if not image_id:
            continue
        local_image_path = images_dir / f"{image_id}.bmp"
        try:
            validate_manifest_image_path(
                row.get("image_path", ""),
                image_id,
                local_image_path,
            )
        except (ValueError, FileNotFoundError) as exc:
            failures.append(str(exc))

    for row in included:
        image_id = row["image_id"].strip()
        expected_label = PH2_MAPPING.get(row.get("external_label", "").strip().lower())
        if row.get("ham_label", "").strip().lower() != expected_label:
            failures.append(f"Unexpected manifest label mapping for {image_id}")
            break
        previous = previous_by_id.get(image_id)
        if previous is None or previous["true_label"].strip().lower() != expected_label:
            failures.append(f"Existing prediction label mismatch for {image_id}")
            break

    if failures:
        print(json.dumps({"preflight": "FAIL", "checks": checks, "failures": failures}, indent=2))
        raise SystemExit("Preflight failed; inference was not run.")

    info = {
        "checkpoint_sha256": checkpoint_hash,
        "gpu": gpu_name,
        "previous_predictions": previous_by_id,
        "preflight_checks": checks,
    }
    print(json.dumps({"preflight": "PASS", "checks": checks}, indent=2))
    return included, previous_by_id, info


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class DiagnosticDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform, preprocess_ham: bool):
        self.rows = rows
        self.transform = transform
        self.preprocess_ham = preprocess_ham

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = ROOT / "data" / "external" / "ph2" / "images" / f"{row['image_id']}.bmp"
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            if self.preprocess_ham:
                from preprocessing import preprocess_image

                rgb_array = np.asarray(rgb, dtype=np.uint8)
                bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                processed_bgr = preprocess_image(bgr)
                # Match HAM's cv2.imwrite(".jpg") then PIL RGB load without
                # creating or modifying any dataset files.
                success, encoded = cv2.imencode(".jpg", processed_bgr)
                if not success:
                    raise RuntimeError(f"JPEG encoding failed for {row['image_id']}")
                decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded_bgr is None:
                    raise RuntimeError(f"JPEG decoding failed for {row['image_id']}")
                processed_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
                rgb = Image.fromarray(processed_rgb, mode="RGB")
            tensor = self.transform(rgb)
        return tensor, row["image_id"]


def infer(model, rows, transform, preprocess_ham: bool) -> dict[str, np.ndarray]:
    dataset = DiagnosticDataset(rows, transform, preprocess_ham)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    output: dict[str, np.ndarray] = {}
    model.eval()
    with torch.inference_mode():
        for images, image_ids in loader:
            logits = model(images.to("cuda", non_blocking=True))
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            for image_id, probability in zip(image_ids, probabilities):
                output[image_id] = probability
    return output


def auc_rank(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def probability_stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "percentiles": {
            str(p): float(np.percentile(array, p)) for p in (0, 25, 50, 75, 90, 95, 100)
        },
    }


def summarize(rows, probabilities):
    predictions = {
        image_id: CLASS_NAMES[int(np.argmax(probability))]
        for image_id, probability in probabilities.items()
    }
    truth = {row["image_id"]: row["ham_label"].strip().lower() for row in rows}
    distribution = Counter(predictions.values())
    matrix = [[0, 0, 0], [0, 0, 0]]
    row_index = {"nv": 0, "mel": 1}
    col_index = {"nv": 0, "mel": 1}
    for image_id, actual in truth.items():
        pred = predictions[image_id]
        column = col_index.get(pred, 2)
        matrix[row_index[actual]][column] += 1

    tn = matrix[0][0]
    fp = matrix[0][1] + matrix[0][2]
    fn = matrix[1][0] + matrix[1][2]
    tp = matrix[1][1]
    denominator = len(rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    correct = sum(predictions[key] == truth[key] for key in truth)

    grouped = {
        "all": list(probabilities.values()),
        "true_melanoma": [probabilities[k] for k in truth if truth[k] == "mel"],
        "true_common_nevus": [probabilities[k] for k in truth if truth[k] == "nv"],
    }
    feature_stats = {}
    for class_name in ("mel", "nv", "vasc"):
        feature_stats[class_name] = {
            group_name: probability_stats([float(p[CLASS_INDEX[class_name]]) for p in group])
            for group_name, group in grouped.items()
        }

    melanoma_scores = [float(probabilities[k][CLASS_INDEX["mel"]]) for k in truth]
    labels = [int(truth[k] == "mel") for k in truth]
    return {
        "sample_count": denominator,
        "true_class_counts": dict(Counter(truth.values())),
        "prediction_distribution_7class": {
            name: int(distribution.get(name, 0)) for name in CLASS_NAMES
        },
        "number_of_mel_predictions": int(distribution.get("mel", 0)),
        "number_of_vasc_predictions": int(distribution.get("vasc", 0)),
        "number_of_other_predictions": int(sum(distribution.get(name, 0) for name in ("akiec", "bcc", "bkl", "df"))),
        "binary_confusion_matrix_rows_nv_mel_cols_nv_mel_other": {
            "columns": ["nv", "mel", "other"],
            "rows": {"nv": matrix[0], "mel": matrix[1]},
        },
        "binary_metrics": {
            "accuracy": correct / denominator,
            "precision": precision,
            "recall_sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
        "melanoma_roc_auc": auc_rank(labels, melanoma_scores),
        "probability_statistics": feature_stats,
        "predictions": predictions,
        "probabilities": probabilities,
    }


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    included, previous, info = preflight()
    from models.baseline_effnet import build_model

    device = torch.device("cuda")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model, _ = build_model(pretrained=False)
    model.load_state_dict(state, strict=True)
    if checkpoint.get("config", {}).get("classes") != list(CLASS_NAMES):
        raise SystemExit("Checkpoint embedded class order mismatch; inference was not run.")
    model.to(device).eval()
    transform = build_transform()

    print("Running RAW inference only; validating exact reproduction before HAM preprocessing.")
    raw_probabilities = infer(model, included, transform, preprocess_ham=False)
    raw_predictions = {
        image_id: CLASS_NAMES[int(np.argmax(probability))]
        for image_id, probability in raw_probabilities.items()
    }
    raw_mismatches = [
        (image_id, previous[image_id]["predicted_ham10000_label"], raw_predictions[image_id])
        for image_id in raw_predictions
        if previous[image_id]["predicted_ham10000_label"].strip().lower() != raw_predictions[image_id]
    ]
    if set(raw_predictions) != set(previous):
        raise SystemExit("RAW inference ID set mismatch; HAM comparison was not run.")
    if raw_mismatches:
        print(json.dumps({"raw_reproduction": "FAIL", "mismatch_count": len(raw_mismatches), "examples": raw_mismatches[:10]}, indent=2))
        raise SystemExit("RAW predictions differ from verified PH2 output; stopped before HAM inference.")
    print("RAW reproduction: PASS for all 120 IDs.")

    print("Running HAM-matched preprocessing inference.")
    ham_probabilities = infer(model, included, transform, preprocess_ham=True)
    if set(ham_probabilities) != set(raw_probabilities):
        raise SystemExit("HAM inference ID set differs from RAW; no outputs written.")

    raw = summarize(included, raw_probabilities)
    ham = summarize(included, ham_probabilities)
    transition = {
        source: {
            target: sum(
                raw["predictions"][image_id] == source and ham["predictions"][image_id] == target
                for image_id in raw["predictions"]
            )
            for target in CLASS_NAMES
        }
        for source in CLASS_NAMES
    }
    binary_transition_classes = ("nv", "mel", "other")
    binary_transition = {source: {target: 0 for target in binary_transition_classes} for source in binary_transition_classes}
    for image_id in raw["predictions"]:
        source = raw["predictions"][image_id]
        target = ham["predictions"][image_id]
        source = source if source in ("nv", "mel") else "other"
        target = target if target in ("nv", "mel") else "other"
        binary_transition[source][target] += 1

    comparison_rows = []
    deltas = []
    for row in included:
        image_id = row["image_id"]
        rp = raw_probabilities[image_id]
        hp = ham_probabilities[image_id]
        raw_class = raw["predictions"][image_id]
        ham_class = ham["predictions"][image_id]
        delta = float(hp[CLASS_INDEX["mel"]] - rp[CLASS_INDEX["mel"]])
        deltas.append((image_id, delta))
        record = {
            "image_id": image_id,
            "true_label": row["ham_label"].strip().lower(),
            "raw_predicted_class": raw_class,
            "raw_melanoma_probability": float(rp[CLASS_INDEX["mel"]]),
            "raw_nv_probability": float(rp[CLASS_INDEX["nv"]]),
            "raw_vasc_probability": float(rp[CLASS_INDEX["vasc"]]),
            "raw_max_probability": float(rp.max()),
            "raw_binary_prediction": raw_class if raw_class in ("nv", "mel") else "other",
            "ham_preprocessed_predicted_class": ham_class,
            "ham_melanoma_probability": float(hp[CLASS_INDEX["mel"]]),
            "ham_nv_probability": float(hp[CLASS_INDEX["nv"]]),
            "ham_vasc_probability": float(hp[CLASS_INDEX["vasc"]]),
            "ham_max_probability": float(hp.max()),
            "ham_binary_prediction": ham_class if ham_class in ("nv", "mel") else "other",
            "prediction_changed": raw_class != ham_class,
            "delta_melanoma_probability": delta,
        }
        for index, name in enumerate(CLASS_NAMES):
            record[f"raw_prob_{name}"] = float(rp[index])
            record[f"ham_prob_{name}"] = float(hp[index])
        comparison_rows.append(record)

    previous_vasc_ids = {
        image_id for image_id, row in previous.items()
        if row["predicted_ham10000_label"].strip().lower() == "vasc"
    }
    if len(previous_vasc_ids) != 18:
        raise SystemExit(f"Expected 18 previous vasc cases, found {len(previous_vasc_ids)}")
    vasc_rows = []
    for row in included:
        image_id = row["image_id"]
        if image_id not in previous_vasc_ids:
            continue
        rp, hp = raw_probabilities[image_id], ham_probabilities[image_id]
        raw_class, ham_class = raw["predictions"][image_id], ham["predictions"][image_id]
        vasc_rows.append(
            {
                "image_id": image_id,
                "true_label": row["ham_label"].strip().lower(),
                "previous_prediction": previous[image_id]["predicted_ham10000_label"],
                "raw_predicted_class": raw_class,
                "raw_top1_probability": float(rp.max()),
                "raw_vasc_probability": float(rp[CLASS_INDEX["vasc"]]),
                "raw_mel_probability": float(rp[CLASS_INDEX["mel"]]),
                "ham_predicted_class": ham_class,
                "ham_top1_probability": float(hp.max()),
                "ham_vasc_probability": float(hp[CLASS_INDEX["vasc"]]),
                "ham_mel_probability": float(hp[CLASS_INDEX["mel"]]),
                "prediction_changed": raw_class != ham_class,
            }
        )

    delta_values = [delta for _, delta in deltas]
    summary = {
        "raw": {key: value for key, value in raw.items() if key not in ("predictions", "probabilities")},
        "ham_preprocessed": {key: value for key, value in ham.items() if key not in ("predictions", "probabilities")},
        "number_of_prediction_changes": sum(
            raw["predictions"][image_id] != ham["predictions"][image_id]
            for image_id in raw["predictions"]
        ),
        "prediction_transition_matrix_raw_to_ham": transition,
        "binary_prediction_transition_matrix_raw_to_ham": binary_transition,
        "melanoma_probability_delta_ham_minus_raw": {
            "mean": float(np.mean(delta_values)),
            "median": float(np.median(delta_values)),
            "positive_count": sum(value > 0 for value in delta_values),
            "negative_count": sum(value < 0 for value in delta_values),
            "zero_count": sum(value == 0 for value in delta_values),
            "largest_increases": [
                {"image_id": image_id, "delta": value}
                for image_id, value in sorted(deltas, key=lambda item: item[1], reverse=True)[:10]
            ],
            "largest_decreases": [
                {"image_id": image_id, "delta": value}
                for image_id, value in sorted(deltas, key=lambda item: item[1])[:10]
            ],
        },
        "previous_vasc_case_transition_counts": dict(
            Counter(row["ham_predicted_class"] for row in vasc_rows)
        ),
        "interpretation_note": (
            "This paired comparison measures sensitivity to the specified preprocessing "
            "change; it does not alone establish preprocessing as the cause of external "
            "performance."
        ),
    }

    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("Checkpoint hash changed during diagnostic; outputs not written.")
    current_outputs = [name for name in EXPECTED_FILES if (OUTPUT_DIR / name).exists()]
    if current_outputs:
        raise SystemExit(f"Refusing to overwrite diagnostic outputs: {current_outputs}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    class_fields = [
        f"{pipeline}_prob_{name}"
        for pipeline in ("raw", "ham") for name in CLASS_NAMES
    ]
    comparison_fields = [
        "image_id", "true_label", "raw_predicted_class",
        "raw_melanoma_probability", "raw_nv_probability", "raw_vasc_probability",
        "raw_max_probability", "raw_binary_prediction",
        "ham_preprocessed_predicted_class", "ham_melanoma_probability",
        "ham_nv_probability", "ham_vasc_probability", "ham_max_probability",
        "ham_binary_prediction", "prediction_changed", "delta_melanoma_probability",
        *class_fields,
    ]
    write_csv(OUTPUT_DIR / "preprocessing_comparison.csv", comparison_fields, comparison_rows)
    write_csv(
        OUTPUT_DIR / "vasc_case_analysis.csv",
        list(vasc_rows[0]),
        vasc_rows,
    )
    for name, condition in (
        ("confusion_matrix_raw.csv", raw),
        ("confusion_matrix_ham_preprocessed.csv", ham),
    ):
        matrix = condition["binary_confusion_matrix_rows_nv_mel_cols_nv_mel_other"]
        write_csv(
            OUTPUT_DIR / name,
            ["actual\\predicted", "nv", "mel", "other"],
            [
                {"actual\\predicted": label, **dict(zip(("nv", "mel", "other"), counts))}
                for label, counts in matrix["rows"].items()
            ],
        )

    config = {
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "model_architecture": "torchvision EfficientNetV2S",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "class_order": list(CLASS_NAMES),
        "raw_preprocessing": {
            "source": "original PH2 BMP",
            "operations": [
                "PIL open",
                "convert RGB",
                "torchvision Resize((384, 384))",
                "ToTensor",
                "ImageNet normalization",
            ],
        },
        "ham_matched_preprocessing": {
            "source": "same original PH2 BMP files",
            "operations": [
                "decode original as RGB",
                "convert RGB to BGR for src/preprocessing.py",
                "blackhat hair mask and Telea inpainting (kernel=17, threshold=10, radius=1.0)",
                "Gray World channel balancing",
                "multi-scale Retinex (sigmas=15,80,250; equal weights; per-channel 1st-99th percentile normalization)",
                "in-memory JPEG encode/decode with OpenCV default JPEG settings to match saved HAM JPG representation",
                "convert BGR to RGB",
                "torchvision Resize((384, 384))",
                "ToTensor",
                "ImageNet normalization",
            ],
            "augmentation": "none",
            "masks_or_roi": "not used",
        },
        "input_count": len(included),
        "ph2_manifest_path": str(MANIFEST),
        "existing_predictions_path": str(PREVIOUS_PREDICTIONS),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_files": [
            "experiments/ph2_preprocessing_diagnostic/run_diagnostic.py",
            "src/preprocessing.py",
            "src/dataset.py",
            "src/train.py",
            "src/external_eval.py",
            "src/models/baseline_effnet.py",
        ],
        "output_directory": str(OUTPUT_DIR),
        "raw_reproduction": {"status": "PASS", "matched_ids": 120},
        "training_performed": False,
        "checkpoint_modified": False,
        "existing_evaluation_artifacts_modified": False,
    }
    (OUTPUT_DIR / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "diagnostic_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "README.md").write_text(
        readme_text(summary, config), encoding="utf-8"
    )

    verify_outputs(included, previous, raw, ham)
    print_final_report(summary, config, len(comparison_rows), len(vasc_rows))


def readme_text(summary, config) -> str:
    return f"""# PH² Preprocessing Sensitivity Diagnostic

## Purpose

This is a paired, frozen-checkpoint diagnostic to determine whether applying
the actual HAM10000 preprocessing pipeline changes predictions on the same PH²
images. It is not a model-improvement or causal-identification experiment.

## Experimental design

- Samples: {config['input_count']} included PH² images (80 common nevi, 40 melanomas).
- Checkpoint SHA256: `{config['checkpoint_sha256']}` (epoch {config['checkpoint_epoch']}).
- No training or fine-tuning occurred. No checkpoint, PH² data, split, or
  existing external-evaluation artifact was modified.
- Image IDs, labels, model, class order, resize, tensor conversion,
  normalization, inference mode, and ordering were held constant.
- The changed variable was whether the frozen HAM image preprocessing was
  applied before model input.

## RAW pipeline

Original PH² BMP → RGB → Resize 384×384 → ToTensor → ImageNet normalization.
This is the existing external-evaluation transform.

## HAM-matched pipeline

The same original PH² BMP is converted to BGR for `src/preprocessing.py`, then
processed with blackhat/Telea hair removal (kernel 17, threshold 10, radius
1.0), Gray World, and multi-scale Retinex (sigmas 15/80/250, equal weights,
per-channel 1st–99th percentile normalization). The result is JPEG-encoded and
decoded in memory to match the saved processed-HAM JPG representation, then
converted to RGB and passed through the same resize/tensor/normalization
transform. No augmentation, masks, or ROI crops are used.

## Results snapshot

- RAW predictions: {summary['raw']['prediction_distribution_7class']}
- HAM-preprocessed predictions: {summary['ham_preprocessed']['prediction_distribution_7class']}
- Changed predicted classes: {summary['number_of_prediction_changes']} / {config['input_count']}
- Mean change in melanoma probability (HAM − RAW): {summary['melanoma_probability_delta_ham_minus_raw']['mean']:.6f}

See `summary_metrics.json`, `preprocessing_comparison.csv`, and
`vasc_case_analysis.csv` for complete results.

## Limitations and interpretation

Any score or prediction change is evidence of preprocessing sensitivity under
this checkpoint and these inputs. It does not, by itself, prove preprocessing
caused the original external-evaluation result. PH² has only 40 included
melanomas and 80 common nevi; there is no independent replication here. The
HAM preprocessing includes image-specific normalization operations and is
applied to an external dataset for this diagnostic only. The RAW condition
must reproduce all 120 previously saved predictions; otherwise no comparison
is interpreted.
"""


def verify_outputs(included, previous, raw, ham):
    comparison = read_csv(OUTPUT_DIR / "preprocessing_comparison.csv")
    vasc = read_csv(OUTPUT_DIR / "vasc_case_analysis.csv")
    ids = [row["image_id"] for row in comparison]
    expected_ids = [row["image_id"] for row in included]
    if len(comparison) != 120 or len(set(ids)) != 120 or set(ids) != set(expected_ids):
        raise SystemExit("Output comparison row/ID verification failed.")
    if len(vasc) != 18 or len({row["image_id"] for row in vasc}) != 18:
        raise SystemExit("Output vasc-case row/ID verification failed.")
    for image_id in ids:
        if raw["predictions"][image_id] != previous[image_id]["predicted_ham10000_label"].strip().lower():
            raise SystemExit(f"RAW output no longer matches prior result for {image_id}")
    missing = [name for name in EXPECTED_FILES if not (OUTPUT_DIR / name).is_file()]
    if missing:
        raise SystemExit(f"Expected output files missing: {missing}")


def print_final_report(summary, config, comparison_count, vasc_count):
    files = [
        name for name in EXPECTED_FILES
        if (OUTPUT_DIR / name).is_file()
    ]
    print(
        json.dumps(
            {
                "preflight": "PASS",
                "raw_reproduction": "PASS",
                "ham_diagnostic": "PASS",
                "output_directory": str(OUTPUT_DIR),
                "output_files": files,
                "comparison_rows": comparison_count,
                "vasc_case_rows": vasc_count,
                "raw_prediction_distribution": summary["raw"]["prediction_distribution_7class"],
                "ham_prediction_distribution": summary["ham_preprocessed"]["prediction_distribution_7class"],
                "number_of_prediction_changes": summary["number_of_prediction_changes"],
                "mel_probability_delta": summary["melanoma_probability_delta_ham_minus_raw"],
                "previous_vasc_transitions": summary["previous_vasc_case_transition_counts"],
                "checkpoint_sha256": config["checkpoint_sha256"],
                "gpu": config["runtime"]["gpu"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
