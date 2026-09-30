"""Prepare or run inference-only PH2 external follow-up evaluation for Exp5.

The default workflow is a read-only preflight. Inference requires the explicit
--run-inference and --require-t4 flags. No training or fine-tuning is provided.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_CHECKPOINT_SHA256 = (
    "81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a"
)
EXPECTED_SPLIT_SHA256 = (
    "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
)
EXPECTED_EPOCH = 15
HAM_CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
PH2_TO_HAM = {
    "common nevus": "nv",
    "melanoma": "mel",
    "atypical nevus": None,
}
EXPECTED_DIAGNOSIS_COUNTS = {
    "common nevus": 80,
    "atypical nevus": 80,
    "melanoma": 40,
}
IMAGE_SIZE = 384
BATCH_SIZE = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OUTPUT_FILES = (
    "ph2_manifest.csv",
    "ph2_predictions.csv",
    "ph2_metrics.json",
    "confusion_matrix.csv",
    "external_validation_config.json",
    "external_validation_report.md",
)
FOLLOW_UP_LIMITATION = (
    "PH² had previously been used for preprocessing diagnostics and an "
    "earlier model evaluation before Experiment #5. Therefore this analysis "
    "is reported as an external follow-up evaluation rather than a completely "
    "untouched independent external validation."
)


class EvaluationError(RuntimeError):
    """Raised when a preflight invariant fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"Expected JSON object in {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise EvaluationError(f"Could not read CSV {path}: {exc}") from exc


def read_clinical_labels(path: Path) -> dict[str, str]:
    code_to_label = {"0": "common nevus", "1": "atypical nevus", "2": "melanoma"}
    pattern = re.compile(r"^\|\|\s*(IMD\d+)\s*\|\|(.*?)\|\|\s*([^|]+?)\s*\|\|")
    labels = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        image_id, _, diagnosis_code = match.groups()
        if diagnosis_code.strip() not in code_to_label:
            raise EvaluationError(
                f"Unknown PH2 diagnosis code {diagnosis_code!r} for {image_id}"
            )
        if image_id in labels:
            raise EvaluationError(f"Duplicate PH2 metadata ID: {image_id}")
        labels[image_id] = code_to_label[diagnosis_code.strip()]
    if not labels:
        raise EvaluationError(f"No clinical labels parsed from {path}")
    return labels


def preprocessing_contract() -> dict:
    return {
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
    }


def verify_preprocessing_contract(root: Path) -> dict:
    diagnostic_path = root / "experiments/ph2_preprocessing_diagnostic/diagnostic_config.json"
    if not diagnostic_path.is_file():
        raise EvaluationError(f"Prior preprocessing diagnostic config missing: {diagnostic_path}")
    diagnostic = read_json(diagnostic_path)
    actual_contract = diagnostic.get("ham_matched_preprocessing")
    expected_contract = preprocessing_contract()
    if actual_contract != expected_contract:
        raise EvaluationError("HAM-matched transform differs from diagnostic_config.json")

    sys.path.insert(0, str(root / "src"))
    from preprocessing import (
        HAIR_KERNEL_SIZE,
        HAIR_THRESHOLD,
        INPAINT_RADIUS,
        RETINEX_SIGMAS,
        RETINEX_WEIGHTS,
        preprocess_image,
    )

    if (
        HAIR_KERNEL_SIZE != 17
        or HAIR_THRESHOLD != 10
        or not math.isclose(INPAINT_RADIUS, 1.0)
        or tuple(RETINEX_SIGMAS) != (15.0, 80.0, 250.0)
        or any(not math.isclose(value, 1 / 3) for value in RETINEX_WEIGHTS)
    ):
        raise EvaluationError("src/preprocessing.py constants differ from diagnostic")
    if not callable(preprocess_image):
        raise EvaluationError("Frozen preprocess_image function is unavailable")
    return {
        "status": "PASS",
        "matches_diagnostic_config": True,
        "implementation": "src/preprocessing.py:preprocess_image",
        "operation_contract": actual_contract,
        "resize": [IMAGE_SIZE, IMAGE_SIZE],
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
    }


def audit_cohort(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    ph2_root = root / "data/external/ph2"
    manifest_path = ph2_root / "metadata/ph2_manifest.csv"
    clinical_path = ph2_root / "metadata/PH2_dataset.txt"
    images_dir = ph2_root / "images"
    for path in (manifest_path, clinical_path):
        if not path.is_file():
            raise EvaluationError(f"Required PH2 metadata file missing: {path}")
    if not images_dir.is_dir():
        raise EvaluationError(f"PH2 image directory missing: {images_dir}")

    rows = read_csv(manifest_path)
    clinical = read_clinical_labels(clinical_path)
    if len(rows) != 200:
        raise EvaluationError(f"Expected 200 PH2 manifest rows; found {len(rows)}")
    required = {"image_id", "image_path", "clinical_code", "external_label", "ham_label", "included"}
    if not rows or not required.issubset(rows[0]):
        raise EvaluationError(f"PH2 manifest missing columns: {sorted(required - set(rows[0] if rows else []))}")

    ids = [row["image_id"].strip() for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(clinical):
        raise EvaluationError("PH2 manifest IDs are duplicate or differ from source metadata")

    diagnosis_counts = Counter()
    included_rows = []
    excluded_rows = []
    missing_images = []
    for row in rows:
        image_id = row["image_id"].strip()
        diagnosis = " ".join(row["external_label"].strip().lower().split())
        if diagnosis != clinical[image_id]:
            raise EvaluationError(f"Source diagnosis disagrees with manifest for {image_id}")
        if diagnosis not in PH2_TO_HAM:
            raise EvaluationError(f"Unapproved PH2 diagnosis {diagnosis!r} for {image_id}")
        expected_ham = PH2_TO_HAM[diagnosis]
        included = row["included"].strip().lower() == "true"
        ham_label = row["ham_label"].strip().lower()
        if expected_ham is None:
            if included or ham_label:
                raise EvaluationError(f"Atypical nevus must be excluded: {image_id}")
            excluded_rows.append(row)
        else:
            if not included or ham_label != expected_ham:
                raise EvaluationError(f"Invalid direct-overlap mapping for {image_id}")
            included_rows.append(row)
        diagnosis_counts[diagnosis] += 1
        if not (images_dir / f"{image_id}.bmp").is_file():
            missing_images.append(image_id)

    if dict(diagnosis_counts) != EXPECTED_DIAGNOSIS_COUNTS:
        raise EvaluationError(f"Unexpected diagnosis counts: {dict(diagnosis_counts)}")
    if missing_images:
        raise EvaluationError(f"Missing PH2 images: {missing_images[:10]}")
    included_counts = Counter(row["ham_label"].strip().lower() for row in included_rows)
    if included_counts != Counter({"nv": 80, "mel": 40}) or len(included_rows) != 120:
        raise EvaluationError(f"Unexpected direct-overlap cohort: {dict(included_counts)}")

    normalized = []
    for row in rows:
        normalized.append({
            "image_id": row["image_id"].strip(),
            "relative_image_path": f"data/external/ph2/images/{row['image_id'].strip()}.bmp",
            "clinical_code": row["clinical_code"].strip(),
            "true_external_label": row["external_label"].strip().lower(),
            "true_ham_label": row["ham_label"].strip().lower(),
            "included": row["included"].strip().lower() == "true",
        })
    return included_rows, normalized, {
        "total_images": len(rows),
        "diagnosis_counts": dict(diagnosis_counts),
        "included_count": len(included_rows),
        "included_ham_counts": dict(included_counts),
        "excluded_atypical_nevi": len(excluded_rows),
        "included_rows": included_rows,
        "normalized_manifest_rows": normalized,
        "images_dir": images_dir,
        "source_manifest": manifest_path,
    }


def check_output_safety(output_dir: Path) -> dict:
    if output_dir.exists():
        existing = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*"))
        if existing:
            raise EvaluationError(
                f"Refusing to overwrite existing output path {output_dir}: {existing[:20]}"
            )
        raise EvaluationError(f"Refusing to reuse existing output directory: {output_dir}")
    return {"status": "PASS", "output_dir_absent": True, "files_will_be_overwritten": False}


def audit_project(root: Path, output_dir: Path, require_t4: bool = False) -> dict:
    root = root.resolve()
    output_dir = output_dir.resolve()
    if root.name != "EG-VAN":
        raise EvaluationError(f"Unexpected project root: {root}")

    checkpoint_path = root / "experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt"
    config_path = checkpoint_path.parent / "config.json"
    experiment_manifest_path = checkpoint_path.parent / "experiment_manifest.json"
    split_path = root / "data/splits/split_leakage_aware.csv"
    for path in (checkpoint_path, config_path, experiment_manifest_path, split_path):
        if not path.is_file():
            raise EvaluationError(f"Required frozen artifact missing: {path}")

    checkpoint_hash = sha256_file(checkpoint_path)
    split_hash = sha256_file(split_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise EvaluationError(f"Experiment #5 checkpoint SHA256 mismatch: {checkpoint_hash}")
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise EvaluationError(f"Frozen HAM split SHA256 mismatch: {split_hash}")

    config = read_json(config_path)
    experiment_manifest = read_json(experiment_manifest_path)
    if config.get("classes") != list(HAM_CLASSES):
        raise EvaluationError("Experiment #5 config class order mismatch")
    if experiment_manifest.get("selected_epoch") != EXPECTED_EPOCH:
        raise EvaluationError("Experiment #5 selected epoch is not 15")
    if experiment_manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise EvaluationError("Experiment #5 manifest checkpoint hash mismatch")
    if config.get("checkpoint_selection_metric") != "validation_threshold_constrained_min_loss":
        raise EvaluationError("Unexpected Experiment #5 checkpoint selection rule")

    provenance_path = root / "data/external/ph2/metadata/provenance.json"
    provenance = read_json(provenance_path)
    if provenance.get("status") != "VERIFIED":
        raise EvaluationError("PH2 provenance is not VERIFIED")
    if provenance.get("intended_use", {}).get("scope") != "non-commercial academic research and external validation":
        raise EvaluationError("PH2 provenance does not allow this scoped follow-up evaluation")
    if provenance.get("intended_use", {}).get("commercial_use") is not False:
        raise EvaluationError("PH2 commercial-use restriction missing")
    if provenance.get("intended_use", {}).get("redistribution") is not False:
        raise EvaluationError("PH2 redistribution restriction missing")

    included_rows, normalized_manifest, cohort = audit_cohort(root)
    preprocessing = verify_preprocessing_contract(root)
    output_safety = check_output_safety(output_dir)

    existing_followup_outputs = root / "experiments/ph2_external_eval_exp5"
    prior_evaluation_dir = root / "experiments/ph2_external_eval"
    prior_evaluation_config = prior_evaluation_dir / "evaluation_config.json"
    prior_config = read_json(prior_evaluation_config) if prior_evaluation_config.is_file() else {}

    runtime = {"python": platform.python_version()}
    if require_t4:
        import torch
        runtime.update({
            "torch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
        if not runtime["cuda_available"] or "t4" not in runtime["gpu"].lower():
            raise EvaluationError(f"A Colab Tesla T4 is required; runtime reported {runtime}")
    else:
        runtime["gpu_check"] = "deferred to Colab preflight"

    return {
        "project_root": str(root),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "split_path": str(split_path),
        "split_sha256": split_hash,
        "selected_epoch": EXPECTED_EPOCH,
        "checkpoint_selection_rule": "validation_threshold_constrained_min_loss",
        "class_order": list(HAM_CLASSES),
        "cohort": {
            key: value for key, value in cohort.items()
            if key not in {"included_rows", "normalized_manifest_rows", "images_dir", "source_manifest"}
        },
        "included_rows": included_rows,
        "normalized_manifest_rows": normalized_manifest,
        "images_dir": str(cohort["images_dir"]),
        "preprocessing": preprocessing,
        "output_safety": output_safety,
        "output_dir": str(output_dir),
        "provenance": provenance,
        "runtime": runtime,
        "previous_ph2_evaluation": {
            "exists": bool(prior_config),
            "checkpoint_sha256": prior_config.get("checkpoint", {}).get("sha256"),
            "interpretation": "historical PH2 use; this run is an external follow-up, not an untouched independent external validation",
        },
        "follow_up_limitation": FOLLOW_UP_LIMITATION,
        "ph2_inference_performed": False,
    }


def confusion_matrix(actual: list[str], predicted: list[str]) -> list[list[int]]:
    if len(actual) != len(predicted):
        raise EvaluationError("Actual and predicted arrays differ in length")
    matrix = [[0, 0, 0], [0, 0, 0]]
    row_index = {"nv": 0, "mel": 1}
    column_index = {"nv": 0, "mel": 1}
    for true_label, predicted_label in zip(actual, predicted):
        if true_label not in row_index:
            raise EvaluationError(f"Unsupported true HAM label: {true_label}")
        predicted_column = column_index.get(predicted_label, 2)
        matrix[row_index[true_label]][predicted_column] += 1
    return matrix


def class_metrics(matrix: list[list[int]], row: int, column: int) -> dict:
    true_positive = matrix[row][column]
    false_negative = sum(matrix[row]) - true_positive
    false_positive = sum(matrix[index][column] for index in range(2) if index != row)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def binary_roc_auc(labels: list[int], scores: list[float]) -> float | None:
    if len(labels) != len(scores):
        raise EvaluationError("AUROC labels and scores differ in length")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def calculate_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise EvaluationError("Cannot calculate metrics for an empty cohort")
    actual = [row["true_ham_label"] for row in rows]
    predicted = [row["predicted_class"] for row in rows]
    matrix = confusion_matrix(actual, predicted)
    nv = class_metrics(matrix, row=0, column=0)
    melanoma = class_metrics(matrix, row=1, column=1)
    total = len(rows)
    accuracy = (matrix[0][0] + matrix[1][1]) / total
    macro_precision = (nv["precision"] + melanoma["precision"]) / 2.0
    macro_recall = (nv["recall"] + melanoma["recall"]) / 2.0
    macro_f1 = (nv["f1"] + melanoma["f1"]) / 2.0
    auc = binary_roc_auc(
        [1 if label == "mel" else 0 for label in actual],
        [float(row["probabilities"]["p_mel"]) for row in rows],
    )
    return {
        "total_samples": total,
        "accuracy": accuracy,
        "balanced_accuracy": macro_recall,
        "class_metrics": {"nv": nv, "mel": melanoma},
        "melanoma_sensitivity_recall": melanoma["recall"],
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "melanoma_auroc": auc,
        "auroc_score": "p_mel",
        "confusion_matrix": {
            "true_rows": ["nv", "mel"],
            "predicted_columns": ["nv", "mel", "other"],
            "counts": matrix,
            "other_policy": "Predictions outside nv/mel remain in other and count as incorrect for accuracy and recall of the true class.",
        },
        "prediction_distribution": dict(Counter(predicted)),
        "metric_scope": "Only externally represented classes nv and mel; no seven-class macro metric is reported.",
    }


def validate_metrics_implementation() -> dict:
    sample_rows = [
        {"true_ham_label": "nv", "predicted_class": "nv", "probabilities": {"p_mel": 0.1}},
        {"true_ham_label": "nv", "predicted_class": "mel", "probabilities": {"p_mel": 0.2}},
        {"true_ham_label": "nv", "predicted_class": "vasc", "probabilities": {"p_mel": 0.05}},
        {"true_ham_label": "mel", "predicted_class": "mel", "probabilities": {"p_mel": 0.9}},
        {"true_ham_label": "mel", "predicted_class": "nv", "probabilities": {"p_mel": 0.8}},
        {"true_ham_label": "mel", "predicted_class": "bkl", "probabilities": {"p_mel": 0.7}},
    ]
    metrics = calculate_metrics(sample_rows)
    expected_matrix = [[1, 1, 1], [1, 1, 1]]
    if metrics["confusion_matrix"]["counts"] != expected_matrix:
        raise EvaluationError("Synthetic confusion-matrix check failed")
    if metrics["total_samples"] != 6 or not math.isclose(metrics["accuracy"], 1 / 3):
        raise EvaluationError("Synthetic accuracy check failed")
    if not math.isclose(metrics["balanced_accuracy"], 1 / 3):
        raise EvaluationError("Synthetic balanced-accuracy check failed")
    if not math.isclose(metrics["melanoma_auroc"], 1.0):
        raise EvaluationError("Synthetic melanoma AUROC check failed")
    return {"status": "PASS", "synthetic_confusion_auc_accuracy_checks": True}


def build_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def preprocess_ph2_image(image_path: Path):
    import cv2
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from preprocessing import preprocess_image

    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        rgb_array = np.asarray(rgb, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    processed_bgr = preprocess_image(bgr)
    success, encoded = cv2.imencode(".jpg", processed_bgr)
    if not success:
        raise EvaluationError(f"In-memory JPEG encoding failed: {image_path}")
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise EvaluationError(f"In-memory JPEG decoding failed: {image_path}")
    return Image.fromarray(cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB), mode="RGB")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_inference(audit: dict) -> dict:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    if not audit["runtime"].get("cuda_available") or "t4" not in audit["runtime"].get("gpu", "").lower():
        raise EvaluationError("Inference is restricted to the preflighted Colab Tesla T4")
    root = Path(audit["project_root"])
    checkpoint_path = Path(audit["checkpoint_path"])
    output_dir = Path(audit["output_dir"])
    analysis_dir = output_dir.parent
    initial_checkpoint_hash = sha256_file(checkpoint_path)
    initial_split_hash = sha256_file(Path(audit["split_path"]))
    if initial_checkpoint_hash != EXPECTED_CHECKPOINT_SHA256 or initial_split_hash != EXPECTED_SPLIT_SHA256:
        raise EvaluationError("Frozen checkpoint or split changed after preflight")
    if output_dir.exists():
        raise EvaluationError(f"Refusing to overwrite output directory: {output_dir}")

    sys.path.insert(0, str(root / "src"))
    from models.baseline_effnet import build_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model, _ = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to("cuda").eval()
    transform = build_transform()

    class FollowupDataset(Dataset):
        def __init__(self, manifest_rows):
            self.rows = manifest_rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            image_path = Path(audit["images_dir"]) / f"{row['image_id']}.bmp"
            image = preprocess_ph2_image(image_path)
            return transform(image), row["image_id"], row["ham_label"], row["external_label"]

    loader = DataLoader(
        FollowupDataset(audit["included_rows"]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    predictions = []
    with torch.inference_mode():
        for images, image_ids, ham_labels, external_labels in loader:
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(images.to("cuda", non_blocking=True))
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            predictions.extend(zip(image_ids, ham_labels, external_labels, probabilities))

    prediction_rows = []
    metric_rows = []
    for image_id, ham_label, external_label, probability in predictions:
        probability_map = {f"p_{label}": float(probability[index]) for index, label in enumerate(HAM_CLASSES)}
        predicted_class = HAM_CLASSES[int(np.argmax(probability))]
        output_row = {
            "image_id": image_id,
            "true_external_label": external_label,
            "true_ham_label": ham_label,
            "predicted_class": predicted_class,
            **probability_map,
        }
        prediction_rows.append(output_row)
        metric_rows.append({
            "true_ham_label": ham_label,
            "predicted_class": predicted_class,
            "probabilities": probability_map,
        })

    metrics = calculate_metrics(metric_rows)
    if metrics["total_samples"] != 120:
        raise EvaluationError(f"Expected 120 predicted cases; got {metrics['total_samples']}")

    after_checkpoint_hash = sha256_file(checkpoint_path)
    after_split_hash = sha256_file(Path(audit["split_path"]))
    if after_checkpoint_hash != EXPECTED_CHECKPOINT_SHA256 or after_split_hash != EXPECTED_SPLIT_SHA256:
        raise EvaluationError("Checkpoint or split hash changed during inference")
    if output_dir.exists():
        raise EvaluationError(f"Refusing to overwrite output directory: {output_dir}")

    stage = Path(tempfile.mkdtemp(prefix=".ph2_external_validation_exp5.", dir=analysis_dir))
    try:
        manifest_fields = [
            "image_id", "relative_image_path", "clinical_code",
            "true_external_label", "true_ham_label", "included",
        ]
        write_csv(stage / "ph2_manifest.csv", manifest_fields, audit["normalized_manifest_rows"])
        probability_columns = [f"p_{label}" for label in HAM_CLASSES]
        prediction_fields = [
            "image_id", "true_external_label", "true_ham_label", "predicted_class",
            *probability_columns,
        ]
        write_csv(stage / "ph2_predictions.csv", prediction_fields, prediction_rows)
        (stage / "ph2_metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        matrix = metrics["confusion_matrix"]["counts"]
        write_csv(
            stage / "confusion_matrix.csv",
            ["true_label", "predicted_nv", "predicted_mel", "predicted_other"],
            [
                {"true_label": "nv", "predicted_nv": matrix[0][0], "predicted_mel": matrix[0][1], "predicted_other": matrix[0][2]},
                {"true_label": "mel", "predicted_nv": matrix[1][0], "predicted_mel": matrix[1][1], "predicted_other": matrix[1][2]},
            ],
        )

        config = {
            "evaluation_type": "PH2 external follow-up evaluation",
            "checkpoint_path": str(checkpoint_path.relative_to(root)).replace("\\", "/"),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "selected_epoch": EXPECTED_EPOCH,
            "checkpoint_selection_rule": "validation_threshold_constrained_min_loss",
            "class_order": list(HAM_CLASSES),
            "cohort": {
                "total_ph2": 200,
                "included": 120,
                "included_ham_counts": {"nv": 80, "mel": 40},
                "excluded_atypical_nevus": 80,
                "mapping": PH2_TO_HAM,
            },
            "preprocessing": audit["preprocessing"],
            "inference": {
                "device": "cuda",
                "gpu": audit["runtime"]["gpu"],
                "model_eval": True,
                "inference_mode": True,
                "cuda_autocast": True,
                "batch_size": BATCH_SIZE,
                "shuffle": False,
                "softmax": "torch.softmax(logits.float(), dim=1)",
            },
            "metrics": metrics,
            "prior_ph2_use_limitation": FOLLOW_UP_LIMITATION,
            "model_policy": "Experiment #5 remains frozen; PH2 results will not modify it.",
            "provenance": {
                "status": "VERIFIED",
                "scope": "non-commercial academic research and external validation only",
                "citation_required": audit["provenance"]["intended_use"]["citation_required"],
                "commercial_use": False,
                "redistribution": False,
                "source_url": audit["provenance"]["dataset_url"],
                "revision": audit["provenance"]["revision"],
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torchvision": str(__import__("torchvision").__version__),
                "cuda": torch.version.cuda,
                "gpu": audit["runtime"]["gpu"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "training_performed": False,
            "fine_tuning_performed": False,
            "checkpoint_modified": False,
            "split_modified": False,
        }
        (stage / "external_validation_config.json").write_text(
            json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        report = f"""# PH2 External Follow-Up Evaluation: Experiment #5

## Scope and frozen model

This is a PH2 external follow-up evaluation of the frozen Controlled Experiment #5 checkpoint. It is not a completely untouched independent external validation. The Experiment #5 checkpoint and HAM split are required to remain unchanged, and PH2 results must not be used to modify the model.

{FOLLOW_UP_LIMITATION}

- Checkpoint SHA256: `{EXPECTED_CHECKPOINT_SHA256}`
- Frozen split SHA256: `{EXPECTED_SPLIT_SHA256}`
- Selected epoch: {EXPECTED_EPOCH}
- Class order: `{', '.join(HAM_CLASSES)}`

## Cohort and mapping

- Total PH2 records: 200
- Evaluated direct-overlap cases: 120 (80 common nevus to `nv`; 40 melanoma to `mel`)
- Excluded: 80 atypical nevi; no HAM label is assigned.
- No absent HAM classes are treated as evaluation failures.

## Preprocessing

The approved HAM-matched diagnostic transform was reused: original PH2 BMP decoded as RGB, converted to BGR, processed by the frozen Phase 4B hair-removal, Gray World, and multi-scale Retinex functions, JPEG-encoded/decoded in memory, converted to RGB, resized to 384x384, converted to tensor, and ImageNet-normalized. No augmentation, masks, ROI crops, test-time augmentation, or threshold tuning were used.

## Metrics

- Total samples: {metrics['total_samples']}
- Accuracy: {metrics['accuracy']:.6f}
- Balanced accuracy: {metrics['balanced_accuracy']:.6f}
- `nv` precision / recall / F1: {metrics['class_metrics']['nv']['precision']:.6f} / {metrics['class_metrics']['nv']['recall']:.6f} / {metrics['class_metrics']['nv']['f1']:.6f}
- `mel` precision / sensitivity-recall / F1: {metrics['class_metrics']['mel']['precision']:.6f} / {metrics['class_metrics']['mel']['recall']:.6f} / {metrics['class_metrics']['mel']['f1']:.6f}
- Macro precision / recall / F1 over `nv` and `mel`: {metrics['macro_precision']:.6f} / {metrics['macro_recall']:.6f} / {metrics['macro_f1']:.6f}
- Melanoma AUROC using `p_mel`: {metrics['melanoma_auroc'] if metrics['melanoma_auroc'] is not None else 'undefined'}

Confusion rows are true `[nv, mel]`; columns are predicted `[nv, mel, other]`. Predictions of `akiec`, `bcc`, `bkl`, `df`, or `vasc` are retained as `other` and count as incorrect. Metrics are restricted to externally represented `nv` and `mel`; no seven-class macro-F1 is reported.

## Limitations

PH2 was previously used for preprocessing diagnostics and an earlier model evaluation before Experiment #5. Interpret this as a follow-up evaluation with potential prior-data exposure, not as an untouched independent test. The cohort is small and limited to two mapped diagnoses. This evaluation does not establish clinical validity.

Experiment #5 is frozen and will not be modified based on these PH2 results. No PH2 training or fine-tuning is performed.
"""
        (stage / "external_validation_report.md").write_text(report, encoding="utf-8")
        if output_dir.exists():
            raise EvaluationError(f"Refusing to overwrite output directory: {output_dir}")
        os.replace(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    return {
        "status": "COMPLETE",
        "output_dir": str(output_dir),
        "total_samples": metrics["total_samples"],
        "prediction_rows": len(prediction_rows),
        "checkpoint_sha256_after": sha256_file(checkpoint_path),
        "split_sha256_after": sha256_file(Path(audit["split_path"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--run-inference", action="store_true")
    parser.add_argument("--require-t4", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/ph2_external_validation_exp5"),
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    try:
        audit = audit_project(root, output_dir, require_t4=args.require_t4)
        validate_metrics_implementation()
        if args.preflight_only:
            print(json.dumps({
                "preflight": "PASS",
                "project_root": audit["project_root"],
                "checkpoint_sha256": audit["checkpoint_sha256"],
                "split_sha256": audit["split_sha256"],
                "selected_epoch": audit["selected_epoch"],
                "cohort": audit["cohort"],
                "preprocessing": audit["preprocessing"],
                "output_safety": audit["output_safety"],
                "runtime": audit["runtime"],
                "previous_ph2_use_limitation": audit["follow_up_limitation"],
                "metrics_test": "PASS",
                "inference_performed": False,
            }, indent=2, allow_nan=False))
            return 0

        if not args.require_t4:
            raise EvaluationError("--run-inference requires --require-t4")
        result = run_inference(audit)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())