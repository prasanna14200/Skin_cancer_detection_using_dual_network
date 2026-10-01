"""Run the approved 12-case PH2/HAM Grad-CAM comparison on Colab/T4 only.

--preflight-only checks provenance/runtime/output safety without inference.
--run reproduces only the fixed 12 saved PH2 cases and generates maps only if
all top-1 and probability checks pass. It never trains or fine-tunes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_V2_S_Weights

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.dont_write_bytecode = True

PH2_RUNNER_DIR = ROOT / "experiments" / "ph2_external_eval_exp5"
HAM_EXPLAINABILITY_DIR = ROOT / "experiments" / "explainability"
for path in (str(PH2_RUNNER_DIR), str(HAM_EXPLAINABILITY_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from explainability.gradcam import gradcam  # noqa: E402
from models.baseline_effnet import build_model  # noqa: E402
from run_external_eval import (  # noqa: E402
    build_transform,
    preprocess_ph2_image,
    verify_preprocessing_contract,
)
from run_internal_gradcam import make_visuals, save_rgb  # noqa: E402

EXPECTED_CHECKPOINT_SHA256 = (
    "81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a"
)
EXPECTED_SPLIT_SHA256 = (
    "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
)
EXPECTED_EPOCH = 15
EXPECTED_SELECTION_RULE = "validation_threshold_constrained_min_loss"
CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
IMAGE_SIZE = 384
BATCH_SIZE = 16
REPRODUCTION_TOLERANCE = 5e-4
TARGET_LAYER_NAME = "model.features[-1]"
FOLLOW_UP_LIMITATION = (
    "PH² is treated as an external follow-up dataset and is not used for "
    "model selection, hyperparameter tuning, threshold optimization, or training."
)
CAUSAL_LIMITATION = (
    "Grad-CAM is an exploratory visualization of model sensitivity and does "
    "not establish causal reasoning or clinical validity."
)

APPROVED_CASES = {
    "IMD065": "correct melanoma",
    "IMD168": "correct melanoma",
    "IMD061": "melanoma to nv",
    "IMD063": "melanoma to nv",
    "IMD058": "melanoma to other",
    "IMD085": "melanoma to other",
    "IMD003": "correct nevus",
    "IMD009": "correct nevus",
    "IMD035": "nevus to melanoma",
    "IMD045": "nevus to melanoma",
    "IMD010": "nevus to other",
    "IMD020": "nevus to other",
}
EXPECTED_LABELS = {
    "IMD065": ("mel", "mel"),
    "IMD168": ("mel", "mel"),
    "IMD061": ("mel", "nv"),
    "IMD063": ("mel", "nv"),
    "IMD058": ("mel", "bkl"),
    "IMD085": ("mel", "bkl"),
    "IMD003": ("nv", "nv"),
    "IMD009": ("nv", "nv"),
    "IMD035": ("nv", "mel"),
    "IMD045": ("nv", "mel"),
    "IMD010": ("nv", "bkl"),
    "IMD020": ("nv", "bkl"),
}

EXPERIMENT_DIR = ROOT / "experiments" / "efficientnetv2s_controlled_exp5"
CHECKPOINT = EXPERIMENT_DIR / "best_checkpoint.pt"
CONFIG = EXPERIMENT_DIR / "config.json"
EXPERIMENT_MANIFEST = EXPERIMENT_DIR / "experiment_manifest.json"
SPLIT = ROOT / "data" / "splits" / "split_leakage_aware.csv"
PH2_ROOT = ROOT / "data" / "external" / "ph2"
PH2_MANIFEST = PH2_ROOT / "metadata" / "ph2_manifest.csv"
PH2_METADATA = PH2_ROOT / "metadata" / "PH2_dataset.txt"
PH2_PROVENANCE = PH2_ROOT / "metadata" / "provenance.json"
PH2_IMAGES = PH2_ROOT / "images"
PH2_EVAL_DIR = ROOT / "analysis" / "ph2_external_validation_exp5"
PH2_PREDICTIONS = PH2_EVAL_DIR / "ph2_predictions.csv"
PH2_EVAL_CONFIG = PH2_EVAL_DIR / "external_validation_config.json"
STAGE1_DIR = ROOT / "analysis" / "domain_shift_exp5"
HAM_EXPLAINABILITY_OUTPUT = ROOT / "analysis" / "explainability_exp5"
OUTPUT_DIR = STAGE1_DIR / "gradcam_comparison"


class Stage2Error(RuntimeError):
    """Raised when provenance, reproduction, or technical checks fail."""


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
        raise Stage2Error(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Stage2Error(f"Expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise Stage2Error(f"Could not read CSV {path}: {exc}") from exc


def require_tesla_t4() -> str:
    if not torch.cuda.is_available():
        raise Stage2Error("CUDA is unavailable; Stage 2 requires Colab Tesla T4")
    gpu_name = torch.cuda.get_device_name(0)
    if "t4" not in gpu_name.lower():
        raise Stage2Error(f"Expected Tesla T4, found {gpu_name}")
    return gpu_name


def validate_artifacts() -> dict:
    if ROOT.name != "EG-VAN":
        raise Stage2Error(f"Unexpected project root: {ROOT}")

    required = (
        CHECKPOINT,
        CONFIG,
        EXPERIMENT_MANIFEST,
        SPLIT,
        PH2_MANIFEST,
        PH2_METADATA,
        PH2_PROVENANCE,
        PH2_PREDICTIONS,
        PH2_EVAL_CONFIG,
        STAGE1_DIR / "domain_features.csv",
        STAGE1_DIR / "domain_feature_summary.csv",
        STAGE1_DIR / "prediction_shift_summary.json",
        STAGE1_DIR / "domain_shift_metrics.json",
        STAGE1_DIR / "domain_shift_stage1_report.md",
        HAM_EXPLAINABILITY_OUTPUT / "case_manifest.csv",
        HAM_EXPLAINABILITY_OUTPUT / "explainability_config.json",
        HAM_EXPLAINABILITY_OUTPUT / "qualitative_analysis.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise Stage2Error(f"Required frozen input/artifact files missing: {missing}")

    checkpoint_hash = sha256_file(CHECKPOINT)
    split_hash = sha256_file(SPLIT)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise Stage2Error(f"Checkpoint SHA256 mismatch: {checkpoint_hash}")
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise Stage2Error(f"Split SHA256 mismatch: {split_hash}")

    config = read_json(CONFIG)
    experiment_manifest = read_json(EXPERIMENT_MANIFEST)
    if config.get("classes") != list(CLASS_ORDER):
        raise Stage2Error("Experiment #5 config class order mismatch")
    if config.get("checkpoint_selection_metric") != EXPECTED_SELECTION_RULE:
        raise Stage2Error("Experiment #5 checkpoint-selection rule mismatch")
    if experiment_manifest.get("selected_epoch") != EXPECTED_EPOCH:
        raise Stage2Error("Experiment #5 selected epoch is not 15")
    if experiment_manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise Stage2Error("Experiment #5 manifest checkpoint hash mismatch")

    ph2_config = read_json(PH2_EVAL_CONFIG)
    if ph2_config.get("checkpoint_sha256") != checkpoint_hash:
        raise Stage2Error("Saved PH2 evaluation was not produced by the frozen Experiment #5 checkpoint")
    if ph2_config.get("split_sha256") != split_hash:
        raise Stage2Error("Saved PH2 evaluation split hash mismatch")
    if ph2_config.get("selected_epoch") != EXPECTED_EPOCH:
        raise Stage2Error("Saved PH2 evaluation selected epoch mismatch")
    if ph2_config.get("inference", {}).get("gpu") != "Tesla T4":
        raise Stage2Error("Saved PH2 evaluation runtime was not Tesla T4")
    if not ph2_config.get("preprocessing", {}).get("matches_diagnostic_config"):
        raise Stage2Error("Saved PH2 evaluation preprocessing did not match the diagnostic")
    preprocessing_audit = verify_preprocessing_contract(ROOT)
    if preprocessing_audit.get("status") != "PASS":
        raise Stage2Error("Frozen PH2 HAM-matched preprocessing audit failed")

    if OUTPUT_DIR.exists():
        existing = sorted(str(path.relative_to(OUTPUT_DIR)) for path in OUTPUT_DIR.rglob("*"))
        raise Stage2Error(f"Refusing to overwrite existing Stage 2 output path: {existing[:30]}")

    prediction_rows = read_csv(PH2_PREDICTIONS)
    prediction_by_id = {row["image_id"]: row for row in prediction_rows}
    if len(prediction_by_id) != len(prediction_rows):
        raise Stage2Error("Duplicate IDs in saved PH2 predictions")
    if not set(APPROVED_CASES).issubset(prediction_by_id):
        absent = sorted(set(APPROVED_CASES) - set(prediction_by_id))
        raise Stage2Error(f"Approved case IDs missing from saved PH2 predictions: {absent}")

    manifest_rows = read_csv(PH2_MANIFEST)
    manifest_by_id = {row["image_id"]: row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise Stage2Error("Duplicate IDs in PH2 manifest")
    if len(APPROVED_CASES) != 12:
        raise Stage2Error("Internal approved-case list must contain exactly 12 IDs")

    for image_id in APPROVED_CASES:
        row = prediction_by_id[image_id]
        manifest = manifest_by_id.get(image_id)
        image_path = PH2_IMAGES / f"{image_id}.bmp"
        if manifest is None or manifest.get("included", "").lower() != "true":
            raise Stage2Error(f"Approved case is absent/excluded in PH2 manifest: {image_id}")
        if not image_path.is_file():
            raise Stage2Error(f"Approved PH2 BMP missing: {image_path}")
        if row.get("true_ham_label") not in ("nv", "mel"):
            raise Stage2Error(f"Unexpected saved true label for {image_id}")
        if manifest.get("ham_label") != row.get("true_ham_label"):
            raise Stage2Error(f"PH2 manifest/prediction label mismatch for {image_id}")
        if row.get("predicted_class") not in CLASS_ORDER:
            raise Stage2Error(f"Unknown saved predicted class for {image_id}")
        if (row["true_ham_label"], row["predicted_class"]) != EXPECTED_LABELS[image_id]:
            raise Stage2Error(
                f"Saved PH2 labels do not match the approved outcome category for {image_id}: "
                f"{row['true_ham_label']} -> {row['predicted_class']}"
            )
        saved_probs = [float(row[f"p_{label}"]) for label in CLASS_ORDER]
        if not all(math.isfinite(value) for value in saved_probs):
            raise Stage2Error(f"Non-finite saved probability for {image_id}")
        if len(saved_probs) != 7 or abs(sum(saved_probs) - 1.0) > 1e-5:
            raise Stage2Error(f"Invalid saved probability vector for {image_id}")

    ham_config = read_json(HAM_EXPLAINABILITY_OUTPUT / "explainability_config.json")
    if ham_config.get("target_layer") != TARGET_LAYER_NAME:
        raise Stage2Error("Existing HAM Grad-CAM target layer does not match model.features[-1]")
    if ham_config.get("checkpoint_sha256") != checkpoint_hash:
        raise Stage2Error("Existing HAM Grad-CAM checkpoint hash mismatch")
    if ham_config.get("split_sha256") != split_hash:
        raise Stage2Error("Existing HAM Grad-CAM split hash mismatch")

    return {
        "root": ROOT,
        "checkpoint_hash": checkpoint_hash,
        "split_hash": split_hash,
        "prediction_by_id": prediction_by_id,
        "manifest_by_id": manifest_by_id,
        "ph2_config": ph2_config,
    }


def safe_load_checkpoint(path: Path) -> dict:
    version_module = getattr(torch, "torch_version", None)
    version_type = getattr(version_module, "TorchVersion", None)
    if version_type is not None:
        torch.serialization.add_safe_globals([version_type])
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise Stage2Error("Unexpected checkpoint structure")
    return value


class ApprovedPH2Dataset(Dataset):
    def __init__(self, image_ids: list[str], saved_rows: dict[str, dict], transform):
        self.image_ids = image_ids
        self.saved_rows = saved_rows
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        source_path = PH2_IMAGES / f"{image_id}.bmp"
        processed_image = preprocess_ph2_image(source_path)
        tensor = self.transform(processed_image)
        return tensor, image_id


def reproduce_approved_cases(model, transform, saved_rows, device):
    dataset = ApprovedPH2Dataset(sorted(APPROVED_CASES), saved_rows, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model.to(device).eval()
    reproduced = {}
    with torch.inference_mode():
        for images, image_ids in loader:
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            for image_id, probability in zip(image_ids, probabilities):
                reproduced[image_id] = [float(value) for value in probability]

    mismatches = []
    per_case = {}
    for image_id in sorted(APPROVED_CASES):
        new_probabilities = reproduced[image_id]
        saved = saved_rows[image_id]
        saved_probabilities = [float(saved[f"p_{label}"]) for label in CLASS_ORDER]
        new_predicted = CLASS_ORDER[max(range(7), key=new_probabilities.__getitem__)]
        max_delta = max(abs(a - b) for a, b in zip(new_probabilities, saved_probabilities))
        per_case[image_id] = {
            "saved_predicted_class": saved["predicted_class"],
            "reproduced_predicted_class": new_predicted,
            "max_abs_probability_difference": max_delta,
        }
        if new_predicted != saved["predicted_class"] or max_delta > REPRODUCTION_TOLERANCE:
            mismatches.append({
                "image_id": image_id,
                "saved_predicted_class": saved["predicted_class"],
                "reproduced_predicted_class": new_predicted,
                "max_abs_probability_difference": max_delta,
            })

    result = {
        "cases_checked": len(reproduced),
        "top1_matches": len(reproduced) - sum(
            item["saved_predicted_class"] != item["reproduced_predicted_class"]
            for item in per_case.values()
        ),
        "probability_tolerance": REPRODUCTION_TOLERANCE,
        "maximum_probability_difference": max(
            item["max_abs_probability_difference"] for item in per_case.values()
        ),
        "mismatches": mismatches,
        "per_case": per_case,
    }
    if mismatches:
        raise Stage2Error(
            "PH2_SELECTED_CASE_REPRODUCTION_FAILED " + json.dumps(result, indent=2)
        )
    return reproduced, result


def validate_cam(cam: np.ndarray, image_id: str, target_class: str) -> None:
    if cam.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise Stage2Error(f"Unexpected CAM size for {image_id}/{target_class}: {cam.shape}")
    if not np.isfinite(cam).all():
        raise Stage2Error(f"Non-finite CAM for {image_id}/{target_class}")
    if float(cam.min()) < 0.0 or float(cam.max()) > 1.0:
        raise Stage2Error(f"CAM outside [0, 1] for {image_id}/{target_class}")


def validate_written_image(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise Stage2Error(f"Expected image not written: {path}")
    with Image.open(path) as image:
        image.load()
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise Stage2Error(f"Unexpected saved image size for {path}: {image.size}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_stage2(audit: dict, gpu_name: str) -> dict:
    if OUTPUT_DIR.exists():
        raise Stage2Error(f"Refusing to overwrite Stage 2 output: {OUTPUT_DIR}")

    checkpoint = safe_load_checkpoint(CHECKPOINT)
    if checkpoint.get("epoch") != EXPECTED_EPOCH:
        raise Stage2Error(f"Checkpoint epoch mismatch: {checkpoint.get('epoch')}")
    if checkpoint.get("class_order") != list(CLASS_ORDER):
        raise Stage2Error("Checkpoint class order mismatch")

    model, _ = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to("cuda").eval()
    if not hasattr(model, "features") or not isinstance(model.features[-1], torch.nn.Module):
        raise Stage2Error("Expected model.features[-1] target layer")
    if model.classifier[-1].out_features != len(CLASS_ORDER):
        raise Stage2Error("Unexpected classifier output count")
    target_layer = model.features[-1]
    transform = build_transform()

    before_checkpoint_hash = sha256_file(CHECKPOINT)
    before_split_hash = sha256_file(SPLIT)
    if before_checkpoint_hash != EXPECTED_CHECKPOINT_SHA256 or before_split_hash != EXPECTED_SPLIT_SHA256:
        raise Stage2Error("Frozen hashes changed before selected-case reproduction")

    reproduced_probabilities, reproduction = reproduce_approved_cases(
        model, transform, audit["prediction_by_id"], torch.device("cuda")
    )
    if reproduction["cases_checked"] != len(APPROVED_CASES) or reproduction["mismatches"]:
        raise Stage2Error("Reproduction gate failed; no Grad-CAM was generated")

    analysis_dir = OUTPUT_DIR.parent
    stage_dir = Path(tempfile.mkdtemp(prefix=".gradcam_comparison.", dir=analysis_dir))
    ph2_cases_dir = stage_dir / "ph2_cases"
    ph2_cases_dir.mkdir()
    manifest_rows = []
    review_rows = []
    computed_predicted_maps = 0
    computed_true_maps = 0
    reused_true_maps = 0

    try:
        for image_id in sorted(APPROVED_CASES):
            saved = audit["prediction_by_id"][image_id]
            true_class = saved["true_ham_label"]
            predicted_class = saved["predicted_class"]
            true_index = CLASS_ORDER.index(true_class)
            predicted_index = CLASS_ORDER.index(predicted_class)
            same_target = true_class == predicted_class

            processed_image = preprocess_ph2_image(PH2_IMAGES / f"{image_id}.bmp")
            input_tensor = transform(processed_image).unsqueeze(0).to("cuda")
            display_image = processed_image.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                resample=Image.Resampling.BILINEAR,
            )
            original_rgb = np.asarray(display_image, dtype=np.uint8)
            if original_rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise Stage2Error(f"Unexpected processed input dimensions: {image_id}")

            case_dir = ph2_cases_dir / image_id
            case_dir.mkdir()
            input_path = case_dir / "input.png"
            save_rgb(input_path, original_rgb)
            validate_written_image(input_path)

            predicted_cam = gradcam(
                model, input_tensor, target_layer, predicted_index
            ).cpu().numpy()
            validate_cam(predicted_cam, image_id, predicted_class)
            predicted_heat, predicted_overlay = make_visuals(original_rgb, predicted_cam)
            predicted_heat_path = case_dir / "predicted_heatmap.png"
            predicted_overlay_path = case_dir / "predicted_overlay.png"
            save_rgb(predicted_heat_path, predicted_heat)
            save_rgb(predicted_overlay_path, predicted_overlay)
            validate_written_image(predicted_heat_path)
            validate_written_image(predicted_overlay_path)
            computed_predicted_maps += 1

            if same_target:
                true_heat_path = case_dir / "true_heatmap.png"
                true_overlay_path = case_dir / "true_overlay.png"
                shutil.copyfile(predicted_heat_path, true_heat_path)
                shutil.copyfile(predicted_overlay_path, true_overlay_path)
                reused_true_maps += 1
            else:
                true_cam = gradcam(model, input_tensor, target_layer, true_index).cpu().numpy()
                validate_cam(true_cam, image_id, true_class)
                true_heat, true_overlay = make_visuals(original_rgb, true_cam)
                true_heat_path = case_dir / "true_heatmap.png"
                true_overlay_path = case_dir / "true_overlay.png"
                save_rgb(true_heat_path, true_heat)
                save_rgb(true_overlay_path, true_overlay)
                computed_true_maps += 1
            validate_written_image(true_heat_path)
            validate_written_image(true_overlay_path)

            saved_probabilities = [float(saved[f"p_{label}"]) for label in CLASS_ORDER]
            repro_probabilities = reproduced_probabilities[image_id]
            confidence = max(saved_probabilities)
            p_mel = saved_probabilities[CLASS_ORDER.index("mel")]
            generated = [
                f"ph2_cases/{image_id}/input.png",
                f"ph2_cases/{image_id}/predicted_heatmap.png",
                f"ph2_cases/{image_id}/predicted_overlay.png",
                f"ph2_cases/{image_id}/true_heatmap.png",
                f"ph2_cases/{image_id}/true_overlay.png",
            ]
            reproduction_detail = reproduction["per_case"][image_id]
            manifest_rows.append({
                "dataset": "PH2",
                "image_id": image_id,
                "case_category": APPROVED_CASES[image_id],
                "true_class": true_class,
                "predicted_class": predicted_class,
                "correct": true_class == predicted_class,
                "confidence": confidence,
                "p_mel": p_mel,
                "saved_probabilities": json.dumps(saved_probabilities, separators=(",", ":")),
                "reproduced_probabilities": json.dumps(repro_probabilities, separators=(",", ":")),
                "max_abs_probability_difference": reproduction_detail["max_abs_probability_difference"],
                "predicted_target_class": predicted_class,
                "true_target_class": true_class,
                "true_equals_predicted": same_target,
                "true_map_reused": same_target,
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "split_sha256": EXPECTED_SPLIT_SHA256,
                "selected_epoch": EXPECTED_EPOCH,
                "target_layer": TARGET_LAYER_NAME,
                "preprocessing": "Frozen PH2 HAM-matched pipeline from experiments/ph2_external_eval_exp5/run_external_eval.py",
                "runtime": f"Python {platform.python_version()}, PyTorch {torch.__version__}, torchvision {torchvision.__version__}, CUDA {torch.version.cuda}",
                "gpu": gpu_name,
                "generated_files": json.dumps(generated, separators=(",", ":")),
            })
            review_rows.append({
                "dataset": "PH2",
                "image_id": image_id,
                "case_category": APPROVED_CASES[image_id],
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "p_mel": p_mel,
                "lesion_centered": "UNCERTAIN",
                "border_attention": "UNCERTAIN",
                "background_attention": "UNCERTAIN",
                "artifact_attention": "UNCERTAIN",
                "corner_or_frame_attention": "UNCERTAIN",
                "diffuse_attention": "UNCERTAIN",
                "ambiguous": "UNCERTAIN",
                "review_notes": "Pending human visual review; no automated qualitative judgment was made.",
            })

        manifest_fields = list(manifest_rows[0])
        write_csv(stage_dir / "ph2_gradcam_manifest.csv", manifest_fields, manifest_rows)
        write_csv(stage_dir / "gradcam_review.csv", list(review_rows[0]), review_rows)

        category_counts = dict(Counter(APPROVED_CASES.values()))
        summary = {
            "status": "COMPLETE",
            "analysis_type": "exploratory qualitative Grad-CAM comparison",
            "cases_requested": len(APPROVED_CASES),
            "cases_completed": len(manifest_rows),
            "case_categories": category_counts,
            "correct_cases": sum(row["correct"] for row in manifest_rows),
            "incorrect_cases": sum(not row["correct"] for row in manifest_rows),
            "target_layer": TARGET_LAYER_NAME,
            "predicted_class_maps_computed": computed_predicted_maps,
            "true_class_maps_computed": computed_true_maps,
            "true_maps_reused_when_target_equal": reused_true_maps,
            "reproduction_gate": reproduction,
            "invalid_maps": [],
            "ph2_full_dataset_inference_rerun": False,
            "training_or_finetuning": False,
            "threshold_tuning": False,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "follow_up_limitation": FOLLOW_UP_LIMITATION,
            "interpretation_limit": CAUSAL_LIMITATION,
        }
        (stage_dir / "gradcam_comparison_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

        report = f"""# Domain Shift Stage 2: HAM–PH² Grad-CAM Comparison

## Status

The fixed 12-case PH² subset passed the saved-prediction reproduction gate on the Colab Tesla T4 before Grad-CAM generation. This is exploratory visual review only; no causal interpretation has been made. The per-case review table intentionally remains `UNCERTAIN` until a human inspects the generated images.

{FOLLOW_UP_LIMITATION}

{CAUSAL_LIMITATION}

## Frozen model and method

- Experiment #5 checkpoint SHA256: `{EXPECTED_CHECKPOINT_SHA256}` (epoch {EXPECTED_EPOCH}).
- Frozen split SHA256: `{EXPECTED_SPLIT_SHA256}`.
- Target layer: `{TARGET_LAYER_NAME}`.
- Cases: exactly the 12 fixed IDs; no other PH² images were passed through the model.
- PH² input: existing HAM-matched preprocessing helper, then the same 384×384 ImageNet-normalized transform. No masks, ROI crops, augmentation, or threshold changes.
- Grad-CAM: existing `src/explainability/gradcam.py` utility and HAM overlay renderer. Correct cases reuse one target map for true and predicted labels.

## Reproduction gate

- Required absolute per-class probability tolerance: {REPRODUCTION_TOLERANCE}.
- Cases checked: {reproduction['cases_checked']}.
- Top-1 matches: {reproduction['top1_matches']}/{reproduction['cases_checked']}.
- Maximum absolute probability difference: {reproduction['maximum_probability_difference']:.10g}.
- Mismatches: {len(reproduction['mismatches'])}.

## Case maps and review

Each `ph2_cases/<image_id>/` directory contains the processed 384×384 input, predicted-class heatmap/overlay, and true-class heatmap/overlay. Correct cases use byte-identical copies for true/predicted targets and are marked in `ph2_gradcam_manifest.csv`. `gradcam_review.csv` is a blank qualitative review worksheet; the runner does not label attention regions automatically.

Compare these selected maps with existing HAM maps only descriptively and only where case/outcome groups are comparable. No lesion masks are used to calculate localization accuracy. The visualization does not establish causal reasoning or clinical validity.
"""
        (stage_dir / "gradcam_comparison_report.md").write_text(report, encoding="utf-8")
        readme = f"""# Stage 2 Grad-CAM Comparison

- Run type: exploratory, selected-case PH² follow-up comparison.
- Approved PH² cases completed: {len(manifest_rows)}.
- Reproduction tolerance: {REPRODUCTION_TOLERANCE}; gate passed before maps.
- Target layer: `{TARGET_LAYER_NAME}`.
- Training/fine-tuning: NO.
- Full PH² inference rerun: NO.
- Threshold tuning: NO.
- {FOLLOW_UP_LIMITATION}
- {CAUSAL_LIMITATION}

Complete `gradcam_review.csv` only after visually reviewing the saved case images. Do not treat qualitative entries as segmentation metrics.
"""
        (stage_dir / "README.md").write_text(readme, encoding="utf-8")

        png_paths = list(ph2_cases_dir.rglob("*.png"))
        expected_png_count = len(APPROVED_CASES) * 3 + computed_true_maps * 2 + reused_true_maps * 2
        if len(png_paths) != expected_png_count:
            raise Stage2Error(f"Unexpected PH2 image count: {len(png_paths)} != {expected_png_count}")
        if len(manifest_rows) != 12 or len(review_rows) != 12:
            raise Stage2Error("Expected exactly 12 completed cases and review rows")

        after_checkpoint_hash = sha256_file(CHECKPOINT)
        after_split_hash = sha256_file(SPLIT)
        if after_checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
            raise Stage2Error("Checkpoint changed during Stage 2")
        if after_split_hash != EXPECTED_SPLIT_SHA256:
            raise Stage2Error("Frozen split changed during Stage 2")
        if OUTPUT_DIR.exists():
            raise Stage2Error(f"Refusing to overwrite Stage 2 output: {OUTPUT_DIR}")
        os.replace(stage_dir, OUTPUT_DIR)
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        if args.project_root.resolve() != ROOT.resolve():
            raise Stage2Error(
                f"--project-root does not match runner location: {args.project_root.resolve()} != {ROOT.resolve()}"
            )
        audit = validate_artifacts()
        gpu_name = require_tesla_t4()
        preflight = {
            "status": "PASS",
            "project_root": str(ROOT),
            "checkpoint_sha256": audit["checkpoint_hash"],
            "split_sha256": audit["split_hash"],
            "selected_epoch": EXPECTED_EPOCH,
            "target_layer": TARGET_LAYER_NAME,
            "gpu": gpu_name,
            "approved_case_count": len(APPROVED_CASES),
            "approved_case_ids": sorted(APPROVED_CASES),
            "existing_ham_gradcam_cases": 24,
            "saved_ph2_prediction_rows": len(audit["prediction_by_id"]),
            "stage2_output_absent": not OUTPUT_DIR.exists(),
            "inference_performed": False,
        }
        if args.preflight_only:
            print(json.dumps(preflight, indent=2, allow_nan=False))
            return 0

        summary = run_stage2(audit, gpu_name)
        print(json.dumps({"preflight": preflight, "stage2": summary}, indent=2, allow_nan=False))
        print("GRAD-CAM COMPARISON STATUS: PASS")
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "BLOCKED",
            "reason": f"{type(exc).__name__}: {exc}",
            "gradcam_generated": OUTPUT_DIR.exists(),
            "training_performed": False,
            "fine_tuning_performed": False,
            "threshold_tuning": False,
        }, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())