"""Generate Grad-CAM only for preapproved frozen Experiment #5 validation cases.

This script verifies prediction reproduction before writing any output. It never
trains, changes model weights, accesses PH2, or evaluates unapproved image IDs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from torchvision.models import EfficientNet_V2_S_Weights
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.dont_write_bytecode = True
from models.baseline_effnet import build_model  # noqa: E402
from explainability.gradcam import gradcam  # noqa: E402

EXPERIMENT_DIR = ROOT / "experiments" / "efficientnetv2s_controlled_exp5"
OUTPUT_DIR = ROOT / "analysis" / "explainability_exp5"
CHECKPOINT = EXPERIMENT_DIR / "best_checkpoint.pt"
CONFIG = EXPERIMENT_DIR / "config.json"
PREDICTIONS = EXPERIMENT_DIR / "validation_predictions.csv"
SPLIT = ROOT / "data" / "splits" / "split_leakage_aware.csv"
PROCESSED_IMAGES = ROOT / "data" / "processed" / "images"
EXPECTED_CHECKPOINT_SHA256 = "81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a"
EXPECTED_SPLIT_SHA256 = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
EXPECTED_EPOCH = 15
EXPECTED_RULE = "validation_threshold_constrained_min_loss"
REPRODUCTION_TOLERANCE = 5e-4
IMAGE_SIZE = 384
CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
APPROVED_CASES = {
    "ISIC_0024449": "correctly classified melanoma",
    "ISIC_0024482": "correctly classified melanoma",
    "ISIC_0024898": "melanoma false negative predicted nv",
    "ISIC_0025018": "melanoma false negative predicted nv",
    "ISIC_0024406": "melanoma false positive from nv",
    "ISIC_0024620": "melanoma false positive from nv",
    "ISIC_0026301": "melanoma false positive from bkl",
    "ISIC_0026675": "melanoma false positive from bkl",
    "ISIC_0024321": "correctly classified nevus",
    "ISIC_0024322": "correctly classified nevus",
    "ISIC_0024646": "correctly classified akiec",
    "ISIC_0024843": "correctly classified akiec",
    "ISIC_0024573": "correctly classified bcc",
    "ISIC_0024595": "correctly classified bcc",
    "ISIC_0024358": "correctly classified bkl",
    "ISIC_0024760": "correctly classified bkl",
    "ISIC_0028651": "correctly classified df",
    "ISIC_0029130": "correctly classified df",
    "ISIC_0024402": "correctly classified vasc",
    "ISIC_0026393": "correctly classified vasc",
    "ISIC_0025275": "other-class error bkl to nv",
    "ISIC_0025986": "other-class error bkl to nv",
    "ISIC_0024411": "other-class error bcc to nv",
    "ISIC_0024432": "other-class error bcc to nv",
}
PROBABILITY_COLUMNS = tuple(f"prob_{label}" for label in CLASS_ORDER)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def safe_load_checkpoint(path: Path) -> dict:
    version_type = getattr(torch.torch_version, "TorchVersion", None)
    if version_type is not None:
        torch.serialization.add_safe_globals([version_type])
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("Unexpected checkpoint structure")
    return value


def load_rows() -> tuple[dict[str, dict], dict[str, str]]:
    with PREDICTIONS.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    prediction_rows = {row["image_id"]: row for row in rows}
    if len(prediction_rows) != len(rows):
        raise ValueError("Duplicate IDs in saved validation predictions")
    if not set(APPROVED_CASES).issubset(prediction_rows):
        raise ValueError(f"Approved IDs absent from validation predictions: {sorted(set(APPROVED_CASES) - set(prediction_rows))}")
    with SPLIT.open("r", newline="", encoding="utf-8") as handle:
        split_rows = {row["image_id"]: row for row in csv.DictReader(handle)}
    return prediction_rows, split_rows


def validate_frozen_state() -> tuple[dict, dict, dict[str, dict], dict[str, str], dict, object, object]:
    for required in (CHECKPOINT, CONFIG, PREDICTIONS, SPLIT):
        if not required.is_file():
            raise FileNotFoundError(required)
    checkpoint_hash = sha256_file(CHECKPOINT)
    split_hash = sha256_file(SPLIT)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint_hash}")
    if split_hash != EXPECTED_SPLIT_SHA256:
        raise ValueError(f"Split hash mismatch: {split_hash}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Explainability output directory already has content: {OUTPUT_DIR}")

    config = read_json(CONFIG)
    if config.get("model") != "EfficientNetV2S" or config.get("classes") != list(CLASS_ORDER):
        raise ValueError("Experiment #5 model or class order mismatch")
    if config.get("checkpoint_selection_metric") != EXPECTED_RULE or int(config.get("epochs", -1)) != 25:
        raise ValueError("Experiment #5 selection rule/epoch budget mismatch")
    if config.get("split_sha256") != EXPECTED_SPLIT_SHA256:
        raise ValueError("Experiment #5 config split hash mismatch")

    checkpoint = safe_load_checkpoint(CHECKPOINT)
    if int(checkpoint.get("epoch", -1)) != EXPECTED_EPOCH:
        raise ValueError(f"Checkpoint selected epoch mismatch: {checkpoint.get('epoch')}")
    if checkpoint.get("class_order") != list(CLASS_ORDER):
        raise ValueError("Checkpoint class order mismatch")
    manifest = read_json(EXPERIMENT_DIR / "experiment_manifest.json")
    if manifest.get("checkpoint_sha256") != checkpoint_hash or int(manifest.get("selected_epoch", -1)) != EXPECTED_EPOCH:
        raise ValueError("Final model manifest does not match Experiment #5 checkpoint")

    prediction_rows, split_rows = load_rows()
    missing_images = [case for case in APPROVED_CASES if not (PROCESSED_IMAGES / f"{case}.jpg").is_file()]
    if missing_images:
        raise FileNotFoundError(f"Approved processed images missing: {missing_images}")
    wrong_split = [case for case in APPROVED_CASES if split_rows.get(case, {}).get("split") != "val"]
    if wrong_split:
        raise ValueError(f"Approved examples must all come from validation split: {wrong_split}")
    for image_id in APPROVED_CASES:
        saved = prediction_rows[image_id]
        source = split_rows[image_id]
        if saved["true_label"] != source["dx"]:
            raise ValueError(f"Prediction true label mismatches frozen split for {image_id}")
        if saved["true_label"] not in CLASS_ORDER or saved["predicted_label"] not in CLASS_ORDER:
            raise ValueError(f"Unknown class for {image_id}")
        expected_correct = saved["true_label"] == saved["predicted_label"]
        if (saved["correct"].lower() == "true") != expected_correct:
            raise ValueError(f"Saved correct flag mismatch for {image_id}")
        probabilities = json.loads(saved["probabilities"])
        if len(probabilities) != len(CLASS_ORDER) or abs(sum(probabilities) - 1.0) > 1e-5:
            raise ValueError(f"Invalid saved probability vector for {image_id}")

    model, _ = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    if not hasattr(model, "features") or not isinstance(model.features[-1], torch.nn.Module):
        raise ValueError("Expected EfficientNetV2-S final spatial block at model.features[-1]")
    if model.classifier[-1].in_features != 1280 or model.classifier[-1].out_features != 7:
        raise ValueError("Unexpected EfficientNetV2-S classifier shape")
    target_layer = model.features[-1]
    normalization = EfficientNet_V2_S_Weights.DEFAULT.transforms()
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor(),
        Normalize(mean=normalization.mean, std=normalization.std),
    ])
    return config, checkpoint, prediction_rows, split_rows, manifest, model, (target_layer, transform, checkpoint_hash, split_hash)


def reproduce_approved_cases(model: torch.nn.Module, transform, prediction_rows: dict[str, dict], device: torch.device) -> tuple[dict[str, dict], dict]:
    reproduced = {}
    max_abs_delta = 0.0
    mismatches = []
    model.to(device).eval()
    with torch.inference_mode():
        for image_id in sorted(APPROVED_CASES):
            image = Image.open(PROCESSED_IMAGES / f"{image_id}.jpg").convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            probabilities = torch.softmax(model(tensor).float(), dim=1)[0].cpu().tolist()
            predicted = CLASS_ORDER[max(range(len(probabilities)), key=probabilities.__getitem__)]
            saved = prediction_rows[image_id]
            saved_probabilities = [float(value) for value in json.loads(saved["probabilities"])]
            delta = max(abs(a - b) for a, b in zip(probabilities, saved_probabilities))
            max_abs_delta = max(max_abs_delta, delta)
            reproduced[image_id] = {"predicted_class": predicted, "probabilities": probabilities, "max_abs_probability_delta": delta}
            if predicted != saved["predicted_label"] or delta > REPRODUCTION_TOLERANCE:
                mismatches.append({"image_id": image_id, "saved_class": saved["predicted_label"],
                                   "reproduced_class": predicted, "max_abs_probability_delta": delta})
    result = {
        "approved_case_count": len(APPROVED_CASES),
        "predicted_classes_match": not any(x["saved_class"] != x["reproduced_class"] for x in mismatches),
        "probabilities_within_tolerance": not any(x["max_abs_probability_delta"] > REPRODUCTION_TOLERANCE for x in mismatches),
        "maximum_absolute_probability_difference": max_abs_delta,
        "absolute_probability_tolerance": REPRODUCTION_TOLERANCE,
        "mismatches": mismatches,
    }
    if mismatches:
        raise RuntimeError("APPROVED_CASE_REPRODUCTION_FAILED " + json.dumps(result, indent=2))
    return reproduced, result


def save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def make_visuals(input_rgb: np.ndarray, cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(cam, (input_rgb.shape[1], input_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    normalized = np.clip(resized, 0.0, 1.0)
    heat_bgr = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(input_rgb, 0.55, heat_rgb, 0.45, 0)
    return heat_rgb, overlay


def generate_case(model, target_layer, transform, image_id: str, saved: dict, reproduced: dict, device: torch.device) -> dict:
    image = Image.open(PROCESSED_IMAGES / f"{image_id}.jpg").convert("RGB")
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR)
    tensor = transform(image).unsqueeze(0).to(device)
    true_class = saved["true_label"]
    predicted_class = saved["predicted_label"]
    targets = [("predicted_class", predicted_class)]
    if true_class != predicted_class:
        targets.append(("true_class", true_class))
    case_dir = OUTPUT_DIR / "images" / image_id
    case_dir.mkdir(parents=True, exist_ok=False)
    input_rgb = np.asarray(resized, dtype=np.uint8)
    save_rgb(case_dir / "input.png", input_rgb)
    generated = [str((case_dir / "input.png").relative_to(OUTPUT_DIR)).replace("\\", "/")]
    for suffix, label in targets:
        cam = gradcam(model, tensor, target_layer, CLASS_ORDER.index(label)).cpu().numpy()
        heat, overlay = make_visuals(input_rgb, cam)
        heat_path = case_dir / f"{suffix}_heatmap.png"
        overlay_path = case_dir / f"{suffix}_overlay.png"
        save_rgb(heat_path, heat)
        save_rgb(overlay_path, overlay)
        generated.extend(str(path.relative_to(OUTPUT_DIR)).replace("\\", "/") for path in (heat_path, overlay_path))
    probs = [float(x) for x in json.loads(saved["probabilities"])]
    return {
        "image_id": image_id,
        "split": "val",
        "true_class": true_class,
        "predicted_class": predicted_class,
        "correct": true_class == predicted_class,
        "prediction_confidence": max(probs),
        **{f"prob_{label}": probs[i] for i, label in enumerate(CLASS_ORDER)},
        "selection_category_reason": APPROVED_CASES[image_id],
        "gradcam_target_layer": "model.features[-1]",
        "generated_files": generated,
        "reproduction_max_abs_probability_delta": reproduced[image_id]["max_abs_probability_delta"],
    }


def main() -> int:
    before_checkpoint_hash = sha256_file(CHECKPOINT)
    before_split_hash = sha256_file(SPLIT)
    config, checkpoint, prediction_rows, split_rows, final_manifest, model, layer_info = validate_frozen_state()
    target_layer, transform, checkpoint_hash, split_hash = layer_info
    if checkpoint_hash != before_checkpoint_hash or split_hash != before_split_hash:
        raise RuntimeError("Frozen hashes changed during preflight")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reproduced, reproduction = reproduce_approved_cases(model, transform, prediction_rows, device)
    # The output directory is created only after every approved prediction reproduces.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    manifest_rows = []
    for image_id in sorted(APPROVED_CASES):
        manifest_rows.append(generate_case(model, target_layer, transform, image_id,
                                           prediction_rows[image_id], reproduced[image_id], device))

    fieldnames = ["image_id", "split", "true_class", "predicted_class", "correct", "prediction_confidence",
                  *[f"prob_{label}" for label in CLASS_ORDER], "selection_category_reason",
                  "gradcam_target_layer", "generated_files", "reproduction_max_abs_probability_delta"]
    with (OUTPUT_DIR / "case_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({**row, "generated_files": json.dumps(row["generated_files"], separators=(",", ":"))})

    versions = {"python": platform.python_version(), "torch": str(torch.__version__),
                "torchvision": __import__("torchvision").__version__, "numpy": np.__version__,
                "opencv": cv2.__version__, "pillow": PIL.__version__}
    explainability_config = {
        "status": "COMPLETE",
        "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_hash,
        "split_path": str(SPLIT.relative_to(ROOT)).replace("\\", "/"),
        "split_sha256": split_hash,
        "selected_epoch": EXPECTED_EPOCH,
        "selection_rule": EXPECTED_RULE,
        "model_architecture": "torchvision EfficientNetV2S with seven-class linear classifier",
        "class_order": list(CLASS_ORDER),
        "preprocessing": {
            "source_images": "already processed HAM10000 data/processed/images; no reprocessing",
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "to_tensor": True,
            "normalization_mean": [float(value) for value in transform.transforms[-1].mean],
            "normalization_std": [float(value) for value in transform.transforms[-1].std],
            "augmentation": "none; evaluation transform",
        },
        "gradcam_method": "Grad-CAM: spatial mean of gradients as channel weights; weighted activation sum; ReLU; bilinear resize; min-max normalize",
        "target_layer": "model.features[-1]",
        "approved_image_ids": sorted(APPROVED_CASES),
        "selection_method": "fixed preapproved IDs from validation_predictions.csv; deterministic sorted-ID selection; no visual-based selection",
        "reproduction_check": reproduction,
        "device": str(device),
        "software_versions": versions,
        "ph2_accessed": False,
        "training_or_finetuning": False,
    }
    (OUTPUT_DIR / "explainability_config.json").write_text(json.dumps(explainability_config, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    category_counts = {}
    for reason in APPROVED_CASES.values():
        category_counts[reason] = category_counts.get(reason, 0) + 1
    summary = {
        "status": "COMPLETE",
        "case_count_requested": len(APPROVED_CASES),
        "case_count_completed": len(manifest_rows),
        "prediction_reproduction": reproduction,
        "categories": {
            "correct_melanoma": {"count": 2, "observation": "manual_review_required; no automated lesion-region judgment made"},
            "melanoma_to_nevus": {"count": 2, "observation": "manual_review_required; no automated lesion-region judgment made"},
            "nevus_or_bkl_to_melanoma": {"count": 4, "observation": "manual_review_required; no automated lesion-region judgment made"},
            "correct_nevus": {"count": 2, "observation": "manual_review_required; no automated lesion-region judgment made"},
            "other_representative_classes_and_errors": {"count": 14, "observation": "manual_review_required; no automated lesion-region judgment made"},
        },
        "class_reason_counts": category_counts,
        "interpretation_limit": "Grad-CAM is qualitative and does not prove causal model reasoning or clinical validity.",
        "ph2_accessed": False,
    }
    (OUTPUT_DIR / "explainability_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    readme = f"""# Experiment #5 Internal Grad-CAM Review

- Frozen checkpoint SHA256: `{checkpoint_hash}` (epoch {EXPECTED_EPOCH})
- Target layer: `model.features[-1]`
- Approved validation cases processed: {len(manifest_rows)}
- Saved-prediction reproduction: PASS; maximum absolute probability delta {reproduction['maximum_absolute_probability_difference']:.8g}, tolerance {REPRODUCTION_TOLERANCE}
- Training/fine-tuning: NO
- PH2 accessed: NO

Cases were selected from the preapproved validation prediction IDs and categories before examining maps. Files contain the processed model input, class-targeted heatmaps, and overlays. Misclassified cases have separate predicted-class and true-class maps; correct cases have a single map because the classes coincide.

All activation-pattern conclusions require manual review. This output is qualitative and does not establish causal model reasoning, clinical validity, or model-selection evidence.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    after_checkpoint_hash = sha256_file(CHECKPOINT)
    after_split_hash = sha256_file(SPLIT)
    if after_checkpoint_hash != before_checkpoint_hash or after_split_hash != before_split_hash:
        raise RuntimeError("Frozen checkpoint or split hash changed during explainability")
    unexpected = set(path.name for path in (OUTPUT_DIR / "images").iterdir()) - set(APPROVED_CASES)
    if unexpected or len(manifest_rows) != len(APPROVED_CASES):
        raise RuntimeError(f"Unexpected/missing explainability cases: unexpected={sorted(unexpected)}")
    print(json.dumps({
        "status": "COMPLETE",
        "checkpoint_sha256": after_checkpoint_hash,
        "split_sha256": after_split_hash,
        "selected_epoch": EXPECTED_EPOCH,
        "target_layer": "model.features[-1]",
        "prediction_reproduction": reproduction,
        "cases_requested": len(APPROVED_CASES),
        "cases_completed": len(manifest_rows),
        "ph2_accessed": False,
        "output_directory": str(OUTPUT_DIR),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
