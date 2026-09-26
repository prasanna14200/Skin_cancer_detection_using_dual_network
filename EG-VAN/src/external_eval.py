"""Fail-fast, inference-only PH2 evaluation for a frozen HAM10000 checkpoint.

The default CLI dry-run audits source labels, images, checkpoint, architecture,
transform definition, provenance status, and CUDA without running inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

PH2_LABEL_TO_HAM = {
    "common nevus": "nv",
    "melanoma": "mel",
    "atypical nevus": None,
}
PH2_EXPECTED_COUNTS = {"common nevus": 80, "atypical nevus": 80, "melanoma": 40}
HAM_CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
PH2_CLASSES = ("nv", "mel")
PREDICTION_COLUMNS = ("nv", "mel", "other")
IMAGE_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AuditError(ValueError):
    """Raised when PH2 inputs fail a strict pre-inference audit."""


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def map_ph2_label(label: str) -> str | None:
    """Map only direct overlap; return None for the explicitly excluded class."""
    normalized = normalize_label(label)
    if normalized not in PH2_LABEL_TO_HAM:
        raise AuditError(f"Unexpected PH2 clinical diagnosis: {label!r}")
    return PH2_LABEL_TO_HAM[normalized]


def read_clinical_labels(path: str | Path) -> dict[str, str]:
    """Read IMD case ID and clinical diagnosis code from PH2_dataset.txt."""
    label_codes = {"0": "common nevus", "1": "atypical nevus", "2": "melanoma"}
    labels: dict[str, str] = {}
    pattern = re.compile(r"^\|\|\s*(IMD\d+)\s*\|\|(.*?)\|\|\s*([^|]+?)\s*\|\|")
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        image_id, _, code = match.groups()
        code = code.strip()
        if code not in label_codes:
            raise AuditError(f"Unexpected clinical diagnosis code {code!r} for {image_id}")
        if image_id in labels:
            raise AuditError(f"Duplicate metadata image ID: {image_id}")
        labels[image_id] = label_codes[code]
    if not labels:
        raise AuditError(f"No PH2 clinical labels parsed from {path}")
    return labels


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "external_label", "ham_label", "included"}
    if not rows or not required.issubset(rows[0]):
        raise AuditError(f"Manifest must contain columns {sorted(required)}")
    return rows


def audit_manifest_rows(
    rows: list[dict[str, str]],
    clinical_labels: dict[str, str],
    images_dir: str | Path | None = None,
    expected_total: int | None = None,
    expected_class_counts: dict[str, int] | None = None,
) -> dict:
    """Validate metadata, approved mappings, uniqueness, and optional image files."""
    if expected_total is not None and len(rows) != expected_total:
        raise AuditError(f"Expected {expected_total} PH2 rows; found {len(rows)}")
    image_ids = [row["image_id"].strip() for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise AuditError("Duplicate image IDs found in PH2 manifest")
    if set(image_ids) != set(clinical_labels):
        missing_labels = sorted(set(image_ids) - set(clinical_labels))
        missing_manifest = sorted(set(clinical_labels) - set(image_ids))
        raise AuditError(f"Manifest/metadata ID mismatch; missing_labels={missing_labels[:5]}, missing_manifest_rows={missing_manifest[:5]}")

    counts = Counter()
    included_rows = []
    excluded_rows = []
    missing_images = []
    for row, image_id in zip(rows, image_ids):
        external_label = normalize_label(row["external_label"])
        actual_label = normalize_label(clinical_labels[image_id])
        if external_label != actual_label:
            raise AuditError(f"Manifest/source label mismatch for {image_id}: {external_label!r} != {actual_label!r}")
        expected_ham = map_ph2_label(external_label)
        ham_label = row["ham_label"].strip().lower()
        included = row["included"].strip().lower() == "true"
        if expected_ham is None:
            if ham_label or included:
                raise AuditError(f"Atypical nevus must be excluded with no HAM label: {image_id}")
            excluded_rows.append(row)
        else:
            if ham_label != expected_ham or not included:
                raise AuditError(f"Invalid approved mapping for {image_id}: expected {expected_ham}")
            included_rows.append(row)
        counts[external_label] += 1
        if images_dir is not None and not (Path(images_dir) / f"{image_id}.bmp").is_file():
            missing_images.append(image_id)
    if expected_class_counts is not None and dict(counts) != expected_class_counts:
        raise AuditError(f"Unexpected PH2 class counts: {dict(counts)}")
    if missing_images:
        raise AuditError(f"Missing PH2 original images for IDs: {missing_images[:10]}")
    if {row["ham_label"].strip().lower() for row in included_rows} != set(PH2_CLASSES):
        raise AuditError("Included labels must be exactly nv and mel")
    return {
        "total_images": len(rows),
        "class_counts": dict(counts),
        "included_count": len(included_rows),
        "excluded_atypical_count": len(excluded_rows),
        "included_rows": included_rows,
        "excluded_rows": excluded_rows,
        "missing_images": missing_images,
    }


def preprocessing_config() -> dict:
    return {
        "resize": [IMAGE_SIZE, IMAGE_SIZE],
        "resize_policy": "torchvision transforms.Resize((384, 384))",
        "to_tensor": True,
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "augmentation": "none; evaluation only",
        "phase4b_rerun": False,
    }


def build_eval_transform():
    """Reproduce train.py evaluation transform, independent of pretrained weights object."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def binary_confusion_matrix(actual: Iterable[str], predicted: Iterable[str]) -> list[list[int]]:
    """Return actual rows (nv,mel) by predicted columns (nv,mel,other).

    Every out-of-overlap seven-class prediction is retained as `other`; it is
    never dropped from the denominator or silently forced into nv/mel.
    """
    actual_values, predicted_values = list(actual), list(predicted)
    if len(actual_values) != len(predicted_values):
        raise ValueError("Actual and predicted lengths differ")
    matrix = [[0, 0, 0], [0, 0, 0]]
    row_index = {"nv": 0, "mel": 1}
    col_index = {"nv": 0, "mel": 1}
    for true_label, pred_label in zip(actual_values, predicted_values):
        if true_label not in row_index:
            raise AuditError(f"Unexpected binary target: {true_label}")
        if pred_label not in HAM_CLASSES:
            raise AuditError(f"Unexpected seven-class model prediction: {pred_label}")
        predicted_overlap = pred_label if pred_label in col_index else "other"
        matrix[row_index[true_label]][col_index.get(predicted_overlap, 2)] += 1
    return matrix


def binary_metrics(matrix: list[list[int]]) -> dict:
    """Treat `other` and opposite-class outputs as errors for the binary task.

    Positive class is melanoma. NV is negative. Any other seven-class output is
    explicitly counted as a false positive for true NV or false negative for
    true melanoma, while remaining visible in the 2x3 confusion matrix.
    """
    if len(matrix) != 2 or any(len(row) != 3 for row in matrix):
        raise ValueError("Expected a 2x3 matrix with predicted columns nv,mel,other")
    tn = matrix[0][0]
    fp = matrix[0][1] + matrix[0][2]
    fn = matrix[1][0] + matrix[1][2]
    tp = matrix[1][1]
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "positive_class": "mel",
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "total": total,
        "other_prediction_policy": "Out-of-overlap HAM10000 predictions count as errors and remain a separate predicted-other column in the 2x3 matrix.",
    }


def checkpoint_audit(checkpoint_path: Path, project_root: Path) -> dict:
    import torch
    if not checkpoint_path.is_file():
        raise AuditError(f"Checkpoint missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint or "config" not in checkpoint:
        raise AuditError("Checkpoint must contain model_state and config")
    config = checkpoint["config"]
    if config.get("model") != "EfficientNetV2S":
        raise AuditError(f"Unexpected checkpoint model: {config.get('model')}")
    if tuple(config.get("classes", ())) != HAM_CLASSES:
        raise AuditError(f"Unexpected checkpoint class list: {config.get('classes')}")
    if config.get("image_size") != IMAGE_SIZE:
        raise AuditError(f"Unexpected checkpoint input size: {config.get('image_size')}")
    state = checkpoint["model_state"]
    weight_shape = tuple(state.get("classifier.1.weight", torch.empty(0)).shape)
    bias_shape = tuple(state.get("classifier.1.bias", torch.empty(0)).shape)
    if weight_shape != (7, 1280) or bias_shape != (7,):
        raise AuditError(f"Unexpected classifier state shapes: {weight_shape}, {bias_shape}")

    # Instantiate the repository architecture and strict-load the checkpoint;
    # this is a structural model check and does not run a forward pass.
    try:
        sys.path.insert(0, str(project_root / "src"))
        from models.baseline_effnet import build_model
        model, _ = build_model(pretrained=False)
        model.load_state_dict(state, strict=True)
        if model.classifier[-1].out_features != 7:
            raise AuditError("Constructed baseline model does not output seven classes")
    except Exception as exc:
        raise AuditError(f"Checkpoint does not load strictly into repository EfficientNetV2S: {exc}") from exc
    return {
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "model": config["model"],
        "classes": list(config["classes"]),
        "classifier_weight_shape": list(weight_shape),
        "classifier_bias_shape": list(bias_shape),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "strict_architecture_load": True,
        "preprocessing": preprocessing_config(),
    }


def provenance_status(path: Path) -> dict:
    if not path.is_file():
        return {"status": "PARTIALLY VERIFIED", "record": None, "missing": ["source URL/name", "uploader/revision", "license/access permission", "traceability evidence"]}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "PARTIALLY VERIFIED", "record": None, "missing": [f"unreadable provenance record: {exc}"]}

    missing = []
    if str(record.get("status", "")).upper() != "VERIFIED":
        missing.append("status must be VERIFIED")

    kaggle = record.get("kaggle_secondary_source", {})
    if (
        kaggle.get("url") != "https://www.kaggle.com/datasets/spacesurfer/ph2-dataset"
        or kaggle.get("dataset_id") != 6220095
        or kaggle.get("ref") != "spacesurfer/ph2-dataset"
        or kaggle.get("uploader") != "Dmitrii K (spacesurfer)"
        or kaggle.get("version") != 2
    ):
        missing.append("Kaggle secondary-source identity is incomplete or does not match the audited dataset")
    if (
        record.get("dataset_url") != kaggle.get("url")
        or record.get("dataset_name") != "PH2 Dataset"
        or record.get("uploader") != kaggle.get("uploader")
        or record.get("revision") != "Kaggle dataset version 2"
    ):
        missing.append("top-level dataset identity must match the documented Kaggle source")
    if kaggle.get("license_field") != "Other (specified in description)" or kaggle.get("license_url") is not None:
        missing.append("Kaggle license metadata must remain recorded as Other with no invented license URL")

    original = record.get("original_dataset_provenance", {})
    terms = original.get("access_reuse_evidence", {})
    quotations = terms.get("quotations", [])
    has_research_terms = any("research and educational purposes" in str(quote).lower() for quote in quotations)
    has_commercial_restriction = any("commercial use is not allowed" in str(quote).lower() for quote in quotations)
    has_redistribution_restriction = any("redistribution" in str(quote).lower() and "not allowed" in str(quote).lower() for quote in quotations)
    if (
        original.get("source_url") != "https://www.fc.up.pt/addi/ph2%20database.html"
        or str(original.get("status", "")).upper() != "DOCUMENTED"
        or not has_research_terms
        or not has_commercial_restriction
        or not has_redistribution_restriction
        or terms.get("commercial_use_permitted") is not False
        or terms.get("redistribution_permitted") is not False
        or not terms.get("citation_required")
    ):
        missing.append("authoritative PH2 research/educational terms and their restrictions are not fully documented")

    traceability = record.get("source_traceability", {})
    trace_evidence = traceability.get("evidence", [])
    if (
        str(traceability.get("status", "")).upper() != "VERIFIED"
        or not isinstance(trace_evidence, list)
        or not trace_evidence
        or not traceability.get("integrity_report_sha256")
    ):
        missing.append("verified source-identity traceability and linked integrity report are required")
    else:
        report_ref = Path(traceability.get("integrity_report", ""))
        report_candidates = [path.parent / report_ref]
        if not report_ref.is_absolute():
            report_candidates.extend(
                repo_root / report_ref
                for repo_root in Path(__file__).resolve().parents[:3]
            )
        report_path = next((candidate for candidate in report_candidates if candidate.is_file()), None)
        if report_path is None:
            missing.append("linked local integrity report is missing")
        else:
            report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            if report_digest != traceability.get("integrity_report_sha256"):
                missing.append("linked local integrity report SHA256 does not match provenance")
            else:
                try:
                    integrity_report = json.loads(report_path.read_text(encoding="utf-8"))
                    if (
                        integrity_report.get("C_local_file_integrity", {}).get("status") != "VERIFIED"
                        or not str(integrity_report.get("C_local_file_integrity", {}).get("source_traceability_status", "")).startswith("VERIFIED")
                        or integrity_report.get("D_metadata_label_integrity", {}).get("status") != "VERIFIED"
                    ):
                        missing.append("linked local integrity report does not verify package and metadata concordance")
                except (OSError, json.JSONDecodeError, AttributeError) as exc:
                    missing.append(f"linked local integrity report is unreadable: {exc}")

    intended_use = record.get("intended_use", {})
    if (
        intended_use.get("scope") != "non-commercial academic research and external validation"
        or intended_use.get("commercial_use") is not False
        or intended_use.get("redistribution") is not False
    ):
        missing.append("intended use must be limited to non-commercial academic research without redistribution")

    if record.get("explicit_permission_verified") is not False:
        missing.append("do not claim an explicit license grant that the evidence does not establish")

    if missing:
        return {"status": "PARTIALLY VERIFIED", "record": record, "missing": missing}
    warnings = []
    if not record.get("download_date"):
        warnings.append("download date is not recorded; this does not replace the documented source-identity evidence")
    return {
        "status": "VERIFIED",
        "record": record,
        "missing": [],
        "authorization_scope": intended_use["scope"],
        "warnings": warnings,
    }


def audit_project(project_root: str | Path, checkpoint_path: str | Path, manifest_override: str | Path | None = None) -> dict:
    root = Path(project_root).resolve()
    ph2_root = root / "data/external/ph2"
    original_dir = ph2_root / "original"
    metadata_dir = ph2_root / "metadata"
    images_dir = ph2_root / "images"
    for directory in (original_dir, metadata_dir, images_dir):
        if not directory.is_dir():
            raise AuditError(f"Required PH2 directory missing: {directory}")
    metadata_path = metadata_dir / "PH2_dataset.txt"
    manifest_path = Path(manifest_override).resolve() if manifest_override is not None else metadata_dir / "ph2_manifest.csv"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise AuditError(f"Required PH2 metadata/manifest missing: {metadata_path} or {manifest_path}")
    clinical_labels = read_clinical_labels(metadata_path)
    rows = read_manifest(manifest_path)
    bmp_images = list(images_dir.glob("*.bmp"))
    if len(bmp_images) != 200:
        raise AuditError(f"Expected exactly 200 derived original BMP images; found {len(bmp_images)}")
    source_images_dir = original_dir / "PH2 Dataset images"
    source_images = [p for p in source_images_dir.rglob("*.bmp") if p.parent.name.endswith("_Dermoscopic_Image")]
    lesion_masks = [p for p in source_images_dir.rglob("*.bmp") if p.parent.name.endswith("_lesion")]
    roi_masks = [p for p in source_images_dir.rglob("*.bmp") if p.parent.name.endswith("_roi")]
    if len(source_images) != 200 or len(lesion_masks) != 200 or len(roi_masks) != 50:
        raise AuditError(f"Preserved original package structure mismatch: dermoscopic={len(source_images)}, lesion_masks={len(lesion_masks)}, roi_masks={len(roi_masks)}")
    if {p.stem for p in source_images} != set(clinical_labels):
        raise AuditError("Preserved original dermoscopic image IDs do not match clinical metadata IDs")
    expected_counts = dict(PH2_EXPECTED_COUNTS)
    data_audit = audit_manifest_rows(rows, clinical_labels, images_dir, expected_total=200, expected_class_counts=expected_counts)
    provenance = provenance_status(metadata_dir / "provenance.json")
    checkpoint = checkpoint_audit(Path(checkpoint_path).resolve(), root)
    import torch
    runtime = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    return {"project_root": str(root), "source_package_path": str(root / "data/external/PH2Dataset"), "original_dir": str(original_dir), "metadata_dir": str(metadata_dir), "images_dir": str(images_dir), "manifest_path": str(manifest_path), "provenance": provenance, "data": {k: v for k, v in data_audit.items() if k not in {"included_rows", "excluded_rows", "missing_images"}} | {"source_dermoscopic_images": len(source_images), "lesion_masks": len(lesion_masks), "roi_masks": len(roi_masks)}, "checkpoint": checkpoint, "runtime": runtime, "inference_ready": provenance["status"] == "VERIFIED" and runtime["cuda_available"]}


def run_evaluation(audit: dict, project_root: str | Path, checkpoint_path: str | Path, output_dir: str | Path) -> dict:
    """CUDA inference only; caller must first obtain VERIFIED provenance."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    root = Path(project_root).resolve()
    if audit["provenance"]["status"] != "VERIFIED":
        raise AuditError("External inference blocked: PH2 provenance/access is not VERIFIED")
    if not torch.cuda.is_available():
        raise AuditError("External inference requires CUDA; CPU inference is not permitted")
    rows = read_manifest(audit["manifest_path"])
    clinical = read_clinical_labels(Path(audit["metadata_dir"]) / "PH2_dataset.txt")
    details = audit_manifest_rows(rows, clinical, audit["images_dir"], expected_total=200, expected_class_counts=PH2_EXPECTED_COUNTS)
    included = details["included_rows"]
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise AuditError(f"Refusing to overwrite non-empty evaluation directory: {output}")

    from models.baseline_effnet import build_model
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model, _ = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.cuda().eval()
    transform = build_eval_transform()

    class PH2Dataset(Dataset):
        def __len__(self):
            return len(included)
        def __getitem__(self, idx):
            row = included[idx]
            image_path = Path(audit["images_dir"]) / f"{row['image_id']}.bmp"
            with Image.open(image_path) as image:
                tensor = transform(image.convert("RGB"))
            return tensor, row["ham_label"], row["image_id"]

    loader = DataLoader(PH2Dataset(), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    predictions = []
    with torch.inference_mode():
        for images, targets, image_ids in loader:
            logits = model(images.cuda(non_blocking=True))
            predicted_indices = logits.argmax(dim=1).cpu().tolist()
            for actual, predicted_index, image_id in zip(targets, predicted_indices, image_ids):
                predicted_label = HAM_CLASSES[predicted_index]
                binary_label = predicted_label if predicted_label in PH2_CLASSES else "other"
                predictions.append({"image_id": image_id, "true_label": actual, "predicted_ham10000_label": predicted_label, "binary_prediction": binary_label})
    matrix = binary_confusion_matrix([row["true_label"] for row in predictions], [row["binary_prediction"] for row in predictions])
    metrics = binary_metrics(matrix)
    pred_counts = Counter(row["predicted_ham10000_label"] for row in predictions)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["actual\\predicted", *PREDICTION_COLUMNS]); writer.writerow(["nv", *matrix[0]]); writer.writerow(["mel", *matrix[1]])
    result = {"total_ph2": 200, "included": len(included), "excluded_atypical_nevus": len(details["excluded_rows"]), "class_counts": details["class_counts"], "mapping": PH2_LABEL_TO_HAM, "prediction_distribution_7class": dict(pred_counts), "binary_confusion_matrix_rows_nv_mel_cols_nv_mel_other": matrix, "binary_metrics": metrics, "provenance_status": audit["provenance"]["status"]}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "evaluation_config.json").write_text(json.dumps({"checkpoint": audit["checkpoint"], "preprocessing": preprocessing_config(), "mapping": PH2_LABEL_TO_HAM, "provenance": audit["provenance"]}, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Current command: external_eval.py --project-root ROOT "
            "--checkpoint CHECKPOINT --output-dir OUTPUT --dry-run. "
            "Legacy positional form is also accepted: external_eval.py MANIFEST CHECKPOINT OUTPUT."
        ),
    )
    parser.add_argument("legacy", nargs="*", help=argparse.SUPPRESS)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Audit package/checkpoint/runtime without inference")
    parser.add_argument("--version", action="version", version="EG-VAN PH2 evaluator 2.0")
    args = parser.parse_args()
    if args.legacy:
        if len(args.legacy) != 3 or any((args.project_root, args.checkpoint, args.output_dir)):
            parser.error("legacy form requires exactly MANIFEST CHECKPOINT OUTPUT and cannot be mixed with options")
        legacy_manifest, legacy_checkpoint, legacy_output = map(Path, args.legacy)
        root = Path.cwd().resolve()
        manifest_override = legacy_manifest if legacy_manifest.is_absolute() else root / legacy_manifest
        checkpoint_arg = legacy_checkpoint
        output_arg = legacy_output
    else:
        root = (args.project_root or Path.cwd()).resolve()
        manifest_override = None
        checkpoint_arg = args.checkpoint or Path("experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt")
        output_arg = args.output_dir or Path("experiments/ph2_external_eval")
    checkpoint = checkpoint_arg if checkpoint_arg.is_absolute() else root / checkpoint_arg
    try:
        audit = audit_project(root, checkpoint, manifest_override)
        if args.dry_run:
            print(json.dumps({"dry_run": "PASS", **audit}, indent=2))
            return 0
        output_dir = output_arg if output_arg.is_absolute() else root / output_arg
        result = run_evaluation(audit, root, checkpoint, output_dir)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
