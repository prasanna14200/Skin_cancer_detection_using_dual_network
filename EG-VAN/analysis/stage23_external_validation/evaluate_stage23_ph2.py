"""Frozen Stage 23 PH2 external follow-up. --preflight never runs inference.

PH2 was used earlier in this project; outputs must not be called untouched
independent validation or used to change the Stage 23 classifier.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import uuid
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL_PATH = HERE / "phase23_ph2_protocol.json"
PROTOCOL_SHA = "3e6e31165e6afcb6b270e940f3abfe94dc99e2c805d84e08db2a156a91754ed8"
MANIFEST_REL = Path("analysis/ph2_external_validation_exp5/ph2_manifest.csv")
MANIFEST_SHA = "3c130b3b44dcc1052ae29656510a95a4e91df4b8f65c121838941efb375d3842"
CHECKPOINT_SHA = "42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab"
SPLIT_SHA = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
MAPPING = {"0": ("common nevus", "nv", True),
           "1": ("atypical nevus", "", False),
           "2": ("melanoma", "mel", True)}
PRED_FIELDS = ["image_id", "true_external_label", "true_ham_label", "predicted_class",
               "mapped_prediction", "binary_mel_prediction", "confidence", "entropy_nats",
               *(f"p_{name}" for name in CLASSES)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def metadata_codes(path: Path) -> dict[str, str]:
    import re
    pattern = re.compile(r"^\|\|\s*(IMD\d+)\s*\|\|(.*?)\|\|\s*([^|]+?)\s*\|\|")
    codes = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            image_id, _, code = match.groups()
            if image_id in codes or code.strip() not in MAPPING:
                raise ValueError(f"Invalid PH2 metadata row: {image_id}")
            codes[image_id] = code.strip()
    return codes


def validate_cohort(rows: list[dict], codes: dict[str, str], root: Path,
                    *, inspect_images: bool = True) -> dict:
    if len(rows) != 200 or len(codes) != 200 or len({r["image_id"] for r in rows}) != 200:
        raise ValueError("PH2 cohort/metadata must have 200 unique matching IDs")
    if {r["image_id"] for r in rows} != set(codes):
        raise ValueError("PH2 manifest and metadata IDs differ")
    from PIL import Image
    included = []
    image_hashes = {}
    for row in rows:
        image_id = row["image_id"]
        label, ham_label, expected_included = MAPPING[codes[image_id]]
        relative = Path(f"data/external/ph2/images/{image_id}.bmp")
        if (row["clinical_code"] != codes[image_id] or
                row["true_external_label"] != label or
                row["true_ham_label"] != ham_label or
                row["included"] != str(expected_included) or
                Path(row["relative_image_path"]) != relative):
            raise ValueError(f"PH2 mapping/path drift: {image_id}")
        image = root / relative
        original = (root / "data/external/ph2/original/PH2 Dataset images" / image_id /
                    f"{image_id}_Dermoscopic_Image" / f"{image_id}.bmp")
        if not image.is_file() or not original.is_file():
            raise FileNotFoundError(f"PH2 image or original missing: {image_id}")
        image_hashes[image_id] = sha256(image)
        if image_hashes[image_id] != sha256(original):
            raise ValueError(f"PH2 flat image differs from original package: {image_id}")
        if inspect_images:
            with Image.open(image) as bitmap:
                if bitmap.width < 384 or bitmap.height < 384:
                    raise ValueError(f"Unexpected PH2 image size: {image_id}")
                bitmap.verify()
        if expected_included:
            included.append(row)
    if (Counter(r["true_ham_label"] for r in included) != {"nv": 80, "mel": 40} or
            Counter(r["true_external_label"] for r in rows) !=
            {"common nevus": 80, "atypical nevus": 80, "melanoma": 40}):
        raise ValueError("PH2 frozen cohort distribution changed")
    return {"included": included, "image_sha256": image_hashes}


def preflight(root: Path, *, inspect_images: bool = True) -> dict:
    root = root.resolve()
    if root != ROOT.resolve():
        raise ValueError("Expected the pinned EG-VAN project root")
    if sha256(PROTOCOL_PATH) != PROTOCOL_SHA:
        raise ValueError("Stage 23 PH2 preregistration changed")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (protocol["status"] != "FROZEN_BEFORE_STAGE23_PH2_INFERENCE" or
            protocol["checkpoint_sha256"] != CHECKPOINT_SHA or
            tuple(protocol["class_order"]) != CLASSES or
            protocol["ph2_manifest_sha256"] != MANIFEST_SHA or
            protocol["cohort"] != {"total": 200, "included": 120,
                "common_nevus_as_nv": 80, "melanoma_as_mel": 40,
                "excluded_atypical_nevus": 80} or
            protocol["inference"]["batch_size"] != 16):
        raise ValueError("Frozen PH2 protocol is incomplete or changed")
    registry_path = root / "models/frozen_stage23/model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if (registry["model_identifier"] != "EGVAN_STAGE23_FINAL" or
            registry["selected_epoch"] != 14 or
            tuple(registry["class_order"]) != CLASSES or
            registry["checkpoint"]["sha256"] != CHECKPOINT_SHA):
        raise ValueError("Stage 23 frozen registry drift")
    for relative, expected in registry["source_artifact_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"Frozen Stage 23 source drift: {relative}")
    if sha256(root / "data/splits/split_leakage_aware.csv") != SPLIT_SHA:
        raise ValueError("Frozen HAM split drift")
    cfg = json.loads((root / "experiments/stage23_resnet50_imagenet_init_exp1/config.json").read_text())
    run_manifest = json.loads((root / "experiments/stage23_resnet50_imagenet_init_exp1/run/experiment_manifest.json").read_text())
    if (cfg["physical_batch_size"] != 16 or cfg["amp"] != "cuda_fp16_grad_scaler" or
            cfg["classes"] != list(CLASSES) or run_manifest["selected_epoch"] != 14 or
            run_manifest["ph2_accessed"] is not False):
        raise ValueError("Frozen Stage 23 inference/training boundary changed")
    manifest_path = root / MANIFEST_REL
    if sha256(manifest_path) != MANIFEST_SHA:
        raise ValueError("Frozen PH2 cohort manifest SHA256 mismatch")
    codes = metadata_codes(root / "data/external/ph2/metadata/PH2_dataset.txt")
    audit = validate_cohort(read_csv(manifest_path), codes, root, inspect_images=inspect_images)
    # IDs can exclude exact identifier overlap, not patient-level linkage.
    ham_ids = {r["image_id"] for r in read_csv(root / "data/splits/split_leakage_aware.csv")}
    if ham_ids.intersection(audit["image_sha256"]):
        raise ValueError("PH2/HAM source image IDs overlap")
    return {"status": "PASS_QUALIFIED_EXTERNAL_FOLLOWUP", "checkpoint_sha256": CHECKPOINT_SHA,
            "split_sha256": SPLIT_SHA, "protocol_sha256": PROTOCOL_SHA,
            "ph2_manifest_sha256": MANIFEST_SHA, "metadata_sha256": sha256(root / "data/external/ph2/metadata/PH2_dataset.txt"),
            "included": 120, "excluded_atypical": 80,
            "class_counts": {"nv": 80, "mel": 40}, "class_order": list(CLASSES),
            "source_image_sha256": audit["image_sha256"],
            "prior_ph2_project_use": True, "patient_identifiers_available": False,
            "stage23_training_manifest_ph2_accessed": False}


def wilson(success: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = success / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [center-half, center+half]


def summarize(rows: list[dict]) -> dict:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(rows) != 120 or len({r["image_id"] for r in rows}) != 120:
        raise ValueError("Stage 23 PH2 predictions incomplete or duplicated")
    matrix = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        truth = row["true_ham_label"]
        pred = row["predicted_class"]
        if truth not in ("nv", "mel") or pred not in CLASSES:
            raise ValueError("Unknown PH2 mapped truth or predicted class")
        matrix[0 if truth == "nv" else 1][0 if pred == "nv" else 1 if pred == "mel" else 2] += 1
    if list(map(sum, matrix)) != [80, 40]:
        raise ValueError("PH2 truth distribution changed")
    mel_tp = matrix[1][1]
    mel_fp = matrix[0][1]
    nv_tp = matrix[0][0]
    correct = mel_tp + nv_tp
    binary_specificity = (80-mel_fp)/80
    truth_binary = np.array([r["true_ham_label"] == "mel" for r in rows], dtype=int)
    score = np.array([float(r["p_mel"]) for r in rows])
    entropy = np.array([float(r["entropy_nats"]) for r in rows])
    confidence = np.array([float(r["confidence"]) for r in rows])
    if not (np.isfinite(score).all() and np.isfinite(entropy).all() and np.isfinite(confidence).all()):
        raise ValueError("Nonfinite Stage 23 PH2 uncertainty")
    precision = mel_tp/(mel_tp+mel_fp) if mel_tp+mel_fp else 0.0
    recall = mel_tp/40
    mel_f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    output = {"sample_count": 120, "class_counts": {"nv": 80, "mel": 40},
              "true_rows": ["nv", "mel"], "predicted_columns": ["nv", "mel", "other"],
              "confusion_matrix": matrix, "mapped_accuracy": correct/120,
              "mel_recall": recall, "mel_precision": precision, "mel_f1": mel_f1,
              "nv_exact_label_recall": nv_tp/80,
              "mapped_balanced_accuracy": (nv_tp/80 + recall)/2,
              "binary_mel_specificity": binary_specificity,
              "binary_specificity_note": "NV predicted OTHER is binary-negative but wrong in mapped 2x3 accuracy.",
              "mel_p_roc_auc": float(roc_auc_score(truth_binary, score)),
              "mel_p_average_precision": float(average_precision_score(truth_binary, score)),
              "mean_entropy_nats": float(entropy.mean()), "mean_confidence": float(confidence.mean()),
              "wilson_95": {"mapped_accuracy": wilson(correct,120), "mel_recall": wilson(mel_tp,40),
                            "nv_exact_label_recall": wilson(nv_tp,80),
                            "binary_mel_specificity": wilson(80-mel_fp,80)},
              "uncertainty_unit": "image ID as lesion proxy; patient identifiers unavailable"}
    correctness = np.array([r["predicted_class"] == r["true_ham_label"] for r in rows])
    output["entropy_error_detection_auc"] = float(roc_auc_score(~correctness, entropy)) if len(set(correctness)) == 2 else None
    return output


def bootstrap(rows: list[dict], *, seed: int = 42, repetitions: int = 2000) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    groups = {name: [r for r in rows if r["true_ham_label"] == name] for name in ("nv", "mel")}
    values = {name: [] for name in ("mapped_accuracy", "mel_f1", "mel_p_roc_auc", "mean_entropy_nats")}
    undefined = {name: 0 for name in values}
    for _ in range(repetitions):
        sample = [groups[name][int(i)] for name in ("nv", "mel")
                  for i in rng.integers(0, len(groups[name]), len(groups[name]))]
        matrix = [[0, 0, 0], [0, 0, 0]]
        for r in sample:
            matrix[0 if r["true_ham_label"] == "nv" else 1][0 if r["predicted_class"] == "nv" else 1 if r["predicted_class"] == "mel" else 2] += 1
        tp, fp = matrix[1][1], matrix[0][1]
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/40
        scored = {"mapped_accuracy": (matrix[0][0]+tp)/120,
                  "mel_f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
                  "mel_p_roc_auc": float(roc_auc_score(
                      [r["true_ham_label"] == "mel" for r in sample],
                      [float(r["p_mel"]) for r in sample])),
                  "mean_entropy_nats": float(np.mean([float(r["entropy_nats"]) for r in sample]))}
        for name, value in scored.items():
            if math.isfinite(value):
                values[name].append(value)
            else:
                undefined[name] += 1
    return {"method": "stratified image-ID bootstrap; patient groups unavailable",
            "seed": seed, "repetitions": repetitions,
            "percentile_95": {name: list(map(float, np.percentile(items,[2.5,97.5]))) if items else None
                              for name, items in values.items()}, "undefined_replicates": undefined}


def run(root: Path) -> None:
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    audit = preflight(root)
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Frozen Stage 23 external follow-up requires Tesla T4 CUDA")
    output = HERE / "run"
    if output.exists() or list(HERE.glob(".run_inprogress_*")):
        raise FileExistsError("Stage 23 PH2 output or prior partial run exists; refusing repeat or overwrite")
    sys.path.insert(0, str(root / "models/frozen_stage23"))
    sys.path.insert(0, str(root / "src"))
    from load_frozen import load_classifier
    from preprocessing import preprocess_image
    model, transform, classes = load_classifier("cuda", root=root)
    if tuple(classes) != CLASSES:
        raise ValueError("Frozen class order drift")
    included = [r for r in read_csv(root / MANIFEST_REL) if r["included"] == "True"]
    class PH2Images(Dataset):
        def __len__(self):
            return len(included)
        def __getitem__(self, index):
            row = included[index]
            with Image.open(root / row["relative_image_path"]) as bitmap:
                rgb = np.asarray(bitmap.convert("RGB"), dtype=np.uint8)
            processed = preprocess_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            ok, encoded = cv2.imencode(".jpg", processed)
            if not ok:
                raise ValueError(f"PH2 JPEG encode failed: {row['image_id']}")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"PH2 JPEG decode failed: {row['image_id']}")
            return transform(Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), "RGB"))
    loader = DataLoader(PH2Images(), batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    predictions = []
    with torch.inference_mode():
        for images in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images.cuda(non_blocking=True))
            if logits.shape != (len(images), 7) or not bool(torch.isfinite(logits).all()):
                raise ValueError("Nonfinite or malformed PH2 logits")
            probs = torch.softmax(logits.float(), dim=1).cpu().tolist()
            for p in probs:
                source = included[len(predictions)]
                if len(p) != 7 or not all(math.isfinite(x) and 0 <= x <= 1 for x in p) or abs(sum(p)-1) > 1e-5:
                    raise ValueError(f"Invalid PH2 probability: {source['image_id']}")
                pred = CLASSES[max(range(7), key=p.__getitem__)]
                entropy = -sum(x*math.log(x) for x in p if x > 0)
                predictions.append({"image_id": source["image_id"],
                    "true_external_label": source["true_external_label"],
                    "true_ham_label": source["true_ham_label"], "predicted_class": pred,
                    "mapped_prediction": pred if pred in ("nv", "mel") else "other",
                    "binary_mel_prediction": int(pred == "mel"),
                    "confidence": max(p), "entropy_nats": entropy,
                    **{f"p_{name}": p[i] for i, name in enumerate(CLASSES)}})
    if [r["image_id"] for r in predictions] != [r["image_id"] for r in included]:
        raise ValueError("PH2 inference order changed")
    if sha256(root / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt") != CHECKPOINT_SHA:
        raise ValueError("Frozen checkpoint changed during external evaluation")
    metrics = summarize(predictions)
    metrics["bootstrap_95"] = bootstrap(predictions)
    temp = HERE / f".run_inprogress_{uuid.uuid4().hex}"
    temp.mkdir(exist_ok=False)
    try:
        with (temp / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=PRED_FIELDS)
            writer.writeheader(); writer.writerows(predictions)
        with (temp / "metrics.json").open("x", encoding="utf-8") as stream:
            json.dump(metrics, stream, indent=2, allow_nan=False); stream.write("\n")
        with (temp / "confusion_matrix.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream); writer.writerow(["true_class", "nv", "mel", "other"])
            writer.writerow(["nv", *metrics["confusion_matrix"][0]])
            writer.writerow(["mel", *metrics["confusion_matrix"][1]])
        provenance = {k:v for k,v in audit.items() if k != "source_image_sha256"}
        provenance.update({"status": "COMPLETE_QUALIFIED_EXTERNAL_FOLLOWUP",
            "source_image_sha256": audit["source_image_sha256"],
            "runner_sha256": sha256(Path(__file__)), "preprocessing_sha256": sha256(root / "src/preprocessing.py"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
            "batch_size": 16, "training_performed": False, "threshold_changed": False,
            "checkpoint_changed": False, "prior_ph2_project_use": True,
            "artifact_sha256": {name: sha256(temp / name) for name in
                ("predictions.csv", "metrics.json", "confusion_matrix.csv")}})
        with (temp / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(provenance, stream, indent=2, allow_nan=False); stream.write("\n")
        os.rename(temp, output)
    except Exception:
        # Keep temporary evidence for diagnosis; never overwrite a prior run.
        raise
    print(json.dumps({"status": provenance["status"], "output": str(output),
                      "sample_count": 120, "prior_ph2_project_use": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--preflight", action="store_true")
    choice.add_argument("--run", action="store_true")
    parser.add_argument("--accept-prior-ph2-use", action="store_true",
                        help="Acknowledge that this is external follow-up, not untouched validation")
    args = parser.parse_args()
    if args.preflight:
        result = preflight(args.project_root)
        print(json.dumps({k:v for k,v in result.items() if k != "source_image_sha256"}, indent=2))
    else:
        if not args.accept_prior_ph2_use:
            parser.error("--run requires --accept-prior-ph2-use")
        run(args.project_root.resolve())


if __name__ == "__main__":
    main()
