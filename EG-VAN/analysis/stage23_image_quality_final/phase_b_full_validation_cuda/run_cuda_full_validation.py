"""Frozen Stage 23 CUDA FP16 batch-16 validation degradation runner.

Writes only to this new CUDA directory. Never reads CPU Phase B chunks, test or PH2.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / "analysis/stage23_image_quality_final/phase_b_full_validation/phase_b_full_validation_protocol.json"
PROBE = ROOT / "analysis/stage23_image_quality_final/phase_b_full_validation/cuda_baseline_mismatch_probe.json"
QUALITY_MANIFEST = ROOT / "analysis/stage23_image_quality_final/quality_analysis_manifest.json"
QUALITY_TABLE = ROOT / "analysis/stage23_image_quality_final/image_quality_metrics.csv"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CHECKPOINT_SHA = "42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab"
BATCH_SIZE = 16
CSV_COLUMNS = ("image_id", "lesion_id", "true_class", "condition", "batch_index", "batch_offset",
               "predicted_class", "correct", "confidence", "entropy_nats", "probabilities",
               "reference_predicted_class", "baseline_max_abs_probability_delta")

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models/frozen_stage23"))
from preprocessing import preprocess_image  # noqa: E402
from load_frozen import load_classifier  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_new(path: Path, obj: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write("\n")


@lru_cache(maxsize=1)
def frozen_processed_hashes() -> dict[str, str]:
    """Previously audited Stage 23 validation JPEG identities, not new image processing."""
    manifest = json.loads(QUALITY_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["output_sha256"]["image_quality_metrics.csv"]
    if sha(QUALITY_TABLE) != expected:
        raise ValueError("Frozen validation image-hash table changed")
    table = pd.read_csv(QUALITY_TABLE, usecols=["image_id", "processed_sha256"])
    if len(table) != 986 or table.image_id.duplicated().any():
        raise ValueError("Frozen validation image-hash table is incomplete or duplicated")
    return dict(zip(table.image_id, table.processed_sha256))


def frozen_inputs():
    protocol = json.loads(FROZEN.read_text(encoding="utf-8"))
    cfg_path = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg["physical_batch_size"] != BATCH_SIZE or cfg["amp"] != "cuda_fp16_grad_scaler":
        raise ValueError("Original Stage 23 batch/AMP configuration changed")
    if (protocol["status"] != "FROZEN_BEFORE_INFERENCE" or protocol["expected_images"] != 986 or
            protocol["expected_predictions"] != 6902 or protocol["baseline_max_abs_probability_delta_tolerance"] != .005 or
            protocol["primary_application"] != "raw RGB before frozen paper preprocessing" or
            protocol["checkpoint_sha256"] != CHECKPOINT_SHA):
        raise ValueError("Frozen Phase B protocol mismatch")
    expected_conditions = [
        {"name": "baseline", "type": "identity"},
        {"name": "blur_r1", "type": "gaussian_blur", "radius_pixels": 1.0},
        {"name": "blur_r2", "type": "gaussian_blur", "radius_pixels": 2.0},
        {"name": "underexposure_070", "type": "brightness", "factor": .7},
        {"name": "overexposure_130", "type": "brightness", "factor": 1.3},
        {"name": "contrast_070", "type": "contrast", "factor": .7},
        {"name": "jpeg_q40", "type": "jpeg", "quality": 40, "subsampling": 0},
    ]
    if protocol["degradation_conditions"] != expected_conditions:
        raise ValueError("Frozen degradation conditions changed")
    files = {
        "checkpoint": ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt",
        "stage23_config": cfg_path,
        "cuda_probe": PROBE,
        "split": ROOT / protocol["split_path"],
        "reference": ROOT / protocol["validation_reference_path"],
        "registry": ROOT / protocol["checkpoint_registry_path"],
        "preprocessing": ROOT / protocol["preprocessing_source_path"],
        "loader": ROOT / protocol["frozen_loader_source_path"],
    }
    for key, expected in (("checkpoint", CHECKPOINT_SHA), ("split", protocol["split_sha256"]),
                          ("reference", protocol["validation_reference_sha256"]),
                          ("registry", protocol["checkpoint_registry_sha256"]),
                          ("preprocessing", protocol["preprocessing_source_sha256"]),
                          ("loader", protocol["frozen_loader_source_sha256"])):
        if sha(files[key]) != expected:
            raise ValueError(f"Frozen input changed: {key}")
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    if (probe["checkpoint_sha256_after"] != CHECKPOINT_SHA or probe["batch_size"] != BATCH_SIZE or
            probe["results"]["cuda_fp16_original_batch"]["max_abs_delta_from_reference"] != 0):
        raise ValueError("CUDA path is not supported by the bounded probe")
    split = pd.read_csv(files["split"])
    validation = split.loc[split.split.eq("val"), ["image_id", "lesion_id", "dx"]].reset_index(drop=True)
    reference = pd.read_csv(files["reference"]).set_index("image_id", verify_integrity=True)
    if (len(validation) != 986 or validation.image_id.duplicated().any() or
            set(validation.image_id) != set(reference.index) or
            not (validation.set_index("image_id").dx == reference.loc[validation.image_id, "true_label"].to_numpy()).all()):
        raise ValueError("Validation split/reference mismatch")
    if tuple(json.loads(files["registry"].read_text())["class_order"]) != CLASSES:
        raise ValueError("Class order changed")
    processed_hashes = frozen_processed_hashes()
    if set(processed_hashes) != set(validation.image_id):
        raise ValueError("Frozen processed-image identities differ from validation split")
    for image_id in validation.image_id:
        for view in ("raw", "processed"):
            if not (ROOT / "data" / view / "images" / f"{image_id}.jpg").is_file():
                raise FileNotFoundError(f"Missing {view} validation image: {image_id}")
        processed_path = ROOT / "data/processed/images" / f"{image_id}.jpg"
        if sha(processed_path) != processed_hashes[image_id]:
            raise ValueError(f"Frozen processed JPEG identity changed: {image_id}")
    for value in reference.probabilities:
        p = np.asarray(json.loads(value), dtype=float)
        if p.shape != (7,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-5):
            raise ValueError("Reference probabilities invalid")
    return protocol, validation, reference, {key: sha(path) for key,path in files.items()}


def batches(validation: pd.DataFrame):
    return [(index, validation.iloc[index*BATCH_SIZE:(index+1)*BATCH_SIZE])
            for index in range((len(validation)+BATCH_SIZE-1)//BATCH_SIZE)]


def degraded_rgb(rgb: np.ndarray, condition: dict) -> np.ndarray:
    kind = condition["type"]
    if kind == "identity":
        return rgb.copy()
    pil = Image.fromarray(rgb, "RGB")
    if kind == "gaussian_blur":
        return np.asarray(pil.filter(ImageFilter.GaussianBlur(float(condition["radius_pixels"]))), dtype=np.uint8)
    if kind == "jpeg":
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=int(condition["quality"]), subsampling=int(condition["subsampling"]))
        buffer.seek(0)
        return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)
    arr = rgb.astype(np.float64)
    if kind == "brightness":
        arr *= float(condition["factor"])
    elif kind == "contrast":
        center = float(arr.mean())
        arr = (arr-center)*float(condition["factor"])+center
    else:
        raise ValueError(f"Unknown condition {kind}")
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def assert_baseline_pixel_identity(candidate: Image.Image, saved_path: Path, image_id: str) -> None:
    """Require the actual RGB input to equal the historical Stage 23 JPEG pixels."""
    with Image.open(saved_path) as reference:
        reference_rgb = reference.convert("RGB")
    if not np.array_equal(np.asarray(candidate), np.asarray(reference_rgb)):
        raise ValueError(f"Baseline preprocessing is not pixel-identical: {image_id}")


def processed_image(image_id: str, condition: dict) -> Image.Image:
    if condition["name"] == "baseline":
        if condition != {"name": "baseline", "type": "identity"}:
            raise ValueError("Baseline condition changed")
        saved_path = ROOT / "data/processed/images" / f"{image_id}.jpg"
        if sha(saved_path) != frozen_processed_hashes()[image_id]:
            raise ValueError(f"Frozen processed JPEG identity changed: {image_id}")
        # Stage 23 validation read this exact JPEG with PIL. The baseline must
        # not preprocess or re-encode it on a different OpenCV/JPEG runtime.
        with Image.open(saved_path) as saved:
            image = saved.convert("RGB")
        assert_baseline_pixel_identity(image, saved_path, image_id)
        return image
    raw_path = ROOT / "data/raw/images" / f"{image_id}.jpg"
    raw_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if raw_bgr is None or raw_bgr.shape[:2] != (450, 600):
        raise ValueError(f"Invalid raw validation image: {image_id}")
    changed = degraded_rgb(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), condition)
    processed = preprocess_image(cv2.cvtColor(changed, cv2.COLOR_RGB2BGR))
    ok, encoded = cv2.imencode(".jpg", processed)
    if not ok:
        raise ValueError(f"Processed JPEG encoding failed: {image_id}")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"Processed JPEG decoding failed: {image_id}")
    return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), "RGB")


def predict_batch(model, transform, rows: pd.DataFrame, condition: dict, reference: pd.DataFrame, batch_index: int):
    images = [transform(processed_image(str(row.image_id), condition)) for row in rows.itertuples(index=False)]
    tensor = torch.stack(images).cuda(non_blocking=False)
    model.eval()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(tensor)
        if logits.shape != (len(rows), 7) or not bool(torch.isfinite(logits).all().item()):
            raise ValueError(f"Invalid logits for {condition['name']} batch {batch_index}")
        predicted = logits.argmax(dim=1).cpu().numpy()
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
    del tensor, images, logits
    result = []
    for offset, row in enumerate(rows.itertuples(index=False)):
        image_id = str(row.image_id)
        p = probabilities[offset].astype(float)
        rp = np.asarray(json.loads(reference.loc[image_id, "probabilities"]), dtype=float)
        label = CLASSES[int(predicted[offset])]
        delta = float(np.max(np.abs(p-rp))) if condition["name"] == "baseline" else None
        if condition["name"] == "baseline" and (
                label != reference.loc[image_id, "predicted_label"] or delta > .005):
            raise ValueError(f"Baseline gate failed for {image_id}: label={label}, reference={reference.loc[image_id, 'predicted_label']}, delta={delta}")
        if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(),1,atol=1e-5):
            raise ValueError(f"Invalid probability vector: {image_id}")
        positive = p[p>0]
        result.append({"image_id": image_id, "lesion_id": str(row.lesion_id), "true_class": str(row.dx),
                       "condition": condition["name"], "batch_index": batch_index, "batch_offset": offset,
                       "predicted_class": label, "correct": int(label == row.dx),
                       "confidence": float(p.max()), "entropy_nats": float(-(positive*np.log(positive)).sum()),
                       "probabilities": json.dumps(p.tolist()),
                       "reference_predicted_class": str(reference.loc[image_id, "predicted_label"]),
                       "baseline_max_abs_probability_delta": delta})
    return pd.DataFrame(result, columns=CSV_COLUMNS)


def batch_dir(condition: str, index: int):
    return HERE / "batches" / condition / f"batch_{index:04d}"


def commit_batch(frame: pd.DataFrame, condition: str, index: int, protocol_sha: str):
    target = batch_dir(condition, index)
    if target.exists():
        raise FileExistsError(f"Refusing batch overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".batch_{index:04d}_", dir=target.parent))
    try:
        csv_path = temporary / "records.csv"
        frame.to_csv(csv_path, index=False, float_format="%.17g")
        with csv_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        metadata = {"condition": condition, "batch_index": index, "image_ids": frame.image_id.tolist(),
                    "rows": len(frame), "records_sha256": sha(csv_path), "protocol_sha256": protocol_sha,
                    "checkpoint_sha256": CHECKPOINT_SHA, "batch_size": BATCH_SIZE,
                    "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__}
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
        with (temporary / "metadata.json").open("r+b") as handle:
            os.fsync(handle.fileno())
        if target.exists():
            raise FileExistsError(f"Refusing batch overwrite: {target}")
        os.rename(temporary, target)  # Atomic directory commit on the same filesystem.
    finally:
        if temporary.exists():
            # Uncommitted staging directory is safe to remove; committed batches are never touched.
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()


def validate_batch(frame: pd.DataFrame, rows: pd.DataFrame, reference: pd.DataFrame, condition: str, index: int):
    if tuple(frame.columns) != CSV_COLUMNS or len(frame) != len(rows):
        raise ValueError(f"Batch schema/count invalid: {condition}/{index}")
    if frame.image_id.tolist() != rows.image_id.tolist() or frame.image_id.duplicated().any():
        raise ValueError(f"Batch ordering/IDs invalid: {condition}/{index}")
    for offset, (saved, expected) in enumerate(zip(frame.itertuples(index=False), rows.itertuples(index=False))):
        if (saved.condition != condition or int(saved.batch_index) != index or int(saved.batch_offset) != offset or
                saved.lesion_id != expected.lesion_id or saved.true_class != expected.dx or
                saved.reference_predicted_class != reference.loc[saved.image_id, "predicted_label"] or
                saved.predicted_class not in CLASSES or int(saved.correct) != int(saved.predicted_class == expected.dx)):
            raise ValueError(f"Batch record metadata invalid: {condition}/{index}/{saved.image_id}")
        p = np.asarray(json.loads(saved.probabilities), dtype=float)
        if p.shape != (7,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-5):
            raise ValueError(f"Batch probability invalid: {saved.image_id}")
        if not np.isclose(p[CLASSES.index(saved.predicted_class)], p.max(), atol=1e-7, rtol=0):
            raise ValueError(f"Batch predicted class disagrees with probability ranking: {saved.image_id}")
        if not np.isclose(p.max(),float(saved.confidence),atol=1e-12):
            raise ValueError(f"Batch confidence invalid: {saved.image_id}")
        entropy = float(-(p[p>0]*np.log(p[p>0])).sum())
        if not np.isclose(entropy,float(saved.entropy_nats),atol=1e-12):
            raise ValueError(f"Batch entropy invalid: {saved.image_id}")
        if condition == "baseline":
            rp = np.asarray(json.loads(reference.loc[saved.image_id,"probabilities"]),dtype=float)
            delta = float(np.max(np.abs(p-rp)))
            if (saved.predicted_class != reference.loc[saved.image_id,"predicted_label"] or delta>.005 or
                    not np.isclose(delta,float(saved.baseline_max_abs_probability_delta),atol=1e-12)):
                raise ValueError(f"Baseline batch gate invalid: {saved.image_id}")


def existing_batches(condition: str, validation: pd.DataFrame, reference: pd.DataFrame, protocol_sha: str):
    condition_dir = HERE / "batches" / condition
    if not condition_dir.exists():
        return {}
    found = {}
    expected = dict(batches(validation))
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    for path in condition_dir.iterdir():
        if path.name.startswith(".batch_"):
            continue  # An interrupted uncommitted staging directory has no research status.
        if not path.is_dir() or not re.fullmatch(r"batch_\d{4}",path.name):
            raise ValueError(f"Unexpected item in batch directory: {path}")
        index = int(path.name[-4:])
        if index not in expected or index in found:
            raise ValueError(f"Unexpected/duplicate batch index: {condition}/{index}")
        csv_path = path / "records.csv"
        metadata_path = path / "metadata.json"
        if not csv_path.is_file() or not metadata_path.is_file() or set(x.name for x in path.iterdir()) != {"records.csv","metadata.json"}:
            raise ValueError(f"Incomplete committed batch: {path}")
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("condition") != condition or metadata.get("batch_index") != index or
                metadata.get("image_ids") != expected[index].image_id.tolist() or
                metadata.get("rows") != len(expected[index]) or metadata.get("records_sha256") != sha(csv_path) or
                metadata.get("protocol_sha256") != protocol_sha or metadata.get("checkpoint_sha256") != CHECKPOINT_SHA or
                metadata.get("batch_size") != BATCH_SIZE or metadata.get("gpu") != probe["gpu"] or
                metadata.get("torch_version") != probe["torch_version"]):
            raise ValueError(f"Committed batch provenance mismatch: {path}")
        frame = pd.read_csv(csv_path)
        validate_batch(frame, expected[index], reference, condition, index)
        found[index] = frame
    return found


def complete_baseline_gate(validation, reference, protocol_sha):
    existing = existing_batches("baseline", validation, reference, protocol_sha)
    expected = len(batches(validation))
    if len(existing) != expected:
        raise ValueError(f"Baseline incomplete: {len(existing)}/{expected} batches; degradations blocked")
    frame = pd.concat([existing[i] for i in range(expected)],ignore_index=True)
    if len(frame) != 986 or frame.image_id.duplicated().any():
        raise ValueError("Baseline rows incomplete or duplicate")
    max_delta = float(frame.baseline_max_abs_probability_delta.max())
    disagreements = frame.loc[frame.predicted_class.ne(frame.reference_predicted_class),"image_id"].tolist()
    if disagreements or max_delta > .005:
        raise ValueError(f"Baseline gate failed: disagreements={disagreements}, max_delta={max_delta}")
    gate = {"status": "PASS", "images": 986, "batches": expected, "batch_size": BATCH_SIZE,
            "max_abs_probability_delta": max_delta, "class_disagreements": disagreements,
            "checkpoint_sha256": CHECKPOINT_SHA, "protocol_sha256": protocol_sha}
    path = HERE / "baseline_gate.json"
    if path.exists():
        if json.loads(path.read_text()) != gate:
            raise ValueError("Existing baseline gate report differs from validated batches")
    else:
        write_json_new(path, gate)
    return gate


def execute(args):
    protocol, validation, reference, hashes = frozen_inputs()
    protocol_sha = sha(FROZEN)
    names = [c["name"] for c in protocol["degradation_conditions"]]
    inventory = {name: len(existing_batches(name,validation,reference,protocol_sha)) for name in names}
    if args.check:
        print(json.dumps({"status":"PREFLIGHT_PASS","batches":inventory,"expected_batches_per_condition":len(batches(validation)),
                          "frozen_input_sha256":hashes,"protocol_sha256":protocol_sha},indent=2))
        return
    if args.resume and not any(inventory.values()):
        raise FileNotFoundError("--resume requires existing CUDA batch commits")
    if not args.resume and any(inventory.values()):
        raise FileExistsError("Existing CUDA batches require --resume; no overwrite")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Tesla T4 required for original Stage 23 execution policy")
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    if torch.cuda.get_device_name() != probe["gpu"] or torch.__version__ != probe["torch_version"]:
        raise RuntimeError("GPU/PyTorch runtime differs from verified original-batch CUDA probe")
    model, transform, classes = load_classifier("cuda")
    if model.training or tuple(classes) != CLASSES:
        raise ValueError("Frozen model eval/class order mismatch")
    for condition in protocol["degradation_conditions"]:
        name = condition["name"]
        if name != "baseline":
            complete_baseline_gate(validation,reference,protocol_sha)
            if args.baseline_only:
                break
        done = existing_batches(name,validation,reference,protocol_sha)
        for index, rows in batches(validation):
            if index in done:
                continue
            frame = predict_batch(model,transform,rows,condition,reference,index)
            validate_batch(frame,rows,reference,name,index)
            commit_batch(frame,name,index,protocol_sha)
            print(f"{name}: committed batch {index+1}/{len(batches(validation))} ({len(frame)} rows)",flush=True)
        if name == "baseline":
            gate = complete_baseline_gate(validation,reference,protocol_sha)
            print("BASELINE_GATE_PASS",json.dumps(gate),flush=True)
            if args.baseline_only:
                return
    final = []
    for name in names:
        committed = existing_batches(name,validation,reference,protocol_sha)
        if len(committed) != len(batches(validation)):
            raise ValueError(f"Incomplete {name}; final table blocked")
        final.extend(committed[i] for i in range(len(batches(validation))))
    table = pd.concat(final,ignore_index=True)
    if len(table)!=6902 or table.duplicated(["image_id","condition"]).any():
        raise ValueError("Final prediction table incomplete or duplicate")
    target = HERE / "cuda_full_validation_predictions.csv"
    if target.exists():
        raise FileExistsError("Final table already exists; refusing overwrite")
    temporary = HERE / ".cuda_full_validation_predictions.tmp"
    if temporary.exists():
        raise FileExistsError("Stale final table staging file requires manual inspection")
    table.to_csv(temporary,index=False,float_format="%.17g")
    os.rename(temporary,target)
    if sha(ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt") != CHECKPOINT_SHA:
        raise ValueError("Frozen checkpoint changed during inference")
    print(f"COMPLETE {len(table)} rows; run analyze_cuda_full_validation.py",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true",help="Read-only input/batch integrity check; no model load")
    parser.add_argument("--run",action="store_true",help="Execute bounded CUDA inference")
    parser.add_argument("--resume",action="store_true",help="Require and validate existing CUDA batch commits")
    parser.add_argument("--baseline-only",action="store_true",help="Stop after all 986 baseline predictions pass")
    args = parser.parse_args()
    if args.check == args.run or (args.baseline_only and not args.run) or (args.resume and not args.run):
        parser.error("Choose --check or --run; --baseline-only/--resume require --run")
    execute(args)


if __name__ == "__main__":
    main()
