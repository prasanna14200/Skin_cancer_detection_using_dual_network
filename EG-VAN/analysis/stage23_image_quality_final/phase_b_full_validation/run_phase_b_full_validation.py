from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE

CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CHECKPOINT_SHA = "42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models/frozen_stage23"))
from preprocessing import preprocess_image  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def peak_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def degraded_rgb(rgb: np.ndarray, condition: dict) -> np.ndarray:
    kind = condition["type"]
    if kind == "identity":
        return rgb.copy()
    pil = Image.fromarray(rgb, "RGB")
    if kind == "gaussian_blur":
        return np.asarray(pil.filter(ImageFilter.GaussianBlur(float(condition["radius_pixels"]))), dtype=np.uint8)
    if kind == "jpeg":
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=int(condition["quality"]), subsampling=int(condition["subsampling"]))
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    arr = rgb.astype(np.float64)
    if kind == "brightness":
        arr *= float(condition["factor"])
    elif kind == "contrast":
        mean = float(arr.mean())
        arr = (arr - mean) * float(condition["factor"]) + mean
    else:
        raise ValueError(f"Unknown degradation type: {kind}")
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def paper_preprocess_to_pil(rgb: np.ndarray) -> Image.Image:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    processed = preprocess_image(bgr)
    ok, encoded = cv2.imencode(".jpg", processed)
    if not ok:
        raise ValueError("OpenCV JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != bgr.shape:
        raise ValueError("Processed JPEG decoding failed")
    return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), "RGB")


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    split_path = ROOT / "data/splits/split_leakage_aware.csv"
    split = pd.read_csv(split_path)
    val = split.loc[split["split"].eq("val"), ["image_id", "lesion_id", "dx"]].copy().sort_values("image_id").reset_index(drop=True)
    if len(val) != 986:
        raise ValueError(f"Expected 986 validation images, found {len(val)}")
    if val["image_id"].duplicated().any():
        raise ValueError("Validation image IDs are duplicated")
    ref_path = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_predictions.csv"
    ref = pd.read_csv(ref_path).sort_values("image_id").reset_index(drop=True)
    if len(ref) != 986:
        raise ValueError(f"Expected 986 reference predictions, found {len(ref)}")
    merged = val.merge(ref[["image_id", "true_label", "predicted_label", "probabilities"]], on="image_id", validate="one_to_one")
    if not (merged["dx"] == merged["true_label"]).all():
        raise ValueError("Frozen validation split and reference true labels disagree")
    return val, ref


def record_from_prediction(record, condition_name: str, prediction: dict, reference_pred: str, ref_probs: np.ndarray, model_delta: float) -> dict:
    probs = np.asarray([prediction["probabilities"][c] for c in CLASSES], dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(), 1.0, atol=1e-5):
        raise ValueError(f"Invalid probability vector for {record.image_id} / {condition_name}")
    pos = probs[probs > 0]
    entropy = float(-np.sum(pos * np.log(pos)))
    confidence = float(np.max(probs))
    return {
        "image_id": record.image_id,
        "lesion_id": record.lesion_id,
        "true_class": record.dx,
        "condition": condition_name,
        "predicted_class": prediction["predicted_class"],
        "correct": int(prediction["predicted_class"] == record.dx),
        "confidence": confidence,
        "entropy_nats": entropy,
        "probabilities": json.dumps(probs.tolist()),
        "baseline_reference_predicted_class": reference_pred,
        "baseline_max_abs_probability_delta": float(model_delta),
        "processing_status": "success",
    }


CHUNK_COLUMNS = (
    "image_id", "lesion_id", "true_class", "condition", "predicted_class",
    "correct", "confidence", "entropy_nats", "probabilities",
    "baseline_reference_predicted_class", "baseline_max_abs_probability_delta",
    "processing_status",
)


def chunk_path(condition: str, chunk_idx: int) -> Path:
    return OUT / f"{condition}_chunk_{chunk_idx:04d}.csv"


def parse_chunks(condition: str) -> list[Path]:
    paths = sorted(OUT.glob(f"{condition}_chunk_*.csv"))
    for path in paths:
        suffix = path.stem.removeprefix(f"{condition}_chunk_")
        if not suffix.isdigit():
            raise ValueError(f"Invalid chunk filename: {path.name}")
    return paths


def all_chunk_paths() -> list[Path]:
    return sorted(OUT.glob("*_chunk_*.csv"))


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            frame.to_csv(handle, index=False, float_format="%.17g")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def flush_chunk(rows: list[dict], condition: str) -> Path | None:
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=CHUNK_COLUMNS)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=OUT,
            prefix=f".{condition}_chunk.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            frame.to_csv(handle, index=False, float_format="%.17g")
            handle.flush()
            os.fsync(handle.fileno())
        used = {int(path.stem.rsplit("_", 1)[1]) for path in parse_chunks(condition)}
        chunk_idx = 0
        while chunk_idx in used:
            chunk_idx += 1
        target = chunk_path(condition, chunk_idx)
        try:
            os.link(temp_path, target)
        except FileExistsError:
            raise RuntimeError(f"Refusing to overwrite existing chunk: {target}")
        return target
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def validate_chunk_records(
    condition: str,
    records: pd.DataFrame,
    reference: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, set[str]]:
    record_by_id = records.set_index("image_id")
    reference_by_id = reference.set_index("image_id")
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    for path in parse_chunks(condition):
        frame = pd.read_csv(path)
        if tuple(frame.columns) != CHUNK_COLUMNS:
            raise ValueError(f"Invalid columns in chunk {path.name}")
        if frame.empty:
            raise ValueError(f"Empty chunk is not a completed output: {path.name}")
        for row in frame.itertuples(index=False):
            image_id = str(row.image_id)
            if image_id not in record_by_id.index:
                raise ValueError(f"Chunk {path.name} contains an image outside validation cohort: {image_id}")
            if image_id in seen:
                raise ValueError(f"Duplicate {condition} record across chunks for {image_id}")
            if str(row.condition) != condition or str(row.processing_status) != "success":
                raise ValueError(f"Incomplete or mismatched row in {path.name} for {image_id}")
            expected = record_by_id.loc[image_id]
            ref_row = reference_by_id.loc[image_id]
            if str(row.lesion_id) != str(expected["lesion_id"]) or str(row.true_class) != str(expected["dx"]):
                raise ValueError(f"Cohort metadata mismatch in {path.name} for {image_id}")
            if str(row.baseline_reference_predicted_class) != str(ref_row["predicted_label"]):
                raise ValueError(f"Reference prediction mismatch in {path.name} for {image_id}")
            probs = np.asarray(json.loads(row.probabilities), dtype=float)
            if probs.shape != (len(CLASSES),) or not np.isfinite(probs).all() or (probs < 0).any():
                raise ValueError(f"Invalid probability vector in {path.name} for {image_id}")
            if not np.isclose(probs.sum(), 1.0, atol=1e-5):
                raise ValueError(f"Probability vector does not sum to one in {path.name} for {image_id}")
            predicted = CLASSES[int(np.argmax(probs))]
            if predicted != str(row.predicted_class):
                raise ValueError(f"Predicted class does not match probabilities in {path.name} for {image_id}")
            if int(row.correct) != int(predicted == str(expected["dx"])):
                raise ValueError(f"Correctness field invalid in {path.name} for {image_id}")
            confidence = float(np.max(probs))
            entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
            if not np.isclose(float(row.confidence), confidence, atol=1e-12):
                raise ValueError(f"Confidence field invalid in {path.name} for {image_id}")
            if not np.isclose(float(row.entropy_nats), entropy, atol=1e-12):
                raise ValueError(f"Entropy field invalid in {path.name} for {image_id}")
            delta = float(row.baseline_max_abs_probability_delta)
            if not np.isfinite(delta) or delta < 0:
                raise ValueError(f"Invalid baseline delta in {path.name} for {image_id}")
            if condition == "baseline":
                reference_probs = np.asarray(json.loads(ref_row["probabilities"]), dtype=float)
                actual_delta = float(np.max(np.abs(probs - reference_probs)))
                if predicted != str(ref_row["predicted_label"]) or actual_delta > tolerance:
                    raise ValueError(f"Saved baseline does not reproduce reference for {image_id}")
                if not np.isclose(delta, actual_delta, atol=1e-9):
                    raise ValueError(f"Saved baseline delta is inconsistent for {image_id}")
            seen.add(image_id)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CHUNK_COLUMNS)
    return combined, seen


def check_processed_identity(image_id: str) -> None:
    raw_path = ROOT / "data/raw/images" / f"{image_id}.jpg"
    raw_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if raw_bgr is None or raw_bgr.shape[:2] != (450, 600):
        raise ValueError(f"Invalid raw validation JPEG at {raw_path}")
    image = paper_preprocess_to_pil(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
    expected_path = ROOT / "data/processed/images" / f"{image_id}.jpg"
    expected_bgr = cv2.imread(str(expected_path), cv2.IMREAD_COLOR)
    if expected_bgr is None:
        raise ValueError(f"Missing processed baseline JPEG at {expected_path}")
    actual_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if not np.array_equal(actual_bgr, expected_bgr):
        diff = int(np.max(np.abs(actual_bgr.astype(np.int16) - expected_bgr.astype(np.int16))))
        raise ValueError(f"Baseline processed JPEG mismatch for {image_id}: max pixel delta={diff}")


def validate_baseline_gate(
    records: pd.DataFrame,
    reference: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    tolerance: float,
    *,
    progress_every: int,
) -> dict:
    if baseline_rows["image_id"].duplicated().any():
        raise ValueError("Baseline records contain duplicate image IDs")
    by_id = baseline_rows.set_index("image_id")
    ref_by_id = reference.set_index("image_id")
    start = time.perf_counter()
    max_delta = 0.0
    total = len(records)
    for index, record in enumerate(records.itertuples(index=False), start=1):
        image_id = str(record.image_id)
        if image_id not in by_id.index:
            raise ValueError(f"Baseline record missing for {image_id}; degradation inference is prohibited")
        check_processed_identity(image_id)
        row = by_id.loc[image_id]
        ref_row = ref_by_id.loc[image_id]
        probs = np.asarray(json.loads(row["probabilities"]), dtype=float)
        reference_probs = np.asarray(json.loads(ref_row["probabilities"]), dtype=float)
        delta = float(np.max(np.abs(probs - reference_probs)))
        if str(row["predicted_class"]) != str(ref_row["predicted_label"]):
            raise ValueError(f"Baseline argmax mismatch for {image_id}")
        if delta > tolerance:
            raise ValueError(f"Baseline probability tolerance failed for {image_id}: {delta}")
        max_delta = max(max_delta, delta)
        if index % progress_every == 0 or index == total:
            elapsed = time.perf_counter() - start
            print(f"baseline-gate {index}/{total} elapsed={format_duration(elapsed)} peak_rss={peak_rss_mb():.1f}MB", flush=True)
    return {"baseline_verified": True, "images": total, "max_abs_probability_delta": max_delta}


def condition_chunks(
    condition: str,
    records: pd.DataFrame,
    reference: pd.DataFrame,
    tolerance: float,
    *,
    resume: bool,
) -> tuple[pd.DataFrame, set[str]]:
    paths = parse_chunks(condition)
    if paths and not resume:
        raise FileExistsError(f"Existing {condition} chunks found; use --resume to validate and continue")
    return validate_chunk_records(condition, records, reference, tolerance)


def run_single_condition(
    condition: str,
    records: pd.DataFrame,
    reference: pd.DataFrame,
    model,
    transform,
    *,
    chunk_size: int,
    resume: bool,
    progress_every: int,
    state: dict,
    state_path: Path,
) -> dict:
    from load_frozen import predict_rgb_image

    _, done = condition_chunks(
        condition, records, reference,
        PROTOCOL["baseline_max_abs_probability_delta_tolerance"], resume=resume,
    )
    processed = 0
    current_rows: list[dict] = []
    missing = len(records) - len(done)
    stats = {"condition": condition, "processed": 0, "skipped": len(done), "valid_existing": len(done)}
    if missing == 0:
        return stats
    started = time.perf_counter()
    reference_by_id = reference.set_index("image_id")
    condition_spec = next(item for item in PROTOCOL["degradation_conditions"] if item["name"] == condition)
    print(f"{condition}: {len(done)} valid records retained; {missing} records remain", flush=True)

    for record in records.itertuples(index=False):
        image_id = str(record.image_id)
        if image_id in done:
            continue
        raw_path = ROOT / "data/raw/images" / f"{image_id}.jpg"
        raw_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw_bgr is None or raw_bgr.shape[:2] != (450, 600):
            raise ValueError(f"Invalid raw validation JPEG at {raw_path}")
        rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        if condition == "baseline":
            image = paper_preprocess_to_pil(rgb)
            expected = cv2.imread(str(ROOT / "data/processed/images" / f"{image_id}.jpg"), cv2.IMREAD_COLOR)
            if expected is None:
                raise ValueError(f"Missing processed baseline output for {image_id}")
            actual = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            if not np.array_equal(actual, expected):
                diff = int(np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16))))
                raise ValueError(f"Baseline processed JPEG mismatch for {image_id}: max delta={diff}")
        else:
            image = paper_preprocess_to_pil(degraded_rgb(rgb, condition_spec))
        prediction = predict_rgb_image(model, transform, image, CLASSES)
        ref_row = reference_by_id.loc[image_id]
        ref_probs = np.asarray(json.loads(ref_row["probabilities"]), dtype=float)
        probs = np.asarray([prediction["probabilities"][name] for name in CLASSES], dtype=float)
        delta = float(np.max(np.abs(probs - ref_probs))) if condition == "baseline" else 0.0
        if condition == "baseline":
            if prediction["predicted_class"] != str(ref_row["predicted_label"]):
                raise ValueError(f"Baseline argmax mismatch for {image_id}")
            if delta > PROTOCOL["baseline_max_abs_probability_delta_tolerance"]:
                raise ValueError(f"Baseline probability tolerance failed for {image_id}: {delta}")
        current_rows.append(record_from_prediction(
            record, condition, prediction, str(ref_row["predicted_label"]), ref_probs, delta,
        ))
        processed += 1

        if len(current_rows) >= chunk_size or processed == missing:
            chunk_file = flush_chunk(current_rows, condition)
            if chunk_file is None:
                raise RuntimeError("Expected a non-empty chunk write")
            current_rows = []
            state["last_durable_chunk"] = chunk_file.name
            state["peak_rss_mb"] = max(state.get("peak_rss_mb", 0.0), peak_rss_mb())
            state["last_progress_at"] = utc_now()
            condition_progress = state.setdefault("condition_progress", {})
            condition_progress[condition] = {
                "valid_records": len(done) + processed,
                "missing_records": len(records) - len(done) - processed,
            }
            state["completed_records"] = sum(
                item["valid_records"] for item in condition_progress.values()
            )
            state["total_expected_records"] = int(PROTOCOL["expected_predictions"])
            write_json(state_path, state)

        if processed % progress_every == 0 or processed == missing:
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta_seconds = (missing - processed) / rate if rate > 0 else float("inf")
            eta = format_duration(eta_seconds) if np.isfinite(eta_seconds) else "unknown"
            print(
                f"{condition}: {len(done) + processed}/{len(records)} "
                f"(elapsed={format_duration(elapsed)}, ETA={eta}, peak_rss={peak_rss_mb():.1f}MB)",
                flush=True,
            )
            state["last_progress"] = {
                "condition": condition, "valid_records": len(done) + processed,
                "total_records": len(records), "elapsed_seconds": round(elapsed, 3),
                "estimated_remaining_seconds": eta_seconds if np.isfinite(eta_seconds) else None,
                "peak_rss_mb": peak_rss_mb(),
            }
            write_json(state_path, state)
    stats["processed"] = processed
    stats["chunks"] = len(parse_chunks(condition))
    return stats


def merge_condition_chunks(condition: str, records: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    frame, seen = validate_chunk_records(
        condition, records, reference,
        PROTOCOL["baseline_max_abs_probability_delta_tolerance"],
    )
    expected_ids = set(records["image_id"].astype(str))
    if seen != expected_ids:
        raise ValueError(f"{condition} is incomplete: {len(expected_ids - seen)} image-condition records missing")
    if frame.duplicated(["image_id", "condition"]).any():
        raise ValueError(f"{condition} contains duplicate records")
    frame = frame.sort_values(["condition", "image_id"]).reset_index(drop=True)
    atomic_write_csv(frame, OUT / f"{condition}.csv")
    return frame


def build_final_predictions(records: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    all_frames = [
        merge_condition_chunks(item["name"], records, reference)
        for item in PROTOCOL["degradation_conditions"]
    ]
    final = pd.concat(all_frames, ignore_index=True)
    expected = len(records) * len(PROTOCOL["degradation_conditions"])
    if len(final) != expected or final.duplicated(["image_id", "condition"]).any():
        raise ValueError(f"Final predictions incomplete or duplicated: {len(final)} / {expected}")
    final = final.sort_values(["condition", "image_id"]).reset_index(drop=True)
    atomic_write_csv(final, OUT / "phase_b_full_validation_predictions.csv")
    return final


def run(args) -> None:
    global PROTOCOL
    PROTOCOL = json.loads((OUT / "phase_b_full_validation_protocol.json").read_text(encoding="utf-8"))
    state_path = OUT / "execution_state.json"
    state = read_json(state_path, {
        "started_at": None, "finished_at": None, "conditions": {},
        "peak_rss_mb": 0.0, "completed_records": 0,
        "completed_conditions": [],
    })
    state["started_at"] = state.get("started_at") or utc_now()
    state["finished_at"] = None
    state["resume_supported"] = True
    state["last_error"] = None

    checkpoint_path = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt"
    if sha256(checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Frozen Stage 23 checkpoint hash mismatch")
    if args.limit is not None and not 1 <= args.limit <= 986:
        raise ValueError("--limit must be between 1 and 986")
    if args.chunk_size < 1 or args.progress_every < 1:
        raise ValueError("--chunk-size and --progress-every must be positive")
    if all_chunk_paths() and not args.resume and not args.verify_existing_chunks:
        raise FileExistsError("Existing chunks detected; rerun with --resume to validate and continue safely")

    records, reference = load_reference()
    if args.verify_existing_chunks:
        checked = {}
        for item in PROTOCOL["degradation_conditions"]:
            condition = item["name"]
            frame, ids = validate_chunk_records(
                condition, records, reference,
                PROTOCOL["baseline_max_abs_probability_delta_tolerance"],
            )
            checked[condition] = {
                "valid_records": len(frame),
                "unique_images": len(ids),
                "duplicate_free": True,
            }
        print(json.dumps({
            "checkpoint_sha256": sha256(checkpoint_path),
            "validation_images": len(records),
            "conditions": checked,
            "verified_records": sum(item["valid_records"] for item in checked.values()),
        }, indent=2))
        return
    if args.limit is not None:
        records = records.head(args.limit).copy()
    state["execution_scope_images"] = len(records)
    state["execution_scope_is_full_validation"] = len(records) == 986
    state["expected_scope_records"] = len(records) * len(PROTOCOL["degradation_conditions"])
    state["total_expected_records"] = int(PROTOCOL["expected_predictions"])
    state["baseline_completed"] = False
    state["baseline_verified"] = False
    state["baseline_verified_full_cohort"] = False
    state["condition_progress"] = {}
    tolerance = float(PROTOCOL["baseline_max_abs_probability_delta_tolerance"])
    for item in PROTOCOL["degradation_conditions"]:
        name = item["name"]
        _, ids = validate_chunk_records(name, records, reference, tolerance)
        state["condition_progress"][name] = {
            "valid_records": len(ids),
            "missing_records": len(records) - len(ids),
        }
    state["completed_records"] = sum(
        item["valid_records"] for item in state["condition_progress"].values()
    )
    state["completed_scope_conditions"] = [
        name for name, result in state["condition_progress"].items()
        if result["valid_records"] == len(records)
    ]
    state["completed_conditions"] = (
        state["completed_scope_conditions"] if len(records) == 986 else []
    )
    write_json(state_path, state)

    import torch
    from load_frozen import load_classifier

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model, transform, classes = load_classifier("cpu")
    if tuple(classes) != CLASSES:
        raise ValueError("Frozen class order mismatch")

    try:
        baseline_stats = run_single_condition(
            "baseline", records, reference, model, transform,
            chunk_size=args.chunk_size, resume=args.resume,
            progress_every=args.progress_every, state=state, state_path=state_path,
        )
        baseline_rows, baseline_ids = condition_chunks(
            "baseline", records, reference, tolerance, resume=True,
        )
        gate = validate_baseline_gate(
            records, reference, baseline_rows, tolerance,
            progress_every=args.progress_every,
        )
        expected_ids = set(records["image_id"].astype(str))
        if baseline_ids != expected_ids or gate["images"] != len(records):
            raise ValueError("Baseline gate incomplete; degradation inference is prohibited")
        state["baseline_completed"] = True
        state["baseline_verified"] = True
        state["baseline_verified_images"] = gate["images"]
        state["baseline_scope_completed"] = gate["images"] == len(records)
        state["baseline_verified_full_cohort"] = len(records) == 986
        state["baseline_max_abs_probability_delta"] = gate["max_abs_probability_delta"]
        state["conditions"]["baseline"] = baseline_stats
        state["condition_progress"] = {}
        for item in PROTOCOL["degradation_conditions"]:
            name = item["name"]
            _, condition_ids = validate_chunk_records(name, records, reference, tolerance)
            state["condition_progress"][name] = {
                "valid_records": len(condition_ids),
                "missing_records": len(records) - len(condition_ids),
            }
        state["completed_records"] = sum(
            result["valid_records"] for result in state["condition_progress"].values()
        )
        state["completed_scope_conditions"] = [
            name for name, result in state["condition_progress"].items()
            if result["valid_records"] == len(records)
        ]
        state["completed_conditions"] = (
            state["completed_scope_conditions"] if len(records) == 986 else []
        )
        state["peak_rss_mb"] = max(state.get("peak_rss_mb", 0.0), peak_rss_mb())
        write_json(state_path, state)
        print(
            f"BASELINE GATE PASSED: {gate['images']}/{len(records)} images; "
            f"max_abs_probability_delta={gate['max_abs_probability_delta']:.12g}",
            flush=True,
        )

        if args.baseline_only:
            state["baseline_only_run"] = True
            state["finished_at"] = utc_now()
            state["final_results_available"] = False
            write_json(state_path, state)
            return

        for condition in [item["name"] for item in PROTOCOL["degradation_conditions"] if item["name"] != "baseline"]:
            condition_start = time.perf_counter()
            stats = run_single_condition(
                condition, records, reference, model, transform,
                chunk_size=args.chunk_size, resume=args.resume,
                progress_every=args.progress_every, state=state, state_path=state_path,
            )
            state["conditions"][condition] = stats
            state["completed_conditions"] = [
                name for name in state["conditions"]
                if len(validate_chunk_records(name, records, reference, tolerance)[1]) == len(records)
            ]
            state["last_condition_runtime_seconds"] = round(time.perf_counter() - condition_start, 3)
            state["peak_rss_mb"] = max(state.get("peak_rss_mb", 0.0), peak_rss_mb())
            write_json(state_path, state)
            gc.collect()

        final = build_final_predictions(records, reference)
        state["finished_at"] = utc_now()
        state["completed_records"] = int(len(final))
        state["total_expected_records"] = int(PROTOCOL["expected_predictions"])
        state["final_results_available"] = len(records) == 986 and len(final) == 6902
        state["peak_rss_mb"] = max(state.get("peak_rss_mb", 0.0), peak_rss_mb())
        write_json(state_path, state)
        print(json.dumps({
            "baseline_verified": state["baseline_verified"],
            "baseline_verified_images": state["baseline_verified_images"],
            "baseline_max_abs_probability_delta": state["baseline_max_abs_probability_delta"],
            "completed_records": state["completed_records"],
            "total_expected_records": state["total_expected_records"],
            "peak_rss_mb": round(state["peak_rss_mb"], 2),
            "final_results_available": bool(state["final_results_available"]),
        }, indent=2))
    except Exception as exc:
        state["last_error"] = {"at": utc_now(), "type": type(exc).__name__, "message": str(exc)}
        state["peak_rss_mb"] = max(state.get("peak_rss_mb", 0.0), peak_rss_mb())
        write_json(state_path, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional bounded test on the first N sorted validation images")
    parser.add_argument("--chunk-size", type=int, default=25, help="Maximum records per durable atomic chunk")
    parser.add_argument("--progress-every", type=int, default=10, help="Report progress and estimated remaining time every N new records")
    parser.add_argument("--baseline-only", action="store_true", help="Run and verify the baseline only; never start degradations")
    parser.add_argument("--resume", action="store_true", help="Validate existing chunks and process only records not yet complete")
    parser.add_argument("--verify-existing-chunks", action="store_true", help="Validate all existing condition chunks and exit without inference")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
