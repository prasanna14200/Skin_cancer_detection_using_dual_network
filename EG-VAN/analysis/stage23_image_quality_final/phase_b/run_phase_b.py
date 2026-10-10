"""Execute frozen Stage 23 validation-only raw-image degradation protocol."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models/frozen_stage23"))
from preprocessing import preprocess_image  # noqa: E402
from load_frozen import load_classifier, predict_rgb_image  # noqa: E402

spec = importlib.util.spec_from_file_location("stage23_validation_audit", ROOT / "analysis/stage23_uncertainty_final/analyze_stage23_uncertainty.py")
assert spec and spec.loader
day1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day1)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def preflight():
    p = json.loads((HERE / "phase_b_protocol.json").read_text())
    if p["status"] != "FROZEN_BEFORE_INFERENCE" or len(p["cases"]) != 21 or len(p["conditions"]) != 7:
        raise ValueError("Frozen protocol status/count mismatch")
    checks = {
        "checkpoint_sha256": ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt",
        "split_sha256": ROOT / "data/splits/split_leakage_aware.csv",
        "validation_predictions_sha256": day1.PREDICTIONS,
        "preprocessing_source_sha256": ROOT / "src/preprocessing.py",
        "frozen_loader_source_sha256": ROOT / "models/frozen_stage23/load_frozen.py",
        "protocol_md_sha256": HERE / "PHASE_B_PROTOCOL.md",
    }
    for key, path in checks.items():
        if sha(path) != p[key]:
            raise ValueError(f"Frozen source changed: {key}")
    frame, _, _ = day1.audit_inputs()
    q = frame.set_index("image_id")
    assert len({x["lesion_id"] for x in p["cases"]}) == 21
    for case in p["cases"]:
        if case["image_id"] not in q.index or q.loc[case["image_id"], "true_label"] != case["true_class"]:
            raise ValueError("Case absent from frozen validation prediction table")
    return p, q


def degraded_rgb(rgb, condition):
    kind = condition["type"]
    if kind == "identity":
        return rgb.copy()
    pil = Image.fromarray(rgb, "RGB")
    if kind == "gaussian_blur":
        return np.asarray(pil.filter(ImageFilter.GaussianBlur(condition["radius_pixels"])), dtype=np.uint8)
    if kind == "jpeg":
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=condition["quality"], subsampling=condition["subsampling"])
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
    arr = rgb.astype(np.float64)
    if kind == "brightness":
        arr *= condition["factor"]
    elif kind == "contrast":
        mean = float(arr.mean())
        arr = (arr - mean) * condition["factor"] + mean
    else:
        raise ValueError(f"Unknown condition {kind}")
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def paper_preprocess_to_pil(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    processed = preprocess_image(bgr)
    ok, encoded = cv2.imencode(".jpg", processed)
    if not ok:
        raise ValueError("OpenCV JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != bgr.shape:
        raise ValueError("Processed JPEG decoding failed")
    return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), "RGB"), decoded


def one_prediction(model, transform, classes, case, condition, saved):
    raw_path = ROOT / "data/raw/images" / f"{case['image_id']}.jpg"
    raw_bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if raw_bgr is None or raw_bgr.shape[:2] != (450, 600):
        raise ValueError(f"Invalid raw validation JPEG: {raw_path}")
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    changed = degraded_rgb(raw_rgb, condition)
    image, processed_bgr = paper_preprocess_to_pil(changed)
    if condition["name"] == "baseline":
        expected_path = ROOT / "data/processed/images" / f"{case['image_id']}.jpg"
        expected = cv2.imread(str(expected_path), cv2.IMREAD_COLOR)
        if expected is None or not np.array_equal(processed_bgr, expected):
            max_diff = None if expected is None or expected.shape != processed_bgr.shape else int(np.max(np.abs(processed_bgr.astype(np.int16)-expected.astype(np.int16))))
            raise ValueError(f"Baseline processed JPEG mismatch: {case['image_id']}, max pixel delta={max_diff}")
    prediction = predict_rgb_image(model, transform, image, classes)
    probs = np.array([prediction["probabilities"][c] for c in classes], dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(), 1, atol=1e-5):
        raise ValueError(f"Invalid probability: {case['image_id']}/{condition['name']}")
    pos = probs[probs > 0]
    saved_probs = np.asarray(json.loads(saved.probabilities), dtype=float)
    return {"image_id": case["image_id"], "lesion_id": case["lesion_id"], "true_class": case["true_class"],
            "condition": condition["name"], "raw_sha256": sha(raw_path),
            "degraded_raw_rgb_sha256": hashlib.sha256(changed.tobytes()).hexdigest(),
            "predicted_class": prediction["predicted_class"],
            "correct": int(prediction["predicted_class"] == case["true_class"]),
            "confidence": float(probs.max()), "entropy_nats": float(-np.sum(pos * np.log(pos))),
            "probabilities": json.dumps(probs.tolist()),
            "saved_stage23_predicted_class": saved.predicted_label,
            "baseline_max_abs_probability_delta": float(np.max(np.abs(probs-saved_probs))) if condition["name"] == "baseline" else None}


def run():
    protocol, frame = preflight()
    if (HERE / "phase_b_predictions.csv").exists():
        raise FileExistsError("Phase B predictions already exist; preserve prior evidence")
    torch.set_num_threads(4)
    model, transform, classes = load_classifier("cpu")
    if model.training or tuple(classes) != day1.CLASSES:
        raise ValueError("Frozen classifier evaluation state/class order mismatch")
    before_hash = sha(day1.CHECKPOINT)
    baseline_rows = []
    try:
        for case in protocol["cases"]:
            row = one_prediction(model, transform, classes, case, protocol["conditions"][0], frame.loc[case["image_id"]])
            baseline_rows.append(row)
            print(f"baseline {len(baseline_rows)}/21 {case['image_id']} delta={row['baseline_max_abs_probability_delta']:.6g}", flush=True)
        bad = [r for r in baseline_rows if r["predicted_class"] != r["saved_stage23_predicted_class"] or r["baseline_max_abs_probability_delta"] > protocol["baseline_max_abs_probability_delta_tolerance"]]
        if bad:
            raise ValueError(f"Baseline saved-prediction gate failed for {[r['image_id'] for r in bad]}")
    except Exception as exc:
        (HERE / "baseline_failure.json").write_text(json.dumps({"status": "STOPPED_BEFORE_DEGRADATIONS", "reason": str(exc), "baseline_completed": len(baseline_rows), "checkpoint_hash": before_hash}, indent=2)+"\n")
        raise
    rows = list(baseline_rows)
    for condition in protocol["conditions"][1:]:
        for case in protocol["cases"]:
            rows.append(one_prediction(model, transform, classes, case, condition, frame.loc[case["image_id"]]))
        print(f"condition {condition['name']} complete {len(rows)}/{protocol['expected_predictions']}", flush=True)
    if len(rows) != protocol["expected_predictions"] or sha(day1.CHECKPOINT) != before_hash:
        raise ValueError("Incomplete predictions or frozen checkpoint changed")
    pd.DataFrame(rows).to_csv(HERE / "phase_b_predictions.csv", index=False, float_format="%.17g")
    print("COMPLETE", len(rows), "predictions", flush=True)


if __name__ == "__main__":
    run()
