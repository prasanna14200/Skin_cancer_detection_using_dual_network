"""Predeclared seven-image Stage 23 validation synthetic-degradation pilot.

Uses processed HAM validation images only. No training, test/PH2 access, or
checkpoint mutation. This is not acquisition-quality validation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, __version__ as PIL_VERSION

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PROTOCOL_PATH = OUT / "degradation_protocol.json"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models/frozen_stage23"))
from image_quality import compute_image_quality  # noqa: E402
from load_frozen import load_classifier, predict_rgb_image  # noqa: E402

spec = importlib.util.spec_from_file_location("stage23_uncertainty_for_degradation", ROOT / "analysis/stage23_uncertainty_final/analyze_stage23_uncertainty.py")
assert spec and spec.loader
day1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(day1)


def sha256(path: Path) -> str:
    return day1.sha256(path)


def preflight() -> tuple[dict, pd.DataFrame]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (protocol["status"] != "FIXED_BEFORE_PILOT_EXECUTION" or
            protocol["checkpoint_sha256"] != day1.CHECKPOINT_SHA or
            protocol["validation_predictions_sha256"] != sha256(day1.PREDICTIONS) or
            sha256(day1.CHECKPOINT) != day1.CHECKPOINT_SHA):
        raise ValueError("Degradation protocol or frozen model/prediction provenance mismatch")
    frame, _, _ = day1.audit_inputs()
    expected = {cls: min(frame.loc[frame.true_label == cls, "image_id"]) for cls in day1.CLASSES}
    if protocol["cases"] != expected or len(protocol["variants"]) != 7:
        raise ValueError("Pilot cases/variants differ from frozen selection")
    for image_id in expected.values():
        path = ROOT / "data/processed/images" / f"{image_id}.jpg"
        with Image.open(path) as image:
            if image.format != "JPEG" or image.size[0] < 16 or image.size[1] < 16:
                raise ValueError(f"Invalid pilot image: {path}")
            image.verify()
    return protocol, frame


def degrade(image: Image.Image, variant: dict) -> Image.Image:
    name = variant["name"]
    if name == "baseline":
        return image.copy()
    if name in ("blur_mild", "blur_moderate"):
        return image.filter(ImageFilter.GaussianBlur(radius=float(variant["radius_pixels"])))
    arr = np.asarray(image, dtype=np.float64)
    if name in ("brightness_down", "brightness_up"):
        changed = np.clip(np.rint(arr * float(variant["factor"])), 0, 255).astype(np.uint8)
        return Image.fromarray(changed, "RGB")
    if name == "contrast_down":
        center = float(arr.mean())
        changed = np.clip(np.rint((arr - center) * float(variant["factor"]) + center), 0, 255).astype(np.uint8)
        return Image.fromarray(changed, "RGB")
    if name == "jpeg_q40":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(variant["quality"]), subsampling=int(variant["subsampling"]))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")
    raise ValueError(f"Unknown fixed variant: {name}")


def run() -> None:
    protocol, frame = preflight()
    torch.set_num_threads(2)
    model, transform, classes = load_classifier("cpu")
    if model.training or tuple(classes) != day1.CLASSES:
        raise ValueError("Frozen model is not in eval mode/class order")
    results = []
    for true_class, image_id in protocol["cases"].items():
        source = ROOT / "data/processed/images" / f"{image_id}.jpg"
        original = Image.open(source).convert("RGB")
        original.load()
        saved = frame.loc[frame.image_id == image_id].iloc[0]
        saved_prob = np.asarray(json.loads(saved.probabilities), dtype=float)
        for variant in protocol["variants"]:
            changed = degrade(original, variant)
            rgb = np.asarray(changed, dtype=np.uint8)
            prediction = predict_rgb_image(model, transform, changed, classes)
            probs = np.asarray([prediction["probabilities"][name] for name in classes], dtype=float)
            if probs.shape != (7,) or not np.isfinite(probs).all() or not np.isclose(probs.sum(), 1, atol=1e-5):
                raise ValueError(f"Invalid pilot probability: {image_id}/{variant['name']}")
            positive = probs[probs > 0]
            quality = compute_image_quality(rgb)
            results.append({
                "image_id": image_id, "true_class": true_class, "variant": variant["name"],
                "processed_source_sha256": sha256(source),
                "degraded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "predicted_class": prediction["predicted_class"],
                "correct": int(prediction["predicted_class"] == true_class),
                "confidence": float(probs.max()), "entropy_nats": float(-np.sum(positive * np.log(positive))),
                "probabilities": json.dumps(probs.tolist()),
                "quality_sharpness": quality["sharpness"], "quality_brightness": quality["brightness"],
                "quality_contrast": quality["contrast"],
                "quality_dark_pixel_ratio": quality["dark_pixel_ratio"],
                "quality_bright_pixel_ratio": quality["bright_pixel_ratio"],
                "quality_saturation": quality["saturation"],
                "saved_baseline_label": saved.predicted_label,
                "saved_baseline_max_abs_probability_delta": float(np.max(np.abs(probs - saved_prob))) if variant["name"] == "baseline" else None,
                "baseline_argmax_matches_saved": bool(prediction["predicted_class"] == saved.predicted_label) if variant["name"] == "baseline" else None,
            })
    table = pd.DataFrame(results)
    if len(table) != 49 or table.groupby("image_id").size().ne(7).any():
        raise ValueError("Incomplete pilot")
    baseline = table[table.variant == "baseline"]
    mean_by_variant = []
    for variant in protocol["variants"]:
        part = table[table.variant == variant["name"]]
        mean_by_variant.append({"variant": variant["name"], "n": len(part),
                                "correct": int(part.correct.sum()), "accuracy": float(part.correct.mean()),
                                "mean_confidence": float(part.confidence.mean()),
                                "mean_entropy": float(part.entropy_nats.mean()),
                                "mean_sharpness": float(part.quality_sharpness.mean()),
                                "mean_brightness": float(part.quality_brightness.mean())})
    summary = {"scope": "Seven predeclared validation images, synthetic degradation after paper preprocessing, CPU FP32 inference",
               "checkpoint_sha256": day1.CHECKPOINT_SHA, "protocol_sha256": sha256(PROTOCOL_PATH),
               "prediction_source_sha256": sha256(day1.PREDICTIONS), "script_sha256": sha256(Path(__file__).resolve()),
               "python_torch_pillow": {"torch": torch.__version__, "pillow": PIL_VERSION},
               "pilot_images": 7, "forwards": len(table),
               "baseline_argmax_match_count": int(baseline.baseline_argmax_matches_saved.sum()),
               "baseline_max_abs_probability_delta": float(baseline.saved_baseline_max_abs_probability_delta.max()),
               "by_variant": mean_by_variant,
               "interpretation_limit": "Small, non-random within-class lexicographic pilot; post-preprocessing synthetic perturbations, not original acquisition-quality validation or a representative accuracy estimate",
               "training_performed": False, "ham_test_accessed": False, "ph2_accessed": False}
    table.to_csv(OUT / "degradation_results.csv", index=False, float_format="%.17g")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    names = [row["variant"] for row in mean_by_variant]
    for ax, key, title in zip(axes, ("accuracy", "mean_confidence", "mean_entropy"),
                              ("Correct / 7", "Mean max-softmax", "Mean entropy (nats)")):
        ax.bar(range(len(names)), [row[key] for row in mean_by_variant], color="#477f9c")
        ax.set_xticks(range(len(names)), names, rotation=50, ha="right")
        ax.set(title=title, ylabel="Pilot value")
    fig.suptitle("Stage 23 predeclared synthetic-degradation pilot (n=7)")
    fig.tight_layout(); fig.savefig(OUT / "degradation_robustness.png", dpi=350); plt.close(fig)
    summary["output_sha256"] = {name: sha256(OUT / name) for name in ("degradation_results.csv", "degradation_robustness.png")}
    (OUT / "degradation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE_BOUNDED_PILOT", "forwards": len(table),
                      "baseline_argmax_match_count": summary["baseline_argmax_match_count"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate fixed protocol without model inference")
    parser.add_argument("--run", action="store_true", help="Run only the frozen 7x7 validation pilot")
    args = parser.parse_args()
    if args.check == args.run:
        parser.error("Choose exactly one of --check or --run")
    if args.check:
        protocol, _ = preflight()
        print(json.dumps({"status": "PREFLIGHT_PASS", "cases": protocol["cases"], "variants": len(protocol["variants"])}, indent=2))
    else:
        run()


if __name__ == "__main__":
    main()
