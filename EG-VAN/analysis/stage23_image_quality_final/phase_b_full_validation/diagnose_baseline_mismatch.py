"""Read-only CPU diagnosis of one Stage 23 validation baseline disagreement."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "models/frozen_stage23"))
from load_frozen import load_classifier, predict_rgb_image  # noqa: E402

spec = importlib.util.spec_from_file_location("phase_b_full_runner", HERE / "run_phase_b_full_validation.py")
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

CASES = ("ISIC_0029017", "ISIC_0029026", "ISIC_0029036")
CLASSES = tuple(runner.CLASSES)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def detail(probs):
    x = np.asarray(probs, dtype=float)
    idx = np.argsort(x)[::-1]
    return {"probabilities": {c: float(x[i]) for i,c in enumerate(CLASSES)},
            "predicted_class": CLASSES[int(idx[0])],
            "top_two": [{"class": CLASSES[int(j)], "probability": float(x[j])} for j in idx[:2]],
            "top_two_margin": float(x[idx[0]]-x[idx[1]])}


def main():
    torch.set_num_threads(1)
    ckpt = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt"
    assert sha(ckpt) == runner.CHECKPOINT_SHA
    split = pd.read_csv(ROOT / "data/splits/split_leakage_aware.csv").set_index("image_id")
    ref = pd.read_csv(ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_predictions.csv").set_index("image_id")
    model, transform, classes = load_classifier("cpu")
    assert not model.training and tuple(classes) == CLASSES
    result = {"scope": "controlled validation-only diagnosis", "device": "cpu", "torch_version": torch.__version__,
              "checkpoint_sha256_before": sha(ckpt), "class_order": list(CLASSES), "cases": {}}
    for image_id in CASES:
        assert split.loc[image_id, "split"] == "val"
        raw = ROOT / "data/raw/images" / f"{image_id}.jpg"
        processed = ROOT / "data/processed/images" / f"{image_id}.jpg"
        raw_bgr = cv2.imread(str(raw), cv2.IMREAD_COLOR)
        saved_bgr = cv2.imread(str(processed), cv2.IMREAD_COLOR)
        regenerated = runner.paper_preprocess_to_pil(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
        regenerated_bgr = cv2.cvtColor(np.asarray(regenerated), cv2.COLOR_RGB2BGR)
        pixel_identical = bool(np.array_equal(regenerated_bgr, saved_bgr))
        with Image.open(processed) as image:
            direct = image.convert("RGB")
            direct.load()
        raw_runs, direct_runs = [], []
        for _ in range(3 if image_id == "ISIC_0029026" else 2):
            raw_prediction = predict_rgb_image(model, transform, regenerated, classes)
            direct_prediction = predict_rgb_image(model, transform, direct, classes)
            raw_runs.append(detail([raw_prediction["probabilities"][c] for c in CLASSES]))
            direct_runs.append(detail([direct_prediction["probabilities"][c] for c in CLASSES]))
        reference = detail(json.loads(ref.loc[image_id, "probabilities"]))
        now = np.array(list(raw_runs[0]["probabilities"].values()))
        old = np.array(list(reference["probabilities"].values()))
        result["cases"][image_id] = {
            "true_class": str(split.loc[image_id, "dx"]), "lesion_id": str(split.loc[image_id, "lesion_id"]),
            "raw_sha256": sha(raw), "processed_sha256": sha(processed),
            "raw_shape": list(raw_bgr.shape), "processed_shape": list(saved_bgr.shape),
            "regenerated_processed_pixel_identical": pixel_identical,
            "regenerated_processed_max_pixel_delta": int(np.max(np.abs(regenerated_bgr.astype(np.int16)-saved_bgr.astype(np.int16)))),
            "reference": reference, "raw_replay": raw_runs, "direct_processed_replay": direct_runs,
            "max_abs_probability_delta": float(np.max(np.abs(now-old))),
            "raw_repeats_exact": all(x["probabilities"] == raw_runs[0]["probabilities"] for x in raw_runs),
            "direct_repeats_exact": all(x["probabilities"] == direct_runs[0]["probabilities"] for x in direct_runs),
            "raw_equals_direct": raw_runs[0]["probabilities"] == direct_runs[0]["probabilities"]}
        print(image_id, reference["predicted_class"], raw_runs[0]["predicted_class"],
              result["cases"][image_id]["max_abs_probability_delta"], flush=True)
    result["checkpoint_sha256_after"] = sha(ckpt)
    assert result["checkpoint_sha256_after"] == result["checkpoint_sha256_before"]
    (HERE / "baseline_mismatch_diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
