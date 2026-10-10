"""Freeze a lesion-aware validation sample and fixed degradations before inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CHECKPOINT_SHA = "42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    protocol = OUT / "phase_b_protocol.json"
    if protocol.exists():
        raise FileExistsError(f"Protocol already frozen: {protocol}")
    split_path = ROOT / "data/splits/split_leakage_aware.csv"
    pred_path = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_predictions.csv"
    ckpt_path = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt"
    frozen = json.loads((ROOT / "models/frozen_stage23/model_registry.json").read_text())
    assert sha(ckpt_path) == CHECKPOINT_SHA == frozen["checkpoint"]["sha256"]
    assert sha(split_path) == "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
    assert sha(pred_path) == frozen["source_artifact_sha256"][str(pred_path.relative_to(ROOT)).replace("\\", "/")]
    split = pd.read_csv(split_path)
    val = split.loc[split.split.eq("val")].copy()
    pred = pd.read_csv(pred_path, usecols=["image_id", "true_label"])
    assert len(val) == len(pred) == 986 and val.image_id.is_unique and pred.image_id.is_unique
    check = val.merge(pred, on="image_id", validate="one_to_one")
    assert len(check) == 986 and (check.dx == check.true_label).all()
    rng = np.random.default_rng(2309)
    cases = []
    for cls in CLASSES:
        sub = val.loc[val.dx.eq(cls)]
        lesions = np.array(sorted(sub.lesion_id.unique()))
        selected = rng.choice(lesions, size=3, replace=False)
        for lesion in selected:
            image_id = min(sub.loc[sub.lesion_id.eq(lesion), "image_id"])
            for stage in ("raw", "processed"):
                path = ROOT / "data" / stage / "images" / f"{image_id}.jpg"
                assert path.is_file(), path
            cases.append({"image_id": image_id, "lesion_id": str(lesion), "true_class": cls})
    cases.sort(key=lambda c: (CLASSES.index(c["true_class"]), c["image_id"]))
    assert len(cases) == 21 and len({c["lesion_id"] for c in cases}) == 21
    result = {
        "status": "FROZEN_BEFORE_INFERENCE", "scope": "Stage 23 validation only, class-balanced 21-image CPU-bounded sample",
        "selection_seed": 2309, "bootstrap_seed": 2310, "bootstrap_repetitions": 1000,
        "sample_rule": "3 unique lesions per true class, RNG choice from sorted lesion IDs; first sorted validation image per lesion",
        "checkpoint_sha256": CHECKPOINT_SHA, "split_sha256": sha(split_path),
        "validation_predictions_sha256": sha(pred_path),
        "preprocessing_source_sha256": sha(ROOT / "src/preprocessing.py"),
        "frozen_loader_source_sha256": sha(ROOT / "models/frozen_stage23/load_frozen.py"),
        "protocol_md_sha256": sha(OUT / "PHASE_B_PROTOCOL.md"),
        "cases": cases,
        "conditions": [
            {"name": "baseline", "type": "identity"},
            {"name": "blur_r1", "type": "gaussian_blur", "radius_pixels": 1.0},
            {"name": "blur_r2", "type": "gaussian_blur", "radius_pixels": 2.0},
            {"name": "underexposure_070", "type": "brightness", "factor": 0.7},
            {"name": "overexposure_130", "type": "brightness", "factor": 1.3},
            {"name": "contrast_070", "type": "contrast", "factor": 0.7},
            {"name": "jpeg_q40", "type": "jpeg", "quality": 40, "subsampling": 0}
        ],
        "primary_application": "raw RGB before frozen paper preprocessing",
        "expected_predictions": 147, "baseline_argmax_required": "21/21",
        "baseline_max_abs_probability_delta_tolerance": 0.005,
        "baseline_processed_decoded_pixels": "exact identity required"
    }
    protocol.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"FROZEN {len(cases)} cases, {len(result['conditions'])} conditions, SHA256 {sha(protocol)}")


if __name__ == "__main__":
    main()
