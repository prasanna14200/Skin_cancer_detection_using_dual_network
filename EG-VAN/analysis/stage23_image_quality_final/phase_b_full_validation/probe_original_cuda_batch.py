"""Bounded validation-only T4 probe for the original Stage 23 batch/AMP path.

No training, checkpoint writes, test/PH2 access, or Phase B chunk updates.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "models/frozen_stage23"))
from load_frozen import load_classifier  # noqa: E402

TARGET = "ISIC_0029026"
EXPECTED_SHA = "42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def probabilities(model, tensor, amp):
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(tensor)
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite diagnostic logits")
        return torch.softmax(logits.float(), dim=1).cpu().numpy()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires the original CUDA/T4 execution environment")
    output = HERE / "cuda_baseline_mismatch_probe.json"
    if output.exists():
        raise FileExistsError(f"Preserve existing probe: {output}")
    checkpoint = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt"
    assert sha(checkpoint) == EXPECTED_SHA
    split = pd.read_csv(ROOT / "data/splits/split_leakage_aware.csv")
    val = split.loc[split.split.eq("val")].reset_index(drop=True)
    assert len(val) == 986
    idx = val.index[val.image_id.eq(TARGET)].tolist()
    assert len(idx) == 1
    idx = idx[0]
    cfg = json.loads((ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/config.json").read_text())
    batch_size = int(cfg["physical_batch_size"])
    start = idx // batch_size * batch_size
    rows = val.iloc[start:start+batch_size]
    offset = idx-start
    model, transform, classes = load_classifier("cuda")
    assert not model.training and tuple(classes) == CLASSES
    images = []
    for image_id in rows.image_id:
        with Image.open(ROOT / "data/processed/images" / f"{image_id}.jpg") as image:
            images.append(transform(image.convert("RGB")))
    batch = torch.stack(images).cuda()
    reference = pd.read_csv(ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_predictions.csv").set_index("image_id")
    ref = np.asarray(json.loads(reference.loc[TARGET, "probabilities"]), dtype=float)
    results = {}
    for name, tensor, amp in (
        ("cuda_fp16_singleton", batch[offset:offset+1], True),
        ("cuda_fp16_original_batch", batch, True),
        ("cuda_fp32_singleton", batch[offset:offset+1], False),
        ("cuda_fp32_original_batch", batch, False),
    ):
        p = probabilities(model, tensor, amp)[0 if tensor.shape[0] == 1 else offset]
        order = np.argsort(p)[::-1]
        results[name] = {"probabilities": p.tolist(), "predicted_class": CLASSES[int(order[0])],
                         "top_two_margin": float(p[order[0]]-p[order[1]]),
                         "max_abs_delta_from_reference": float(np.max(np.abs(p-ref)))}
    artifact = {"checkpoint_sha256_before": EXPECTED_SHA, "checkpoint_sha256_after": sha(checkpoint),
                "target": TARGET, "class_order": list(CLASSES), "validation_order_index": idx,
                "batch_start": start, "batch_size": batch_size, "target_offset": offset,
                "batch_image_ids": rows.image_id.tolist(), "gpu": torch.cuda.get_device_name(),
                "torch_version": torch.__version__, "reference_probabilities": ref.tolist(),
                "results": results, "training_performed": False, "test_or_ph2_accessed": False}
    assert artifact["checkpoint_sha256_after"] == EXPECTED_SHA
    output.write_text(json.dumps(artifact, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"target": TARGET, "results": results}, indent=2))


if __name__ == "__main__":
    main()
