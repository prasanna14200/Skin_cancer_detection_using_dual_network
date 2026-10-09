"""Read-only, idempotent verification of the frozen Stage 23 reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from load_frozen import HERE, ROOT, load_classifier, sha256, verify_registry


def verify(*, smoke: bool = False) -> dict:
    registry = verify_registry(ROOT)
    manifest_path = HERE / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = ROOT / registry["checkpoint"]["path"]
    if (manifest["model_identifier"] != registry["model_identifier"] or
            manifest["original_checkpoint_path"] != registry["checkpoint"]["path"] or
            manifest["frozen_checkpoint_path"] != registry["checkpoint"]["path"] or
            manifest["original_checkpoint_sha256"] != registry["checkpoint"]["sha256"] or
            manifest["frozen_checkpoint_sha256"] != sha256(checkpoint_path) or
            manifest["model_registry_sha256"] != sha256(HERE / "model_registry.json") or
            manifest["checkpoint_storage"] != "reference_original_no_copy"):
        raise ValueError("Frozen manifest/registry/checkpoint mismatch")
    for relative, expected in manifest["validation_and_decision_artifact_sha256"].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Original experiment/decision artifact changed: {relative}")
    result = {"status": "PASS", "model_identifier": registry["model_identifier"],
              "selected_epoch": registry["selected_epoch"],
              "checkpoint_sha256": registry["checkpoint"]["sha256"],
              "checkpoint_storage": manifest["checkpoint_storage"],
              "source_artifact_count": len(registry["source_artifact_sha256"]),
              "synthetic_cpu_smoke": "NOT_RUN", "training_performed": False,
              "ham_test_inference_performed": False, "ph2_inference_performed": False}
    if smoke:
        import torch
        from PIL import Image
        from load_frozen import predict_rgb_image

        torch.set_num_threads(2)
        model, transform, classes = load_classifier("cpu")
        image = Image.new("RGB", (384, 384), (127, 127, 127))
        first = transform(image)
        second = transform(image)
        if not torch.equal(first, second) or tuple(first.shape) != (3, 384, 384):
            raise ValueError("Frozen evaluation preprocessing is not deterministic")
        prediction = predict_rgb_image(model, transform, image, classes)
        probabilities = prediction["probabilities"]
        if (model.training or len(probabilities) != 7 or
                tuple(probabilities) != classes or
                prediction["predicted_class"] not in classes or
                abs(sum(probabilities.values()) - 1) > 1e-5):
            raise ValueError("Synthetic CPU inference contract failed")
        result["synthetic_cpu_smoke"] = "PASS_1x7_FINITE"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="CPU synthetic image only")
    args = parser.parse_args()
    print(json.dumps(verify(smoke=args.smoke), indent=2))


if __name__ == "__main__":
    main()
