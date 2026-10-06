"""Read-only verification of frozen Grad-CAM evidence; writes separate audit artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "analysis/stage20_reliability_completion"
OUTPUT = BASE / "final_model_gradcam_output_v2"
CHECKPOINT = ROOT / "experiments/stage15_single_candidate_exp1/stage15_single_candidate_exp1/recovery_epoch16/best_checkpoint.pt"
PREDICTIONS = ROOT / "analysis/stage16_final_evaluation/final_ham_test_predictions.csv"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
EXPECTED_CHECKPOINT_SHA256 = "85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_image(path: Path, mode: str) -> np.ndarray:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.mode != mode or image.size != (384, 384):
            raise ValueError(f"Unexpected image mode/size: {path}: {image.mode} {image.size}")
        values = np.asarray(image).copy()
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite image: {path}")
    return values


def main() -> None:
    selection_path = BASE / "gradcam_selection.json"
    manifest_path = OUTPUT / "gradcam_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256(CHECKPOINT)
    source_hash = sha256(PREDICTIONS)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Frozen checkpoint SHA256 mismatch")
    if (manifest["checkpoint_sha256"] != checkpoint_hash or
        selection["checkpoint_sha256"] != checkpoint_hash or
        manifest["selection_manifest_sha256"] != sha256(selection_path) or
        selection["source_predictions_sha256"] != source_hash or
        manifest["status"] != "MAPS_GENERATED_REVIEW_PENDING" or
        selection["target_layer"] != "model.resnet.nonlocal3" or
        selection["target_class"] != "saved predicted class"):
        raise ValueError("Grad-CAM source/selection manifest mismatch")
    with PREDICTIONS.open(newline="", encoding="utf-8") as stream:
        rows = {r["image_id"]: r for r in csv.DictReader(stream)}
    if len(rows) != 1014:
        raise ValueError("Frozen HAM prediction count mismatch")
    selected = selection["cases"]
    generated = manifest["cases"]
    if len(selected) != 8 or len(generated) != 8:
        raise ValueError("Not exactly eight cases")
    expected_names = {"gradcam_manifest.json"}
    cases = []
    display_rows = []
    for a, b in zip(selected, generated):
        if any(a[key] != b[key] for key in ("group", "image_id", "true_class", "predicted_class", "saved_probability")):
            raise ValueError(f"Selection mismatch: {a['image_id']}")
        row = rows[a["image_id"]]
        if (a["true_class"], a["predicted_class"]) != (row["true_class"], row["predicted_class"]):
            raise ValueError(f"Saved label mismatch: {a['image_id']}")
        probability = float(row[f"p_{a['predicted_class']}"])
        if probability != a["saved_probability"] or abs(b["replayed_probability"] - probability) > 5e-4:
            raise ValueError(f"Probability reproduction mismatch: {a['image_id']}")
        probabilities = np.asarray([float(row[f"p_{c}"]) for c in CLASSES])
        entropy = float(-(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])).sum())
        name = f"{a['group']}_{a['image_id']}"
        map_path = OUTPUT / f"{name}_map.png"
        overlay_path = OUTPUT / f"{name}_overlay.png"
        original_path = ROOT / "data/processed/images" / f"{a['image_id']}.jpg"
        if any((sha256(path) != b[key]) for path, key in ((map_path, "map_sha256"),
                                                          (overlay_path, "overlay_sha256"),
                                                          (original_path, "processed_image_sha256"))):
            raise ValueError(f"Image SHA256 mismatch: {a['image_id']}")
        heatmap = checked_image(map_path, "L")
        overlay = checked_image(overlay_path, "RGB")
        if len(np.unique(heatmap)) < 2 or heatmap.max() <= heatmap.min():
            raise ValueError(f"Degenerate Grad-CAM map: {a['image_id']}")
        if (b["map_min"], b["map_max"]) != (0.0, 1.0):
            raise ValueError(f"Unexpected normalized map range: {a['image_id']}")
        with Image.open(original_path) as original:
            original = original.convert("RGB").resize((384, 384))
        display_rows.append((a, original, Image.fromarray(overlay), Image.fromarray(heatmap)))
        expected_names.update({map_path.name, overlay_path.name})
        cases.append({"image_id": a["image_id"], "group": a["group"],
                      "true_class": a["true_class"], "predicted_class": a["predicted_class"],
                      "correct": a["true_class"] == a["predicted_class"],
                      "target_class": a["predicted_class"], "confidence": probability,
                      "predictive_entropy": entropy, "replayed_probability": b["replayed_probability"],
                      "map_unique_8bit_values": int(len(np.unique(heatmap))),
                      "map_pixel_min": int(heatmap.min()), "map_pixel_max": int(heatmap.max()),
                      "map_nonzero_fraction": float(np.mean(heatmap > 0)),
                      "map_path": str(map_path.relative_to(ROOT)),
                      "overlay_path": str(overlay_path.relative_to(ROOT))})
    actual_names = {p.name for p in OUTPUT.iterdir() if p.is_file()}
    if actual_names != expected_names:
        raise ValueError(f"Unexpected/missing Grad-CAM output names: {actual_names ^ expected_names}")
    audit = {"status": "INTEGRITY_PASS_VISUAL_REVIEW_PENDING",
             "checkpoint_sha256": checkpoint_hash, "selection_sha256": sha256(selection_path),
             "output_manifest_sha256": sha256(manifest_path), "saved_predictions_sha256": source_hash,
             "case_count": len(cases), "map_count": 8, "overlay_count": 8,
             "source_script": "final_model_gradcam.py: frozen preflight, model.eval(), autograd.grad for CAM, no optimizer/step/save_state",
             "cases": cases}
    with (BASE / "gradcam_integrity.json").open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")
    for block in range(2):
        sheet = Image.new("RGB", (1152, 4 * 418), "white")
        pen = ImageDraw.Draw(sheet)
        for offset, (case, original, overlay, heatmap) in enumerate(display_rows[block * 4:(block + 1) * 4]):
            y = offset * 418
            title = f"{case['image_id']}  true={case['true_class']} predicted={case['predicted_class']}  original / overlay / map"
            pen.text((5, y + 3), title, fill="black")
            for column, panel in enumerate((original, overlay, heatmap.convert("RGB"))):
                sheet.paste(panel, (column * 384, y + 26))
        sheet.save(BASE / f"gradcam_audit_contact_sheet_{block + 1}.png")
    print(json.dumps({"status": audit["status"], "cases": len(cases),
                      "selection_sha256": audit["selection_sha256"],
                      "checkpoint_sha256": checkpoint_hash}))


if __name__ == "__main__":
    main()
