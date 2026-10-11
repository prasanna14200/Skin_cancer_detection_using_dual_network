"""Read-only Stage 23 HAM/PH2 melanoma-error analysis from frozen artifacts.

No model loading, preprocessing rerun, image writes, inference, or threshold
selection. PH2 raw-image quality proxies are descriptive measurements only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
HAM_PRED = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_predictions.csv"
PH2_PRED = ROOT / "analysis/stage23_external_validation/run/predictions.csv"
HAM_UNCERTAINTY = ROOT / "analysis/stage23_uncertainty_final/validation_uncertainty.csv"
HAM_QUALITY = ROOT / "analysis/stage23_image_quality_final/image_quality_metrics.csv"
PH2_MANIFEST = ROOT / "analysis/ph2_external_validation_exp5/ph2_manifest.csv"
SPLIT = ROOT / "data/splits/split_leakage_aware.csv"
HAM_CHECKPOINT = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt"
EXPECTED_CHECKPOINT = "42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab"
EXPECTED_SPLIT = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def checked_index(rows: list[dict], *, expected: int, name: str) -> dict[str, dict]:
    by_id = {r["image_id"]: r for r in rows}
    if len(rows) != expected or len(by_id) != expected:
        raise ValueError(f"{name} rows/unique IDs differ from {expected}")
    return by_id


def probability_check(values: list[float], predicted: str) -> None:
    if (len(values) != 7 or any(not math.isfinite(x) or x < 0 or x > 1 for x in values)
            or abs(sum(values)-1) > 1e-5 or
            CLASSES[max(range(7), key=values.__getitem__)] != predicted):
        raise ValueError("Non-finite, unnormalized, or argmax-inconsistent probabilities")


def distribution(rows: list[dict], key: str) -> dict:
    values = [float(r[key]) for r in rows]
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def subgroup_auc(correct: list[dict], missed: list[dict], key: str) -> float:
    from sklearn.metrics import roc_auc_score
    values = [float(r[key]) for r in correct+missed]
    return float(roc_auc_score([0]*len(correct)+[1]*len(missed), values))


def audit_and_build() -> tuple[list[dict], list[dict], dict]:
    import cv2
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT/"src"))
    from image_quality import QUALITY_FEATURES, compute_image_quality

    registry = json.loads((ROOT/"models/frozen_stage23/model_registry.json").read_text())
    ph2_run = json.loads((ROOT/"analysis/stage23_external_validation/run/manifest.json").read_text())
    uncertainty_manifest = json.loads((ROOT/"analysis/stage23_uncertainty_final/analysis_manifest.json").read_text())
    quality_manifest = json.loads((ROOT/"analysis/stage23_image_quality_final/quality_analysis_manifest.json").read_text())
    if (sha(HAM_CHECKPOINT) != EXPECTED_CHECKPOINT or sha(SPLIT) != EXPECTED_SPLIT or
            tuple(registry["class_order"]) != CLASSES or
            registry["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT or
            sha(HAM_PRED) != registry["source_artifact_sha256"][str(HAM_PRED.relative_to(ROOT)).replace("\\", "/")] or
            sha(HAM_UNCERTAINTY) != uncertainty_manifest["output_sha256"]["validation_uncertainty.csv"] or
            sha(HAM_QUALITY) != quality_manifest["output_sha256"]["image_quality_metrics.csv"] or
            sha(PH2_PRED) != ph2_run["artifact_sha256"]["predictions.csv"] or
            sha(PH2_MANIFEST) != ph2_run["ph2_manifest_sha256"] or
            ph2_run["checkpoint_sha256"] != EXPECTED_CHECKPOINT or
            tuple(ph2_run["class_order"]) != CLASSES):
        raise ValueError("Frozen input/source hash or model identity mismatch")
    ham_rows = read_csv(HAM_PRED)
    ph2_rows = read_csv(PH2_PRED)
    ham_unc = checked_index(read_csv(HAM_UNCERTAINTY), expected=986, name="HAM uncertainty")
    ham_qual = checked_index(read_csv(HAM_QUALITY), expected=986, name="HAM quality")
    ham_by_id = checked_index(ham_rows, expected=986, name="HAM predictions")
    ph2_by_id = checked_index(ph2_rows, expected=120, name="PH2 predictions")
    val_rows = [r for r in read_csv(SPLIT) if r["split"] == "val"]
    val = checked_index(val_rows, expected=986, name="HAM frozen val split")
    included = [r for r in read_csv(PH2_MANIFEST) if r["included"] == "True"]
    ph2_fixed = checked_index(included, expected=120, name="PH2 fixed cohort")
    if set(ham_by_id) != set(val) or set(ham_unc) != set(val) or set(ham_qual) != set(val):
        raise ValueError("HAM validation join IDs differ")
    if set(ph2_by_id) != set(ph2_fixed):
        raise ValueError("PH2 fixed-cohort prediction IDs differ")
    if set(ham_by_id) & set(ph2_by_id):
        raise ValueError("HAM and PH2 image IDs overlap")
    if Counter(r["true_ham_label"] for r in ph2_rows) != {"nv":80,"mel":40}:
        raise ValueError("PH2 included class distribution changed")
    ham_mel_all, ph2_mel_all = [], []
    ham_errors, ph2_errors = [], []
    for source in ham_rows:
        image_id = source["image_id"]
        if (source["true_label"] != val[image_id]["dx"] or
                source["true_label"] != ham_qual[image_id]["true_class"] or
                source["predicted_label"] != ham_unc[image_id]["predicted_label"] or
                source["predicted_label"] != ham_qual[image_id]["predicted_class"] or
                source["correct"] != str(source["predicted_label"] == source["true_label"]) or
                ham_qual[image_id]["lesion_id"] != val[image_id]["lesion_id"]):
            raise ValueError(f"HAM label/lesion/prediction mismatch: {image_id}")
        p = [float(x) for x in json.loads(source["probabilities"])]
        probability_check(p, source["predicted_label"])
        entropy = -sum(x*math.log(x) for x in p if x > 0)
        if (abs(entropy-float(ham_unc[image_id]["predictive_entropy_nats"])) > 1e-12 or
                abs(max(p)-float(ham_unc[image_id]["maximum_softmax_probability"])) > 1e-12 or
                abs(entropy-float(ham_qual[image_id]["predictive_entropy_nats"])) > 1e-12):
            raise ValueError(f"HAM saved uncertainty mismatch: {image_id}")
        if source["true_label"] != "mel":
            continue
        row = {"image_id":image_id, "lesion_id":val[image_id]["lesion_id"],
               "true_class":"mel", "predicted_class":source["predicted_label"],
               "correct":int(source["predicted_label"] == "mel"),
               "confidence":max(p), "p_mel":p[4], "entropy_nats":entropy,
               "normalized_entropy":entropy/math.log(7),
               "image_path":f"data/raw/images/{image_id}.jpg",
               "raw_width":int(ham_qual[image_id]["raw_width"]),
               "raw_height":int(ham_qual[image_id]["raw_height"]),
               "quality_source":"saved Stage23 raw JPEG quality table"}
        for feature in QUALITY_FEATURES:
            row[f"raw_{feature}"] = float(ham_qual[image_id][f"raw_{feature}"])
        ham_mel_all.append(row)
        if not row["correct"]:
            ham_errors.append(row)
    source_image_hashes = ph2_run["source_image_sha256"]
    for source in ph2_rows:
        image_id = source["image_id"]
        fixed = ph2_fixed[image_id]
        if (source["true_ham_label"] != fixed["true_ham_label"] or
                source["true_external_label"] != fixed["true_external_label"] or
                source["predicted_class"] not in CLASSES):
            raise ValueError(f"PH2 truth/prediction mismatch: {image_id}")
        p = [float(source[f"p_{name}"]) for name in CLASSES]
        probability_check(p, source["predicted_class"])
        entropy = -sum(x*math.log(x) for x in p if x > 0)
        if (abs(entropy-float(source["entropy_nats"])) > 1e-12 or
                abs(max(p)-float(source["confidence"])) > 1e-12 or
                source["mapped_prediction"] != (source["predicted_class"] if source["predicted_class"] in ("nv","mel") else "other")):
            raise ValueError(f"PH2 saved uncertainty/mapping mismatch: {image_id}")
        if source["true_ham_label"] != "mel":
            continue
        image = ROOT/fixed["relative_image_path"]
        if sha(image) != source_image_hashes[image_id]:
            raise ValueError(f"PH2 image changed since inference: {image_id}")
        with Image.open(image) as bitmap:
            width, height = bitmap.size
        quality = compute_image_quality(image)
        row = {"image_id":image_id, "lesion_id":"unavailable",
               "true_class":"mel", "predicted_class":source["predicted_class"],
               "correct":int(source["predicted_class"] == "mel"),
               "confidence":max(p), "p_mel":p[4], "entropy_nats":entropy,
               "normalized_entropy":entropy/math.log(7),
               "image_path":fixed["relative_image_path"],
               "raw_width":width,"raw_height":height,
               "quality_source":"new descriptive raw PH2 BMP proxy; no quality label"}
        for feature in QUALITY_FEATURES:
            row[f"raw_{feature}"] = quality[feature]
        ph2_mel_all.append(row)
        if not row["correct"]:
            ph2_errors.append(row)
    if (len(ham_mel_all),len(ham_errors),len(ph2_mel_all),len(ph2_errors)) != (107,37,40,27):
        raise ValueError("MEL support or false-negative count mismatch")
    if (Counter(r["predicted_class"] for r in ham_errors)["nv"] != 26 or
            Counter(r["predicted_class"] for r in ph2_errors)["nv"] != 13):
        raise ValueError("MEL-to-NV error count mismatch")
    summary = {"status":"PASS_SAVED_PREDICTION_AND_IMAGE_PROXY_AUDIT",
               "checkpoint_sha256":EXPECTED_CHECKPOINT,
               "source_sha256": {str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in
                   (HAM_PRED,PH2_PRED,HAM_UNCERTAINTY,HAM_QUALITY,PH2_MANIFEST,SPLIT)},
               "ham":{},"ph2":{},"quality_note":"PH2 raw BMP proxies computed read-only; HAM raw JPEG proxies come from saved Stage23 table; format/resolution confound direct comparisons"}
    for name,all_rows,errors in (("ham",ham_mel_all,ham_errors),("ph2",ph2_mel_all,ph2_errors)):
        correct = [r for r in all_rows if r["correct"]]
        result = {"mel_total":len(all_rows),"mel_correct":len(correct),"mel_false_negatives":len(errors),
                  "error_predictions":dict(Counter(r["predicted_class"] for r in errors)),
                  "false_negative_confidence_ge_0_8":sum(r["confidence"]>=.8 for r in errors),
                  "false_negative_confidence_ge_0_9":sum(r["confidence"]>=.9 for r in errors),
                  "entropy_error_auc_within_true_mel":subgroup_auc(correct,errors,"entropy_nats"),
                  "one_minus_confidence_error_auc_within_true_mel":subgroup_auc(correct,errors,"confidence") * -1 + 1}
        for key in ("confidence","p_mel","entropy_nats","raw_brightness","raw_contrast",
                    "raw_sharpness","raw_saturation","raw_dark_pixel_ratio","raw_bright_pixel_ratio"):
            result[key] = {"correct":distribution(correct,key),"false_negative":distribution(errors,key)}
        result["raw_dimensions"] = dict(Counter(f"{r['raw_width']}x{r['raw_height']}" for r in all_rows))
        summary[name] = result
    return ph2_errors,ham_errors,summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    ph2,ham,summary = audit_and_build()
    if args.write:
        for name,rows in (("PH2_MELANOMA_ERRORS.csv",ph2),("HAM_MELANOMA_ERRORS.csv",ham)):
            with (OUT/name).open("x",newline="",encoding="utf-8") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
        with (OUT/"failure_summary.json").open("x",encoding="utf-8") as stream:
            json.dump(summary,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"status":summary["status"],"ham":{k:summary["ham"][k] for k in
        ("mel_total","mel_correct","mel_false_negatives","error_predictions","entropy_error_auc_within_true_mel")},
        "ph2":{k:summary["ph2"][k] for k in
        ("mel_total","mel_correct","mel_false_negatives","error_predictions","entropy_error_auc_within_true_mel")}},indent=2))


if __name__ == "__main__":
    main()
