"""Frozen mapped PH2 follow-up for Stage 15 candidate; --preflight never runs inference."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from common import (CLASSES, ROOT, CHECKPOINT_SHA, PH2_REL, ph2_preflight, load_model,
                    read_csv, require_t4, runtime, sha256, source_hashes, write_csv, write_json)

OUTPUT_NAMES = ("final_ph2_predictions.csv", "final_ph2_metrics.json",
                "final_ph2_confusion_matrix.csv", "final_ph2_manifest.json")


def summarize(rows: list[dict]) -> tuple[dict, list[dict]]:
    matrix = [[0, 0, 0], [0, 0, 0]]
    for row in rows:
        i = 0 if row["true_ham_label"] == "nv" else 1
        j = 0 if row["predicted_class"] == "nv" else 1 if row["predicted_class"] == "mel" else 2
        matrix[i][j] += 1
    if sum(map(sum, matrix)) != 120 or list(map(sum, matrix)) != [80, 40]:
        raise ValueError("PH2 mapped cohort counts changed")
    per = {}
    for i, name in enumerate(("nv", "mel")):
        tp = matrix[i][i]
        fp = matrix[1 - i][i]
        fn = sum(matrix[i]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[name] = {"support": sum(matrix[i]), "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1}
    auc = {"status": "unavailable", "value": None, "reason": None}
    try:
        from sklearn.metrics import roc_auc_score
        auc = {"status": "computed", "value": float(roc_auc_score(
            [int(r["true_ham_label"] == "mel") for r in rows],
            [float(r["p_mel"]) for r in rows])), "reason": None}
    except (ImportError, ValueError) as exc:
        auc["reason"] = str(exc)
    metrics = {"sample_count": 120, "true_rows": ["nv", "mel"],
               "predicted_columns": ["nv", "mel", "other"], "confusion_matrix": matrix,
               "other_policy": "Other seven-class predictions remain other and count as incorrect.",
               "accuracy": (matrix[0][0] + matrix[1][1]) / 120,
               "balanced_accuracy": (per["nv"]["recall"] + per["mel"]["recall"]) / 2,
               "per_class": per, "mel_recall": per["mel"]["recall"],
               "mel_precision": per["mel"]["precision"], "mel_f1": per["mel"]["f1"],
               "nv_recall": per["nv"]["recall"], "melanoma_probability_roc_auc": auc,
               "limitation": "PH2 was previously used in this project; external follow-up, not untouched validation."}
    confusion = [{"true_class": label, "nv": matrix[i][0], "mel": matrix[i][1],
                  "other": matrix[i][2]} for i, label in enumerate(("nv", "mel"))]
    return metrics, confusion


def run(root: Path, batch_size: int, workers: int) -> None:
    import torch
    from torch.utils.data import DataLoader
    print(json.dumps(ph2_preflight(root), indent=2), flush=True)
    require_t4()
    if batch_size < 1 or workers < 0:
        raise ValueError("Invalid loader settings")
    output = root / "analysis/stage16_final_evaluation"
    if any((output / name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError("Stage 16 PH2 result already exists; refusing repeat inference")
    sys.path.insert(0, str(root / "analysis/egvan_ph2_external_followup_exp1"))
    sys.path.insert(0, str(root / "src"))
    from evaluate_stage11 import PH2Dataset
    from preprocessing import preprocess_image
    model, transform = load_model(root)
    manifest_rows = read_csv(root / PH2_REL)
    included = [r for r in manifest_rows if r["included"] == "True"]
    ds = PH2Dataset(root, included, transform, preprocess_image)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    rows = []
    with torch.inference_mode():
        for images in loader:
            with torch.autocast("cuda"):
                logits = model(images.cuda(non_blocking=True))
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Non-finite PH2 logits")
            probabilities = torch.softmax(logits.float(), dim=1).cpu().tolist()
            for probability in probabilities:
                source = included[len(rows)]
                if any(not math.isfinite(v) or v < 0 or v > 1 for v in probability):
                    raise ValueError("Invalid PH2 probability")
                prediction = CLASSES[max(range(7), key=lambda i: probability[i])]
                rows.append({"image_id": source["image_id"],
                             "true_external_label": source["true_external_label"],
                             "true_ham_label": source["true_ham_label"],
                             "predicted_class": prediction,
                             **{f"p_{name}": probability[i] for i, name in enumerate(CLASSES)}})
    if len(rows) != 120 or len({r["image_id"] for r in rows}) != 120:
        raise ValueError("PH2 result count/ID failure")
    metrics, confusion = summarize(rows)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / OUTPUT_NAMES[0], rows, ["image_id", "true_external_label",
              "true_ham_label", "predicted_class", *(f"p_{c}" for c in CLASSES)])
    write_json(output / OUTPUT_NAMES[1], metrics)
    write_csv(output / OUTPUT_NAMES[2], confusion, ["true_class", "nv", "mel", "other"])
    sources = source_hashes(root, Path(__file__))
    for relative in (PH2_REL, "src/preprocessing.py",
                     "analysis/egvan_ph2_external_followup_exp1/evaluate_stage11.py",
                     "analysis/ph2_external_validation_exp5/external_validation_config.json"):
        sources[relative] = sha256(root / relative)
    write_json(output / OUTPUT_NAMES[3], {"status": "COMPLETE", "checkpoint_sha256": CHECKPOINT_SHA,
        "checkpoint_epoch": 16, "ph2_manifest_sha256": sha256(root / PH2_REL),
        "evaluation_script_sha256": sha256(Path(__file__)), "source_sha256": sources,
        "numerical_protocol": "CUDA FP16 autocast with resnet.nonlocal3 q@k torch.bmm in FP32",
        "mapping": {"common nevus": "nv", "melanoma": "mel", "atypical nevus": None},
        "runtime": runtime(), "artifact_sha256":
            {name: sha256(output / name) for name in OUTPUT_NAMES[:-1]},
        "training_performed": False, "ham_test_not_used_for_selection": True,
        "limitation": "Prior PH2 use in project; external follow-up only."})
    print(json.dumps({"status": "COMPLETE", "sample_count": len(rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(ph2_preflight(args.project_root), indent=2))
    else:
        run(args.project_root.resolve(), args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
