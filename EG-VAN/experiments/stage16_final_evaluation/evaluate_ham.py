"""One-time frozen Stage 15 candidate HAM test evaluation; --preflight is read-only."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from common import (CLASSES, ROOT, CHECKPOINT_SHA, SPLIT_REL, preflight, load_model,
                    read_csv, require_t4, runtime, sha256, source_hashes, write_csv, write_json)

OUTPUT_NAMES = ("final_ham_test_predictions.csv", "final_ham_test_metrics.json",
                "final_ham_test_confusion_matrix.csv", "final_ham_test_classwise_metrics.csv",
                "final_ham_test_manifest.json")


def summarize(rows: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    matrix = [[0] * 7 for _ in CLASSES]
    for row in rows:
        matrix[CLASSES.index(row["true_class"])][CLASSES.index(row["predicted_class"])] += 1
    n = sum(map(sum, matrix))
    per = []
    for i, name in enumerate(CLASSES):
        tp = matrix[i][i]
        support = sum(matrix[i])
        fp = sum(matrix[j][i] for j in range(7) if j != i)
        fn = support - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per.append({"class": name, "support": support, "tp": tp, "fp": fp, "fn": fn,
                    "precision": precision, "recall": recall, "f1": f1})
    metrics = {"sample_count": n, "class_order": list(CLASSES),
               "accuracy": sum(matrix[i][i] for i in range(7)) / n,
               "balanced_accuracy": sum(p["recall"] for p in per) / 7,
               "macro_precision": sum(p["precision"] for p in per) / 7,
               "macro_recall": sum(p["recall"] for p in per) / 7,
               "macro_f1": sum(p["f1"] for p in per) / 7,
               "weighted_f1": sum(p["f1"] * p["support"] for p in per) / n,
               "per_class": {p["class"]: {k: p[k] for k in
                            ("support", "tp", "fp", "fn", "precision", "recall", "f1")} for p in per},
               "confusion_matrix": matrix}
    metrics.update({"mel_precision": per[4]["precision"], "mel_recall": per[4]["recall"],
                    "mel_f1": per[4]["f1"], "mel_tp": per[4]["tp"], "mel_fn": per[4]["fn"],
                    "nv_recall": per[5]["recall"]})
    auc = {}
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score
        labels = np.asarray([CLASSES.index(row["true_class"]) for row in rows])
        probabilities = np.asarray([[float(row[f"p_{name}"]) for name in CLASSES] for row in rows])
        auc["ovr_macro"] = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
        auc["ovr_weighted"] = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted"))
        auc["status"] = "computed"
    except (ImportError, ValueError) as exc:
        auc = {"status": "unavailable", "reason": str(exc), "ovr_macro": None, "ovr_weighted": None}
    metrics["multiclass_roc_auc"] = auc
    confusion = [{"true_class": name, **dict(zip(CLASSES, matrix[i]))}
                 for i, name in enumerate(CLASSES)]
    return metrics, per, confusion


def run(root: Path, batch_size: int, workers: int) -> None:
    import torch
    from torch.utils.data import DataLoader
    print(json.dumps(preflight(root), indent=2), flush=True)
    require_t4()
    if batch_size < 1 or workers < 0:
        raise ValueError("Invalid loader settings")
    output = root / "analysis/stage16_final_evaluation"
    if any((output / name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError("Stage 16 HAM result already exists; refusing repeat inference")
    sys.path.insert(0, str(root / "src"))
    from dataset import HAM10000Dataset
    model, transform = load_model(root)
    split = read_csv(root / SPLIT_REL)
    expected = [r for r in split if r["split"] == "test"]
    ds = HAM10000Dataset(root / "data/processed/images", root / SPLIT_REL, "test", transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    rows = []
    with torch.inference_mode():
        for images, labels in loader:
            with torch.autocast("cuda"):
                logits = model(images.cuda(non_blocking=True))
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Non-finite HAM logits")
            probabilities = torch.softmax(logits.float(), dim=1).cpu().tolist()
            for probability in probabilities:
                source = expected[len(rows)]
                prediction = CLASSES[max(range(7), key=lambda i: probability[i])]
                if any(not math.isfinite(v) or v < 0 or v > 1 for v in probability):
                    raise ValueError("Invalid HAM probability")
                rows.append({"image_id": source["image_id"], "true_class": source["dx"],
                             "predicted_class": prediction, "correct": source["dx"] == prediction,
                             **{f"p_{name}": probability[i] for i, name in enumerate(CLASSES)}})
    if len(rows) != 1014 or len({r["image_id"] for r in rows}) != 1014:
        raise ValueError("HAM test result count/ID failure")
    metrics, per, matrix = summarize(rows)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / OUTPUT_NAMES[0], rows,
              ["image_id", "true_class", "predicted_class", "correct", *(f"p_{c}" for c in CLASSES)])
    write_json(output / OUTPUT_NAMES[1], metrics)
    write_csv(output / OUTPUT_NAMES[2], matrix, ["true_class", *CLASSES])
    write_csv(output / OUTPUT_NAMES[3], per,
              ["class", "support", "tp", "fp", "fn", "precision", "recall", "f1"])
    artifact_hashes = {name: sha256(output / name) for name in OUTPUT_NAMES[:-1]}
    manifest = {"status": "COMPLETE", "checkpoint_sha256": CHECKPOINT_SHA, "checkpoint_epoch": 16,
                "split_sha256": sha256(root / SPLIT_REL), "evaluation_script_sha256": sha256(Path(__file__)),
                "source_sha256": source_hashes(root, Path(__file__)), "numerical_protocol":
                "CUDA FP16 autocast with resnet.nonlocal3 q@k torch.bmm in FP32",
                "runtime": runtime(), "artifact_sha256": artifact_hashes,
                "training_performed": False, "ph2_accessed": False}
    write_json(output / OUTPUT_NAMES[4], manifest)
    print(json.dumps({"status": "COMPLETE", "sample_count": len(rows),
                      "checkpoint_sha256": CHECKPOINT_SHA}, indent=2))


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
        print(json.dumps(preflight(args.project_root), indent=2))
    else:
        run(args.project_root.resolve(), args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
