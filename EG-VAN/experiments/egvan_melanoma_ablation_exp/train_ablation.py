"""Stage 13 HAM train/validation-only EG-VAN A/B/C ablation runner."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim import Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import EfficientNet_V2_S_Weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSIS = ROOT / "analysis/stage13_melanoma_ablation"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
TRAIN_COUNTS = {"akiec": 257, "bcc": 398, "bkl": 891, "df": 95, "mel": 899, "nv": 5366, "vasc": 109}
VAL_COUNTS = {"akiec": 30, "bcc": 58, "bkl": 104, "df": 9, "mel": 107, "nv": 663, "vasc": 15}
SPLIT_SHA = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
STAGE9_SHA = "60f34ea4cfa6cbf3dce69e8d8bcc82292d8d8813af30f33b2f666a4f06340de0"
STAGE9_GATES = {"mel_f1": 0.5757731958762886, "macro_f1": 0.6316582381362074, "nv_recall": 0.9}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def variant_config(name: str) -> dict:
    if name not in "ABC" or len(name) != 1:
        raise ValueError("Variant must be A, B, or C")
    return read_json(HERE / f"config_{name}.json")


def imports(root: Path):
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from dataset import CLASS_NAMES, HAM10000Dataset, validate_split_integrity
    from models.egvan import EGVAN
    from train import focal_loss, make_transforms, set_seed
    return CLASS_NAMES, HAM10000Dataset, validate_split_integrity, EGVAN, focal_loss, make_transforms, set_seed


def git_state(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}",
                                "-c", f"safe.directory={root.parent.as_posix()}", "status", "--short"],
                                cwd=root, capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def preflight(root: Path) -> dict:
    root = root.resolve()
    if sha256(root / "data/splits/split_leakage_aware.csv") != SPLIT_SHA:
        raise ValueError("Frozen split SHA mismatch")
    if sha256(root / "experiments/egvan_reconstruction_controlled_exp1/best_checkpoint.pt") != STAGE9_SHA:
        raise ValueError("Stage 9 control checkpoint SHA mismatch")
    names, Dataset, validate, _, _, make_transforms, _ = imports(root)
    if tuple(names) != CLASSES:
        raise ValueError("Class order changed")
    split = root / "data/splits/split_leakage_aware.csv"
    integrity = validate(split, require_lesion_isolation=True)
    if integrity != {"rows": 10015, "unique_lesions": 7470, "crossing_lesions": 0}:
        raise ValueError("Frozen split lesion integrity changed")
    counts = {}
    with split.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] in ("train", "val"):
                counts.setdefault(row["split"], Counter())[row["dx"]] += 1
    if dict(counts["train"]) != TRAIN_COUNTS or dict(counts["val"]) != VAL_COUNTS:
        raise ValueError("HAM train/validation counts changed")
    _, eval_tf = make_transforms(EfficientNet_V2_S_Weights.DEFAULT)
    if [type(x).__name__ for x in eval_tf.transforms] != ["Resize", "ToTensor", "Normalize"]:
        raise ValueError("Stage 9 evaluation transform changed")
    images = root / "data/processed/images"
    train = Dataset(images, split, "train", None)
    val = Dataset(images, split, "val", None)
    missing = [x["image_id"] for x in (*train.rows, *val.rows) if not (images / f"{x['image_id']}.jpg").is_file()]
    if missing or len(train) != 8015 or len(val) != 986:
        raise ValueError(f"Train/validation image integrity failure: {missing[:5]}")
    configs = {name: variant_config(name) for name in "ABC"}
    for name in "ABC":
        cfg = configs[name]
        if (tuple(cfg["classes"]) != CLASSES or cfg["epochs"] != 25 or cfg["seed"] != 42 or
                cfg["physical_batch_size"] != 16 or cfg["effective_batch_size"] != 16 or
                cfg["gradient_accumulation_steps"] != 1 or cfg["image_size"] != 384):
            raise ValueError(f"Variant {name} changes frozen core protocol")
    for name in "BC":
        control = json.loads(json.dumps(configs["A"]))
        candidate = json.loads(json.dumps(configs[name]))
        for cfg in (control, candidate):
            cfg.pop("variant")
            cfg.pop("description")
        if name == "B":
            control["loss"]["name"] = candidate["loss"]["name"]
            control["loss"]["mel_multiplier"] = candidate["loss"]["mel_multiplier"]
        else:
            control["sampler"]["mel_weight"] = candidate["sampler"]["mel_weight"]
        if control != candidate:
            raise ValueError(f"Variant {name} changes fields outside its registered ablation")
    # Validate the registered reference using validation predictions only.
    stage9_val = root / "experiments/egvan_reconstruction_controlled_exp1/validation_predictions.csv"
    with stage9_val.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 986 or len({x["image_id"] for x in rows}) != 986:
        raise ValueError("Stage 9 validation predictions incomplete")
    val_ids = {x["image_id"]: x["dx"] for x in val.rows}
    if {x["image_id"]: x["true_label"] for x in rows} != val_ids:
        raise ValueError("Stage 9 validation IDs differ from frozen validation")
    matrix = [[0] * 7 for _ in range(7)]
    for row in rows:
        matrix[CLASSES.index(row["true_label"])][CLASSES.index(row["predicted_label"])] += 1
    reference = metrics(matrix)
    rule = read_json(root / "analysis/stage13_melanoma_ablation/selection_rule.json")["reference"]
    for a, b in ((reference["accuracy"], rule["accuracy"]),
                 (reference["macro_f1"], rule["macro_f1"]),
                 (reference["per_class"]["mel"]["recall"], rule["melanoma_recall"]),
                 (reference["per_class"]["nv"]["recall"], rule["nevus_recall"])):
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Registered Stage 9 validation reference mismatch")
    return {"status": "PASS", "data_boundary": "train and val loader only; no test or PH2 loader",
            "split_sha256": SPLIT_SHA, "stage9_control_sha256": STAGE9_SHA,
            "counts": {"train": TRAIN_COUNTS, "val": VAL_COUNTS}, "lesion_integrity": integrity,
            "stage9_validation_reference": {"accuracy": reference["accuracy"],
                "macro_f1": reference["macro_f1"], "mel_recall": reference["per_class"]["mel"]["recall"],
                "nv_recall": reference["per_class"]["nv"]["recall"]},
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
            "git_status": git_state(root)}


def loss(logits: torch.Tensor, targets: torch.Tensor, multiplier: float = 1.0) -> torch.Tensor:
    # For multiplier=1, exactly the Stage 9 focal-loss operation order.
    probability = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    ce = nn.functional.cross_entropy(logits, targets, reduction="none")
    terms = 0.25 * (1 - probability).pow(2.0) * ce
    if multiplier != 1.0:
        terms = terms * (1.0 + (multiplier - 1.0) * (targets == CLASSES.index("mel")).to(terms.dtype))
    return terms.mean()


def make_sampler(ds, weight: float, generator: torch.Generator):
    labels = [x["dx"] for x in ds.rows]
    if len(labels) != 8015 or dict(Counter(labels)) != TRAIN_COUNTS:
        raise ValueError("Sampler source is not frozen train")
    weights = torch.tensor([weight if x == "mel" else 1.0 for x in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=8015, replacement=True, generator=generator)


def metrics(matrix: list[list[int]]) -> dict:
    per = {}
    for i, name in enumerate(CLASSES):
        tp = matrix[i][i]; support = sum(matrix[i]); predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[name] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    total = sum(map(sum, matrix))
    return {"sample_count": total, "accuracy": sum(matrix[i][i] for i in range(7))/total,
            "balanced_accuracy": sum(v["recall"] for v in per.values())/7,
            "macro_f1": sum(v["f1"] for v in per.values())/7,
            "per_class": per, "confusion_matrix": matrix}


def eligible(m: dict) -> bool:
    return (m["per_class"]["mel"]["f1"] >= STAGE9_GATES["mel_f1"] and
            m["macro_f1"] >= STAGE9_GATES["macro_f1"] and
            m["per_class"]["nv"]["recall"] >= STAGE9_GATES["nv_recall"])


def epoch_run(model, loader, device, *, train_optimizer=None, scaler=None, multiplier=1.0, predictions=False):
    training = train_optimizer is not None
    model.train(training)
    matrix = [[0] * 7 for _ in range(7)]
    loss_sum = count = 0
    pred_rows = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            target = labels.to(device, non_blocking=True)
            if training:
                train_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                batch_loss = loss(logits, target, multiplier if training else 1.0)
            if training:
                scaler.scale(batch_loss).backward()
                scaler.step(train_optimizer)
                scaler.update()
            pred = logits.argmax(1).detach().cpu().tolist()
            probabilities = torch.softmax(logits.detach().float(), dim=1).cpu().tolist() if predictions else None
            for j, value in enumerate(pred):
                matrix[int(labels[j])][value] += 1
                if predictions:
                    source = loader.dataset.rows[count+j]
                    pred_rows.append({"image_id": source["image_id"], "true_label": source["dx"],
                                      "predicted_label": CLASSES[value], "correct": str(value == int(labels[j])),
                                      "probabilities": json.dumps(probabilities[j])})
            loss_sum += float(batch_loss.detach().float().cpu()) * len(labels)
            count += len(labels)
    return loss_sum/count, metrics(matrix), pred_rows


def optimizer_scheduler(model):
    optimizer = Adamax(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    return optimizer, scheduler


def save_checkpoint(path: Path, state: dict):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".stage13_", suffix=".pt", delete=False) as handle:
        temp = Path(handle.name)
    try:
        torch.save(state, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def select_candidate(summaries: list[dict], reference: dict) -> tuple[dict | None, list[dict]]:
    """Frozen validation-only candidate gate and deterministic ranking."""
    if [row["variant"] for row in summaries] != ["A", "B", "C"]:
        raise ValueError("Final selection requires A/B/C in order")
    control = summaries[0]
    minimum_recall = max(reference["melanoma_recall"], control["mel_recall"]) + 1 / 107
    eligible_candidates = [row for row in summaries[1:]
        if row["mel_recall"] + 1e-12 >= minimum_recall
        and row["accuracy"] + 1e-12 >= reference["accuracy"] - 0.02
        and row["macro_f1"] + 1e-12 >= reference["macro_f1"] - 0.02
        and row["nv_recall"] + 1e-12 >= reference["nevus_recall"] - 0.03
        and row["mel_f1"] + 1e-12 >= STAGE9_GATES["mel_f1"]]
    ranked = sorted(eligible_candidates, key=lambda row: (-row["mel_recall"], -row["mel_f1"],
        -row["macro_f1"], -row["accuracy"], row["validation_loss"], row["variant"]))
    return (ranked[0] if ranked else None), eligible_candidates


def memory_probe(root: Path):
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Stage 13 memory probe requires Tesla T4")
    _, _, _, EGVAN, _, _, set_seed = imports(root)
    set_seed(42)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = EGVAN(num_classes=7, pretrained_efficient=True, pretrained_resnet=False).cuda().train()
    optimizer, _ = optimizer_scheduler(model)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(16, 3, 384, 384, device="cuda")
    y = torch.zeros(16, dtype=torch.long, device="cuda")
    with torch.autocast("cuda"):
        value = loss(model(x), y)
    scaler.scale(value).backward(); scaler.step(optimizer); scaler.update()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    report = {"status": "PASS" if free/total >= 0.15 else "FAIL_INSUFFICIENT_HEADROOM",
              "gpu": torch.cuda.get_device_name(0), "batch": 16, "image_size": 384,
              "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "free_bytes": free,
              "total_bytes": total, "headroom_fraction": free/total, "training_performed": False}
    (HERE / "memory_probe.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Batch 16 probe lacks required 15% GPU headroom")


def train(root: Path, name: str, resume: bool):
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Stage 13 training requires Tesla T4")
    probe = read_json(HERE / "memory_probe.json")
    if probe.get("status") != "PASS" or probe.get("gpu") != torch.cuda.get_device_name(0) or probe.get("batch") != 16:
        raise RuntimeError("Run passing Stage 13 T4 batch-16 memory probe first")
    cfg = variant_config(name)
    _, Dataset, _, EGVAN, _, make_transforms, set_seed = imports(root)
    set_seed(42)
    train_tf, eval_tf = make_transforms(EfficientNet_V2_S_Weights.DEFAULT)
    split = root / "data/splits/split_leakage_aware.csv"
    images = root / "data/processed/images"
    train_ds = Dataset(images, split, "train", train_tf)
    val_ds = Dataset(images, split, "val", eval_tf)
    generator = torch.Generator(device="cpu").manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=16,
        sampler=make_sampler(train_ds, cfg["sampler"]["mel_weight"], generator),
        num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    model = EGVAN(num_classes=7, pretrained_efficient=not resume, pretrained_resnet=False).cuda()
    optimizer, scheduler = optimizer_scheduler(model)
    scaler = torch.amp.GradScaler("cuda")
    out = HERE / name
    if not out.exists():
        out.mkdir()
    start = 1; best = math.inf; best_epoch = None; history = []
    if resume:
        state = torch.load(out / "last_checkpoint.pt", map_location="cpu", weights_only=False)
        if state["variant"] != name or state["configuration"] != cfg:
            raise ValueError("Resume variant/configuration changed")
        model.load_state_dict(state["model_state"]); optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"]); scaler.load_state_dict(state["scaler_state"])
        generator.set_state(state["sampler_generator_state"])
        random.setstate(state["python_rng_state"]); np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"]); torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        start = state["epoch"]+1; best = state["best_validation_loss"]
        best_epoch = state["best_epoch"]; history = state["history"]
    elif any((out / x).exists() for x in ("best_checkpoint.pt", "last_checkpoint.pt", "training_history.csv")):
        raise FileExistsError(f"Variant {name} outputs exist; use --resume only for interrupted same run")
    else:
        (out / "config.json").write_text(json.dumps(cfg, indent=2)+"\n", encoding="utf-8")
    for epoch in range(start, 26):
        train_loss, train_metrics, _ = epoch_run(model, train_loader, torch.device("cuda"),
            train_optimizer=optimizer, scaler=scaler, multiplier=cfg["loss"]["mel_multiplier"])
        val_loss, val_metrics, _ = epoch_run(model, val_loader, torch.device("cuda"))
        scheduler.step(val_loss)
        gate = eligible(val_metrics)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_accuracy": val_metrics["accuracy"], "val_balanced_accuracy": val_metrics["balanced_accuracy"],
               "val_macro_f1": val_metrics["macro_f1"],
               "val_mel_precision": val_metrics["per_class"]["mel"]["precision"],
               "val_mel_recall": val_metrics["per_class"]["mel"]["recall"],
               "val_mel_f1": val_metrics["per_class"]["mel"]["f1"],
               "val_nv_recall": val_metrics["per_class"]["nv"]["recall"],
               "eligible": gate, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if gate and val_loss < best:
            best = val_loss; best_epoch = epoch
            improved = True
        else:
            improved = False
        state = {"variant": name, "epoch": epoch, "best_epoch": best_epoch,
                 "best_validation_loss": best, "configuration": cfg, "class_order": list(CLASSES),
                 "architecture": cfg["architecture"], "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                 "scaler_state": scaler.state_dict(), "history": history,
                 "sampler_generator_state": generator.get_state(), "python_rng_state": random.getstate(),
                 "numpy_rng_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(),
                 "cuda_rng_states": torch.cuda.get_rng_state_all()}
        if improved:
            save_checkpoint(out / "best_checkpoint.pt", state)
        save_checkpoint(out / "last_checkpoint.pt", state)
        write_csv(out / "training_history.csv", history)
        print(f"variant={name} epoch={epoch}/25 train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"mel_recall={row['val_mel_recall']:.6f} eligible={gate} best={best_epoch}", flush=True)
    if best_epoch is None:
        status = "FAIL_NO_ELIGIBLE_CHECKPOINT"
    else:
        state = torch.load(out / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        selected_loss, selected_metrics, pred = epoch_run(model, val_loader, torch.device("cuda"), predictions=True)
        if not math.isclose(selected_loss, best, rel_tol=1e-3, abs_tol=1e-4):
            raise ValueError("Selected validation loss drifted on re-evaluation")
        write_csv(out / "validation_predictions.csv", pred)
        (out / "validation_metrics.json").write_text(json.dumps({"epoch": best_epoch,
            "val_loss": selected_loss, **selected_metrics}, indent=2)+"\n", encoding="utf-8")
        status = "COMPLETE_VALIDATION_ONLY"
    manifest = {"status": status, "variant": name, "epochs": len(history), "selected_epoch": best_epoch,
                "best_validation_loss": best if best_epoch else None,
                "best_checkpoint_sha256": sha256(out / "best_checkpoint.pt") if best_epoch else None,
                "last_checkpoint_sha256": sha256(out / "last_checkpoint.pt"),
                "split_sha256": SPLIT_SHA, "stage9_control_sha256": STAGE9_SHA,
                "class_order": list(CLASSES), "runtime": {"python": platform.python_version(),
                    "torch": torch.__version__, "torchvision": torchvision.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
                "git_status": git_state(root), "ham_test_accessed": False, "ph2_accessed": False,
                "test_inference_performed": False, "ph2_inference_performed": False}
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def finalize(root: Path):
    rule = read_json(root / "analysis/stage13_melanoma_ablation/selection_rule.json")
    reference = rule["reference"]
    summaries = []; perrows = []; stability = []
    for name in "ABC":
        out = HERE / name
        manifest = read_json(out / "experiment_manifest.json")
        if manifest["status"] != "COMPLETE_VALIDATION_ONLY" or manifest["epochs"] != 25:
            raise ValueError(f"Variant {name} not complete with eligible best checkpoint")
        if sha256(out / "best_checkpoint.pt") != manifest["best_checkpoint_sha256"]:
            raise ValueError(f"Variant {name} checkpoint hash changed")
        selected = read_json(out / "validation_metrics.json")
        with (out / "training_history.csv").open(newline="", encoding="utf-8") as f:
            history = list(csv.DictReader(f))
        if len(history) != 25 or [int(r["epoch"]) for r in history] != list(range(1,26)):
            raise ValueError(f"Variant {name} history incomplete")
        eligible_rows = [r for r in history if r["eligible"] == "True"]
        winner = min(eligible_rows, key=lambda r: (float(r["val_loss"]), int(r["epoch"])))
        if int(winner["epoch"]) != manifest["selected_epoch"] or selected["epoch"] != manifest["selected_epoch"]:
            raise ValueError(f"Variant {name} checkpoint rule mismatch")
        if not math.isclose(float(winner["val_loss"]), float(selected["val_loss"]), rel_tol=1e-3, abs_tol=1e-4):
            raise ValueError(f"Variant {name} selected validation loss mismatch")
        # Recompute all seven validation class metrics from saved predictions.
        with (out / "validation_predictions.csv").open(newline="", encoding="utf-8") as f:
            pred = list(csv.DictReader(f))
        with (root / "data/splits/split_leakage_aware.csv").open(newline="", encoding="utf-8") as f:
            val_ids = {r["image_id"]: r["dx"] for r in csv.DictReader(f) if r["split"] == "val"}
        if len(pred) != 986 or {r["image_id"]: r["true_label"] for r in pred} != val_ids:
            raise ValueError(f"Variant {name} validation IDs differ")
        matrix = [[0]*7 for _ in range(7)]
        for row in pred:
            probs = json.loads(row["probabilities"])
            if (len(probs) != 7 or any(not isinstance(p, (int, float)) or not math.isfinite(p)
                    or p < 0 or p > 1 for p in probs)
                    or not math.isclose(sum(probs), 1, abs_tol=1e-4)):
                raise ValueError("Invalid validation probability vector")
            if CLASSES[max(range(7), key=lambda i: probs[i])] != row["predicted_label"]:
                raise ValueError("Validation argmax mismatch")
            if row["correct"] != str(row["predicted_label"] == row["true_label"]):
                raise ValueError("Validation correctness flag mismatch")
            matrix[CLASSES.index(row["true_label"])][CLASSES.index(row["predicted_label"])] += 1
        derived = metrics(matrix)
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            if not math.isclose(derived[key], selected[key], abs_tol=1e-12):
                raise ValueError(f"Variant {name} validation metric mismatch: {key}")
        for class_name in CLASSES:
            for key in ("support", "precision", "recall", "f1"):
                if not math.isclose(derived["per_class"][class_name][key], selected["per_class"][class_name][key], abs_tol=1e-12):
                    raise ValueError(f"Variant {name} class metric mismatch")
                perrows.append({"variant": name, "class": class_name, "metric": key,
                                "value": derived["per_class"][class_name][key]})
        summaries.append({"variant": name, "selected_epoch": selected["epoch"],
            "validation_loss": selected["val_loss"], "accuracy": derived["accuracy"],
            "balanced_accuracy": derived["balanced_accuracy"], "macro_f1": derived["macro_f1"],
            "mel_precision": derived["per_class"]["mel"]["precision"],
            "mel_recall": derived["per_class"]["mel"]["recall"],
            "mel_f1": derived["per_class"]["mel"]["f1"],
            "nv_recall": derived["per_class"]["nv"]["recall"],
            "checkpoint_sha256": manifest["best_checkpoint_sha256"]})
        stability.append({"variant": name, "epochs":25,
                          "eligible_epochs": len(eligible_rows),
                          "first_train_loss": float(history[0]["train_loss"]),
                          "last_train_loss": float(history[-1]["train_loss"]),
                          "first_val_loss": float(history[0]["val_loss"]),
                          "last_val_loss": float(history[-1]["val_loss"]),
                          "minimum_val_loss": min(float(r["val_loss"]) for r in history),
                          "last_learning_rate": float(history[-1]["learning_rate"])})
    selected, eligible_candidates = select_candidate(summaries, reference)
    for r in summaries:
        r["final_candidate_eligible"] = r in eligible_candidates
    output = root / "analysis/stage13_melanoma_ablation"
    for filename in ("validation_ablation_comparison.csv", "per_class_validation_comparison.csv",
                     "training_stability_comparison.csv", "selected_candidate.json", "stage13_results_report.md"):
        if (output / filename).exists():
            raise FileExistsError(f"Refusing to overwrite Stage 13 result: {filename}")
    write_csv(output / "validation_ablation_comparison.csv", summaries)
    write_csv(output / "per_class_validation_comparison.csv", perrows)
    write_csv(output / "training_stability_comparison.csv", stability)
    choice = {"status": "CANDIDATE_SELECTED" if selected else "NO_CANDIDATE_SELECTED",
              "selected_variant": selected["variant"] if selected else None,
              "checkpoint_path": f"experiments/egvan_melanoma_ablation_exp/{selected['variant']}/best_checkpoint.pt" if selected else None,
              "checkpoint_sha256": selected["checkpoint_sha256"] if selected else None,
              "selection_rule": rule, "eligible_candidates": [r["variant"] for r in eligible_candidates],
              "ham_test_inference_performed": False, "ph2_inference_performed": False}
    (output / "selected_candidate.json").write_text(json.dumps(choice, indent=2)+"\n", encoding="utf-8")
    (output / "stage13_results_report.md").write_text(
        "# Stage 13 validation-only ablation results\n\n"
        f"Status: **{choice['status']}**. A/B/C each completed 25 HAM train/validation epochs. "
        "All metrics and the selection decision use validation only. No HAM test or PH2 inference occurred.\n\n"
        f"Variant summaries: {summaries}.\n\n"
        f"Rule: MEL recall must exceed both frozen Stage 9 validation and A by at least 1/107, "
        f"with accuracy >= {reference['accuracy']-0.02:.6f}, macro F1 >= {reference['macro_f1']-0.02:.6f}, "
        f"NV recall >= {reference['nevus_recall']-0.03:.6f}, and MEL F1 >= {STAGE9_GATES['mel_f1']:.6f}. "
        "Tie-breaks are in selection_rule.json. No criteria were relaxed.\n", encoding="utf-8")
    print(json.dumps(choice, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--variant", choices=list("ABC"))
    p.add_argument("--resume", action="store_true")
    mode = p.add_mutually_exclusive_group(required=True)
    for name in ("preflight", "memory-probe", "train", "finalize"):
        mode.add_argument("--"+name, action="store_true")
    args = p.parse_args()
    root = args.project_root.resolve()
    report = preflight(root)
    print(json.dumps(report, indent=2), flush=True)
    if args.memory_probe:
        memory_probe(root)
    elif args.train:
        if not args.variant:
            p.error("--train requires --variant A/B/C")
        train(root, args.variant, args.resume)
    elif args.finalize:
        finalize(root)


if __name__ == "__main__":
    main()
