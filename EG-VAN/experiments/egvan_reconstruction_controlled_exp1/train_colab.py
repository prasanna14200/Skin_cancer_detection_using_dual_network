"""Stage 9 EG-VAN preflight, T4 probe, optional tiny sanity, and gated training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.optim import Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.models import EfficientNet_V2_S_Weights

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1]
EXPECTED_CHECKPOINT = "81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a"
EXPECTED_SPLIT = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config() -> dict:
    return json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def imports(root: Path):
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from dataset import CLASS_NAMES, HAM10000Dataset, validate_split_integrity
    from models.egvan import EGVAN
    from train import focal_loss, make_transforms, set_seed
    return CLASS_NAMES, HAM10000Dataset, validate_split_integrity, EGVAN, focal_loss, make_transforms, set_seed


def preflight(root: Path) -> dict:
    cfg = config()
    classes, Dataset, check_split, _, _, transforms, _ = imports(root)
    checkpoint = root / "experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt"
    split = root / cfg["split_csv"]
    hashes = {"experiment5_checkpoint": digest(checkpoint), "frozen_split": digest(split)}
    if hashes != {"experiment5_checkpoint": EXPECTED_CHECKPOINT, "frozen_split": EXPECTED_SPLIT}:
        raise ValueError(f"Protected hash mismatch: {hashes}")
    if list(classes) != cfg["classes"]:
        raise ValueError("Class order differs from Experiment #5")
    with split.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    integrity = check_split(split, require_lesion_isolation=True)
    ids = defaultdict(set)
    counts = {part: Counter() for part in ("train", "val", "test")}
    for row in rows:
        if row["split"] not in counts or row["dx"] not in classes:
            raise ValueError("Unknown partition or class in frozen split")
        ids[row["split"]].add(row["image_id"])
        counts[row["split"]][row["dx"]] += 1
    if any(ids[a] & ids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Image ID overlap across partitions")
    expected = {
        "train": [257, 398, 891, 95, 899, 5366, 109],
        "val": [30, 58, 104, 9, 107, 663, 15],
        "test": [40, 58, 104, 11, 107, 676, 18],
    }
    actual = {part: [counts[part][c] for c in classes] for part in counts}
    if actual != expected or integrity != {"rows": 10015, "unique_lesions": 7470, "crossing_lesions": 0}:
        raise ValueError(f"Split differs from Experiment #5: {actual}, {integrity}")
    images = root / "data/processed/images"
    missing = [r["image_id"] for r in rows if not (images / (r["image_id"] + ".jpg")).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} processed images missing; first: {missing[:5]}")
    train_tf, val_tf = transforms(EfficientNet_V2_S_Weights.DEFAULT)
    datasets = {part: Dataset(images, split, part, train_tf if part == "train" else val_tf) for part in counts}
    if any(len(datasets[p]) != len(ids[p]) for p in counts):
        raise ValueError("Dataset loader count mismatch")
    return {"status": "PASS", "hashes": hashes, "counts": actual,
            "totals": {p: len(ids[p]) for p in ids}, "integrity": integrity,
            "missing_images": 0, "class_order": list(classes),
            "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}


def make_model(root: Path, *, pretrained: bool) -> torch.nn.Module:
    _, _, _, EGVAN, _, _, _ = imports(root)
    # EfficientNet is pretrained in the actual experiment. ResNet status is
    # unresolved in the paper, so this declared random policy is fixed here.
    return EGVAN(num_classes=7, pretrained_efficient=pretrained, pretrained_resnet=False)


def optimizer_scheduler(model):
    o = Adamax(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
    return o, ReduceLROnPlateau(o, mode="min", factor=0.5, patience=1)


def dataset(root: Path, part: str):
    _, Dataset, _, _, _, transforms, _ = imports(root)
    train_tf, val_tf = transforms(EfficientNet_V2_S_Weights.DEFAULT)
    return Dataset(root / "data/processed/images", root / "data/splits/split_leakage_aware.csv",
                   part, train_tf if part == "train" else val_tf)


def sampler(ds, generator):
    labels = [row["dx"] for row in ds.rows]
    if len(labels) != 8015 or Counter(labels) != Counter(dict(zip(config()["classes"], [257,398,891,95,899,5366,109]))):
        raise ValueError("Sampler is not built from frozen TRAIN partition")
    weights = torch.tensor([1.5630495442733532 if c == "mel" else 1.0 for c in labels], dtype=torch.double)
    return WeightedRandomSampler(weights, len(labels), replacement=True, generator=generator)


def metrics(matrix: torch.Tensor) -> dict:
    m = matrix.tolist()
    names = config()["classes"]
    per = {}
    for i, name in enumerate(names):
        tp, support, predicted = m[i][i], sum(m[i]), sum(row[i] for row in m)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[name] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    return {"accuracy": sum(m[i][i] for i in range(7)) / sum(map(sum, m)),
            "macro_f1": sum(v["f1"] for v in per.values()) / 7,
            "balanced_accuracy": sum(v["recall"] for v in per.values()) / 7,
            "per_class": per, "confusion_matrix": m}


def eligible(m: dict) -> bool:
    rule = config()["checkpoint_selection"]
    return (m["per_class"]["mel"]["f1"] >= rule["melanoma_f1_minimum"] and
            m["macro_f1"] >= rule["validation_macro_f1_minimum"] and
            m["per_class"]["nv"]["recall"] >= rule["nevus_recall_minimum"])


def epoch_run(model, loader, device, loss_fn, *, opt=None, scaler=None, accumulation=1, clip=None, predictions=False):
    training = opt is not None
    model.train(training)
    matrix = torch.zeros((7, 7), dtype=torch.long)
    total_loss = total_count = 0
    prediction_rows = []
    if training:
        opt.zero_grad(set_to_none=True)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_idx, (x, y) in enumerate(loader):
            x, y_device = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y_device, alpha=0.25, gamma=2.0)
            if training:
                # Weight each microbatch by its sample count, including the
                # final 15-sample optimizer window in the 8,015-draw epoch.
                window_start = (batch_idx // accumulation) * accumulation
                window_samples = min(accumulation * loader.batch_size,
                                     len(loader.sampler) - window_start * loader.batch_size)
                scaler.scale(loss * (len(y) / window_samples)).backward()
                if (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == len(loader):
                    if clip is not None:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
            pred = logits.argmax(1).detach().cpu()
            for target, p in zip(y.tolist(), pred.tolist()):
                matrix[target, p] += 1
            if predictions:
                offset = total_count
                probabilities = torch.softmax(logits.detach().float(), dim=1).cpu().tolist()
                for j, p in enumerate(pred.tolist()):
                    row = loader.dataset.rows[offset + j]
                    prediction_rows.append({"image_id": row["image_id"], "true_label": row["dx"],
                                            "predicted_label": config()["classes"][p],
                                            "correct": str(p == y[j].item()),
                                            "probabilities": json.dumps(probabilities[j])})
            total_loss += float(loss.detach().float().cpu()) * len(y)
            total_count += len(y)
    return total_loss / total_count, metrics(matrix), prediction_rows


def atomic_checkpoint(path: Path, state: dict):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint_", suffix=".pt", delete=False) as f:
        temp = Path(f.name)
    try:
        torch.save(state, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def memory_probe(root: Path, args):
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Memory probe requires CUDA Tesla T4")
    _, _, _, _, loss_fn, _, set_seed = imports(root)
    results = []
    batch = 16
    while batch >= 1:
        model = opt = scaler = x = y = loss = None
        try:
            set_seed(42)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = make_model(root, pretrained=True).cuda()
            opt, _ = optimizer_scheduler(model)
            scaler = torch.amp.GradScaler("cuda")
            x = torch.randn(batch, 3, 384, 384, device="cuda")
            y = torch.zeros(batch, dtype=torch.long, device="cuda")
            with torch.autocast("cuda"):
                loss = loss_fn(model(x), y, alpha=0.25, gamma=2.0)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            record = {"batch": batch, "status": "PASS", "allocated_bytes": torch.cuda.memory_allocated(),
                      "reserved_bytes": torch.cuda.memory_reserved(), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "free_bytes": free, "total_bytes": total, "headroom_fraction": free / total}
            results.append(record)
            if record["headroom_fraction"] >= 0.15:
                break
        except torch.cuda.OutOfMemoryError:
            results.append({"batch": batch, "status": "CUDA_OOM"})
        finally:
            del model, opt, scaler, x, y, loss
            torch.cuda.empty_cache()
        batch //= 2
    selected = next((r["batch"] for r in results if r["status"] == "PASS" and r["headroom_fraction"] >= 0.15), None)
    accumulation = math.ceil(16 / selected) if selected else None
    report = {"device": torch.cuda.get_device_name(0), "attempts": results,
              "recommended_physical_batch_size": selected, "gradient_accumulation_steps": accumulation,
              "effective_batch_size": selected * accumulation if selected else None,
              "exact_batch_equivalence": selected * accumulation == 16 if selected else False,
              "note": "Probe only; no research checkpoint or full training."}
    (root / "experiments/egvan_reconstruction_controlled_exp1/memory_probe.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def mini_sanity(root: Path, steps: int):
    _, _, _, _, loss_fn, _, set_seed = imports(root)
    set_seed(42)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    ds = dataset(root, "train")
    # Fixed, unaugmented subset makes the tiny loss trend interpretable.
    ds.transform = imports(root)[5](EfficientNet_V2_S_Weights.DEFAULT)[1]
    subset = Subset(ds, list(range(8)))
    loader = DataLoader(subset, batch_size=2, shuffle=False, num_workers=0)
    model = make_model(root, pretrained=False)
    model.train()
    opt, scheduler = optimizer_scheduler(model)
    losses = []
    first_x, first_y = next(iter(loader))
    model.eval()
    with torch.no_grad():
        initial_loss = float(loss_fn(model(first_x), first_y))
    model.train()
    for i, (x, y) in enumerate(loader):
        if i >= steps:
            break
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    scheduler.step(losses[-1])
    model.eval()
    with torch.no_grad():
        final_loss = float(loss_fn(model(first_x), first_y))
    print(json.dumps({"mode": "mini_sanity", "optimization_steps": len(losses),
                      "subset_samples": len(subset), "initial_loss": initial_loss,
                      "final_loss": final_loss, "first_step_loss": losses[0], "last_step_loss": losses[-1],
                      "research_result": False}))


def train(root: Path, args):
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Full training requires Colab Tesla T4")
    if args.batch_size * args.gradient_accumulation != 16:
        raise ValueError("Effective batch must be exactly 16; select physical batch divisor of 16")
    probe_path = root / "experiments/egvan_reconstruction_controlled_exp1/memory_probe.json"
    if not probe_path.is_file():
        raise RuntimeError("Run --memory-probe on this T4 before full training")
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe.get("device") != torch.cuda.get_device_name(0) or not probe.get("recommended_physical_batch_size"):
        raise RuntimeError("No passing memory probe for this T4")
    if args.batch_size > probe["recommended_physical_batch_size"]:
        raise ValueError("Physical batch exceeds probed safe size")
    _, _, _, _, loss_fn, _, set_seed = imports(root)
    set_seed(42)
    ds, val = dataset(root, "train"), dataset(root, "val")
    generator = torch.Generator(device="cpu").manual_seed(42)
    train_loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler(ds, generator),
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    model = make_model(root, pretrained=not args.resume).cuda()
    opt, scheduler = optimizer_scheduler(model)
    scaler = torch.amp.GradScaler("cuda")
    output = root / "experiments/egvan_reconstruction_controlled_exp1"
    run_cfg = config() | {"physical_batch_size": args.batch_size, "gradient_accumulation_steps": args.gradient_accumulation,
                          "effective_batch_size": args.batch_size * args.gradient_accumulation,
                          "num_workers": args.num_workers, "epochs": args.epochs}
    start, best, best_epoch, history = 1, math.inf, None, []
    if args.resume:
        state = torch.load(output / "last_checkpoint.pt", map_location="cpu", weights_only=False)
        if state["training_configuration"] != run_cfg or state["class_order"] != run_cfg["classes"]:
            raise ValueError("Resume configuration or class order mismatch")
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        generator.set_state(state["sampler_generator_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        start, best, best_epoch, history = state["epoch"] + 1, state["best_validation_metric"], state["best_epoch"], state["history"]
    elif (output / "last_checkpoint.pt").exists() or (output / "best_checkpoint.pt").exists():
        raise FileExistsError("Training artifacts exist; use --resume to continue")
    else:
        run_cfg["status"] = "TRAINING_IN_PROGRESS"
        (output / "config.json").write_text(json.dumps(run_cfg, indent=2) + "\n", encoding="utf-8")
    for epoch in range(start, args.epochs + 1):
        train_loss, _, _ = epoch_run(model, train_loader, torch.device("cuda"), loss_fn, opt=opt, scaler=scaler,
                                     accumulation=args.gradient_accumulation, clip=args.clip_grad)
        val_loss, val_metrics, _ = epoch_run(model, val_loader, torch.device("cuda"), loss_fn)
        scheduler.step(val_loss)
        gate = eligible(val_metrics)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "validation_macro_f1": val_metrics["macro_f1"],
                        "validation_mel_f1": val_metrics["per_class"]["mel"]["f1"],
                        "validation_nv_recall": val_metrics["per_class"]["nv"]["recall"],
                        "eligible": gate, "learning_rate": opt.param_groups[0]["lr"]})
        improved = gate and val_loss < best
        if improved:
            best, best_epoch = val_loss, epoch
        state = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                 "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                 "best_validation_metric": best, "best_epoch": best_epoch, "class_order": run_cfg["classes"],
                 "training_configuration": run_cfg, "architecture": run_cfg["architecture"], "history": history,
                 "sampler_generator_state": generator.get_state(), "python_rng_state": random.getstate(),
                 "numpy_rng_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(),
                 "cuda_rng_states": torch.cuda.get_rng_state_all()}
        if improved:
            atomic_checkpoint(output / "best_checkpoint.pt", state)
        atomic_checkpoint(output / "last_checkpoint.pt", state)
        write_csv(output / "training_history.csv", history, list(history[0]))
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} eligible={gate} best={best_epoch}", flush=True)
    if best_epoch is None:
        status = "FAIL_NO_ELIGIBLE_CHECKPOINT"
    else:
        state = torch.load(output / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        val_loss, val_metrics, pred = epoch_run(model, val_loader, torch.device("cuda"), loss_fn, predictions=True)
        write_csv(output / "validation_predictions.csv", pred,
                  ["image_id", "true_label", "predicted_label", "correct", "probabilities"])
        rows = [{"class": k, **v} for k, v in val_metrics["per_class"].items()]
        write_csv(output / "validation_metrics.csv", rows, ["class", "support", "precision", "recall", "f1"])
        status = "TRAINED_VALIDATION_ONLY"
    (output / "experiment_manifest.json").write_text(json.dumps({"status": status, "best_epoch": best_epoch,
        "best_validation_loss": best if best_epoch else None, "completed_epochs": len(history),
        "ham_test_inference_performed": False, "ph2_inference_performed": False}, indent=2) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--clip-grad", type=float)
    modes = p.add_mutually_exclusive_group(required=True)
    for mode in ("preflight", "memory-probe", "mini-sanity", "train"):
        modes.add_argument("--" + mode, action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.gradient_accumulation < 1 or args.epochs < 1 or args.num_workers < 0:
        p.error("Batch, accumulation, epochs must be positive; workers cannot be negative")
    root = args.project_root.resolve()
    report = preflight(root)
    print(json.dumps(report, indent=2), flush=True)
    if args.memory_probe:
        memory_probe(root, args)
    elif args.mini_sanity:
        mini_sanity(root, steps=3)
    elif args.train:
        train(root, args)


if __name__ == "__main__":
    main()
