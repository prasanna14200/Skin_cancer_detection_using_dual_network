"""Proposed Stage 13C numerical protocol; isolated from frozen Stage 13 A/B/C runs.

The bounded gate is executable now. Full runs are disabled until that gate
passes on the Tesla T4. No original A/B/C artifact is read for training or
overwritten. Scientific A/B/C configs are imported unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_S_Weights

import train_ablation as base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSIS = ROOT / "analysis/stage13_melanoma_ablation"
GATE_PATH = ANALYSIS / "stage13c_bounded_t4_gate.json"
INITIAL_SCALE = 16384.0
GATE_BATCHES = 8


class NumericalFailure(RuntimeError):
    def __init__(self, context: dict, tensor: str, value: torch.Tensor | None = None):
        status = tensor_summary(value) if value is not None else {}
        self.record = {**context, "affected_tensor": tensor, **status}
        super().__init__(json.dumps(self.record, allow_nan=False))


def tensor_summary(value: torch.Tensor) -> dict:
    x = value.detach()
    finite = torch.isfinite(x)
    valid = x[finite]
    return {"finite": bool(finite.all().item()), "nan_count": int(torch.isnan(x).sum().item()),
            "inf_count": int(torch.isinf(x).sum().item()),
            "finite_min": float(valid.min().item()) if valid.numel() else None,
            "finite_max": float(valid.max().item()) if valid.numel() else None,
            "dtype": str(x.dtype), "shape": list(x.shape)}


def require_finite(value: torch.Tensor, context: dict, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise NumericalFailure(context, name, value)


def context_for(variant: str, epoch: int, batch: int, labels: torch.Tensor, scaler) -> dict:
    counts = Counter(base.CLASSES[int(v)] if 0 <= int(v) < len(base.CLASSES) else f"INVALID_{int(v)}"
                     for v in labels.detach().cpu())
    return {"variant": variant, "epoch": epoch, "batch_index": batch,
            "target_class_distribution": dict(counts), "contains_mel": counts.get("mel", 0) > 0,
            "amp_scale": float(scaler.get_scale()) if scaler is not None else None}


def require_valid_scale(scaler, context: dict) -> None:
    scale = float(scaler.get_scale())
    if not math.isfinite(scale) or scale <= 0:
        raise NumericalFailure(context, "GradScaler.scale")


def require_finite_states(model, optimizer, context: dict) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            require_finite(value, context, "model_state/" + name)
    for key, state in optimizer.state.items():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                require_finite(value, context, "optimizer_state/" + name)


def focal_terms(logits: torch.Tensor, targets: torch.Tensor, multiplier: float) -> torch.Tensor:
    # Preserve the Stage 9/13 operation order and global focal alpha/gamma.
    probability = torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    ce = nn.functional.cross_entropy(logits, targets, reduction="none")
    terms = 0.25 * (1 - probability).pow(2.0) * ce
    if multiplier != 1.0:
        terms = terms * (1.0 + (multiplier - 1.0) *
                         (targets == base.CLASSES.index("mel")).to(terms.dtype))
    return terms


def guarded_step(model, optimizer, scaler, images, labels, variant: str, epoch: int, batch: int):
    context = context_for(variant, epoch, batch, labels, scaler)
    require_valid_scale(scaler, context)
    require_finite(images, context, "input")
    if labels.dtype != torch.long or not bool(((labels >= 0) & (labels < 7)).all().item()):
        raise NumericalFailure(context, "labels")
    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda"):
        logits = model(images)
        require_finite(logits, context, "logits")
        terms = focal_terms(logits, labels, base.variant_config(variant)["loss"]["mel_multiplier"])
        require_finite(terms, context, "per_example_loss")
        loss = terms.mean()
        require_finite(loss, context, "reduced_loss")
    scaled = scaler.scale(loss)
    require_finite(scaled, context, "scaled_loss")
    scaled.backward()
    scaler.unscale_(optimizer)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            require_finite(parameter.grad, context, "unscaled_gradient/" + name)
    scaler.step(optimizer)
    scaler.update()
    require_valid_scale(scaler, context)
    require_finite_states(model, optimizer, context)
    return float(loss.detach().float().cpu()), logits.detach().argmax(1)


def build_components(root: Path, variant: str):
    _, Dataset, _, EGVAN, _, make_transforms, set_seed = base.imports(root)
    set_seed(42)
    train_tf, eval_tf = make_transforms(EfficientNet_V2_S_Weights.DEFAULT)
    split = root / "data/splits/split_leakage_aware.csv"
    images = root / "data/processed/images"
    train_ds = Dataset(images, split, "train", train_tf)
    val_ds = Dataset(images, split, "val", eval_tf)
    generator = torch.Generator(device="cpu").manual_seed(42)
    cfg = base.variant_config(variant)
    train_loader = DataLoader(train_ds, batch_size=16,
        sampler=base.make_sampler(train_ds, cfg["sampler"]["mel_weight"], generator),
        num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    model = EGVAN(num_classes=7, pretrained_efficient=True, pretrained_resnet=False).cuda()
    optimizer, scheduler = base.optimizer_scheduler(model)
    scaler = torch.amp.GradScaler("cuda", init_scale=INITIAL_SCALE)
    return model, optimizer, scheduler, scaler, train_loader, val_loader, generator


def bounded_gate(root: Path, output: Path = GATE_PATH) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite bounded gate result: {output}")
    base.preflight(root)
    report = {"status": "PASS", "prespecified_batches_per_variant": GATE_BATCHES,
              "initial_scale": INITIAL_SCALE, "normal_dynamic_scaler_updates": True,
              "split_sha256": base.SPLIT_SHA,
              "runner_sha256": base.sha256(Path(__file__)),
              "source_sha256": {name: base.sha256(root / name) for name in (
                  "experiments/egvan_melanoma_ablation_exp/train_ablation.py",
                  "src/train.py", "src/dataset.py", "src/models/egvan.py")},
              "variant_config_sha256": {v: base.sha256(HERE / f"config_{v}.json") for v in "ABC"},
              "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                          "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(0)},
              "variants": {}, "checkpoint_saved": False,
              "ham_test_accessed": False, "ph2_accessed": False}
    for variant in "ABC":
        model, optimizer, _, scaler, train_loader, _, _ = build_components(root, variant)
        losses = []
        try:
            for batch, (images, labels) in enumerate(train_loader, start=1):
                if batch > GATE_BATCHES:
                    break
                value, _ = guarded_step(model, optimizer, scaler,
                    images.cuda(non_blocking=True), labels.cuda(non_blocking=True), variant, 1, batch)
                losses.append(value)
            if len(losses) != GATE_BATCHES:
                raise RuntimeError("Bounded gate did not receive the prespecified batch count")
            report["variants"][variant] = {"status": "PASS", "batches": len(losses),
                "losses": losses, "final_scale": float(scaler.get_scale())}
        except (NumericalFailure, RuntimeError) as exc:
            report["status"] = "FAIL"
            report["variants"][variant] = {"status": "FAIL", "batches_completed": len(losses),
                "failure": exc.record if isinstance(exc, NumericalFailure) else str(exc)}
            break
        finally:
            del model, optimizer, scaler
            torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Stage 13C bounded stability gate failed")
    return report


def guarded_validation_epoch(model, loader, variant: str, epoch: int, *, predictions=False):
    model.eval()
    matrix = [[0] * 7 for _ in range(7)]
    loss_sum = count = 0
    rows = []
    with torch.inference_mode():
        for batch, (images_cpu, labels_cpu) in enumerate(loader, start=1):
            context = context_for(variant, epoch, batch, labels_cpu, None)
            images = images_cpu.cuda(non_blocking=True)
            labels = labels_cpu.cuda(non_blocking=True)
            require_finite(images, context, "validation_input")
            if labels.dtype != torch.long or not bool(((labels >= 0) & (labels < 7)).all().item()):
                raise NumericalFailure(context, "validation_labels")
            with torch.autocast(device_type="cuda"):
                logits = model(images)
                require_finite(logits, context, "validation_logits")
                terms = focal_terms(logits, labels, 1.0)
                require_finite(terms, context, "validation_per_example_loss")
                value = terms.mean()
                require_finite(value, context, "validation_reduced_loss")
            predicted = logits.argmax(1).detach().cpu().tolist()
            probabilities = torch.softmax(logits.detach().float(), dim=1).cpu().tolist() if predictions else None
            for j, p in enumerate(predicted):
                true = int(labels_cpu[j])
                matrix[true][p] += 1
                if predictions:
                    source = loader.dataset.rows[count + j]
                    rows.append({"image_id": source["image_id"], "true_label": source["dx"],
                                 "predicted_label": base.CLASSES[p], "correct": str(true == p),
                                 "probabilities": json.dumps(probabilities[j])})
            loss_sum += float(value.detach().float().cpu()) * len(labels_cpu)
            count += len(labels_cpu)
    loss = loss_sum / count
    if not math.isfinite(loss):
        raise NumericalFailure({"variant": variant, "epoch": epoch}, "validation_epoch_loss")
    return loss, base.metrics(matrix), rows


def require_finite_epoch_losses(train_loss: float, val_loss: float, variant: str, epoch: int) -> None:
    for name, value in (("train_epoch_loss", train_loss), ("validation_epoch_loss", val_loss)):
        if not math.isfinite(value):
            raise NumericalFailure({"variant": variant, "epoch": epoch}, name)


def checked_checkpoint(path: Path, state: dict, model, optimizer, scaler, variant: str, epoch: int) -> None:
    context = {"variant": variant, "epoch": epoch, "checkpoint_path": str(path)}
    require_valid_scale(scaler, context)
    require_finite_states(model, optimizer, context)
    scheduler = state["scheduler_state"]
    if not math.isfinite(float(scheduler["best"])) or not math.isfinite(float(scheduler["_last_lr"][0])):
        raise NumericalFailure(context, "scheduler_state")
    for row in state["history"]:
        require_finite_epoch_losses(row["train_loss"], row["val_loss"], variant, row["epoch"])
    base.save_checkpoint(path, state)


def train_variant(root: Path, variant: str) -> None:
    if not GATE_PATH.is_file():
        raise RuntimeError("Passing Stage 13C bounded T4 gate required before training")
    gate = base.read_json(GATE_PATH)
    if (gate.get("status") != "PASS" or gate.get("runner_sha256") != base.sha256(Path(__file__)) or
            gate.get("initial_scale") != INITIAL_SCALE or
            gate.get("split_sha256") != base.SPLIT_SHA or
            gate.get("prespecified_batches_per_variant") != GATE_BATCHES or
            gate.get("runtime") != {"python": platform.python_version(), "torch": torch.__version__,
                                    "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                                    "gpu": torch.cuda.get_device_name(0)} or
            set(gate.get("variants", {})) != set("ABC") or
            any(gate["variants"][v]["status"] != "PASS" for v in "ABC") or
            any(gate.get("source_sha256", {}).get(name) != base.sha256(root / name) for name in (
                "experiments/egvan_melanoma_ablation_exp/train_ablation.py",
                "src/train.py", "src/dataset.py", "src/models/egvan.py")) or
            any(gate["variant_config_sha256"][v] != base.sha256(HERE / f"config_{v}.json") for v in "ABC")):
        raise RuntimeError("Bounded gate is absent, failed, or does not match this numerical protocol")
    base.preflight(root)
    out = HERE / "stage13c_guarded_runs" / variant
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {out}")
    out.mkdir(parents=True)
    cfg = base.variant_config(variant)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (out / "numerical_protocol.json").write_text(json.dumps({
        "name": "Stage13C guarded AMP", "initial_scale": INITIAL_SCALE,
        "normal_dynamic_scaler_updates": True, "fail_fast_unscaled_gradients": True,
        "gradient_clipping": None, "scientific_variant": variant,
        "gate_sha256": base.sha256(GATE_PATH)}, indent=2) + "\n", encoding="utf-8")
    model, optimizer, scheduler, scaler, train_loader, val_loader, generator = build_components(root, variant)
    history = []; best_loss = math.inf; best_epoch = None
    for epoch in range(1, 26):
        train_sum = train_count = 0
        for batch, (images, labels) in enumerate(train_loader, start=1):
            value, _ = guarded_step(model, optimizer, scaler,
                images.cuda(non_blocking=True), labels.cuda(non_blocking=True), variant, epoch, batch)
            train_sum += value * len(labels)
            train_count += len(labels)
        train_loss = train_sum / train_count
        val_loss, val_metrics, _ = guarded_validation_epoch(model, val_loader, variant, epoch)
        require_finite_epoch_losses(train_loss, val_loss, variant, epoch)
        scheduler.step(val_loss)
        eligible = base.eligible(val_metrics)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_accuracy": val_metrics["accuracy"],
               "val_balanced_accuracy": val_metrics["balanced_accuracy"],
               "val_macro_f1": val_metrics["macro_f1"],
               "val_mel_precision": val_metrics["per_class"]["mel"]["precision"],
               "val_mel_recall": val_metrics["per_class"]["mel"]["recall"],
               "val_mel_f1": val_metrics["per_class"]["mel"]["f1"],
               "val_nv_recall": val_metrics["per_class"]["nv"]["recall"],
               "eligible": eligible, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        improved = eligible and val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
        state = {"variant": variant, "epoch": epoch, "best_epoch": best_epoch,
                 "best_validation_loss": best_loss, "configuration": cfg,
                 "numerical_protocol": "Stage13C guarded AMP", "class_order": list(base.CLASSES),
                 "architecture": cfg["architecture"], "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                 "scaler_state": scaler.state_dict(), "history": history,
                 "sampler_generator_state": generator.get_state(),
                 "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all()}
        if improved:
            checked_checkpoint(out / "best_checkpoint.pt", state, model, optimizer, scaler, variant, epoch)
        checked_checkpoint(out / "last_checkpoint.pt", state, model, optimizer, scaler, variant, epoch)
        base.write_csv(out / "training_history.csv", history)
        print(f"Stage13C {variant} epoch={epoch}/25 train_loss={train_loss:.6f} "
              f"val_loss={val_loss:.6f} eligible={eligible}", flush=True)
    if best_epoch is None:
        status = "FAIL_NO_ELIGIBLE_CHECKPOINT"
    else:
        state = torch.load(out / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        selected_loss, selected_metrics, predictions = guarded_validation_epoch(
            model, val_loader, variant, best_epoch, predictions=True)
        if not math.isclose(selected_loss, best_loss, rel_tol=1e-3, abs_tol=1e-4):
            raise ValueError("Selected validation loss drifted")
        base.write_csv(out / "validation_predictions.csv", predictions)
        (out / "validation_metrics.json").write_text(json.dumps({"epoch": best_epoch,
            "val_loss": selected_loss, **selected_metrics}, indent=2) + "\n", encoding="utf-8")
        status = "COMPLETE_VALIDATION_ONLY"
    (out / "experiment_manifest.json").write_text(json.dumps({
        "status": status, "variant": variant, "epochs": len(history),
        "selected_epoch": best_epoch, "best_validation_loss": best_loss if best_epoch else None,
        "best_checkpoint_sha256": base.sha256(out / "best_checkpoint.pt") if best_epoch else None,
        "last_checkpoint_sha256": base.sha256(out / "last_checkpoint.pt"),
        "split_sha256": base.SPLIT_SHA, "stage9_control_sha256": base.STAGE9_SHA,
        "numerical_protocol": "Stage13C guarded AMP", "class_order": list(base.CLASSES),
        "ham_test_accessed": False, "ph2_accessed": False}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--bounded-gate", action="store_true")
    modes.add_argument("--train", action="store_true")
    parser.add_argument("--variant", choices=list("ABC"))
    parser.add_argument("--output", type=Path, default=GATE_PATH)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Stage 13C bounded gate requires Tesla T4")
    if args.bounded_gate:
        bounded_gate(args.project_root.resolve(), args.output)
    else:
        if not args.variant:
            parser.error("--train requires --variant A/B/C")
        train_variant(args.project_root.resolve(), args.variant)


if __name__ == "__main__":
    main()
