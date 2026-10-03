"""Stage 13B diagnostic only: compare A/B through at most one HAM train epoch.

No checkpoint is saved, no validation/test/PH2 loader is constructed, and this
does not change the registered training runner or any research configuration.
Run on the same Colab T4 environment as the failed B run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_S_Weights

import train_ablation as stage13


def tensor_status(t: torch.Tensor) -> dict:
    detached = t.detach()
    if not (detached.is_floating_point() or detached.is_complex()):
        return {"finite": True, "dtype": str(detached.dtype), "shape": list(detached.shape)}
    finite = torch.isfinite(detached)
    if bool(finite.all().item()):
        return {"finite": True, "dtype": str(detached.dtype), "shape": list(detached.shape)}
    values = detached[finite]
    return {"finite": False, "dtype": str(detached.dtype),
            "shape": list(detached.shape), "nan_count": int(torch.isnan(detached).sum().item()),
            "inf_count": int(torch.isinf(detached).sum().item()),
            "finite_min": float(values.min().item()) if values.numel() else None,
            "finite_max": float(values.max().item()) if values.numel() else None}


def first_bad_tensor(items) -> tuple[str, dict] | None:
    for name, value in items:
        if isinstance(value, torch.Tensor):
            status = tensor_status(value)
            if not status["finite"]:
                return name, status
    return None


def run_variant(root: Path, variant: str, max_batches: int) -> dict:
    cfg = stage13.variant_config(variant)
    _, Dataset, _, EGVAN, _, make_transforms, set_seed = stage13.imports(root)
    set_seed(42)
    train_tf, _ = make_transforms(EfficientNet_V2_S_Weights.DEFAULT)
    split = root / "data/splits/split_leakage_aware.csv"
    images = root / "data/processed/images"
    train_ds = Dataset(images, split, "train", train_tf)
    generator = torch.Generator(device="cpu").manual_seed(42)
    loader = DataLoader(train_ds, batch_size=16,
        sampler=stage13.make_sampler(train_ds, cfg["sampler"]["mel_weight"], generator),
        num_workers=2, pin_memory=True)
    model = EGVAN(num_classes=7, pretrained_efficient=True, pretrained_resnet=False).cuda().train()
    optimizer, _ = stage13.optimizer_scheduler(model)
    scaler = torch.amp.GradScaler("cuda")
    # Hash the identical initial model state before any batch or optimizer step.
    model_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        model_hash.update(name.encode()); model_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    result = {"variant": variant, "configuration": cfg, "initial_model_sha256": model_hash.hexdigest(),
              "batches": [], "first_nonfinite": None, "max_batches": max_batches,
              "training_checkpoint_saved": False, "ham_test_accessed": False, "ph2_accessed": False}

    for batch_index, (x_cpu, y_cpu) in enumerate(loader, start=1):
        if batch_index > max_batches:
            break
        labels = [stage13.CLASSES[int(v)] for v in y_cpu]
        context = {"variant": variant, "epoch": 1, "batch_index": batch_index,
                   "target_class_distribution": dict(Counter(labels)),
                   "contains_mel": "mel" in labels, "amp_scale": float(scaler.get_scale()),
                   "input_sha256": hashlib.sha256(x_cpu.contiguous().numpy().tobytes()).hexdigest(),
                   "labels_sha256": hashlib.sha256(y_cpu.numpy().tobytes()).hexdigest()}

        def inspect(name: str, value: torch.Tensor) -> bool:
            status = tensor_status(value)
            if not status["finite"]:
                result["first_nonfinite"] = {**context, "affected_tensor": name, **status}
                return False
            return True

        if not inspect("input_tensors", x_cpu):
            break
        if y_cpu.dtype != torch.long or not bool(((y_cpu >= 0) & (y_cpu < 7)).all()):
            result["first_nonfinite"] = {**context, "affected_tensor": "labels", "labels": y_cpu.tolist()}
            break
        if not math.isfinite(context["amp_scale"]) or context["amp_scale"] <= 0:
            result["first_nonfinite"] = {**context, "affected_tensor": "amp_scale_before_forward"}
            break
        x = x_cpu.cuda(non_blocking=True); y = y_cpu.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda"):
            logits = model(x)
            if not inspect("model_logits", logits):
                break
            probabilities = torch.softmax(logits, dim=1)
            if not inspect("softmax_values", probabilities):
                break
            log_probabilities = torch.log_softmax(logits, dim=1)
            if not inspect("log_softmax_values", log_probabilities):
                break
            cross_entropy = nn.functional.cross_entropy(logits, y, reduction="none")
            if not inspect("cross_entropy", cross_entropy):
                break
            p_true = probabilities.gather(1, y.unsqueeze(1)).squeeze(1)
            if not inspect("p_true", p_true):
                break
            original_focal = 0.25 * (1 - p_true).pow(2.0) * cross_entropy
            if not inspect("original_focal_term", original_focal):
                break
            multiplier = cfg["loss"]["mel_multiplier"]
            terms = original_focal if multiplier == 1.0 else original_focal * (
                1.0 + (multiplier - 1.0) * (y == stage13.CLASSES.index("mel")).to(original_focal.dtype))
            if not inspect("mel_multiplier_application", terms):
                break
            if not inspect("final_per_example_loss", terms):
                break
            batch_loss = terms.mean()
            if not inspect("batch_loss", batch_loss):
                break
        scaled_loss = scaler.scale(batch_loss)
        if not inspect("amp_scaled_loss", scaled_loss):
            break
        scaled_loss.backward()
        bad = first_bad_tensor((name + ".grad", p.grad) for name, p in model.named_parameters() if p.grad is not None)
        if bad:
            result["first_nonfinite"] = {**context, "affected_tensor": "gradient_before_optimizer_step/" + bad[0], **bad[1]}
            break
        scaler.step(optimizer); scaler.update()
        bad = first_bad_tensor(model.named_parameters())
        if bad:
            result["first_nonfinite"] = {**context, "affected_tensor": "parameter_after_optimizer_step/" + bad[0], **bad[1]}
            break
        bad = first_bad_tensor((str(index) + "/" + name, value)
            for index, state in enumerate(optimizer.state.values()) for name, value in state.items())
        if bad:
            result["first_nonfinite"] = {**context, "affected_tensor": "optimizer_state/" + bad[0], **bad[1]}
            break
        scale_status = tensor_status(torch.tensor(float(scaler.get_scale())))
        if not scale_status["finite"] or scaler.get_scale() <= 0:
            result["first_nonfinite"] = {**context, "affected_tensor": "amp_scale_after_step", **scale_status}
            break
        context["batch_loss"] = float(batch_loss.detach())
        context["amp_scale_after_step"] = float(scaler.get_scale())
        result["batches"].append(context)
        print(f"{variant} batch {batch_index}: finite loss={context['batch_loss']:.8f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=stage13.ROOT)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_batches <= 502:
        parser.error("--max-batches must be 1..502 (at most one train epoch)")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Exact Stage 13B diagnostic requires the Colab Tesla T4")
    root = args.project_root.resolve()
    stage13.preflight(root)
    a = stage13.variant_config("A"); b = stage13.variant_config("B")
    a.pop("variant"); a.pop("description")
    b.pop("variant"); b.pop("description")
    a["loss"]["name"] = b["loss"]["name"]
    a["loss"]["mel_multiplier"] = b["loss"]["mel_multiplier"]
    if a != b:
        raise ValueError("A/B differ outside the preregistered true-MEL loss weighting")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {args.output}")
    b_result = run_variant(root, "B", args.max_batches)
    control_batches = (b_result["first_nonfinite"]["batch_index"]
                       if b_result["first_nonfinite"] else args.max_batches)
    a_result = run_variant(root, "A", control_batches)
    results = {"A": a_result, "B": b_result}
    a_batches, b_batches = a_result["batches"], b_result["batches"]
    shared = min(len(a_batches), len(b_batches))
    results["comparison"] = {"same_initial_model": a_result["initial_model_sha256"] == b_result["initial_model_sha256"],
        "matching_finite_batch_inputs_and_labels": all(
            a_batches[i]["input_sha256"] == b_batches[i]["input_sha256"] and
            a_batches[i]["labels_sha256"] == b_batches[i]["labels_sha256"] for i in range(shared)),
        "matched_finite_batches": shared,
        "B_first_nonfinite": b_result["first_nonfinite"]}
    failure = b_result["first_nonfinite"]
    if failure and failure["batch_index"] <= len(a_batches):
        control_batch = a_batches[failure["batch_index"] - 1]
        results["comparison"]["same_input_at_B_failure"] = (
            control_batch["input_sha256"] == failure["input_sha256"] and
            control_batch["labels_sha256"] == failure["labels_sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(results["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
