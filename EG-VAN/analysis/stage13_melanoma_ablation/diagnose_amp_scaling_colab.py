"""Stage 13B diagnostic: one seed-42 HAM train batch, backward only, no step."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_S_Weights

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/egvan_melanoma_ablation_exp"))
import train_ablation as stage13


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensor_status(t: torch.Tensor) -> dict:
    x = t.detach()
    mask = torch.isfinite(x)
    all_finite = bool(mask.all().item())
    if all_finite:
        minimum, maximum = torch.aminmax(x)
        lo, hi = float(minimum.item()), float(maximum.item())
        return {"finite": True, "nan_count": 0, "inf_count": 0,
                "finite_min": lo, "finite_max": hi, "finite_max_abs": max(abs(lo), abs(hi)),
                "dtype": str(x.dtype), "shape": list(x.shape)}
    values = x[mask]
    return {"finite": False, "nan_count": int(torch.isnan(x).sum().item()),
            "inf_count": int(torch.isinf(x).sum().item()),
            "finite_min": float(values.min().item()) if values.numel() else None,
            "finite_max": float(values.max().item()) if values.numel() else None,
            "finite_max_abs": float(values.abs().max().item()) if values.numel() else None,
            "dtype": str(x.dtype), "shape": list(x.shape)}


def gradient_status(model) -> dict:
    first_bad = None
    all_finite = True
    maximum = 0.0
    minimum_value = None
    maximum_value = None
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        item = tensor_status(parameter.grad)
        all_finite &= item["finite"]
        if item["finite_max_abs"] is not None:
            maximum = max(maximum, item["finite_max_abs"])
        if item["finite_min"] is not None:
            minimum_value = item["finite_min"] if minimum_value is None else min(minimum_value, item["finite_min"])
            maximum_value = item["finite_max"] if maximum_value is None else max(maximum_value, item["finite_max"])
        if not item["finite"] and first_bad is None:
            first_bad = {"parameter": name + ".grad", **item}
    return {"all_finite": bool(all_finite), "first_nonfinite_parameter": first_bad,
            "finite_gradient_min": minimum_value, "finite_gradient_max": maximum_value,
            "maximum_absolute_finite_gradient": maximum}


def get_first_batch(root: Path):
    _, Dataset, _, EGVAN, _, make_transforms, set_seed = stage13.imports(root)
    set_seed(42)
    train_tf, _ = make_transforms(EfficientNet_V2_S_Weights.DEFAULT)
    ds = Dataset(root / "data/processed/images", root / "data/splits/split_leakage_aware.csv", "train", train_tf)
    generator = torch.Generator(device="cpu").manual_seed(42)
    loader = DataLoader(ds, batch_size=16,
        sampler=stage13.make_sampler(ds, stage13.variant_config("A")["sampler"]["mel_weight"], generator),
        num_workers=2, pin_memory=True)
    model = EGVAN(num_classes=7, pretrained_efficient=True, pretrained_resnet=False).cuda().train()
    # Match the original training setup order before the loader iterator begins.
    stage13.optimizer_scheduler(model)
    torch.amp.GradScaler("cuda")
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_hash = hashlib.sha256()
    for name, value in initial.items():
        initial_hash.update(name.encode())
        initial_hash.update(value.contiguous().numpy().tobytes())
    iterator = iter(loader)
    x_cpu, y_cpu = next(iterator)
    del iterator
    if not torch.isfinite(x_cpu).all() or y_cpu.dtype != torch.long or not ((y_cpu >= 0) & (y_cpu < 7)).all():
        raise ValueError("First HAM train batch has invalid inputs or labels")
    context = {"seed": 42, "batch_index": 1, "input_sha256": digest(x_cpu), "labels_sha256": digest(y_cpu),
               "initial_model_sha256": initial_hash.hexdigest(),
               "target_class_distribution": dict(Counter(stage13.CLASSES[int(v)] for v in y_cpu)),
               "contains_mel": bool((y_cpu == stage13.CLASSES.index("mel")).any()),
               "input_status": tensor_status(x_cpu), "labels": y_cpu.tolist()}
    rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
    return model, initial, rng, x_cpu, y_cpu, context


def run_path(model, initial, rng, x_cpu, y_cpu, variant: str, amp: bool, scale: int | None) -> dict:
    model.load_state_dict(initial)
    model.train(True)
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state_all(rng["cuda"])
    optimizer, _ = stage13.optimizer_scheduler(model)
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", init_scale=scale) if amp else None
    result = {"variant": variant, "forward_mode": "cuda_amp_fp16" if amp else "full_fp32",
              "amp_scale": scale, "input_sha256": digest(x_cpu), "labels_sha256": digest(y_cpu),
              "stages": {}, "gradient_before_unscale": None,
              "gradient_after_unscale": None, "first_nonfinite_operation": None,
              "optimizer_step_performed": False}

    def check(name: str, value: torch.Tensor) -> bool:
        status = tensor_status(value)
        result["stages"][name] = status
        if not status["finite"] and result["first_nonfinite_operation"] is None:
            result["first_nonfinite_operation"] = name
        return status["finite"]

    x = x_cpu.cuda(non_blocking=True); y = y_cpu.cuda(non_blocking=True)
    try:
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = model(x)
            if not check("logits", logits):
                return result
            probabilities = torch.softmax(logits, dim=1)
            if not check("softmax", probabilities):
                return result
            log_probabilities = torch.log_softmax(logits, dim=1)
            if not check("log_softmax", log_probabilities):
                return result
            cross_entropy = nn.functional.cross_entropy(logits, y, reduction="none")
            if not check("cross_entropy", cross_entropy):
                return result
            p_true = probabilities.gather(1, y.unsqueeze(1)).squeeze(1)
            if not check("p_true", p_true):
                return result
            focal = 0.25 * (1 - p_true).pow(2.0) * cross_entropy
            if not check("base_focal_per_example", focal):
                return result
            multiplier = stage13.variant_config(variant)["loss"]["mel_multiplier"]
            terms = focal if multiplier == 1.0 else focal * (1.0 +
                (multiplier - 1.0) * (y == stage13.CLASSES.index("mel")).to(focal.dtype))
            if not check("mel_multiplier_application", terms):
                return result
            if not check("final_per_example_loss", terms):
                return result
            loss = terms.mean()
            if not check("mean_batch_loss", loss):
                return result
        backward_loss = scaler.scale(loss) if amp else loss
        if not check("amp_scaled_loss" if amp else "fp32_backward_loss", backward_loss):
            return result
        backward_loss.backward()
        before = gradient_status(model)
        result["gradient_before_unscale"] = before
        if not before["all_finite"]:
            result["first_nonfinite_operation"] = "backward_gradient_before_unscale"
        if amp:
            scaler.unscale_(optimizer)
            after = gradient_status(model)
            result["gradient_after_unscale"] = after
            if not after["all_finite"] and result["first_nonfinite_operation"] is None:
                result["first_nonfinite_operation"] = "gradient_after_unscale"
            result["amp_scale_after_unscale"] = float(scaler.get_scale())
        else:
            result["gradient_after_unscale"] = {**before, "note": "unscale not applicable to FP32"}
        result["optimizer_state_is_empty"] = len(optimizer.state) == 0
        return result
    except torch.cuda.OutOfMemoryError as exc:
        result["error"] = "CUDA_OUT_OF_MEMORY"
        result["error_detail"] = str(exc).split("\n")[0]
        return result
    finally:
        optimizer.zero_grad(set_to_none=True)
        del x, y
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--reference-trace", type=Path,
        help="Previous bounded A/B trace; defaults to the registered Stage 13B trace path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This exact AMP diagnostic requires a Tesla T4")
    root = args.project_root.resolve()
    stage13.preflight(root)
    a = stage13.variant_config("A"); b = stage13.variant_config("B")
    a.pop("variant"); a.pop("description")
    b.pop("variant"); b.pop("description")
    a["loss"]["name"] = b["loss"]["name"]
    a["loss"]["mel_multiplier"] = b["loss"]["mel_multiplier"]
    if a != b:
        raise ValueError("A/B differ outside the registered true-MEL loss multiplier")
    model, initial, rng, x_cpu, y_cpu, context = get_first_batch(root)
    reference_path = args.reference_trace or root / "analysis/stage13_melanoma_ablation/stage13b_t4_first_epoch_trace.json"
    reference = stage13.read_json(reference_path)
    for variant in "AB":
        previous = reference[variant]["first_nonfinite"]
        if (reference[variant]["initial_model_sha256"] != context["initial_model_sha256"] or
                previous["input_sha256"] != context["input_sha256"] or
                previous["labels_sha256"] != context["labels_sha256"]):
            raise ValueError(f"{variant} initial model or first batch differs from bounded reference trace")
    results = {"context": context, "config_isolation_verified": True, "paths": [],
               "reference_trace_sha256": stage13.sha256(reference_path),
               "reference_model_and_first_batch_match": True,
               "optimizer_step_performed": False, "full_training_performed": False,
               "ham_test_accessed": False, "ph2_accessed": False}
    for scale in (65536, 32768, 16384, 8192, 4096):
        item = run_path(model, initial, rng, x_cpu, y_cpu, "A", True, scale)
        results["paths"].append(item)
        print("A AMP", scale, item["first_nonfinite_operation"], flush=True)
    fp32 = run_path(model, initial, rng, x_cpu, y_cpu, "A", False, None)
    results["paths"].append(fp32)
    print("A FP32", fp32["first_nonfinite_operation"], flush=True)
    finite_scales = [item["amp_scale"] for item in results["paths"] if item["variant"] == "A"
                     and item["forward_mode"] == "cuda_amp_fp16" and not item.get("error")
                     and item["first_nonfinite_operation"] is None
                     and item["gradient_after_unscale"]["all_finite"]]
    results["A_finite_tested_scales"] = finite_scales
    results["A_lowest_tested_finite_scale"] = min(finite_scales) if finite_scales else None
    results["A_highest_tested_finite_scale"] = max(finite_scales) if finite_scales else None
    b_scales = [65536]
    if finite_scales and max(finite_scales) != 65536:
        b_scales.append(max(finite_scales))
    for scale in b_scales:
        item = run_path(model, initial, rng, x_cpu, y_cpu, "B", True, scale)
        results["paths"].append(item)
        print("B AMP", scale, item["first_nonfinite_operation"], flush=True)
    results["paths"].append(run_path(model, initial, rng, x_cpu, y_cpu, "B", False, None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Diagnostic written:", args.output, flush=True)


if __name__ == "__main__":
    main()
