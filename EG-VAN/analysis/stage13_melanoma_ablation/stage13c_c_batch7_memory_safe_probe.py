"""Memory-safe C batch-7 AMP/FP32 forward probe; no research checkpoint."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import platform
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/egvan_melanoma_ablation_exp"))
import train_ablation as base  # noqa: E402
import stage13c_guarded as guard  # noqa: E402
import stage13c_abc_8192_gate as gate  # noqa: E402
from stage13c_multistep_amp_diagnostic import (  # noqa: E402
    digest_model, digest_tensor, first_bad, gradient_status, model_status, optimizer_status,
)

GATE_SHA256 = "ccdfe75250492878f759d6e95feda59afc1ad1ea88bdb25b3227eca0b0268c56"
FAILED_PROBE_SHA256 = "fa32d5a2e2326c1dbe102963d494d7cc7676becebb1a764e024ad0b2a6b266c5"
OUTPUT = HERE / "stage13c_C_batch7_memory_safe_probe.json"


def cuda_memory() -> dict:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {"allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "free_bytes": int(free), "total_bytes": int(total)}


def safe_cuda_memory() -> dict:
    try:
        return cuda_memory()
    except Exception as exc:
        return {"unavailable": type(exc).__name__, "message": str(exc)}


def nested_tensors(value, prefix="output"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            yield from nested_tensors(child, f"{prefix}[{i}]")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from nested_tensors(child, f"{prefix}.{key}")


def first_bad_activation(named_tensors):
    """Inspect one leading-dimension slice at a time to bound hook scratch memory."""
    for name, value in named_tensors:
        if not value.is_floating_point():
            continue
        x = value.detach()
        parts = [x] if x.ndim == 0 else (x[i] for i in range(x.shape[0]))
        if all(bool(torch.isfinite(part).all().item()) for part in parts):
            continue
        parts = [x] if x.ndim == 0 else (x[i] for i in range(x.shape[0]))
        summaries = [guard.tensor_summary(part) for part in parts]
        minima = [item["finite_min"] for item in summaries if item["finite_min"] is not None]
        maxima = [item["finite_max"] for item in summaries if item["finite_max"] is not None]
        return {"tensor": name, "finite": False,
                "nan_count": sum(item["nan_count"] for item in summaries),
                "inf_count": sum(item["inf_count"] for item in summaries),
                "finite_min": min(minima) if minima else None,
                "finite_max": max(maxima) if maxima else None,
                "dtype": str(x.dtype), "shape": list(x.shape)}
    return None


def state_parts(model):
    bad_parameters = first_bad(model.named_parameters())
    bad_buffers = first_bad(model.named_buffers())
    return {"parameters": {"finite": bad_parameters is None,
                           "first_nonfinite": bad_parameters},
            "buffers": {"finite": bad_buffers is None,
                        "first_nonfinite": bad_buffers}}


def inspect_forward(model, images, labels, mode: str) -> dict:
    first = {"input": None, "output": None}
    hooks = []

    def pre_hook(name):
        def check(_module, inputs):
            if first["input"] is None:
                bad = first_bad_activation((f"{name}/{p}", x)
                    for p, x in nested_tensors(inputs, "input"))
                if bad is not None:
                    first["input"] = bad
        return check

    def post_hook(name):
        def check(_module, _inputs, output):
            if first["output"] is None:
                bad = first_bad_activation((f"{name}/{p}", x)
                    for p, x in nested_tensors(output))
                if bad is not None:
                    first["output"] = bad
        return check

    for name, module in model.named_modules():
        if name and not any(module.children()):
            hooks.append(module.register_forward_pre_hook(pre_hook(name)))
            hooks.append(module.register_forward_hook(post_hook(name)))
    try:
        model.train(True)
        # Forward-only replay must not retain a training computation graph.
        with torch.no_grad(), torch.autocast("cuda", enabled=(mode == "amp")):
            logits = model(images)
            logits_bad = first_bad([("logits", logits)])
            row = {"mode": mode, "first_nonfinite_leaf_input": first["input"],
                   "first_nonfinite_leaf_output": first["output"],
                   "logits_finite": logits_bad is None,
                   "logits_nonfinite": logits_bad}
            if logits_bad is None:
                terms = guard.focal_terms(logits, labels, 1.0)
                row["focal_terms_finite"] = bool(torch.isfinite(terms).all().item())
                row["mean_loss"] = float(terms.mean().float().cpu())
                row["mean_loss_finite"] = math.isfinite(row["mean_loss"])
            row["model_after_forward"] = model_status(model)
            row["model_state_parts_after_forward"] = state_parts(model)
            return row
    finally:
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root, output = args.project_root.resolve(), args.output.resolve()
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Tesla T4 required")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite probe: {output}")
    if output.parent != root / "analysis/stage13_melanoma_ablation":
        raise ValueError("Output must be in Stage 13 analysis directory")
    prior_path = HERE / "stage13c_abc_8192_gate.json"
    if not prior_path.is_file() or base.sha256(prior_path) != GATE_SHA256:
        raise ValueError("Audited A/B/C gate absent or changed")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (prior["status"] != "FAIL_OR_NO_CLEARANCE" or
            prior["variants"]["C"]["status"] != "STOPPED_NUMERICAL_FAILURE"):
        raise ValueError("Unexpected prior gate status")
    configs = gate.verify_scientific_factors(root)
    failed_probe_path = HERE / "stage13c_C_batch7_forward_probe.json"
    failed_probe_hash = base.sha256(failed_probe_path) if failed_probe_path.is_file() else None
    if failed_probe_hash is not None and failed_probe_hash != FAILED_PROBE_SHA256:
        raise ValueError("Original failed probe changed; preserve it as evidence")
    model, optimizer, scaler, loader = gate.make_components(root, "C", configs["C"])
    report = {"status": "IN_PROGRESS", "scope": "HAM train only; C batches 1-6, batch-7 forward only",
              "seed": 42, "initial_scale": 8192.0, "prior_gate_sha256": GATE_SHA256,
              "failed_probe_sha256": failed_probe_hash,
              "memory_safe_protocol": "CPU model snapshot; no_grad sequential AMP/FP32; optimizer and prior graph released",
              "script_sha256": base.sha256(Path(__file__)),
              "source_sha256": {
                  "analysis/stage13_melanoma_ablation/stage13c_abc_8192_gate.py":
                      base.sha256(HERE / "stage13c_abc_8192_gate.py"),
                  "experiments/egvan_melanoma_ablation_exp/train_ablation.py":
                      base.sha256(root / "experiments/egvan_melanoma_ablation_exp/train_ablation.py"),
                  "src/models/egvan/egvan.py": base.sha256(root / "src/models/egvan/egvan.py")},
              "split_sha256": base.sha256(root / "data/splits/split_leakage_aware.csv"),
              "config_C_sha256": base.sha256(root / "experiments/egvan_melanoma_ablation_exp/config_C.json"),
              "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                          "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(0)},
              "initial_model_sha256": digest_model(model),
              "replayed_batches": [], "batch7": None,
              "cuda_memory": {},
              "research_checkpoint_saved": False, "full_training_performed": False,
              "ham_test_accessed": False, "ph2_accessed": False}
    steps = [0]

    def post_step(_optimizer, _args, _kwargs):
        steps[0] += 1

    hook = optimizer.register_step_post_hook(post_step)
    phase = "replay_setup"
    try:
        report["cuda_memory"]["before_replay"] = cuda_memory()
        if (report["initial_model_sha256"] != prior["variants"]["C"]["initial_model_sha256"] or
                report["runtime"] != prior["runtime"]):
            report["status"] = "INITIAL_STATE_OR_RUNTIME_MISMATCH"
            return
        iterator = iter(loader)
        for index in range(1, 8):
            phase = f"batch_{index}_replay" if index <= 6 else "batch_7_prepare"
            images_cpu, labels_cpu = next(iterator)
            reference = prior["variants"]["C"]["batches"][index - 1]
            input_sha, label_sha = digest_tensor(images_cpu), digest_tensor(labels_cpu)
            if input_sha != reference["input_sha256"] or label_sha != reference["labels_sha256"]:
                report["status"] = "DATA_REPRODUCTION_MISMATCH"
                report["mismatch_batch"] = index
                break
            images = images_cpu.cuda(non_blocking=True)
            labels = labels_cpu.cuda(non_blocking=True)
            if index == 7:
                report["cuda_memory"]["after_batch6_before_cleanup"] = cuda_memory()
                report["batch7"] = {"batch_index": 7, "input_sha256": input_sha,
                                    "labels_sha256": label_sha,
                                    "target_class_distribution": dict(Counter(
                                        base.CLASSES[int(y)] for y in labels_cpu.tolist())),
                                    "scale_before": float(scaler.get_scale()),
                                    "model_before_forward": model_status(model),
                                    "model_state_parts_before_forward": state_parts(model),
                                    "optimizer_before_forward": optimizer_status(optimizer),
                                    "model_sha256_before_forward": digest_model(model)}
                if (not report["batch7"]["model_before_forward"]["finite"] or
                        not report["batch7"]["optimizer_before_forward"]["finite"]):
                    report["status"] = "PRE_FORWARD_STATE_NONFINITE"
                    break
                # Snapshot only in CPU memory; no checkpoint file is written.
                model_snapshot = {name: tensor.detach().cpu().clone()
                                  for name, tensor in model.state_dict().items()}
                report["cuda_memory"]["after_cpu_snapshot"] = cuda_memory()
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
                py_rng = random.getstate()
                np_rng = copy.deepcopy(np.random.get_state())
                # Batch-6 locals otherwise retain the full training graph.
                del logits, terms, loss
                # The optimizer and its GPU moments are unnecessary for either
                # forward-only pass; gradients are unnecessary as well.
                optimizer.zero_grad(set_to_none=True)
                hook.remove()
                hook = None
                del optimizer, scaler, iterator, loader, images_cpu, labels_cpu
                gc.collect()
                torch.cuda.empty_cache()
                report["cuda_memory"]["before_amp_forward"] = cuda_memory()
                phase = "batch_7_amp_forward"
                report["batch7"]["amp_forward"] = inspect_forward(model, images, labels, "amp")
                report["cuda_memory"]["after_amp_forward"] = cuda_memory()
                gc.collect()
                torch.cuda.empty_cache()
                report["cuda_memory"]["after_amp_cleanup"] = cuda_memory()
                phase = "restore_pre_batch7_model"
                model.load_state_dict(model_snapshot)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                random.setstate(py_rng)
                np.random.set_state(np_rng)
                gc.collect()
                torch.cuda.empty_cache()
                report["cuda_memory"]["before_fp32_forward"] = cuda_memory()
                phase = "batch_7_fp32_forward"
                report["batch7"]["fp32_forward"] = inspect_forward(model, images, labels, "fp32")
                report["cuda_memory"]["after_fp32_forward"] = cuda_memory()
                report["batch7"]["optimizer_step_performed_on_batch7"] = False
                report["batch7"]["optimizer_state_released_before_forwards"] = True
                report["batch7"]["amp_reproduced_prior_nonfinite_logits"] = (
                    not report["batch7"]["amp_forward"]["logits_finite"] and
                    not reference["logits_finite"])
                report["status"] = ("COMPLETE_REPRODUCED" if
                    report["batch7"]["amp_reproduced_prior_nonfinite_logits"]
                    else "FORWARD_FAILURE_NOT_REPRODUCED")
                del model_snapshot, images, labels
                gc.collect()
                torch.cuda.empty_cache()
                report["cuda_memory"]["after_final_cleanup"] = cuda_memory()
                break
            model.train(True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                logits = model(images)
                terms = guard.focal_terms(logits, labels, 1.0)
                loss = terms.mean()
            row = {"batch_index": index, "input_sha256": input_sha,
                   "labels_sha256": label_sha, "loss": float(loss.detach().float().cpu()),
                   "scale_before": float(scaler.get_scale())}
            report["replayed_batches"].append(row)
            if not math.isfinite(row["loss"]):
                report["status"] = "EARLY_REPLAY_NONFINITE_LOSS"
                break
            scaler.scale(loss).backward()
            row["scaled_gradients"] = gradient_status(model)
            scaler.unscale_(optimizer)
            row["unscaled_gradients"] = gradient_status(model)
            before = steps[0]
            scaler.step(optimizer)
            row["optimizer_step_performed"] = steps[0] > before
            row["optimizer_step_skipped"] = not row["optimizer_step_performed"]
            scaler.update()
            row["scale_after"] = float(scaler.get_scale())
            row["model_after_batch"] = model_status(model)
            row["optimizer_after_batch"] = optimizer_status(optimizer)
            if (not math.isclose(row["loss"], reference["loss"], rel_tol=1e-5, abs_tol=1e-5)
                    or row["scale_before"] != reference["scale_before"]
                    or row["scale_after"] != reference["scale_after"]
                    or row["optimizer_step_performed"] != reference["optimizer_step_performed"]
                    or row["optimizer_step_skipped"] != reference["optimizer_step_skipped"]
                    or row["scaled_gradients"]["finite"] != reference["scaled_gradients"]["finite"]
                    or row["unscaled_gradients"]["finite"] != reference["unscaled_gradients"]["finite"]
                    or not row["model_after_batch"]["finite"]
                    or not row["optimizer_after_batch"]["finite"]):
                report["status"] = "EARLY_REPLAY_MISMATCH"
                report["mismatch_batch"] = index
                break
        if report["status"] == "IN_PROGRESS":
            report["status"] = "INCOMPLETE"
    except torch.OutOfMemoryError as exc:
        report["status"] = "CUDA_OOM_IN_DIAGNOSTIC"
        report["oom"] = {"phase": phase, "type": type(exc).__name__,
                         "message": str(exc), "cuda_memory": safe_cuda_memory()}
        report["cuda_memory"]["after_oom_in_" + phase] = report["oom"]["cuda_memory"]
        if phase == "batch_7_fp32_forward" and report["batch7"] is not None:
            report["batch7"]["fp32_forward_status"] = "UNDETERMINED_DUE_TO_DIAGNOSTIC_OOM"
    except Exception as exc:
        report["status"] = "EXCEPTION"
        report["exception"] = {"phase": phase, "type": type(exc).__name__,
                               "message": str(exc)}
    finally:
        if hook is not None:
            hook.remove()
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as cleanup_exc:
            report["cleanup_exception"] = {"type": type(cleanup_exc).__name__,
                                           "message": str(cleanup_exc)}
        report["cuda_memory"]["at_exit"] = safe_cuda_memory()
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Probe JSON: {output}", flush=True)
        print(f"Probe status: {report['status']}", flush=True)


if __name__ == "__main__":
    main()
