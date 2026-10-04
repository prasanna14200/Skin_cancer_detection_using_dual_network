"""Reproduce C's first six train steps, then inspect batch-7 AMP/FP32 forward only."""
from __future__ import annotations

import argparse
import copy
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
OUTPUT = HERE / "stage13c_C_batch7_forward_probe.json"


def nested_tensors(value, prefix="output"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            yield from nested_tensors(child, f"{prefix}[{i}]")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from nested_tensors(child, f"{prefix}.{key}")


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
                bad = first_bad((f"{name}/{p}", x) for p, x in nested_tensors(inputs, "input"))
                if bad is not None:
                    first["input"] = bad
        return check

    def post_hook(name):
        def check(_module, _inputs, output):
            if first["output"] is None:
                bad = first_bad((f"{name}/{p}", x) for p, x in nested_tensors(output))
                if bad is not None:
                    first["output"] = bad
        return check

    for name, module in model.named_modules():
        if name and not any(module.children()):
            hooks.append(module.register_forward_pre_hook(pre_hook(name)))
            hooks.append(module.register_forward_hook(post_hook(name)))
    try:
        model.train(True)
        with torch.enable_grad(), torch.autocast("cuda", enabled=(mode == "amp")):
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
    model, optimizer, scaler, loader = gate.make_components(root, "C", configs["C"])
    report = {"status": "IN_PROGRESS", "scope": "HAM train only; C batches 1-6, batch-7 forward only",
              "seed": 42, "initial_scale": 8192.0, "prior_gate_sha256": GATE_SHA256,
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
              "research_checkpoint_saved": False, "full_training_performed": False,
              "ham_test_accessed": False, "ph2_accessed": False}
    steps = [0]

    def post_step(_optimizer, _args, _kwargs):
        steps[0] += 1

    hook = optimizer.register_step_post_hook(post_step)
    try:
        if (report["initial_model_sha256"] != prior["variants"]["C"]["initial_model_sha256"] or
                report["runtime"] != prior["runtime"]):
            report["status"] = "INITIAL_STATE_OR_RUNTIME_MISMATCH"
            return
        for index, (images_cpu, labels_cpu) in enumerate(loader, start=1):
            if index > 7:
                break
            reference = prior["variants"]["C"]["batches"][index - 1]
            input_sha, label_sha = digest_tensor(images_cpu), digest_tensor(labels_cpu)
            if input_sha != reference["input_sha256"] or label_sha != reference["labels_sha256"]:
                report["status"] = "DATA_REPRODUCTION_MISMATCH"
                report["mismatch_batch"] = index
                break
            images = images_cpu.cuda(non_blocking=True)
            labels = labels_cpu.cuda(non_blocking=True)
            if index == 7:
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
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
                py_rng = random.getstate()
                np_rng = copy.deepcopy(np.random.get_state())
                report["batch7"]["amp_forward"] = inspect_forward(model, images, labels, "amp")
                model.load_state_dict(model_snapshot)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                random.setstate(py_rng)
                np.random.set_state(np_rng)
                report["batch7"]["fp32_forward"] = inspect_forward(model, images, labels, "fp32")
                report["batch7"]["optimizer_after_both_forwards"] = optimizer_status(optimizer)
                report["batch7"]["optimizer_step_performed_on_batch7"] = False
                report["batch7"]["amp_reproduced_prior_nonfinite_logits"] = (
                    not report["batch7"]["amp_forward"]["logits_finite"] and
                    not reference["logits_finite"])
                report["status"] = ("COMPLETE_REPRODUCED" if
                    report["batch7"]["amp_reproduced_prior_nonfinite_logits"]
                    else "FORWARD_FAILURE_NOT_REPRODUCED")
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
    except Exception as exc:
        report["status"] = "EXCEPTION"
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        hook.remove()
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Probe JSON: {output}", flush=True)
        print(f"Probe status: {report['status']}", flush=True)


if __name__ == "__main__":
    main()
