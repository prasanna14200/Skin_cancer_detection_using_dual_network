"""Bounded C batch-7 NonLocalBlock arithmetic trace under AMP and FP32."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import platform
import random
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.nn import functional as F

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
REFERENCE_SHA256 = "caad192d6199374a5df4efa49b7d9fdbf3f6c933b5cf8f9abba3afc062a12131"
OUTPUT = HERE / "stage13c_nonlocal3_arithmetic_trace.json"


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


def sliced_summary(value: torch.Tensor) -> dict:
    """Summarize one leading-dimension slice at a time to bound scratch memory."""
    x = value.detach()
    parts = [x] if x.ndim == 0 else (x[i] for i in range(x.shape[0]))
    summaries = [guard.tensor_summary(part) for part in parts]
    minima = [item["finite_min"] for item in summaries if item["finite_min"] is not None]
    maxima = [item["finite_max"] for item in summaries if item["finite_max"] is not None]
    return {"finite": all(item["finite"] for item in summaries),
            "nan_count": sum(item["nan_count"] for item in summaries),
            "inf_count": sum(item["inf_count"] for item in summaries),
            "finite_min": min(minima) if minima else None,
            "finite_max": max(maxima) if maxima else None,
            "dtype": str(x.dtype), "shape": list(x.shape)}


def state_parts(model):
    bad_parameters = first_bad(model.named_parameters())
    bad_buffers = first_bad(model.named_buffers())
    return {"parameters": {"finite": bad_parameters is None,
                           "first_nonfinite": bad_parameters},
            "buffers": {"finite": bad_buffers is None,
                        "first_nonfinite": bad_buffers}}


def inspect_forward(model, images, labels, mode: str) -> dict:
    """Trace the actual NonLocalBlock.forward operation order without editing source."""
    block = model.resnet.nonlocal3
    original_forward = block.forward
    intermediates = []

    def record(name: str, operation: str, value: torch.Tensor) -> None:
        intermediates.append({"name": name, "operation": operation,
                              **sliced_summary(value)})

    def traced_forward(self, x):
        record("block_input", "input_from_resnet.layer3", x)
        b, _, h, w = x.shape
        if h >= self.key_stride and w >= self.key_stride:
            kv = F.avg_pool2d(x, self.key_stride)
            record("key_value_source", "F.avg_pool2d(x, key_stride)", kv)
        else:
            kv = x
            record("key_value_source", "identity(x); no pool", kv)
        theta = self.theta(x)
        record("theta_projection", "self.theta(x)", theta)
        q = theta.flatten(2).transpose(1, 2)
        record("q", "theta.flatten(2).transpose(1, 2)", q)
        phi = self.phi(kv)
        record("phi_projection", "self.phi(kv)", phi)
        k = phi.flatten(2)
        record("k", "phi.flatten(2)", k)
        g = self.g(kv)
        record("g_projection", "self.g(kv)", g)
        v = g.flatten(2).transpose(1, 2)
        record("v", "g.flatten(2).transpose(1, 2)", v)
        scores = torch.bmm(q, k)
        record("qk_affinity", "torch.bmm(q, k)", scores)
        # Actual source applies NO 1/sqrt(d) or other scaling operation.
        record("pre_softmax_scores", "identity(qk_affinity); no scaling", scores)
        attention = torch.softmax(scores, dim=-1)
        record("attention", "torch.softmax(scores, dim=-1)", attention)
        weighted = torch.bmm(attention, v)
        record("attention_value_product", "torch.bmm(attention, v)", weighted)
        attended = weighted.transpose(1, 2).reshape(b, -1, h, w)
        record("tensor_passed_to_project", "transpose(1, 2).reshape(b, -1, h, w)", attended)
        projected = self.project(attended)
        record("output_projection", "self.project(attended)", projected)
        output = x + projected
        record("nonlocal3_output", "x + projected", output)
        return output

    block.forward = types.MethodType(traced_forward, block)
    try:
        model.train(True)
        with torch.no_grad(), torch.autocast("cuda", enabled=(mode == "amp")):
            logits = model(images)
            logits_summary = sliced_summary(logits)
            first_bad_record = next((item for item in intermediates if not item["finite"]), None)
            row = {"mode": mode, "implementation": "src/models/egvan/non_local.py:NonLocalBlock.forward",
                   "scaling_operation": "none in actual implementation",
                   "intermediates": intermediates,
                   "first_nonfinite_record": first_bad_record,
                   "first_nonfinite_arithmetic": (
                       "upstream_of_nonlocal3" if first_bad_record and
                       first_bad_record["name"] == "block_input" else
                       first_bad_record["operation"] if first_bad_record else None),
                   "logits": logits_summary}
            if logits_summary["finite"]:
                terms = guard.focal_terms(logits, labels, 1.0)
                row["focal_terms"] = sliced_summary(terms)
                row["mean_loss"] = float(terms.mean().float().cpu())
                row["mean_loss_finite"] = math.isfinite(row["mean_loss"])
            row["model_after_forward"] = model_status(model)
            row["model_state_parts_after_forward"] = state_parts(model)
            return row
    finally:
        block.forward = original_forward


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
    reference_path = HERE / "stage13c_C_batch7_memory_safe_probe.json"
    if not reference_path.is_file() or base.sha256(reference_path) != REFERENCE_SHA256:
        raise ValueError("Verified memory-safe probe absent or changed")
    reference_probe = json.loads(reference_path.read_text(encoding="utf-8"))
    model, optimizer, scaler, loader = gate.make_components(root, "C", configs["C"])
    report = {"status": "IN_PROGRESS", "scope": "HAM train only; C batches 1-6, batch-7 forward only",
              "seed": 42, "initial_scale": 8192.0, "prior_gate_sha256": GATE_SHA256,
              "reference_probe_sha256": REFERENCE_SHA256,
              "protocol": "C first six exact train steps; CPU state snapshot; batch-7 arithmetic trace under AMP then FP32",
              "script_sha256": base.sha256(Path(__file__)),
              "source_sha256": {
                  "analysis/stage13_melanoma_ablation/stage13c_abc_8192_gate.py":
                      base.sha256(HERE / "stage13c_abc_8192_gate.py"),
                  "experiments/egvan_melanoma_ablation_exp/train_ablation.py":
                      base.sha256(root / "experiments/egvan_melanoma_ablation_exp/train_ablation.py"),
                  "src/models/egvan/egvan.py": base.sha256(root / "src/models/egvan/egvan.py"),
                  "src/models/egvan/non_local.py": base.sha256(root / "src/models/egvan/non_local.py")},
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
                report["runtime"] != prior["runtime"] or
                report["runtime"] != reference_probe["runtime"]):
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
                if (report["batch7"]["model_sha256_before_forward"] !=
                        reference_probe["batch7"]["model_sha256_before_forward"]):
                    report["status"] = "PRE_BATCH7_MODEL_MISMATCH"
                    break
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
                report["batch7"]["model_sha256_after_restore"] = digest_model(model)
                if (report["batch7"]["model_sha256_after_restore"] !=
                        report["batch7"]["model_sha256_before_forward"]):
                    report["status"] = "MODEL_RESTORE_MISMATCH"
                    break
                gc.collect()
                torch.cuda.empty_cache()
                report["cuda_memory"]["before_fp32_forward"] = cuda_memory()
                phase = "batch_7_fp32_forward"
                report["batch7"]["fp32_forward"] = inspect_forward(model, images, labels, "fp32")
                report["cuda_memory"]["after_fp32_forward"] = cuda_memory()
                report["batch7"]["optimizer_step_performed_on_batch7"] = False
                report["batch7"]["optimizer_state_released_before_forwards"] = True
                report["batch7"]["amp_reproduced_prior_nonfinite_logits"] = (
                    not report["batch7"]["amp_forward"]["logits"]["finite"] and
                    not reference["logits_finite"])
                report["batch7"]["fp32_reproduced_reference_finite"] = (
                    report["batch7"]["fp32_forward"]["logits"]["finite"] and
                    report["batch7"]["fp32_forward"].get("mean_loss_finite", False) and
                    math.isclose(report["batch7"]["fp32_forward"]["mean_loss"],
                                 reference_probe["batch7"]["fp32_forward"]["mean_loss"],
                                 rel_tol=1e-5, abs_tol=1e-5))
                report["status"] = ("COMPLETE_ARITHMETIC_TRACE" if
                    report["batch7"]["amp_reproduced_prior_nonfinite_logits"] and
                    report["batch7"]["fp32_reproduced_reference_finite"]
                    else "FORWARD_REPRODUCTION_MISMATCH")
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
