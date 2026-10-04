"""Bounded, train-only T4 diagnostic of Stage 13C AMP scales; never saves checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import torch
import torchvision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNER_DIR = ROOT / "experiments/egvan_melanoma_ablation_exp"
sys.path.insert(0, str(RUNNER_DIR))
import stage13c_guarded as guard  # noqa: E402
import train_ablation as base  # noqa: E402

SCALES = (8192.0, 4096.0, 2048.0)
BATCHES = 16
DEFAULT_OUTPUT = HERE / "stage13c_multistep_amp_trace.json"


def digest_tensor(value: torch.Tensor) -> str:
    x = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(str(tuple(x.shape)).encode())
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def digest_model(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(digest_tensor(tensor).encode())
    return h.hexdigest()


def first_bad(named_tensors):
    for name, value in named_tensors:
        if value is not None and value.is_floating_point():
            if not bool(torch.isfinite(value).all().item()):
                return {"tensor": name, **guard.tensor_summary(value)}
    return None


def gradient_status(model):
    bad = first_bad((name, p.grad) for name, p in model.named_parameters())
    return {"finite": bad is None, "first_nonfinite": bad}


def model_status(model):
    bad = first_bad(model.state_dict().items())
    return {"finite": bad is None, "first_nonfinite": bad}


def optimizer_status(optimizer):
    named = ((f"parameter_{index}/{key}", value)
             for index, state in enumerate(optimizer.state.values())
             for key, value in state.items() if isinstance(value, torch.Tensor))
    bad = first_bad(named)
    return {"finite": bad is None, "first_nonfinite": bad}


def run_scale(root: Path, initial_scale: float) -> dict:
    model, optimizer, _, original_scaler, train_loader, _, _ = guard.build_components(root, "A")
    del original_scaler
    scaler = torch.amp.GradScaler("cuda", init_scale=initial_scale)
    result = {"variant": "A", "initial_scale": initial_scale,
              "initial_model_sha256": digest_model(model), "batches": [],
              "status": "IN_PROGRESS"}
    step_count = [0]

    def record_step(_optimizer, _args, _kwargs):
        step_count[0] += 1

    hook = optimizer.register_step_post_hook(record_step)
    try:
        for index, (images_cpu, labels_cpu) in enumerate(train_loader, start=1):
            if index > BATCHES:
                break
            counts = Counter(base.CLASSES[int(y)] for y in labels_cpu.tolist())
            row = {"batch_index": index, "input_sha256": digest_tensor(images_cpu),
                   "labels_sha256": digest_tensor(labels_cpu),
                   "target_class_distribution": dict(counts),
                   "contains_mel": counts.get("mel", 0) > 0,
                   "scale_before": float(scaler.get_scale())}
            result["batches"].append(row)
            if not math.isfinite(row["scale_before"]) or row["scale_before"] <= 0:
                row["failure"] = "invalid GradScaler scale before batch"
                break
            images = images_cpu.cuda(non_blocking=True)
            labels = labels_cpu.cuda(non_blocking=True)
            row["input_finite"] = bool(torch.isfinite(images).all().item())
            row["labels_valid"] = bool(labels.dtype == torch.long and
                ((labels >= 0) & (labels < len(base.CLASSES))).all().item())
            if not row["input_finite"] or not row["labels_valid"]:
                row["failure"] = "input or label invalid"
                break
            model.train(True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                logits = model(images)
                row["logits_finite"] = bool(torch.isfinite(logits).all().item())
                if row["logits_finite"]:
                    terms = guard.focal_terms(logits, labels, 1.0)
                    row["per_example_loss_finite"] = bool(torch.isfinite(terms).all().item())
                    if row["per_example_loss_finite"]:
                        loss = terms.mean()
                        row["loss"] = float(loss.detach().float().cpu())
                        row["loss_finite"] = math.isfinite(row["loss"])
            if not (row["logits_finite"] and row.get("per_example_loss_finite")
                    and row.get("loss_finite")):
                row["failure"] = "nonfinite forward or loss"
                row["model_after_batch"] = model_status(model)
                break
            scaled = scaler.scale(loss)
            row["scaled_loss_finite"] = bool(torch.isfinite(scaled).all().item())
            if not row["scaled_loss_finite"]:
                row["failure"] = "nonfinite scaled loss"
                break
            scaled.backward()
            row["scaled_gradients"] = gradient_status(model)
            # Observe overflow, but allow GradScaler to handle it normally.
            scaler.unscale_(optimizer)
            row["unscaled_gradients"] = gradient_status(model)
            before = step_count[0]
            scaler.step(optimizer)
            row["optimizer_step_performed"] = step_count[0] > before
            row["optimizer_step_skipped"] = not row["optimizer_step_performed"]
            scaler.update()
            row["scale_after"] = float(scaler.get_scale())
            row["model_after_batch"] = model_status(model)
            row["optimizer_after_batch"] = optimizer_status(optimizer)
            if row["optimizer_step_skipped"]:
                row["skip_consistent_with_overflow"] = (
                    not row["unscaled_gradients"]["finite"]
                    and row["scale_after"] < row["scale_before"])
            if (not row["model_after_batch"]["finite"] or
                    not row["optimizer_after_batch"]["finite"] or
                    not math.isfinite(row["scale_after"]) or row["scale_after"] <= 0):
                row["failure"] = "nonfinite model, optimizer, or scale after step attempt"
                break
            if row["optimizer_step_skipped"] and not row["skip_consistent_with_overflow"]:
                row["failure"] = "unexplained optimizer step skip"
                break
        result["status"] = "COMPLETE" if len(result["batches"]) == BATCHES and not any(
            "failure" in row for row in result["batches"]) else "STOPPED_EARLY"
        result["optimizer_steps_performed"] = sum(
            row.get("optimizer_step_performed", False) for row in result["batches"])
        result["optimizer_steps_skipped"] = sum(
            row.get("optimizer_step_skipped", False) for row in result["batches"])
        return result
    finally:
        hook.remove()
        del model, optimizer, scaler, train_loader
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root, output = args.project_root.resolve(), args.output.resolve()
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This bounded diagnostic requires a Tesla T4")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")
    if output.parent != root / "analysis/stage13_melanoma_ablation":
        raise ValueError("Output must be in the Stage 13 analysis directory")
    preflight = base.preflight(root)
    if preflight["status"] != "PASS":
        raise RuntimeError("Frozen train/validation preflight failed")
    report = {"status": "IN_PROGRESS", "scope": "HAM train only; Variant A",
              "seed": 42, "batches_per_scale": BATCHES, "initial_scales": SCALES,
              "split_sha256": base.sha256(root / "data/splits/split_leakage_aware.csv"),
              "config_A_sha256": base.sha256(RUNNER_DIR / "config_A.json"),
              "script_sha256": base.sha256(Path(__file__)),
              "guarded_runner_sha256": base.sha256(Path(guard.__file__)),
              "source_sha256": {name: base.sha256(root / name) for name in (
                  "experiments/egvan_melanoma_ablation_exp/train_ablation.py",
                  "src/train.py", "src/dataset.py", "src/models/egvan.py")},
              "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                          "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(0)},
              "research_checkpoint_saved": False, "full_training_performed": False,
              "ham_test_accessed": False, "ph2_accessed": False,
              "scales": []}
    try:
        for scale in SCALES:
            run = run_scale(root, scale)
            report["scales"].append(run)
            if len(report["scales"]) > 1:
                reference = report["scales"][0]
                run["same_initial_model_as_8192"] = (
                    run["initial_model_sha256"] == reference["initial_model_sha256"])
                n = min(len(run["batches"]), len(reference["batches"]))
                run["same_data_sequence_as_8192"] = all(
                    run["batches"][j]["input_sha256"] == reference["batches"][j]["input_sha256"]
                    and run["batches"][j]["labels_sha256"] == reference["batches"][j]["labels_sha256"]
                    for j in range(n)) and n == BATCHES
        report["status"] = "COMPLETE" if all(
            r["status"] == "COMPLETE" and
            (r is report["scales"][0] or
             (r["same_initial_model_as_8192"] and r["same_data_sequence_as_8192"]))
            for r in report["scales"]) else "INCOMPLETE_OR_MISMATCH"
    except Exception as exc:
        report["status"] = "EXCEPTION"
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Diagnostic JSON: {output}", flush=True)
        print(f"Diagnostic status: {report['status']}", flush=True)


if __name__ == "__main__":
    main()
