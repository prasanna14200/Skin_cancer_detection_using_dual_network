"""Bounded A/B/C T4 gate: only nonlocal3 q@k uses FP32 under normal AMP."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import types
from collections import Counter
from itertools import islice
from pathlib import Path

import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_S_Weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNER_DIR = ROOT / "experiments/egvan_melanoma_ablation_exp"
sys.path.insert(0, str(RUNNER_DIR))
import train_ablation as base  # noqa: E402
import stage13c_guarded as guard  # noqa: E402
from stage13c_multistep_amp_diagnostic import (  # noqa: E402
    digest_model, digest_tensor, first_bad, gradient_status, model_status, optimizer_status,
)

INITIAL_SCALE = 8192.0
BATCHES = 16
TRACE_SHA256 = "864f8ead863137ff803a7e1e5430b97c6bb1a176fec11104af142e44d8242888"
SELECTIVE_TRACE_SHA256 = "723b8cbabb9ed9e84964fc91f002e3039b24ea4fd3242f27786f2bcce99f5ff5"
PRIOR_GATE_SHA256 = "ccdfe75250492878f759d6e95feda59afc1ad1ea88bdb25b3227eca0b0268c56"
DEFAULT_OUTPUT = HERE / "stage13c_abc_selective_qk_fp32_gate.json"


def install_selective_qk_policy(model):
    """Patch one block instance in memory; preserve source architecture semantics."""
    block = model.resnet.nonlocal3
    original = block.forward

    def selective_forward(self, x):
        b, _, h, w = x.shape
        kv = F.avg_pool2d(x, self.key_stride) if h >= self.key_stride and w >= self.key_stride else x
        q = self.theta(x).flatten(2).transpose(1, 2)
        k = self.phi(kv).flatten(2)
        v = self.g(kv).flatten(2).transpose(1, 2)
        with torch.autocast("cuda", enabled=False):
            scores = torch.bmm(q.float(), k.float())
        attention = torch.softmax(scores, dim=-1)
        attended = torch.bmm(attention, v).transpose(1, 2).reshape(b, -1, h, w)
        return x + self.project(attended)

    block.forward = types.MethodType(selective_forward, block)
    return lambda: setattr(block, "forward", original)


def state_parts(model):
    parameter_bad = first_bad(model.named_parameters())
    buffer_bad = first_bad(model.named_buffers())
    return {"parameters": {"finite": parameter_bad is None, "first_nonfinite": parameter_bad},
            "buffers": {"finite": buffer_bad is None, "first_nonfinite": buffer_bad}}


def finite_scale(scaler):
    value = float(scaler.get_scale())
    return value if math.isfinite(value) else None


def verify_scientific_factors(root: Path) -> dict:
    split = root / "data/splits/split_leakage_aware.csv"
    if base.sha256(split) != base.SPLIT_SHA:
        raise ValueError("Frozen split hash changed")
    configs = {variant: base.variant_config(variant) for variant in "ABC"}
    for variant, cfg in configs.items():
        if (cfg["seed"] != 42 or cfg["epochs"] != 25 or cfg["image_size"] != 384 or
                cfg["physical_batch_size"] != 16 or cfg["effective_batch_size"] != 16 or
                cfg["gradient_accumulation_steps"] != 1 or
                tuple(cfg["classes"]) != base.CLASSES):
            raise ValueError(f"Frozen core configuration changed: {variant}")
    if (configs["A"]["loss"]["mel_multiplier"] != 1.0 or
            configs["B"]["loss"]["mel_multiplier"] != 1.5630495442733532 or
            configs["A"]["sampler"]["mel_weight"] != 1.5630495442733532 or
            configs["B"]["sampler"]["mel_weight"] != 1.5630495442733532 or
            configs["C"]["sampler"]["mel_weight"] != 2.4431238778531372):
        raise ValueError("Registered MEL factors changed")
    for variant in "BC":
        control = json.loads(json.dumps(configs["A"]))
        candidate = json.loads(json.dumps(configs[variant]))
        for cfg in (control, candidate):
            cfg.pop("variant")
            cfg.pop("description")
        if variant == "B":
            control["loss"]["name"] = candidate["loss"]["name"]
            control["loss"]["mel_multiplier"] = candidate["loss"]["mel_multiplier"]
        else:
            control["sampler"]["mel_weight"] = candidate["sampler"]["mel_weight"]
        if control != candidate:
            raise ValueError(f"Unregistered {variant} scientific difference")
    return configs


def make_components(root: Path, variant: str, cfg: dict):
    names, Dataset, _, EGVAN, _, make_transforms, set_seed = base.imports(root)
    if tuple(names) != base.CLASSES:
        raise ValueError("Class order changed")
    set_seed(42)
    train_tf, _ = make_transforms(EfficientNet_V2_S_Weights.DEFAULT)
    train_ds = Dataset(root / "data/processed/images",
                       root / "data/splits/split_leakage_aware.csv", "train", train_tf)
    if len(train_ds) != 8015 or dict(Counter(row["dx"] for row in train_ds.rows)) != base.TRAIN_COUNTS:
        raise ValueError("Frozen HAM train partition changed")
    generator = torch.Generator(device="cpu").manual_seed(42)
    loader = DataLoader(train_ds, batch_size=16,
        sampler=base.make_sampler(train_ds, cfg["sampler"]["mel_weight"], generator),
        num_workers=2, pin_memory=True)
    model = EGVAN(num_classes=7, pretrained_efficient=True, pretrained_resnet=False).cuda()
    optimizer, _ = base.optimizer_scheduler(model)
    scaler = torch.amp.GradScaler("cuda", init_scale=INITIAL_SCALE)
    return model, optimizer, scaler, loader


def run_variant(root: Path, variant: str, cfg: dict, result: dict) -> None:
    model, optimizer, scaler, loader = make_components(root, variant, cfg)
    result["initial_model_sha256"] = digest_model(model)
    restore_policy = install_selective_qk_policy(model)
    result["policy"] = "nonlocal3 q@k = FP32 torch.bmm(q.float(), k.float()); rest under normal AMP"
    step_count = [0]

    def post_step(_optimizer, _args, _kwargs):
        step_count[0] += 1

    hook = optimizer.register_step_post_hook(post_step)
    try:
        def stop_before_step(row: dict, reason: str) -> None:
            row["failure"] = reason
            row["optimizer_step_attempted"] = False
            row["optimizer_step_performed"] = False
            row["optimizer_step_skipped"] = False  # No scaler.step call was attempted.
            row["scale_after"] = finite_scale(scaler)
            row["model_after_batch"] = model_status(model)
            row["state_parts_after_batch"] = state_parts(model)
            row["optimizer_after_batch"] = optimizer_status(optimizer)

        for index, (images_cpu, labels_cpu) in enumerate(islice(loader, BATCHES), start=1):
            counts = Counter(base.CLASSES[int(y)] for y in labels_cpu.tolist())
            row = {"batch_index": index, "input_sha256": digest_tensor(images_cpu),
                   "labels_sha256": digest_tensor(labels_cpu),
                   "target_class_distribution": dict(counts),
                   "contains_mel": counts.get("mel", 0) > 0,
                   "scale_before": finite_scale(scaler),
                   "first_nonfinite_tensor": None}
            result["batches"].append(row)
            if row["scale_before"] is None or row["scale_before"] <= 0:
                row["first_nonfinite_tensor"] = "GradScaler scale"
                stop_before_step(row, "invalid scale before batch")
                break
            images = images_cpu.cuda(non_blocking=True)
            labels = labels_cpu.cuda(non_blocking=True)
            row["input_finite"] = bool(torch.isfinite(images).all().item())
            row["labels_valid"] = bool(labels.dtype == torch.long and
                ((labels >= 0) & (labels < 7)).all().item())
            if not row["input_finite"] or not row["labels_valid"]:
                row["first_nonfinite_tensor"] = "input" if not row["input_finite"] else "labels"
                stop_before_step(row, "invalid input or labels")
                break
            model.train(True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                logits = model(images)
                row["logits_finite"] = bool(torch.isfinite(logits).all().item())
                row["state_parts_after_forward"] = state_parts(model)
                if row["logits_finite"]:
                    terms = guard.focal_terms(logits, labels, cfg["loss"]["mel_multiplier"])
                    row["per_example_loss_finite"] = bool(torch.isfinite(terms).all().item())
                    if row["per_example_loss_finite"]:
                        loss = terms.mean()
                        loss_value = float(loss.detach().float().cpu())
                        row["loss_finite"] = math.isfinite(loss_value)
                        row["loss"] = loss_value if row["loss_finite"] else None
            if not (row["logits_finite"] and
                    row["state_parts_after_forward"]["parameters"]["finite"] and
                    row["state_parts_after_forward"]["buffers"]["finite"] and
                    row.get("per_example_loss_finite")
                    and row.get("loss_finite")):
                if not row["logits_finite"]:
                    row["first_nonfinite_tensor"] = "logits"
                else:
                    row["first_nonfinite_tensor"] = (
                        row["state_parts_after_forward"]["parameters"]["first_nonfinite"] or
                        row["state_parts_after_forward"]["buffers"]["first_nonfinite"] or
                        ("per_example_loss" if not row.get("per_example_loss_finite", False)
                         else "loss"))
                stop_before_step(row, "nonfinite forward, model state, or loss")
                break
            scaled = scaler.scale(loss)
            row["scaled_loss_finite"] = bool(torch.isfinite(scaled).all().item())
            if not row["scaled_loss_finite"]:
                row["first_nonfinite_tensor"] = "scaled_loss"
                stop_before_step(row, "nonfinite scaled loss")
                break
            scaled.backward()
            row["scaled_gradients"] = gradient_status(model)
            scaler.unscale_(optimizer)
            row["unscaled_gradients"] = gradient_status(model)
            if not row["scaled_gradients"]["finite"] or not row["unscaled_gradients"]["finite"]:
                row["first_nonfinite_tensor"] = (
                    row["scaled_gradients"]["first_nonfinite"] or
                    row["unscaled_gradients"]["first_nonfinite"])
            before = step_count[0]
            row["optimizer_step_attempted"] = True
            scaler.step(optimizer)
            row["optimizer_step_performed"] = step_count[0] > before
            row["optimizer_step_skipped"] = not row["optimizer_step_performed"]
            scaler.update()
            row["scale_after"] = finite_scale(scaler)
            row["model_after_batch"] = model_status(model)
            row["state_parts_after_batch"] = state_parts(model)
            row["optimizer_after_batch"] = optimizer_status(optimizer)
            if row["optimizer_step_skipped"]:
                row["skip_consistent_with_overflow"] = (
                    not row["unscaled_gradients"]["finite"] and
                    row["scale_after"] is not None and
                    row["scale_after"] < row["scale_before"])
            if (not row["model_after_batch"]["finite"] or
                    not row["state_parts_after_batch"]["parameters"]["finite"] or
                    not row["state_parts_after_batch"]["buffers"]["finite"] or
                    not row["optimizer_after_batch"]["finite"] or
                    row["scale_after"] is None or row["scale_after"] <= 0):
                row["first_nonfinite_tensor"] = row["first_nonfinite_tensor"] or (
                    row["model_after_batch"]["first_nonfinite"] or
                    row["optimizer_after_batch"]["first_nonfinite"] or "GradScaler scale")
                row["failure"] = "corrupt post-step model/optimizer/scaler state"
                break
            if row["optimizer_step_skipped"] and not row["skip_consistent_with_overflow"]:
                row["failure"] = "unexplained optimizer step skip"
                break
            if row["optimizer_step_performed"] and not row["unscaled_gradients"]["finite"]:
                row["failure"] = "optimizer step performed with nonfinite gradients"
                break
            # A handled overflow is recorded distinctly; continue the bounded
            # window only while model/optimizer state remains finite.
        result["batches_attempted"] = len(result["batches"])
        result["batches_completed"] = sum(
            "failure" not in row and "optimizer_step_performed" in row
            for row in result["batches"])
        result["optimizer_steps_performed"] = sum(
            row.get("optimizer_step_performed", False) for row in result["batches"])
        result["optimizer_steps_skipped"] = sum(
            row.get("optimizer_step_skipped", False) for row in result["batches"])
        result["nonfinite_gradient_batches"] = [row["batch_index"] for row in result["batches"]
            if not row.get("scaled_gradients", {}).get("finite", True) or
               not row.get("unscaled_gradients", {}).get("finite", True)]
        result["status"] = (
            "STOPPED_NUMERICAL_FAILURE" if any("failure" in row for row in result["batches"])
            else "INCOMPLETE" if result["batches_completed"] != BATCHES
            else "PASS_BOUNDED_WITH_RECOVERED_OVERFLOW" if result["optimizer_steps_skipped"]
            else "PASS_BOUNDED_WINDOW")
    finally:
        hook.remove()
        restore_policy()
        del model, optimizer, scaler, loader
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root, output = args.project_root.resolve(), args.output.resolve()
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Tesla T4 required")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite gate result: {output}")
    if output.parent != root / "analysis/stage13_melanoma_ablation":
        raise ValueError("Output must be in Stage 13 analysis directory")
    trace = HERE / "stage13c_multistep_amp_trace.json"
    if not trace.is_file() or base.sha256(trace) != TRACE_SHA256:
        raise ValueError("Audited A-only multistep trace absent or changed")
    previous_A = json.loads(trace.read_text(encoding="utf-8"))["scales"][0]
    selective_path = HERE / "stage13c_selective_qk_fp32_trace.json"
    if not selective_path.is_file() or base.sha256(selective_path) != SELECTIVE_TRACE_SHA256:
        raise ValueError("Audited selective C batch-7 trace absent or changed")
    selective_trace = json.loads(selective_path.read_text(encoding="utf-8"))
    if selective_trace["status"] != "SELECTIVE_BATCH7_PASS":
        raise ValueError("Selective C batch-7 probe did not pass")
    prior_path = HERE / "stage13c_abc_8192_gate.json"
    if not prior_path.is_file() or base.sha256(prior_path) != PRIOR_GATE_SHA256:
        raise ValueError("Audited prior A/B/C gate absent or changed")
    prior_gate = json.loads(prior_path.read_text(encoding="utf-8"))
    configs = verify_scientific_factors(root)
    source_paths = ["experiments/egvan_melanoma_ablation_exp/train_ablation.py",
                    "experiments/egvan_melanoma_ablation_exp/stage13c_guarded.py",
                    "analysis/stage13_melanoma_ablation/stage13c_multistep_amp_diagnostic.py",
                    "src/train.py", "src/dataset.py"]
    source_paths += [str(p.relative_to(root)).replace("\\", "/")
                     for p in sorted((root / "src/models/egvan").glob("*.py"))]
    report = {"status": "IN_PROGRESS", "scope": "HAM train only; no validation/test/PH2 loader",
              "seed": 42, "initial_scale": INITIAL_SCALE, "batches_per_variant": BATCHES,
              "audited_A_multistep_trace_sha256": TRACE_SHA256,
              "audited_selective_batch7_trace_sha256": SELECTIVE_TRACE_SHA256,
              "audited_prior_abc_gate_sha256": PRIOR_GATE_SHA256,
              "numerical_policy": "nonlocal3 q@k FP32 inside normal CUDA AMP, identical for A/B/C",
              "split_sha256": base.sha256(root / "data/splits/split_leakage_aware.csv"),
              "config_sha256": {v: base.sha256(RUNNER_DIR / f"config_{v}.json") for v in "ABC"},
              "gate_script_sha256": base.sha256(Path(__file__)),
              "source_sha256": {p: base.sha256(root / p) for p in source_paths},
              "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                          "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(0)},
              "research_checkpoint_saved": False, "full_training_performed": False,
              "ham_test_accessed": False, "ph2_accessed": False, "variants": {}}
    try:
        for variant in "ABC":
            result = {"variant": variant, "scientific_config_sha256": report["config_sha256"][variant],
                      "batches": [], "status": "IN_PROGRESS"}
            report["variants"][variant] = result
            run_variant(root, variant, configs[variant], result)
            if variant == "A":
                result["same_initial_model_as_a_only_trace"] = (
                    result["initial_model_sha256"] == previous_A["initial_model_sha256"])
                result["same_data_sequence_as_a_only_trace"] = (
                    len(result["batches"]) == len(previous_A["batches"]) == BATCHES and
                    all(x["input_sha256"] == y["input_sha256"] and
                        x["labels_sha256"] == y["labels_sha256"]
                        for x, y in zip(result["batches"], previous_A["batches"])))
            else:
                a = report["variants"]["A"]
                result["same_initial_model_as_A"] = (
                    result["initial_model_sha256"] == a["initial_model_sha256"])
                if variant == "B":
                    result["same_data_sequence_as_A"] = (
                        len(result["batches"]) == len(a["batches"]) == BATCHES and
                        all(x["input_sha256"] == y["input_sha256"] and
                            x["labels_sha256"] == y["labels_sha256"]
                            for x, y in zip(result["batches"], a["batches"])))
            prior_batches = prior_gate["variants"][variant]["batches"]
            result["same_data_prefix_as_prior_gate"] = (
                len(result["batches"]) >= len(prior_batches) and
                all(x["input_sha256"] == y["input_sha256"] and
                    x["labels_sha256"] == y["labels_sha256"]
                    for x, y in zip(result["batches"], prior_batches)))
            if (result["status"] not in
                    ("PASS_BOUNDED_WINDOW", "PASS_BOUNDED_WITH_RECOVERED_OVERFLOW") or
                    not result["same_data_prefix_as_prior_gate"] or
                    (variant == "A" and not result["same_initial_model_as_a_only_trace"]) or
                    (variant == "A" and not result["same_data_sequence_as_a_only_trace"]) or
                    (variant != "A" and not result["same_initial_model_as_A"]) or
                    (variant == "B" and not result["same_data_sequence_as_A"])):
                report["status"] = "FAIL_OR_NO_CLEARANCE"
                break
        else:
            report["status"] = "PASS_BOUNDED_WINDOW_ALL_VARIANTS"
    except Exception as exc:
        report["status"] = "EXCEPTION"
        report["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Gate JSON: {output}", flush=True)
        print(f"Gate status: {report['status']}", flush=True)


if __name__ == "__main__":
    main()
