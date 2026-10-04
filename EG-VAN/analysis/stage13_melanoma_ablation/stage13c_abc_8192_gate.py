"""Bounded HAM-train-only A/B/C T4 AMP gate; writes JSON, never checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_S_Weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNER_DIR = ROOT / "experiments/egvan_melanoma_ablation_exp"
sys.path.insert(0, str(RUNNER_DIR))
import train_ablation as base  # noqa: E402
import stage13c_guarded as guard  # noqa: E402
from stage13c_multistep_amp_diagnostic import (  # noqa: E402
    digest_model, digest_tensor, gradient_status, model_status, optimizer_status,
)

INITIAL_SCALE = 8192.0
BATCHES = 16
TRACE_SHA256 = "864f8ead863137ff803a7e1e5430b97c6bb1a176fec11104af142e44d8242888"
DEFAULT_OUTPUT = HERE / "stage13c_abc_8192_gate.json"


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
    step_count = [0]

    def post_step(_optimizer, _args, _kwargs):
        step_count[0] += 1

    hook = optimizer.register_step_post_hook(post_step)
    try:
        def stop_before_step(row: dict, reason: str) -> None:
            row["failure"] = reason
            row["optimizer_step_performed"] = False
            row["optimizer_step_skipped"] = False  # No scaler.step call was attempted.
            row["scale_after"] = float(scaler.get_scale())
            row["model_after_batch"] = model_status(model)
            row["optimizer_after_batch"] = optimizer_status(optimizer)

        for index, (images_cpu, labels_cpu) in enumerate(loader, start=1):
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
                stop_before_step(row, "invalid scale before batch")
                break
            images = images_cpu.cuda(non_blocking=True)
            labels = labels_cpu.cuda(non_blocking=True)
            row["input_finite"] = bool(torch.isfinite(images).all().item())
            row["labels_valid"] = bool(labels.dtype == torch.long and
                ((labels >= 0) & (labels < 7)).all().item())
            if not row["input_finite"] or not row["labels_valid"]:
                stop_before_step(row, "invalid input or labels")
                break
            model.train(True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                logits = model(images)
                row["logits_finite"] = bool(torch.isfinite(logits).all().item())
                if row["logits_finite"]:
                    terms = guard.focal_terms(logits, labels, cfg["loss"]["mel_multiplier"])
                    row["per_example_loss_finite"] = bool(torch.isfinite(terms).all().item())
                    if row["per_example_loss_finite"]:
                        loss = terms.mean()
                        row["loss"] = float(loss.detach().float().cpu())
                        row["loss_finite"] = math.isfinite(row["loss"])
            if not (row["logits_finite"] and row.get("per_example_loss_finite")
                    and row.get("loss_finite")):
                stop_before_step(row, "nonfinite forward or loss")
                break
            scaled = scaler.scale(loss)
            row["scaled_loss_finite"] = bool(torch.isfinite(scaled).all().item())
            if not row["scaled_loss_finite"]:
                stop_before_step(row, "nonfinite scaled loss")
                break
            scaled.backward()
            row["scaled_gradients"] = gradient_status(model)
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
                    not row["unscaled_gradients"]["finite"] and
                    row["scale_after"] < row["scale_before"])
            if (not row["model_after_batch"]["finite"] or
                    not row["optimizer_after_batch"]["finite"] or
                    not math.isfinite(row["scale_after"]) or row["scale_after"] <= 0):
                row["failure"] = "corrupt post-step model/optimizer/scaler state"
                break
            if row["optimizer_step_skipped"] and not row["skip_consistent_with_overflow"]:
                row["failure"] = "unexplained optimizer step skip"
                break
            # A handled overflow is recorded distinctly; continue the bounded
            # window only while model/optimizer state remains finite.
        result["optimizer_steps_performed"] = sum(
            row.get("optimizer_step_performed", False) for row in result["batches"])
        result["optimizer_steps_skipped"] = sum(
            row.get("optimizer_step_skipped", False) for row in result["batches"])
        result["nonfinite_gradient_batches"] = [row["batch_index"] for row in result["batches"]
            if not row.get("scaled_gradients", {}).get("finite", True) or
               not row.get("unscaled_gradients", {}).get("finite", True)]
        result["status"] = (
            "STOPPED_NUMERICAL_FAILURE" if any("failure" in row for row in result["batches"])
            else "INCOMPLETE" if len(result["batches"]) != BATCHES
            else "RECOVERED_OVERFLOW_NO_CLEARANCE" if result["nonfinite_gradient_batches"] or
                 result["optimizer_steps_skipped"]
            else "PASS_BOUNDED_WINDOW")
    finally:
        hook.remove()
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
            if (result["status"] != "PASS_BOUNDED_WINDOW" or
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
