"""Single Stage 14 HAM train/validation candidate; prepared, never auto-started."""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE13 = ROOT / "experiments/egvan_melanoma_ablation_exp"
STAGE13_ANALYSIS = ROOT / "analysis/stage13_melanoma_ablation"
sys.path.insert(0, str(STAGE13))
sys.path.insert(0, str(STAGE13_ANALYSIS))
import train_ablation as base  # noqa: E402
import stage13c_guarded as guard  # noqa: E402
from stage13c_selective_qk_fp32_train import (  # noqa: E402
    checked_train_batch, validate_saved_scaler,
)
from stage13c_abc_selective_qk_fp32_gate import install_selective_qk_policy  # noqa: E402

NAME = "stage14_single_mel_objective_exp1"
CONFIG_SHA = "b9a83c4f281f747fa6dea5a8c5f64732a880452386864dcbf8f38446fb5a0de9"
RULE_SHA = "41b0b30641458799c9b1c8697a1494bdbfbe4f41f657df17b0f205d36161cb32"
NUMERICAL_SHA = "5329ad4a11af4ad6865f4c21ea9ca9c46f28bf4ea601cceff9a230a793d7e6ea"
GATE_SHA = "721eeaf6c64740d7dd8d7cdb06890a2ffa1c1be49c336f4d07033e4d172cdb00"
SELECTIVE_RUNNER_SHA = "d5cdfbade60e834e56acf9d67fbb3f83d90f0c6f96473dff9b9d3dd4c399ef1a"
INITIAL_SCALE = 8192.0
OUT = HERE / "run"


def runtime() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def read_preregistration() -> tuple[dict, dict, dict]:
    paths = (("config.json", CONFIG_SHA), ("selection_rule.json", RULE_SHA),
             ("numerical_protocol.json", NUMERICAL_SHA))
    for filename, digest in paths:
        if base.sha256(HERE / filename) != digest:
            raise ValueError(f"Frozen preregistration changed: {filename}")
    cfg, rule, numerical = (base.read_json(HERE / filename) for filename, _ in paths)
    old = base.variant_config("B")
    for key in old:
        if key not in ("variant", "description", "checkpoint_rule") and cfg[key] != old[key]:
            raise ValueError(f"Stage 14 training field differs from audited B setup: {key}")
    control = base.variant_config("A")
    for key in old:
        if key not in ("variant", "description", "checkpoint_rule", "loss") and cfg[key] != control[key]:
            raise ValueError(f"Unexpected scientific factor relative to control: {key}")
    if (cfg["experiment"] != NAME or cfg["epochs"] != 25 or cfg["seed"] != 42 or
            cfg["loss"]["mel_multiplier"] != 1.5630495442733532 or
            cfg["loss"]["alpha"] != 0.25 or cfg["loss"]["gamma"] != 2.0 or
            rule["experiment"] != NAME or rule["melanoma_support"] != 107 or
            rule["stage9_correct_melanomas"] != 62 or
            numerical["initial_grad_scaler_scale"] != INITIAL_SCALE or
            numerical["reference_bounded_gate_sha256"] != GATE_SHA):
        raise ValueError("Frozen Stage 14 factor/target/numerical policy mismatch")
    return cfg, rule, numerical


def check_preflight(root: Path) -> tuple[dict, dict, dict]:
    cfg, rule, numerical = read_preregistration()
    if base.sha256(STAGE13 / "stage13c_selective_qk_fp32_train.py") != SELECTIVE_RUNNER_SHA:
        raise ValueError("Audited selective-FP32 training helper changed")
    base.preflight(root)  # HAM train/val and frozen Stage 9 validation integrity only.
    gate_path = STAGE13_ANALYSIS / "stage13c_abc_selective_qk_fp32_gate.json"
    if base.sha256(gate_path) != GATE_SHA:
        raise ValueError("Audited selective-FP32 bounded gate changed")
    gate = base.read_json(gate_path)
    if (gate["status"] != "PASS_BOUNDED_WINDOW_ALL_VARIANTS" or
            gate["variants"]["B"]["status"] != "PASS_BOUNDED_WINDOW" or
            gate["variants"]["B"]["batches_completed"] != 16 or
            gate["config_sha256"]["B"] != base.sha256(STAGE13 / "config_B.json") or
            gate["split_sha256"] != base.SPLIT_SHA or
            gate["gate_script_sha256"] != base.sha256(STAGE13_ANALYSIS / "stage13c_abc_selective_qk_fp32_gate.py")):
        raise ValueError("Prior B numerical/data gate is not applicable")
    for relative, digest in gate["source_sha256"].items():
        if base.sha256(root / relative) != digest:
            raise ValueError(f"Source changed since bounded gate: {relative}")
    return cfg, rule, numerical


def eligible(metrics: dict, rule: dict) -> bool:
    floor = rule["eligible_if_all"]
    mel = metrics["per_class"]["mel"]
    if mel["support"] != rule["melanoma_support"]:
        raise ValueError("HAM validation MEL support changed")
    return (mel["recall"] * mel["support"] + 1e-9 >= floor["melanoma_correct_minimum"] and
            mel["recall"] + 1e-12 >= floor["melanoma_recall_minimum"] and
            mel["f1"] + 1e-12 >= floor["melanoma_f1_minimum"] and
            metrics["macro_f1"] + 1e-12 >= floor["macro_f1_minimum"] and
            metrics["accuracy"] + 1e-12 >= floor["accuracy_minimum"] and
            metrics["per_class"]["nv"]["recall"] + 1e-12 >= floor["nevus_recall_minimum"])


def eligible_history_row(row: dict, rule: dict) -> bool:
    return eligible({"accuracy": row["val_accuracy"], "macro_f1": row["val_macro_f1"],
        "per_class": {"mel": {"support": 107, "recall": row["val_mel_recall"],
                              "f1": row["val_mel_f1"]},
                      "nv": {"recall": row["val_nv_recall"]}}}, rule)


def finite_tensors(named_values) -> bool:
    return all(not isinstance(value, torch.Tensor) or not value.is_floating_point() or
               bool(torch.isfinite(value).all().item()) for _, value in named_values)


def validate_resume(state: dict, cfg: dict, rule: dict) -> dict:
    required = {"experiment", "epoch", "configuration", "config_sha256", "selection_rule_sha256",
        "numerical_protocol_sha256", "model_state", "optimizer_state", "scheduler_state",
        "scaler_state", "history", "best_epoch", "best_validation_loss",
        "sampler_generator_state", "torch_rng_state", "cuda_rng_states",
        "python_rng_state", "numpy_rng_state", "numerical_events",
        "class_order", "architecture", "runner_sha256"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Resume checkpoint is incomplete")
    if (state["experiment"] != NAME or state["configuration"] != cfg or
            state["config_sha256"] != CONFIG_SHA or state["selection_rule_sha256"] != RULE_SHA or
            state["numerical_protocol_sha256"] != NUMERICAL_SHA or
            state["class_order"] != list(base.CLASSES) or
            state["architecture"] != cfg["architecture"] or
            state["runner_sha256"] != base.sha256(Path(__file__))):
        raise ValueError("Resume checkpoint provenance/configuration mismatch")
    epoch, history = state["epoch"], state["history"]
    if type(epoch) is not int or not 1 <= epoch < 25 or not isinstance(history, list) or len(history) != epoch:
        raise ValueError("Resume checkpoint epoch/history mismatch or already complete")
    if [row.get("epoch") for row in history] != list(range(1, epoch + 1)):
        raise ValueError("Resume history numbering mismatch")
    for row in history:
        if (not math.isfinite(float(row["train_loss"])) or
                not math.isfinite(float(row["val_loss"])) or
                eligible_history_row(row, rule) is not row["eligible"]):
            raise ValueError("Resume history loss/eligibility mismatch")
    eligible_rows = [row for row in history if row["eligible"]]
    winner = min(eligible_rows, key=lambda row: (row["val_loss"], row["epoch"])) if eligible_rows else None
    if winner is None:
        if state["best_epoch"] is not None or state["best_validation_loss"] != math.inf:
            raise ValueError("Resume best selection inconsistent with history")
    elif state["best_epoch"] != winner["epoch"] or state["best_validation_loss"] != winner["val_loss"]:
        raise ValueError("Resume best selection inconsistent with history")
    if (not finite_tensors(state["model_state"].items()) or
            not finite_tensors((key, value) for slot in state["optimizer_state"]["state"].values()
                               for key, value in slot.items())):
        raise ValueError("Resume model/optimizer contains NaN/Inf")
    group = state["optimizer_state"]["param_groups"][0]
    scheduler = state["scheduler_state"]
    if (group["betas"] != (0.9, 0.999) or group["weight_decay"] != 0.0001 or
            group["eps"] != 1e-8 or not math.isfinite(float(group["lr"])) or
            scheduler["last_epoch"] != epoch or not math.isfinite(float(scheduler["best"])) or
            float(scheduler["_last_lr"][0]) != float(group["lr"])):
        raise ValueError("Resume optimizer/scheduler state invalid")
    saved_scale = validate_saved_scaler(state["scaler_state"])
    for key in ("sampler_generator_state", "torch_rng_state"):
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or not value.numel():
            raise ValueError(f"Resume RNG state invalid: {key}")
    if (not isinstance(state["cuda_rng_states"], (list, tuple)) or
            not state["cuda_rng_states"] or any(
                not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or not value.numel()
                for value in state["cuda_rng_states"])):
        raise ValueError("Resume CUDA RNG states invalid")
    if (not isinstance(state["python_rng_state"], tuple) or
            not isinstance(state["numpy_rng_state"], tuple)):
        raise ValueError("Resume Python/NumPy RNG states invalid")
    if (not isinstance(state["numerical_events"], list) or any(
            not isinstance(event, dict) or event.get("epoch", math.inf) > epoch or
            event.get("step_skipped") is not True for event in state["numerical_events"])):
        raise ValueError("Resume numerical event history invalid")
    return {"saved_epoch": epoch, "start_epoch": epoch + 1, "scale": saved_scale,
            "best_epoch": state["best_epoch"], "history_rows": len(history)}


def restore_resume(state, model, optimizer, scheduler, scaler, generator, *, cuda_rng_setter=None):
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["scaler_state"])
    if float(scaler.get_scale()) != float(state["scaler_state"]["scale"]):
        raise ValueError("GradScaler scale was reset during resume")
    generator.set_state(state["sampler_generator_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    setter = torch.cuda.set_rng_state_all if cuda_rng_setter is None else cuda_rng_setter
    setter(state["cuda_rng_states"])
    return state["epoch"] + 1, copy.deepcopy(state["history"]), state["best_epoch"], float(state["best_validation_loss"])


def verify_run_files_for_resume(state: dict) -> None:
    if (base.read_json(OUT / "config.json") != state["configuration"] or
            base.read_json(OUT / "numerical_protocol.json") != base.read_json(HERE / "numerical_protocol.json") or
            base.read_json(OUT / "selection_rule.json") != base.read_json(HERE / "selection_rule.json")):
        raise ValueError("Run snapshots differ from checkpoint/preregistration")
    history_path = OUT / "training_history.csv"
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            if (len(rows) > len(state["history"]) or reader.fieldnames != list(state["history"][0]) or
                    any(any(text_row[key] != str(saved_row[key]) for key in reader.fieldnames)
                        for text_row, saved_row in zip(rows, state["history"]))):
                raise ValueError("Run CSV does not match saved history")
    event_path = OUT / "numerical_events.json"
    if event_path.exists():
        logged = base.read_json(event_path)
        if (not isinstance(logged, list) or len(logged) > len(state["numerical_events"]) or
                logged != state["numerical_events"][:len(logged)]):
            raise ValueError("Numerical event log differs from saved checkpoint prefix")
    best_path = OUT / "best_checkpoint.pt"
    if state["best_epoch"] is None:
        if best_path.exists():
            raise ValueError("Unexpected best checkpoint without eligible epoch")
    else:
        if not best_path.is_file():
            raise FileNotFoundError("Selected best checkpoint missing")
        best = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
        if (best.get("epoch") != state["best_epoch"] or
                best.get("best_validation_loss") != state["best_validation_loss"] or
                not finite_tensors(best["model_state"].items())):
            raise ValueError("Existing selected checkpoint is inconsistent/corrupt")


def execute(root: Path, cfg: dict, rule: dict, numerical: dict, *, resume: bool) -> None:
    saved = None
    if resume:
        path = OUT / "last_checkpoint.pt"
        if not path.is_file():
            raise FileNotFoundError("--resume requires existing last_checkpoint.pt; no fresh fallback")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        validate_resume(saved, cfg, rule)
        verify_run_files_for_resume(saved)
        if len(saved["cuda_rng_states"]) != torch.cuda.device_count():
            raise ValueError("Saved CUDA RNG device count differs")
    else:
        if OUT.exists():
            raise FileExistsError("Fresh run refuses existing output directory")
        OUT.mkdir(parents=True)
        for filename in ("config.json", "numerical_protocol.json", "selection_rule.json"):
            (OUT / filename).write_bytes((HERE / filename).read_bytes())
    model, optimizer, scheduler, unused_scaler, train_loader, val_loader, generator = guard.build_components(root, "B")
    del unused_scaler
    scaler = torch.amp.GradScaler("cuda", init_scale=INITIAL_SCALE)
    restore_policy = install_selective_qk_policy(model)
    history, best_epoch, best_loss, start_epoch = [], None, math.inf, 1
    events = []
    ready = not resume
    try:
        if resume:
            events = copy.deepcopy(saved["numerical_events"])
            start_epoch, history, best_epoch, best_loss = restore_resume(
                saved, model, optimizer, scheduler, scaler, generator)
            guard.require_finite_states(model, optimizer, {"experiment": NAME, "epoch": saved["epoch"]})
            ready = True
            print(f"Stage14 resume saved_epoch={saved['epoch']} next_epoch={start_epoch} "
                  f"scale={scaler.get_scale():.0f}", flush=True)
            del saved
            gc.collect()
        for epoch in range(start_epoch, 26):
            train_sum = train_count = steps = 0
            for batch, (images_cpu, labels_cpu) in enumerate(train_loader, start=1):
                images = images_cpu.cuda(non_blocking=True)
                labels = labels_cpu.cuda(non_blocking=True)
                value, stepped, event = checked_train_batch(
                    model, optimizer, scaler, images, labels, NAME, epoch, batch,
                    cfg["loss"]["mel_multiplier"])
                train_sum += value * len(labels_cpu)
                train_count += len(labels_cpu)
                steps += int(stepped)
                if event is not None:
                    events.append(event)
            if steps == 0:
                raise RuntimeError(f"No optimizer step in epoch {epoch}")
            train_loss = train_sum / train_count
            val_loss, val_metrics, _ = guard.guarded_validation_epoch(model, val_loader, NAME, epoch)
            guard.require_finite_states(model, optimizer, {"experiment": NAME, "epoch": epoch})
            guard.require_finite_epoch_losses(train_loss, val_loss, NAME, epoch)
            scheduler.step(val_loss)
            is_eligible = eligible(val_metrics, rule)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "val_accuracy": val_metrics["accuracy"],
                   "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                   "val_macro_f1": val_metrics["macro_f1"],
                   "val_mel_precision": val_metrics["per_class"]["mel"]["precision"],
                   "val_mel_recall": val_metrics["per_class"]["mel"]["recall"],
                   "val_mel_f1": val_metrics["per_class"]["mel"]["f1"],
                   "val_nv_recall": val_metrics["per_class"]["nv"]["recall"],
                   "eligible": is_eligible, "learning_rate": optimizer.param_groups[0]["lr"],
                   "optimizer_steps": steps,
                   "amp_skips": sum(event["epoch"] == epoch for event in events)}
            history.append(row)
            improved = is_eligible and val_loss < best_loss
            if improved:
                best_epoch, best_loss = epoch, val_loss
            state = {"experiment": NAME, "epoch": epoch,
                     "configuration": cfg, "config_sha256": CONFIG_SHA,
                     "selection_rule_sha256": RULE_SHA,
                     "numerical_protocol_sha256": NUMERICAL_SHA,
                     "runner_sha256": base.sha256(Path(__file__)),
                     "class_order": list(base.CLASSES), "architecture": cfg["architecture"],
                     "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                     "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                     "history": history, "best_epoch": best_epoch,
                     "best_validation_loss": best_loss,
                     "numerical_events": copy.deepcopy(events),
                     "sampler_generator_state": generator.get_state(),
                     "python_rng_state": random.getstate(),
                     "numpy_rng_state": np.random.get_state(),
                     "torch_rng_state": torch.get_rng_state(),
                     "cuda_rng_states": torch.cuda.get_rng_state_all()}
            if improved:
                guard.checked_checkpoint(OUT / "best_checkpoint.pt", state, model, optimizer, scaler, NAME, epoch)
            guard.checked_checkpoint(OUT / "last_checkpoint.pt", state, model, optimizer, scaler, NAME, epoch)
            base.write_csv(OUT / "training_history.csv", history)
            (OUT / "numerical_events.json").write_text(
                json.dumps(events, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Stage14 epoch={epoch}/25 train_loss={train_loss:.6f} "
                  f"val_loss={val_loss:.6f} eligible={is_eligible} scale={scaler.get_scale():.0f}",
                  flush=True)
        if best_epoch is None:
            status = "NO_CANDIDATE_SELECTED"
        else:
            selected = torch.load(OUT / "best_checkpoint.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(selected["model_state"])
            selected_loss, selected_metrics, predictions = guard.guarded_validation_epoch(
                model, val_loader, NAME, best_epoch, predictions=True)
            if not math.isclose(selected_loss, best_loss, rel_tol=1e-3, abs_tol=1e-4):
                raise ValueError("Selected validation loss drifted")
            if not eligible(selected_metrics, rule):
                raise ValueError("Selected checkpoint fails registered eligibility on replay")
            base.write_csv(OUT / "validation_predictions.csv", predictions)
            (OUT / "validation_metrics.json").write_text(json.dumps({
                "epoch": best_epoch, "val_loss": selected_loss, **selected_metrics},
                indent=2, allow_nan=False) + "\n", encoding="utf-8")
            status = "CANDIDATE_SELECTED_VALIDATION_ONLY"
        artifacts = ["config.json", "selection_rule.json", "numerical_protocol.json",
                     "training_history.csv", "last_checkpoint.pt"]
        artifacts += (["best_checkpoint.pt", "validation_predictions.csv", "validation_metrics.json"]
                      if best_epoch is not None else [])
        artifacts.append("numerical_events.json")
        (OUT / "experiment_manifest.json").write_text(json.dumps({
            "experiment": NAME, "status": status, "epochs": len(history),
            "selected_epoch": best_epoch,
            "best_validation_loss": best_loss if best_epoch is not None else None,
            "config_sha256": CONFIG_SHA, "selection_rule_sha256": RULE_SHA,
            "numerical_protocol_sha256": NUMERICAL_SHA,
            "split_sha256": base.SPLIT_SHA, "stage9_reference_checkpoint_sha256": base.STAGE9_SHA,
            "numerical_gate_sha256": GATE_SHA, "runner_sha256": base.sha256(Path(__file__)),
            "artifact_sha256": {name: base.sha256(OUT / name) for name in artifacts},
            "runtime": runtime(), "ham_test_accessed": False, "ph2_accessed": False},
            indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception as exc:
        if ready:
            failure = (exc.record if isinstance(exc, guard.NumericalFailure) else
                       {"type": type(exc).__name__, "message": str(exc)})
            (OUT / "failure.json").write_text(json.dumps({
                "experiment": NAME, "status": "STOPPED", "failure": failure,
                "completed_epochs": len(history), "ham_test_accessed": False,
                "ph2_accessed": False}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    finally:
        restore_policy()
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true", help="read-only preregistration/data check")
    modes.add_argument("--train", action="store_true", help="start or continue the one candidate run")
    parser.add_argument("--resume", action="store_true", help="requires existing last_checkpoint.pt")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if root != ROOT.resolve():
        raise ValueError("Use this script from its pinned repository root")
    if args.resume and not args.train:
        parser.error("--resume requires --train")
    if args.resume and not (OUT / "last_checkpoint.pt").is_file():
        raise FileNotFoundError("--resume requires existing last_checkpoint.pt")
    cfg, rule, numerical = check_preflight(root)
    if args.check:
        print(json.dumps({"status": "PREFLIGHT_PASS", "experiment": NAME,
                          "config_sha256": CONFIG_SHA, "selection_rule_sha256": RULE_SHA,
                          "numerical_protocol_sha256": NUMERICAL_SHA,
                          "bounded_B_gate_sha256": GATE_SHA,
                          "full_training_started": False,
                          "ham_test_accessed": False, "ph2_accessed": False}, indent=2))
        return
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Full experiment requires Tesla T4")
    execute(root, cfg, rule, numerical, resume=args.resume)


if __name__ == "__main__":
    main()
