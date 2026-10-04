"""Prepared Stage 13C A/B/C full runner; execute only after explicit review.

The scientific variant configurations are imported unchanged. This separate
numerical protocol runs only nonlocal3's q@k affinity in FP32 under CUDA AMP.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
from pathlib import Path

import torch
import torchvision

import train_ablation as base
import stage13c_guarded as guard

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANALYSIS = ROOT / "analysis/stage13_melanoma_ablation"
GATE_SHA256 = "721eeaf6c64740d7dd8d7cdb06890a2ffa1c1be49c336f4d07033e4d172cdb00"
GATE_PATH = ANALYSIS / "stage13c_abc_selective_qk_fp32_gate.json"
INITIAL_SCALE = 8192.0
OUTPUT_ROOT = HERE / "stage13c_selective_qk_fp32_runs"
PROTOCOL = "Stage13C selective nonlocal3 qk FP32 under normal CUDA AMP"


def runtime() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0)}


def verify_gate_and_config(root: Path) -> dict:
    # Import only the audited bounded-gate helper, which checks the complete
    # registered A/B/C scientific configuration and frozen split.
    import sys
    if str(ANALYSIS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS))
    import stage13c_abc_selective_qk_fp32_gate as gate

    base.preflight(root)
    configs = gate.verify_scientific_factors(root)
    if not GATE_PATH.is_file() or base.sha256(GATE_PATH) != GATE_SHA256:
        raise RuntimeError("Audited 16-batch selective gate absent or changed")
    evidence = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    if (evidence["status"] != "PASS_BOUNDED_WINDOW_ALL_VARIANTS" or
            evidence["runtime"] != runtime() or evidence["seed"] != 42 or
            evidence["initial_scale"] != INITIAL_SCALE or evidence["batches_per_variant"] != 16 or
            evidence["gate_script_sha256"] != base.sha256(ANALYSIS / "stage13c_abc_selective_qk_fp32_gate.py") or
            evidence["split_sha256"] != base.SPLIT_SHA or
            evidence["numerical_policy"] != "nonlocal3 q@k FP32 inside normal CUDA AMP, identical for A/B/C" or
            set(evidence["variants"]) != set("ABC")):
        raise RuntimeError("Selective gate is inconsistent with this full-run protocol")
    for variant in "ABC":
        result = evidence["variants"][variant]
        if (result["status"] not in ("PASS_BOUNDED_WINDOW", "PASS_BOUNDED_WITH_RECOVERED_OVERFLOW") or
                result["batches_completed"] != 16 or
                evidence["config_sha256"][variant] != base.sha256(HERE / f"config_{variant}.json") or
                configs[variant] != base.variant_config(variant)):
            raise RuntimeError(f"Variant {variant} gate/config mismatch")
    for path, digest in evidence["source_sha256"].items():
        if base.sha256(root / path) != digest:
            raise RuntimeError(f"Gate source changed: {path}")
    return configs


def install_policy(model):
    # The exact bounded-gate policy is reused for every training and validation
    # forward; no production architecture file is changed.
    import sys
    if str(ANALYSIS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS))
    from stage13c_abc_selective_qk_fp32_gate import install_selective_qk_policy
    return install_selective_qk_policy(model)


def checked_train_batch(model, optimizer, scaler, images, labels, variant, epoch, batch, multiplier):
    context = guard.context_for(variant, epoch, batch, labels, scaler)
    guard.require_valid_scale(scaler, context)
    guard.require_finite(images, context, "train_input")
    if labels.dtype != torch.long or not bool(((labels >= 0) & (labels < 7)).all().item()):
        raise guard.NumericalFailure(context, "train_labels")
    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda"):
        logits = model(images)
        guard.require_finite(logits, context, "train_logits")
        guard.require_finite_states(model, optimizer, context)
        terms = guard.focal_terms(logits, labels, multiplier)
        guard.require_finite(terms, context, "train_per_example_loss")
        loss = terms.mean()
        guard.require_finite(loss, context, "train_batch_loss")
    scaled_loss = scaler.scale(loss)
    guard.require_finite(scaled_loss, context, "scaled_loss")
    scaled_loss.backward()

    def first_bad_gradient():
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                return name, parameter.grad
        return None, None

    scaled_bad, scaled_tensor = first_bad_gradient()
    scaler.unscale_(optimizer)
    unscaled_bad, unscaled_tensor = first_bad_gradient()
    scale_before = float(context["amp_scale"])
    steps = [0]
    hook = optimizer.register_step_post_hook(lambda *_: steps.__setitem__(0, steps[0] + 1))
    try:
        scaler.step(optimizer)
    finally:
        hook.remove()
    scaler.update()
    scale_after = float(scaler.get_scale())
    guard.require_valid_scale(scaler, context)
    guard.require_finite_states(model, optimizer, context)
    skipped = steps[0] == 0
    if skipped:
        if unscaled_bad is None or not scale_after < scale_before:
            raise guard.NumericalFailure(context, "unexplained_GradScaler_skip")
    elif scaled_bad is not None or unscaled_bad is not None:
        raise guard.NumericalFailure(context, "optimizer_step_with_nonfinite_gradient",
                                     unscaled_tensor if unscaled_tensor is not None else scaled_tensor)
    event = None
    if skipped:
        event = {"variant": variant, "epoch": epoch, "batch_index": batch,
                 "scale_before": scale_before, "scale_after": scale_after,
                 "step_skipped": True, "scaled_first_nonfinite_gradient": scaled_bad,
                 "unscaled_first_nonfinite_gradient": unscaled_bad,
                 "target_class_distribution": context["target_class_distribution"],
                 "contains_mel": context["contains_mel"]}
    return float(loss.detach().float().cpu()), not skipped, event


def validate_saved_scaler(state: dict) -> float:
    if not isinstance(state, dict):
        raise ValueError("Checkpoint GradScaler state is missing")
    expected = {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}
    if set(state) != expected:
        raise ValueError("Checkpoint GradScaler state has missing or unexpected keys")
    scale = float(state["scale"])
    growth = float(state["growth_factor"])
    backoff = float(state["backoff_factor"])
    interval = state["growth_interval"]
    tracker = state["_growth_tracker"]
    if (not math.isfinite(scale) or scale <= 0 or
            not math.isfinite(growth) or growth <= 1 or
            not math.isfinite(backoff) or not 0 < backoff < 1 or
            type(interval) is not int or interval <= 0 or
            type(tracker) is not int or not 0 <= tracker < interval):
        raise ValueError("Checkpoint GradScaler state is invalid")
    return scale


def _all_floating_tensors_finite(named_values) -> bool:
    return all(not isinstance(value, torch.Tensor) or not value.is_floating_point() or
               bool(torch.isfinite(value).all().item()) for _, value in named_values)


def validate_resume_checkpoint(state: dict, variant: str, cfg: dict) -> dict:
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is not a mapping")
    required = {"variant", "epoch", "best_epoch", "best_validation_loss", "configuration",
                "numerical_protocol", "class_order", "architecture", "model_state",
                "optimizer_state", "scheduler_state", "scaler_state", "history",
                "sampler_generator_state", "torch_rng_state", "cuda_rng_states"}
    if not required.issubset(state):
        raise ValueError(f"Checkpoint keys missing: {sorted(required - set(state))}")
    if state["variant"] != variant:
        raise ValueError("Checkpoint variant mismatch")
    if state["configuration"] != cfg:
        raise ValueError("Checkpoint configuration mismatch")
    if state["numerical_protocol"] != PROTOCOL:
        raise ValueError("Checkpoint selective qk FP32 protocol mismatch")
    if state["class_order"] != list(base.CLASSES):
        raise ValueError("Checkpoint class order mismatch")
    if state["architecture"] != cfg["architecture"]:
        raise ValueError("Checkpoint architecture mismatch")
    epoch = state["epoch"]
    history = state["history"]
    if type(epoch) is not int or not 1 <= epoch < 25 or not isinstance(history, list) or len(history) != epoch:
        raise ValueError("Checkpoint epoch/history length mismatch or no epochs remain")
    if [row.get("epoch") for row in history] != list(range(1, epoch + 1)):
        raise ValueError("Checkpoint history epoch numbering mismatch")
    for row in history:
        if (not math.isfinite(float(row["train_loss"])) or
                not math.isfinite(float(row["val_loss"])) or
                not math.isfinite(float(row["learning_rate"]))):
            raise ValueError("Checkpoint history contains non-finite losses or LR")
        expected_eligible = base.eligible({
            "macro_f1": row["val_macro_f1"],
            "per_class": {"mel": {"f1": row["val_mel_f1"]},
                          "nv": {"recall": row["val_nv_recall"]}}})
        if row["eligible"] is not expected_eligible:
            raise ValueError("Checkpoint eligibility history changed")
    eligible_rows = [row for row in history if row["eligible"]]
    expected_best = min(eligible_rows, key=lambda row: row["val_loss"]) if eligible_rows else None
    best_epoch = state["best_epoch"]
    best_loss = float(state["best_validation_loss"])
    if expected_best is None:
        if best_epoch is not None or best_loss != math.inf:
            raise ValueError("Checkpoint best selection inconsistent with history")
    elif best_epoch != expected_best["epoch"] or best_loss != expected_best["val_loss"]:
        raise ValueError("Checkpoint best selection inconsistent with eligibility/loss rule")
    model_state = state["model_state"]
    if not isinstance(model_state, dict) or not model_state or not _all_floating_tensors_finite(model_state.items()):
        raise ValueError("Checkpoint model state contains NaN/Inf or is missing")
    optimizer_state = state["optimizer_state"]
    if not isinstance(optimizer_state, dict) or "state" not in optimizer_state or "param_groups" not in optimizer_state:
        raise ValueError("Checkpoint optimizer state is missing")
    if not _all_floating_tensors_finite(
            (name, value) for slots in optimizer_state["state"].values()
            for name, value in slots.items()):
        raise ValueError("Checkpoint optimizer state contains NaN/Inf")
    groups = optimizer_state["param_groups"]
    if len(groups) != 1 or not groups[0]["params"]:
        raise ValueError("Checkpoint Adamax parameter groups are invalid")
    group = groups[0]
    if (not math.isfinite(float(group["lr"])) or group["lr"] <= 0 or
            group["betas"] != (0.9, 0.999) or group["eps"] != 1e-8 or
            group["weight_decay"] != 0.0001):
        raise ValueError("Checkpoint Adamax settings differ from registered settings")
    scheduler = state["scheduler_state"]
    if (not isinstance(scheduler, dict) or scheduler.get("last_epoch") != epoch or
            not math.isfinite(float(scheduler["best"])) or
            not scheduler.get("_last_lr") or
            not math.isfinite(float(scheduler["_last_lr"][0])) or
            not math.isclose(float(scheduler["_last_lr"][0]), float(group["lr"]), rel_tol=0, abs_tol=0)):
        raise ValueError("Checkpoint scheduler state is invalid")
    scale = validate_saved_scaler(state["scaler_state"])
    for key in ("sampler_generator_state", "torch_rng_state"):
        rng = state[key]
        if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1 or rng.numel() == 0:
            raise ValueError(f"Checkpoint {key} is invalid")
    cuda_rng = state["cuda_rng_states"]
    if not isinstance(cuda_rng, (list, tuple)) or not cuda_rng or any(
            not isinstance(x, torch.Tensor) or x.dtype != torch.uint8 or x.ndim != 1 or x.numel() == 0
            for x in cuda_rng):
        raise ValueError("Checkpoint CUDA RNG states are invalid")
    return {"saved_epoch": epoch, "start_epoch": epoch + 1, "best_epoch": best_epoch,
            "best_validation_loss": best_loss, "history_rows": len(history),
            "saved_amp_scale": scale}


def validate_existing_run_files(out: Path, state: dict, cfg: dict) -> None:
    if json.loads((out / "config.json").read_text(encoding="utf-8")) != cfg:
        raise ValueError("Existing run config.json differs from registered configuration")
    protocol = json.loads((out / "numerical_protocol.json").read_text(encoding="utf-8"))
    if (protocol.get("name") != PROTOCOL or protocol.get("scientific_variant") != state["variant"] or
            protocol.get("initial_scale") != INITIAL_SCALE or protocol.get("gate_sha256") != GATE_SHA256):
        raise ValueError("Existing run numerical protocol differs")
    with (out / "training_history.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if (len(rows) != len(state["history"]) or
                reader.fieldnames != list(state["history"][0]) or
                any(any(csv_row[key] != str(history_row[key]) for key in reader.fieldnames)
                    for csv_row, history_row in zip(rows, state["history"]))):
            raise ValueError("Existing training_history.csv differs from checkpoint history")
    best_path = out / "best_checkpoint.pt"
    if state["best_epoch"] is None:
        if best_path.exists():
            raise ValueError("Unexpected best checkpoint for run with no eligible epoch")
    else:
        if not best_path.is_file():
            raise FileNotFoundError("Selected best checkpoint is missing; refusing resume")
        best = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
        if (best.get("variant") != state["variant"] or best.get("epoch") != state["best_epoch"] or
                best.get("configuration") != cfg or best.get("numerical_protocol") != PROTOCOL or
                best.get("best_validation_loss") != state["best_validation_loss"]):
            raise ValueError("Existing best checkpoint does not match saved selection")
        if not _all_floating_tensors_finite(best["model_state"].items()):
            raise ValueError("Existing best checkpoint model state is non-finite")


def restore_resume_state(state: dict, model, optimizer, scheduler, scaler, generator,
                         *, cuda_rng_setter=None) -> tuple[int, list, int | None, float]:
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["scaler_state"])
    if float(scaler.get_scale()) != float(state["scaler_state"]["scale"]):
        raise ValueError("GradScaler scale did not restore exactly")
    generator.set_state(state["sampler_generator_state"])
    torch.set_rng_state(state["torch_rng_state"])
    setter = torch.cuda.set_rng_state_all if cuda_rng_setter is None else cuda_rng_setter
    setter(state["cuda_rng_states"])
    return (state["epoch"] + 1, list(state["history"]), state["best_epoch"],
            float(state["best_validation_loss"]))


def train_variant(root: Path, variant: str, cfg: dict, *, resume: bool = False) -> None:
    out = OUTPUT_ROOT / variant
    saved = None
    if resume:
        checkpoint_path = out / "last_checkpoint.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume requires an existing checkpoint: {checkpoint_path}")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(saved, variant, cfg)
        validate_existing_run_files(out, saved, cfg)
        if len(saved["cuda_rng_states"]) != torch.cuda.device_count():
            raise ValueError("Saved CUDA RNG state count differs from current runtime")
    else:
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite any existing run artifact: {out}")
        out.mkdir(parents=True)
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        (out / "numerical_protocol.json").write_text(json.dumps({
            "name": PROTOCOL, "initial_scale": INITIAL_SCALE,
            "dynamic_GradScaler_updates": True, "recoverable_overflow_skip": True,
            "gradient_clipping": None, "scientific_variant": variant,
            "gate_sha256": GATE_SHA256, "runner_sha256": base.sha256(Path(__file__))}, indent=2) + "\n",
            encoding="utf-8")
    model, optimizer, scheduler, old_scaler, train_loader, val_loader, generator = guard.build_components(root, variant)
    del old_scaler
    scaler = torch.amp.GradScaler("cuda", init_scale=INITIAL_SCALE)
    restore_policy = install_policy(model)
    history = []
    best_loss = math.inf
    best_epoch = None
    start_epoch = 1
    numerical_events = []
    run_ready = not resume
    try:
        if resume:
            event_path = out / "numerical_events.json"
            if event_path.exists():
                numerical_events = json.loads(event_path.read_text(encoding="utf-8"))
                if not isinstance(numerical_events, list) or any(
                        not isinstance(e, dict) or e.get("epoch", math.inf) > saved["epoch"]
                        for e in numerical_events):
                    raise ValueError("Existing numerical event log extends beyond checkpoint epoch")
            start_epoch, history, best_epoch, best_loss = restore_resume_state(
                saved, model, optimizer, scheduler, scaler, generator)
            guard.require_finite_states(model, optimizer, {"variant": variant, "epoch": saved["epoch"]})
            guard.require_valid_scale(scaler, {"variant": variant, "epoch": saved["epoch"]})
            run_ready = True
            print(f"Stage13C {variant} resuming saved_epoch={saved['epoch']} "
                  f"next_epoch={start_epoch} scale={scaler.get_scale():.0f}", flush=True)
            del saved
            gc.collect()
        for epoch in range(start_epoch, 26):
            train_sum = train_count = optimizer_steps = 0
            for batch, (images_cpu, labels_cpu) in enumerate(train_loader, start=1):
                images = images_cpu.cuda(non_blocking=True)
                labels = labels_cpu.cuda(non_blocking=True)
                value, stepped, event = checked_train_batch(
                    model, optimizer, scaler, images, labels, variant, epoch, batch,
                    cfg["loss"]["mel_multiplier"])
                train_sum += value * len(labels_cpu)
                train_count += len(labels_cpu)
                optimizer_steps += int(stepped)
                if event is not None:
                    numerical_events.append(event)
                    (out / "numerical_events.json").write_text(
                        json.dumps(numerical_events, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if optimizer_steps == 0:
                raise RuntimeError(f"No optimizer update in epoch {epoch}; numerical clearance lost")
            train_loss = train_sum / train_count
            val_loss, val_metrics, _ = guard.guarded_validation_epoch(model, val_loader, variant, epoch)
            guard.require_finite_states(model, optimizer, {"variant": variant, "epoch": epoch})
            guard.require_finite_epoch_losses(train_loss, val_loss, variant, epoch)
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
                   "eligible": eligible, "learning_rate": optimizer.param_groups[0]["lr"],
                   "optimizer_steps": optimizer_steps,
                   "amp_skips": sum(e["epoch"] == epoch for e in numerical_events)}
            history.append(row)
            improved = eligible and val_loss < best_loss
            if improved:
                best_loss, best_epoch = val_loss, epoch
            state = {"variant": variant, "epoch": epoch, "best_epoch": best_epoch,
                     "best_validation_loss": best_loss, "configuration": cfg,
                     "numerical_protocol": PROTOCOL, "class_order": list(base.CLASSES),
                     "architecture": cfg["architecture"], "model_state": model.state_dict(),
                     "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                     "scaler_state": scaler.state_dict(), "history": history,
                     "sampler_generator_state": generator.get_state(),
                     "torch_rng_state": torch.get_rng_state(),
                     "cuda_rng_states": torch.cuda.get_rng_state_all()}
            if improved:
                guard.checked_checkpoint(out / "best_checkpoint.pt", state, model, optimizer, scaler, variant, epoch)
            guard.checked_checkpoint(out / "last_checkpoint.pt", state, model, optimizer, scaler, variant, epoch)
            base.write_csv(out / "training_history.csv", history)
            print(f"Stage13C {variant} epoch={epoch}/25 train_loss={train_loss:.6f} "
                  f"val_loss={val_loss:.6f} eligible={eligible} scale={scaler.get_scale():.0f}", flush=True)
        if best_epoch is None:
            status = "FAIL_NO_ELIGIBLE_CHECKPOINT"
        else:
            state = torch.load(out / "best_checkpoint.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state"])
            selected_loss, selected_metrics, predictions = guard.guarded_validation_epoch(
                model, val_loader, variant, best_epoch, predictions=True)
            if not math.isclose(selected_loss, best_loss, rel_tol=1e-3, abs_tol=1e-4):
                raise ValueError("Selected validation loss drifted")
            base.write_csv(out / "validation_predictions.csv", predictions)
            (out / "validation_metrics.json").write_text(json.dumps({"epoch": best_epoch,
                "val_loss": selected_loss, **selected_metrics}, indent=2) + "\n", encoding="utf-8")
            status = "COMPLETE_VALIDATION_ONLY"
        (out / "experiment_manifest.json").write_text(json.dumps({
            "status": status, "variant": variant, "epochs": len(history),
            "selected_epoch": best_epoch,
            "best_validation_loss": best_loss if best_epoch is not None else None,
            "best_checkpoint_sha256": base.sha256(out / "best_checkpoint.pt") if best_epoch else None,
            "last_checkpoint_sha256": base.sha256(out / "last_checkpoint.pt"),
            "split_sha256": base.SPLIT_SHA, "stage9_control_sha256": base.STAGE9_SHA,
            "numerical_protocol": PROTOCOL, "numerical_events": len(numerical_events),
            "class_order": list(base.CLASSES), "ham_test_accessed": False,
            "ph2_accessed": False}, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        if run_ready:
            failure = (exc.record if isinstance(exc, guard.NumericalFailure) else
                       {"type": type(exc).__name__, "message": str(exc)})
            (out / "failure.json").write_text(json.dumps({
                "status": "STOPPED", "variant": variant, "failure": failure,
                "epochs_in_history": len(history), "gate_sha256": GATE_SHA256,
                "ham_test_accessed": False, "ph2_accessed": False},
                indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    finally:
        restore_policy()
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--variant", choices=list("ABC"), required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Continue only from an existing validated last_checkpoint.pt")
    parser.add_argument("--train", action="store_true", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This prepared training protocol requires Tesla T4")
    root = args.project_root.resolve()
    if root != ROOT.resolve():
        raise ValueError("Runner must execute from the pinned repository root")
    if args.resume and not (OUTPUT_ROOT / args.variant / "last_checkpoint.pt").is_file():
        raise FileNotFoundError("--resume requires existing last_checkpoint.pt for requested variant")
    cfg = verify_gate_and_config(root)[args.variant]
    train_variant(root, args.variant, cfg, resume=args.resume)


if __name__ == "__main__":
    main()
