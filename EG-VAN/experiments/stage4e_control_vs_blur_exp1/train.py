"""Stage 4E fresh paired EG-VAN control/treatment runner; never loads Stage 23 weights."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import platform
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

import common


ROOT = common.ROOT
HERE = common.HERE
RUNS = HERE / "runs"
for entry in (str(ROOT / "experiments/egvan_melanoma_ablation_exp"),
              str(ROOT / "experiments/Stage13_melanoma_ablation"),
              str(ROOT / "analysis/stage13_melanoma_ablation")):
    if entry not in sys.path:
        sys.path.insert(0, entry)
import train_ablation as base  # noqa: E402
import stage13c_guarded as guard  # noqa: E402
from stage13c_selective_qk_fp32_train import checked_train_batch, validate_saved_scaler  # noqa: E402
from stage13c_abc_selective_qk_fp32_gate import install_selective_qk_policy  # noqa: E402


def runtime():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def finite_state(state):
    for name, value in state["model_state"].items():
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite saved model parameter/buffer: {name}")
    for param_id, slot in state["optimizer_state"]["state"].items():
        for name, value in slot.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite optimizer state: {param_id}/{name}")


def history_csv(history):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(history)
    return stream.getvalue().encode("utf-8")


def predictions_csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_once(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if common.sha256(path) != hashlib.sha256(data).hexdigest():
            raise ValueError(f"Existing output differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def init_receipt_path():
    return RUNS / "shared_initialization.json"


def read_receipt():
    path = init_receipt_path()
    if not path.is_file():
        raise FileNotFoundError("Shared initial-state receipt required; run --check-initialization first")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "MATCHED_TWO_FRESH_MODELS" or
            receipt.get("stage4d_manifest_sha256") != common.V2_MANIFEST_SHA or
            receipt.get("config_sha256") != common.CONFIG_SHA or
            receipt.get("source_hashes_sha256") != common.SOURCE_HASHES_SHA or
            receipt.get("common_sha256") != common.sha256(HERE / "common.py") or
            receipt.get("runner_sha256") != common.sha256(Path(__file__)) or
            receipt.get("runtime") != runtime() or
            not isinstance(receipt.get("initial_state_sha256"), str)):
        raise ValueError("Shared initialization receipt/source/runtime mismatch")
    return receipt


def check_initialization(cfg):
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Initialization check is registered for Tesla T4")
    if RUNS.exists():
        raise FileExistsError("Initialization check refuses an existing run directory")
    fingerprints = []
    reports = []
    for _ in ("control", "treatment"):
        model, report = common.fresh_model()
        fingerprints.append(common.model_fingerprint(model))
        reports.append(report)
        del model
        gc.collect()
    if fingerprints[0] != fingerprints[1]:
        raise ValueError("Fresh control/treatment initial states differ")
    record = {"status": "MATCHED_TWO_FRESH_MODELS", "initial_state_sha256": fingerprints[0],
              "initialization_reports": reports, "seed": cfg["seed"],
              "stage4d_manifest_sha256": common.V2_MANIFEST_SHA,
              "config_sha256": common.CONFIG_SHA,
              "source_hashes_sha256": common.SOURCE_HASHES_SHA,
              "common_sha256": common.sha256(HERE / "common.py"),
              "runner_sha256": common.sha256(Path(__file__)), "runtime": runtime(),
              "training_performed": False, "selection_inference_performed": False}
    RUNS.mkdir(parents=True, exist_ok=False)
    write_once(init_receipt_path(), (json.dumps(record, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return record


def epoch_folder(arm_dir, epoch):
    return arm_dir / "epochs" / f"epoch_{epoch:03d}"


def verify_epoch(folder: Path, cfg: dict, arm: str):
    commit = json.loads((folder / "commit.json").read_text(encoding="utf-8"))
    if (commit.get("arm") != arm or commit.get("epoch") != int(folder.name[-3:]) or
            commit.get("config_sha256") != common.CONFIG_SHA or
            commit.get("stage4d_manifest_sha256") != common.V2_MANIFEST_SHA):
        raise ValueError(f"Epoch commit provenance mismatch: {folder}")
    for name, digest in commit["artifact_sha256"].items():
        if common.sha256(folder / name) != digest:
            raise ValueError(f"Epoch artifact hash mismatch: {folder / name}")
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    history = common.read_csv(folder / "history.csv")
    rows = common.read_csv(folder / "predictions.csv")
    expected = common.read_csv(common.V2 / common.SELECT_NAME)
    rebuilt = common.validate_predictions(rows, expected)
    if (len(history) != commit["epoch"] or
            [int(r["epoch"]) for r in history] != list(range(1, commit["epoch"] + 1)) or
            metrics["epoch"] != commit["epoch"] or
            not math.isclose(metrics["validation_loss"], float(history[-1]["val_loss"]), rel_tol=0, abs_tol=0)):
        raise ValueError("Epoch history/metrics mismatch")
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "mel_specificity", "confusion_matrix", "per_class"):
        if rebuilt[key] != metrics[key]:
            raise ValueError(f"Epoch saved predictions/metric mismatch: {key}")
    if not math.isfinite(float(history[-1]["train_loss"])) or not math.isfinite(float(history[-1]["val_loss"])):
        raise ValueError("Non-finite epoch loss")
    return commit


def check_resume_state(state, cfg, arm, receipt, latest_epoch):
    required = {"arm", "epoch", "configuration", "config_sha256", "stage4d_manifest_sha256",
                "runner_sha256", "common_sha256", "initial_state_sha256", "model_state",
                "optimizer_state", "scheduler_state", "scaler_state", "history", "best_epoch",
                "best_validation_loss", "sampler_generator_state", "python_rng_state",
                "numpy_rng_state", "torch_rng_state", "cuda_rng_states", "numerical_events"}
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError("Incomplete resume checkpoint")
    if (state["arm"] != arm or state["epoch"] != latest_epoch or
            state["configuration"] != common.arm_config(cfg, arm) or
            state["config_sha256"] != common.CONFIG_SHA or
            state["stage4d_manifest_sha256"] != common.V2_MANIFEST_SHA or
            state["runner_sha256"] != common.sha256(Path(__file__)) or
            state["common_sha256"] != common.sha256(HERE / "common.py") or
            state["initial_state_sha256"] != receipt["initial_state_sha256"] or
            len(state["history"]) != latest_epoch or
            [r["epoch"] for r in state["history"]] != list(range(1, latest_epoch + 1)) or
            not 1 <= latest_epoch <= 25):
        raise ValueError("Resume checkpoint provenance/history mismatch")
    winner = min(state["history"], key=lambda r: (r["val_loss"], r["epoch"]))
    if (state["best_epoch"] != winner["epoch"] or
            state["best_validation_loss"] != winner["val_loss"] or
            any(not math.isfinite(float(r["train_loss"])) or not math.isfinite(float(r["val_loss"]))
                for r in state["history"])):
        raise ValueError("Resume checkpoint selection/loss mismatch")
    finite_state(state)
    validate_saved_scaler(state["scaler_state"])
    optimizer = state["optimizer_state"]
    scheduler = state["scheduler_state"]
    if (len(optimizer["param_groups"]) != 1 or
            optimizer["param_groups"][0]["betas"] != (0.9, 0.999) or
            optimizer["param_groups"][0]["weight_decay"] != cfg["optimizer"]["weight_decay"] or
            optimizer["param_groups"][0]["eps"] != cfg["optimizer"]["eps"] or
            scheduler["last_epoch"] != latest_epoch or
            not math.isfinite(float(scheduler["best"])) or
            scheduler["_last_lr"][0] != optimizer["param_groups"][0]["lr"]):
        raise ValueError("Resume optimizer/scheduler invalid")
    for key in ("torch_rng_state", "sampler_generator_state"):
        if not isinstance(state[key], torch.Tensor) or state[key].dtype != torch.uint8 or not state[key].numel():
            raise ValueError(f"Resume RNG missing: {key}")
    if (not isinstance(state["python_rng_state"], tuple) or
            not isinstance(state["numpy_rng_state"], tuple) or
            not isinstance(state["cuda_rng_states"], list) or
            len(state["cuda_rng_states"]) != torch.cuda.device_count()):
        raise ValueError("Resume Python/NumPy/CUDA RNG invalid")


def load_resume(arm_dir, cfg, arm, receipt):
    folders = sorted((arm_dir / "epochs").glob("epoch_[0-9][0-9][0-9]"))
    if not folders:
        raise FileNotFoundError("--resume requires a completed immutable epoch checkpoint")
    numbers = [int(path.name[-3:]) for path in folders]
    if numbers != list(range(1, max(numbers) + 1)):
        raise ValueError("Completed epoch directories have gaps")
    for folder in folders:
        verify_epoch(folder, cfg, arm)
    latest = folders[-1]
    state = torch.load(latest / "checkpoint.pt", map_location="cpu", weights_only=False)
    check_resume_state(state, cfg, arm, receipt, numbers[-1])
    history_csv_saved = common.read_csv(latest / "history.csv")
    if any(str(a[key]) != b[key] for a, b in zip(state["history"], history_csv_saved)
           for key in ("epoch", "train_loss", "val_loss")):
        raise ValueError("Saved checkpoint and history snapshot differ")
    return state


def save_epoch(arm_dir, state, predictions, metrics):
    epoch = state["epoch"]
    final = epoch_folder(arm_dir, epoch)
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite completed epoch: {final}")
    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".pending_epoch_{epoch:03d}_", dir=parent))
    torch.save(state, temp / "checkpoint.pt")
    (temp / "predictions.csv").write_bytes(predictions_csv(predictions))
    (temp / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (temp / "history.csv").write_bytes(history_csv(state["history"]))
    (temp / "numerical_events.json").write_text(json.dumps(state["numerical_events"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    artifacts = {name: common.sha256(temp / name) for name in
                 ("checkpoint.pt", "predictions.csv", "metrics.json", "history.csv", "numerical_events.json")}
    commit = {"arm": state["arm"], "epoch": epoch,
              "config_sha256": common.CONFIG_SHA, "stage4d_manifest_sha256": common.V2_MANIFEST_SHA,
              "artifact_sha256": artifacts}
    (temp / "commit.json").write_text(json.dumps(commit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.rename(temp, final)  # Committed directory appears only after every file exists.
    verify_epoch(final, state["configuration"], state["arm"])
    return commit


def restore_training_state(state, model, optimizer, scheduler, scaler, generator, *, cuda_rng_setter=None):
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["scaler_state"])
    if scaler.get_scale() != state["scaler_state"]["scale"]:
        raise ValueError("GradScaler scale reset on resume")
    generator.set_state(state["sampler_generator_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    setter = torch.cuda.set_rng_state_all if cuda_rng_setter is None else cuda_rng_setter
    setter(state["cuda_rng_states"])
    return state["epoch"] + 1, copy.deepcopy(state["history"]), copy.deepcopy(state["numerical_events"])


def run_arm(cfg, arm, resume):
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Registered training device is Tesla T4; no CPU training")
    receipt = read_receipt()
    arm_dir = RUNS / arm
    if resume:
        if not arm_dir.is_dir() or (arm_dir / "run_manifest.json").exists():
            raise FileNotFoundError("--resume requires an unfinished existing arm run")
        saved = load_resume(arm_dir, cfg, arm, receipt)
    else:
        if arm_dir.exists():
            raise FileExistsError("Fresh training refuses existing arm output")
        saved = None
    model, init_report = common.fresh_model()
    if init_report["initial_state_sha256"] != receipt["initial_state_sha256"]:
        raise ValueError("Fresh arm does not match shared initial model")
    if not resume:
        arm_dir.mkdir(parents=True, exist_ok=False)
    model = model.cuda()
    restore_policy = install_selective_qk_policy(model)
    train_transform, selection_transform = common.transforms_for_arm(arm)
    train_rows = common.read_csv(common.V2 / common.TRAIN_NAME)
    selection_rows = common.read_csv(common.V2 / common.SELECT_NAME)
    train_ds = common.ManifestDataset(ROOT, train_rows, train_transform)
    selection_ds = common.ManifestDataset(ROOT, selection_rows, selection_transform)
    generator = torch.Generator(device="cpu").manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=16, sampler=common.sampler_for(train_rows, generator),
                              num_workers=cfg["num_workers"], pin_memory=True)
    selection_loader = DataLoader(selection_ds, batch_size=16, shuffle=False,
                                  num_workers=cfg["num_workers"], pin_memory=True)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                   eps=1e-8, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler("cuda", init_scale=8192.0)
    history, events, start = [], [], 1
    try:
        if saved is not None:
            start, history, events = restore_training_state(
                saved, model, optimizer, scheduler, scaler, generator)
            guard.require_finite_states(model, optimizer, {"arm": arm, "epoch": saved["epoch"]})
            print(f"Stage4E {arm} resume saved_epoch={saved['epoch']} next_epoch={start} scale={scaler.get_scale():.0f}", flush=True)
            del saved
            gc.collect()
        for epoch in range(start, cfg["epochs"] + 1):
            train_sum = count = steps = 0
            for batch, (images_cpu, labels_cpu) in enumerate(train_loader, 1):
                images = images_cpu.cuda(non_blocking=True)
                labels = labels_cpu.cuda(non_blocking=True)
                loss, stepped, event = checked_train_batch(model, optimizer, scaler, images, labels,
                                                            f"stage4e_{arm}", epoch, batch, 1.2815247721366766)
                train_sum += loss * len(labels_cpu)
                count += len(labels_cpu)
                steps += int(stepped)
                if event is not None:
                    events.append(event)
            if count != cfg["sampler"]["num_samples"] or steps == 0:
                raise ValueError("Training sample/optimizer-step count invalid")
            train_loss = train_sum / count
            val_loss, base_metrics, predictions = guard.guarded_validation_epoch(
                model, selection_loader, f"stage4e_{arm}", epoch, predictions=True)
            guard.require_finite_states(model, optimizer, {"arm": arm, "epoch": epoch})
            guard.require_finite_epoch_losses(train_loss, val_loss, f"stage4e_{arm}", epoch)
            rebuilt = common.validate_predictions(predictions, selection_rows)
            if (rebuilt["accuracy"] != base_metrics["accuracy"] or
                    rebuilt["balanced_accuracy"] != base_metrics["balanced_accuracy"] or
                    rebuilt["macro_f1"] != base_metrics["macro_f1"] or
                    rebuilt["confusion_matrix"] != base_metrics["confusion_matrix"]):
                raise ValueError("Selection metrics disagree with predictions")
            scheduler.step(val_loss)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "accuracy": rebuilt["accuracy"], "balanced_accuracy": rebuilt["balanced_accuracy"],
                   "macro_f1": rebuilt["macro_f1"], "mel_recall": rebuilt["per_class"]["mel"]["recall"],
                   "mel_tp": rebuilt["per_class"]["mel"]["tp"],
                   "mel_fn": rebuilt["per_class"]["mel"]["fn"],
                   "mel_specificity": rebuilt["mel_specificity"],
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "optimizer_steps": steps, "amp_skips": sum(e["epoch"] == epoch for e in events)}
            history.append(row)
            best = min(history, key=lambda r: (r["val_loss"], r["epoch"]))
            evidence = {**rebuilt, "arm": arm, "epoch": epoch, "validation_loss": val_loss,
                        "class_order": list(common.CLASSES), "mel_tp": rebuilt["per_class"]["mel"]["tp"],
                        "mel_fn": rebuilt["per_class"]["mel"]["fn"]}
            state = {"arm": arm, "epoch": epoch, "configuration": common.arm_config(cfg, arm),
                     "config_sha256": common.CONFIG_SHA, "stage4d_manifest_sha256": common.V2_MANIFEST_SHA,
                     "runner_sha256": common.sha256(Path(__file__)),
                     "common_sha256": common.sha256(HERE / "common.py"),
                     "initial_state_sha256": receipt["initial_state_sha256"],
                     "class_order": list(common.CLASSES),
                     "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                     "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                     "history": copy.deepcopy(history), "best_epoch": best["epoch"],
                     "best_validation_loss": best["val_loss"], "numerical_events": copy.deepcopy(events),
                     "sampler_generator_state": generator.get_state(),
                     "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
                     "torch_rng_state": torch.get_rng_state(),
                     "cuda_rng_states": torch.cuda.get_rng_state_all()}
            save_epoch(arm_dir, state, predictions, evidence)
            print(f"Stage4E {arm} epoch={epoch}/25 train_loss={train_loss:.6f} "
                  f"selection_loss={val_loss:.6f} best_epoch={best['epoch']} "
                  f"scale={scaler.get_scale():.0f}", flush=True)
        selected = min(history, key=lambda r: (r["val_loss"], r["epoch"]))
        selected_epoch = selected["epoch"]
        latest = epoch_folder(arm_dir, cfg["epochs"])
        chosen = epoch_folder(arm_dir, selected_epoch)
        write_once(arm_dir / "training_history.csv", (latest / "history.csv").read_bytes())
        manifest = {"status": "COMPLETE_INTERNAL_SELECTION_ONLY", "arm": arm,
                    "epochs": cfg["epochs"], "selected_epoch": selected_epoch,
                    "selected_selection_loss": selected["val_loss"],
                    "selected_checkpoint": f"epochs/epoch_{selected_epoch:03d}/checkpoint.pt",
                    "selected_checkpoint_sha256": common.sha256(chosen / "checkpoint.pt"),
                    "initial_state_sha256": receipt["initial_state_sha256"],
                    "config_sha256": common.CONFIG_SHA,
                    "stage4d_manifest_sha256": common.V2_MANIFEST_SHA,
                    "runner_sha256": common.sha256(Path(__file__)),
                    "common_sha256": common.sha256(HERE / "common.py"),
                    "epoch_commit_sha256": {str(epoch): common.sha256(epoch_folder(arm_dir, epoch) / "commit.json")
                                            for epoch in range(1, cfg["epochs"] + 1)},
                    "runtime": runtime(), "ham_test_accessed": False, "ph2_accessed": False,
                    "frozen_stage23_evaluated": False}
        write_once(arm_dir / "run_manifest.json", (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        print(json.dumps({"arm": arm, "status": manifest["status"], "selected_epoch": selected_epoch}, indent=2))
    finally:
        restore_policy()
        del model, optimizer, scheduler, scaler, train_loader, selection_loader
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="CPU-safe, read-only split/config/image preflight")
    mode.add_argument("--check-initialization", action="store_true", help="T4 model initialization only; no training/inference")
    mode.add_argument("--train", action="store_true", help="requires separate training authorization")
    parser.add_argument("--arm", choices=("control", "treatment"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.train:
        parser.error("--resume requires --train")
    if args.train and not args.arm:
        parser.error("--train requires --arm")
    if not args.train and args.arm:
        parser.error("--arm is only for --train")
    cfg = common.read_config()
    result = common.preflight(args.project_root.resolve(), hash_images=True)
    print(json.dumps(result, indent=2), flush=True)
    if args.check:
        return
    if args.check_initialization:
        print(json.dumps(check_initialization(cfg), indent=2), flush=True)
        return
    run_arm(cfg, args.arm, args.resume)


if __name__ == "__main__":
    main()
