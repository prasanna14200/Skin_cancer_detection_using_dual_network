"""Stage25 conservative-backbone-LR HAM train/validation-only experiment."""
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
from pathlib import Path

import numpy as np
import torch
import torchvision
from initialization import (BACKBONE_PREFIXES, RESNET_LOADER_NAME, RESNET_WEIGHTS,
                            apply_resnet_state, best_checkpoint_for_finalization,
                            build_stage25_components, scheduler_step_preserving_ratio)
from comparison import compare_stage25

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

NAME = "stage25_conservative_backbone_lr_exp1"
CONFIG_SHA = "1400f20794853a47c8262add64476e9d465499bc9dbab31e0b3272b5a90c52ad"
RULE_SHA = "270b527ce8ffb65130a117bda8922c8f4f1ee86d182342149c067a7a541038d8"
NUMERICAL_SHA = "5b5924a65428f76cace43fd8ed2eff6de51f76e635a4e04aa4092d1eef2e7d97"
LR_POLICY_SHA = "ab709c7b02e5aba5cd56601a6f4d9531b28e820866916f1aa9cfefbc2e0c775e"
INITIALIZATION_SPEC_SHA = "947c719b4b62db6fbcf1d9455162d3af78b4ea52a67ff840d38796059eda4522"
STAGE23_CONFIG_SHA = "67a9d77d8464ef51d94f5861004084b179c3c42159e172c680d85952514d7afa"
STAGE23_RULE_SHA = "a03213d5db62af968a425f3557773aa14c85adeabdbaeb8e6d46bf4d5481111d"
STAGE23_NUMERICAL_SHA = "06222c374635f15ad99642e1bdbeaf4c6c69959dd753d6b332cdb753b87abdfb"
STAGE23_VALIDATION_SHA = "beaebb91d9b7043698b5b4580d4af606eac477a985512589e4eb8e99e16af058"
STAGE23_BEST_SHA = "42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab"
GATE_SHA = "721eeaf6c64740d7dd8d7cdb06890a2ffa1c1be49c336f4d07033e4d172cdb00"
SELECTIVE_RUNNER_SHA = "d5cdfbade60e834e56acf9d67fbb3f83d90f0c6f96473dff9b9d3dd4c399ef1a"
STAGE15_CONFIG_SHA = "3ad8c205e06cde8203a869be0a6cc9643ed0ca5b6beb553b57d53d4dce9d25f2"
STAGE15_RULE_SHA = "daf07a1bac3538b5a35ed59121bd5561aba4797a6889e852fd0ca07ad88be735"
STAGE15_NUMERICAL_SHA = "f59010b7e2509c0bf1633b7126ff1d8d629afab39b515f57e95012fb78eb0196"
STAGE15_RUNNER_SHA = "391769f3993e1593064cf9f6460feeb8846e0ca2cc955e3fc469700420d9fdb9"
STAGE15_HISTORY_SHA = "2df7edd11a23658e56dc723ee57638476579cb3620d9c5d10039eefa07941fd0"
STAGE15_VALIDATION_METRICS_SHA = "630bb7a41c41cd5b125576ca0d14beef7d5b523dba7919680e526501f52c0571"
STAGE13_CONFIG_B_SHA = "9f69e862c8d63a2375291343f732c73f5f4bbc4339029b4afced844c52cf7562"
SPLIT_SHA = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
INITIAL_SCALE = 8192.0
FROZEN_SOURCE_HASHES = {
    "experiments/egvan_melanoma_ablation_exp/stage13c_guarded.py": "93e2318afafff8ad1c6907752781d61856cdf3bbde3a614b29686686dac0340d",
    "experiments/egvan_melanoma_ablation_exp/train_ablation.py": "56a71821c1f944028874a76febf00ea68b3b152e481e6450ddd5bd94606e5389",
    "experiments/egvan_melanoma_ablation_exp/config_B.json": STAGE13_CONFIG_B_SHA,
    "experiments/egvan_melanoma_ablation_exp/stage13c_selective_qk_fp32_train.py": SELECTIVE_RUNNER_SHA,
    "analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.py": "198769909e0572b0d5959c849731bb7c7613d578e6c231fd0040580b3519fcff",
    "src/models/egvan/modified_resnet50.py": "5f9a79d4de988f0e2e046bc5b38d90d62dc7c28273c7c922edad6e64f76caeba",
    "src/models/egvan/egvan.py": "f5f5217d6cced6ccc9c1a2d17a5691701bfe454eb46bf52bf97468030cc7b44f",
    "src/models/egvan/efficientnet_branch.py": "e9f25b66247349ff0597ff76a1e4d3dc86a80d8358ad9f63ef1bf4e96cb98b0e",
    "src/models/egvan/scga.py": "6d8a72871c7ae46de12d52f3b0fe973cf75b022434678161b6ab801ffa900d11",
    "src/models/egvan/non_local.py": "d1795018519ae6650a130c44a17aac88fcc58b2ece4336879caf703c24c45c3a",
    "src/models/egvan/mff.py": "ad97108386f6b749bf30d6624d3db88663635393519d0cb6a519e65f96bbcb06",
    "src/models/egvan/__init__.py": "e858a0996e67168ab50bd5524a2ee3609d90db2fec86109ce5ce0273a3e61449",
    "src/models/egvan/spatial_attention.py": "34600fbd2207040de06baff88532d1df684852dcf9086e49789856352cf52ae7",
    "src/preprocessing.py": "cf0a176a610b52ba45b0e91761b027a560252234f81f7b6fd572f02d8baab6ed",
    "src/dataset.py": "02c6d280e468a8c83719bffd7b6bc042b138893d7badd794d0ecfa9291ec8cad",
    "src/train.py": "c195142d2f924b83374c0d6a8d4d93061ecf2e77e947710e6a3a8eb8264686bc",
}
OUT = HERE / "run"
ARCHITECTURE_METADATA = {
    "name": "stage8b_egvan_four_mff_serial_terminal_branches",
    "num_classes": 7,
    "efficientnet_tap_indices": [2, 3, 5, 7],
    "efficientnet_output_channels": [48, 64, 160, 1280],
    "resnet_stage_channels": [256, 512, 1024, 2048],
    "scga_after_resnet_stages": [1, 2],
    "nonlocal_after_resnet_stages": [3, 4],
    "mff_count": 4,
    "fusion_channels": 128,
    "class_order": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
}


def runtime() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def read_preregistration() -> tuple[dict, dict, dict]:
    for filename, digest in (("config.json", CONFIG_SHA), ("selection_rule.json", RULE_SHA),
                             ("numerical_protocol.json", NUMERICAL_SHA),
                             ("lr_policy.json", LR_POLICY_SHA),
                             ("initialization_spec.json", INITIALIZATION_SPEC_SHA)):
        if base.sha256(HERE / filename) != digest:
            raise ValueError(f"Frozen Stage25 preregistration changed: {filename}")
    cfg, rule, numerical = (base.read_json(HERE / filename) for filename in
                            ("config.json", "selection_rule.json", "numerical_protocol.json"))
    reference_dir = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1"
    for filename, digest in (("config.json", STAGE23_CONFIG_SHA),
                             ("selection_rule.json", STAGE23_RULE_SHA),
                             ("numerical_protocol.json", STAGE23_NUMERICAL_SHA)):
        if base.sha256(reference_dir / filename) != digest:
            raise ValueError(f"Frozen Stage23 reference changed: {filename}")
    reference_cfg = base.read_json(reference_dir / "config.json")
    candidate_core, reference_core = copy.deepcopy(cfg), copy.deepcopy(reference_cfg)
    for value in (candidate_core, reference_core):
        value.pop("experiment", None)
        value.pop("description", None)
    candidate_core["optimizer"].pop("parameter_groups", None)
    if candidate_core != reference_core:
        raise ValueError("Stage25 changes a Stage23 scientific factor beyond optimizer LR policy")
    reference_rule = base.read_json(reference_dir / "selection_rule.json")
    for key in ("melanoma_support", "stage9_correct_melanomas", "eligible_if_all",
                "checkpoint_selection", "if_none_eligible"):
        if rule.get(key) != reference_rule.get(key):
            raise ValueError(f"Stage25 selection differs from Stage23: {key}")
    reference_numerical = base.read_json(reference_dir / "numerical_protocol.json")
    if ({key: value for key, value in numerical.items() if key != "name"} !=
            {key: value for key, value in reference_numerical.items() if key != "name"}):
        raise ValueError("Stage25 numerical policy differs from Stage23")
    policy = base.read_json(HERE / "lr_policy.json")
    initialization_spec = base.read_json(HERE / "initialization_spec.json")
    if (cfg["experiment"] != NAME or cfg["epochs"] != 25 or cfg["seed"] != 42 or
            cfg["data"]["split_sha256"] != SPLIT_SHA or tuple(cfg["classes"]) != base.CLASSES or
            rule["experiment"] != NAME or rule["melanoma_support"] != 107 or
            numerical["initial_grad_scaler_scale"] != INITIAL_SCALE or
            numerical["reference_bounded_gate_sha256"] != GATE_SHA or
            RESNET_WEIGHTS.name != "IMAGENET1K_V2" or
            RESNET_WEIGHTS.url != initialization_spec["official_weight_url"] or
            initialization_spec["loader"] != RESNET_LOADER_NAME or
            initialization_spec["stage23_trained_checkpoint_loaded"] is not False or
            policy["all_resnet_trunk_trainable_from_epoch"] != 1 or
            policy["through_epoch"] != 25 or policy["frozen_layers"] or
            policy["forced_batchnorm_eval"] is not False or
            policy["optimizer_groups"] != cfg["optimizer"]["parameter_groups"] or
            [g["initial_learning_rate"] for g in policy["optimizer_groups"]] != [0.001, 0.0001] or
            [g["lr_multiplier"] for g in policy["optimizer_groups"]] != [1.0, 0.1] or
            policy["scheduler"]["factor"] != cfg["scheduler"]["factor"] or
            policy["scheduler"]["patience"] != cfg["scheduler"]["patience"]):
        raise ValueError("Stage25 frozen factor, split, selection, or weight source mismatch")
    return cfg, rule, numerical


def check_preflight(root: Path) -> tuple[dict, dict, dict]:
    if root.resolve() != ROOT.resolve():
        raise ValueError("Use the pinned EG-VAN project root")
    cfg, rule, numerical = read_preregistration()
    stage23 = root / "experiments/stage23_resnet50_imagenet_init_exp1/run"
    if (base.sha256(stage23 / "validation_metrics.json") != STAGE23_VALIDATION_SHA or
            base.sha256(stage23 / "best_checkpoint.pt") != STAGE23_BEST_SHA or
            base.read_json(stage23 / "validation_metrics.json")["epoch"] != 14):
        raise ValueError("Frozen Stage23 selected validation candidate changed")
    for relative, digest in FROZEN_SOURCE_HASHES.items():
        if base.sha256(root / relative) != digest:
            raise ValueError(f"Frozen Stage15 source changed: {relative}")
    reference_dir = ROOT / "experiments/stage15_single_candidate_exp1"
    reference_history = reference_dir / "run/training_history.csv"
    if base.sha256(reference_history) != STAGE15_HISTORY_SHA:
        raise ValueError("Frozen Stage15 training history changed")
    reference_runner = reference_dir / "train.py"
    if base.sha256(reference_runner) != STAGE15_RUNNER_SHA:
        raise ValueError("Frozen Stage15 runner changed")
    reference_metrics = reference_dir / "run/validation_epochs/epoch_016_metrics.json"
    if base.sha256(reference_metrics) != STAGE15_VALIDATION_METRICS_SHA:
        raise ValueError("Frozen Stage15 validation comparison metrics changed")
    if base.sha256(STAGE13 / "config_B.json") != STAGE13_CONFIG_B_SHA:
        raise ValueError("Frozen Stage15 sampler source config changed")
    base.preflight(root)
    gate_path = STAGE13_ANALYSIS / "stage13c_abc_selective_qk_fp32_gate.json"
    if base.sha256(gate_path) != GATE_SHA:
        raise ValueError("Frozen selective-FP32 numerical gate changed")
    gate = base.read_json(gate_path)
    gate_script_sha = FROZEN_SOURCE_HASHES[
        "analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.py"]
    if (gate["status"] != "PASS_BOUNDED_WINDOW_ALL_VARIANTS" or
            gate["variants"]["B"]["status"] != "PASS_BOUNDED_WINDOW" or
            gate["variants"]["B"]["batches_completed"] != 16 or
            gate["config_sha256"]["B"] != STAGE13_CONFIG_B_SHA or
            gate["split_sha256"] != SPLIT_SHA or gate["gate_script_sha256"] != gate_script_sha):
        raise ValueError("Frozen Stage15 numerical/data gate is not applicable")
    for relative, digest in gate["source_sha256"].items():
        if base.sha256(root / relative) != digest:
            raise ValueError(f"Source changed since the numerical gate: {relative}")
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


def failed_gates(metrics: dict, rule: dict) -> list[str]:
    floor = rule["eligible_if_all"]
    mel = metrics["per_class"]["mel"]
    checks = (
        ("melanoma_recall", mel["support"] == rule["melanoma_support"] and
         mel["recall"] * mel["support"] + 1e-9 >= floor["melanoma_correct_minimum"] and
         mel["recall"] + 1e-12 >= floor["melanoma_recall_minimum"]),
        ("melanoma_f1", mel["f1"] + 1e-12 >= floor["melanoma_f1_minimum"]),
        ("macro_f1", metrics["macro_f1"] + 1e-12 >= floor["macro_f1_minimum"]),
        ("accuracy", metrics["accuracy"] + 1e-12 >= floor["accuracy_minimum"]),
        ("nevus_recall", metrics["per_class"]["nv"]["recall"] + 1e-12 >= floor["nevus_recall_minimum"]),
    )
    return [name for name, passed in checks if not passed]


def validation_evidence(epoch: int, loss: float, metrics: dict, predictions: list[dict],
                        val_loader, rule: dict) -> dict:
    source = val_loader.dataset.rows
    if len(predictions) != len(source) or len(source) != 986:
        raise ValueError("Validation prediction count differs from frozen partition")
    matrix = [[0] * 7 for _ in range(7)]
    ids = set()
    for expected, row in zip(source, predictions):
        image_id, true_name, predicted_name = row["image_id"], row["true_label"], row["predicted_label"]
        if (image_id != expected["image_id"] or true_name != expected["dx"] or
                image_id in ids or true_name not in base.CLASSES or predicted_name not in base.CLASSES or
                row["correct"] != str(true_name == predicted_name)):
            raise ValueError("Validation predictions differ from frozen validation IDs/classes")
        ids.add(image_id)
        matrix[base.CLASSES.index(true_name)][base.CLASSES.index(predicted_name)] += 1
        probs = json.loads(row["probabilities"])
        if (len(probs) != 7 or not all(math.isfinite(p) and 0 <= p <= 1 for p in probs) or
                abs(sum(probs) - 1) > 1e-3 or
                base.CLASSES[max(range(7), key=lambda i: probs[i])] != predicted_name):
            raise ValueError("Validation probabilities are invalid")
    rebuilt = base.metrics(matrix)
    if rebuilt != metrics:
        raise ValueError("Validation metrics do not match saved predictions")
    classes = {}
    for index, name in enumerate(base.CLASSES):
        tp = matrix[index][index]
        classes[name] = {**metrics["per_class"][name], "tp": tp,
                         "fp": sum(row[index] for row in matrix) - tp,
                         "fn": sum(matrix[index]) - tp}
    failed = failed_gates(metrics, rule)
    if bool(failed) == eligible(metrics, rule):
        raise ValueError("Failed-gate list differs from frozen eligibility rule")
    total = sum(value["support"] for value in metrics["per_class"].values())
    macro_precision = sum(value["precision"] for value in metrics["per_class"].values()) / 7
    macro_recall = sum(value["recall"] for value in metrics["per_class"].values()) / 7
    weighted_f1 = sum(value["f1"] * value["support"]
                      for value in metrics["per_class"].values()) / total
    return {"experiment": NAME, "epoch": epoch, "sample_count": len(predictions),
            "class_order": list(base.CLASSES), "validation_loss": loss,
            "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_precision": macro_precision, "macro_recall": macro_recall,
            "macro_f1": metrics["macro_f1"], "weighted_f1": weighted_f1,
            "mel_recall": classes["mel"]["recall"],
            "mel_f1": classes["mel"]["f1"], "nv_recall": classes["nv"]["recall"],
            "per_class": classes, "confusion_matrix": matrix,
            "eligible": not failed, "failed_gates": failed}


def epoch_paths(epoch: int) -> tuple[Path, Path]:
    folder = OUT / "validation_epochs"
    return (folder / f"epoch_{epoch:03d}_predictions.csv",
            folder / f"epoch_{epoch:03d}_metrics.json")


def validation_bytes(predictions: list[dict], metrics: dict) -> tuple[bytes, bytes]:
    if not predictions:
        raise ValueError("Cannot save empty validation predictions")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
    writer.writeheader()
    writer.writerows(predictions)
    return (stream.getvalue().encode("utf-8"),
            (json.dumps(metrics, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def validation_hashes(prediction_bytes: bytes, metrics_bytes: bytes) -> dict[str, str]:
    return {"predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
            "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest()}


def write_once(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(contents).hexdigest()
    if path.exists():
        if base.sha256(path) != digest:
            raise ValueError(f"Existing validation artifact differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(contents)
    os.replace(temporary, path)


def write_atomic(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(contents)
    os.replace(temporary, path)


def history_bytes(history: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)
    return stream.getvalue().encode("utf-8")


def require_existing_artifacts(names: list[str], *, context: str) -> None:
    missing = [name for name in names if not (OUT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{context} stopped: expected artifact(s) missing: {missing}")


def finite_tensors(named_values) -> bool:
    return all(not isinstance(value, torch.Tensor) or not value.is_floating_point() or
               bool(torch.isfinite(value).all().item()) for _, value in named_values)


def validate_resume(state: dict, cfg: dict, rule: dict) -> dict:
    required = {"experiment", "epoch", "configuration", "config_sha256", "selection_rule_sha256",
        "numerical_protocol_sha256", "lr_policy_sha256", "model_state", "optimizer_state", "scheduler_state",
        "scaler_state", "history", "best_epoch", "best_validation_loss",
        "sampler_generator_state", "torch_rng_state", "cuda_rng_states",
        "python_rng_state", "numpy_rng_state", "numerical_events",
        "validation_artifacts", "latest_validation_payload",
        "class_order", "architecture", "architecture_metadata", "numerical_protocol",
        "runner_sha256", "initialization_sha256", "comparison_sha256", "initialization_report",
        "initialization_report_sha256"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Resume checkpoint is incomplete")
    if (state["experiment"] != NAME or state["configuration"] != cfg or
            state["config_sha256"] != CONFIG_SHA or state["selection_rule_sha256"] != RULE_SHA or
            state["numerical_protocol_sha256"] != NUMERICAL_SHA or
            state["lr_policy_sha256"] != LR_POLICY_SHA or
            state["class_order"] != list(base.CLASSES) or
            state["architecture"] != cfg["architecture"] or
            state["architecture_metadata"] != ARCHITECTURE_METADATA or
            state["numerical_protocol"] != base.read_json(HERE / "numerical_protocol.json") or
            state["initialization_report"].get("status") != "LOADED" or
            state["initialization_report"].get("weight_enum") != f"ResNet50_Weights.{RESNET_WEIGHTS.name}" or
            state["initialization_report"].get("report_sha256") is None or
            state["initialization_report_sha256"] == "" or
            state["runner_sha256"] != base.sha256(Path(__file__)) or
            state["initialization_sha256"] != base.sha256(HERE / "initialization.py") or
            state["comparison_sha256"] != base.sha256(HERE / "comparison.py")):
        raise ValueError("Resume checkpoint provenance/configuration mismatch")
    epoch, history = state["epoch"], state["history"]
    if type(epoch) is not int or not 1 <= epoch <= 25 or not isinstance(history, list) or len(history) != epoch:
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
    groups = state["optimizer_state"]["param_groups"]
    scheduler = state["scheduler_state"]
    if (len(groups) != 2 or
            [g.get("group_name") for g in groups] !=
            ["new_and_non_resnet", "pretrained_resnet_trunk"] or
            any(g["betas"] != (0.9, 0.999) or g["weight_decay"] != 0.0001 or
                g["eps"] != 1e-8 or not math.isfinite(float(g["lr"])) or
                float(g["lr"]) <= 0 for g in groups) or
            float(groups[1]["lr"]) != float(groups[0]["lr"]) * 0.1 or
            set(groups[0]["params"]) & set(groups[1]["params"]) or
            scheduler["last_epoch"] != epoch or not math.isfinite(float(scheduler["best"])) or
            len(scheduler["_last_lr"]) != 2 or
            any(float(scheduler["_last_lr"][i]) != float(groups[i]["lr"]) for i in range(2))):
        raise ValueError("Resume optimizer/scheduler state invalid")
    for row in history:
        if (float(row["lr_new_modules"]) <= 0 or
                float(row["lr_pretrained_resnet"]) != float(row["lr_new_modules"]) * 0.1):
            raise ValueError("Resume history LR ratio mismatch")
    if (float(history[-1]["lr_new_modules"]) != float(groups[0]["lr"]) or
            float(history[-1]["lr_pretrained_resnet"]) != float(groups[1]["lr"])):
        raise ValueError("Resume optimizer LR differs from history")
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
    artifacts = state["validation_artifacts"]
    latest = state["latest_validation_payload"]
    if (not isinstance(artifacts, dict) or set(artifacts) != set(range(1, epoch + 1)) or
            not isinstance(latest, dict) or set(latest) != {"predictions", "metrics"}):
        raise ValueError("Resume per-epoch validation evidence incomplete")
    latest_prediction_bytes, latest_metrics_bytes = validation_bytes(
        latest["predictions"], latest["metrics"])
    if (validation_hashes(latest_prediction_bytes, latest_metrics_bytes) != artifacts[epoch] or
            latest["metrics"]["epoch"] != epoch or
            latest["metrics"]["validation_loss"] != history[-1]["val_loss"] or
            latest["metrics"]["eligible"] is not history[-1]["eligible"] or
            latest["metrics"]["failed_gates"] != failed_gates(latest["metrics"], rule)):
        raise ValueError("Resume latest validation evidence differs from saved history/rule")
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
            base.read_json(OUT / "lr_policy.json") != base.read_json(HERE / "lr_policy.json") or
            base.read_json(OUT / "selection_rule.json") != base.read_json(HERE / "selection_rule.json") or
            base.read_json(OUT / "initialization_spec.json") != base.read_json(HERE / "initialization_spec.json") or
            base.sha256(OUT / "initialization_report.json") != state["initialization_report_sha256"] or
            base.read_json(OUT / "initialization_report.json") != state["initialization_report"]):
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
    for epoch, expected in state["validation_artifacts"].items():
        predictions_path, metrics_path = epoch_paths(epoch)
        for path, digest in ((predictions_path, expected["predictions_sha256"]),
                             (metrics_path, expected["metrics_sha256"])):
            if path.is_file():
                if base.sha256(path) != digest:
                    raise ValueError(f"Per-epoch validation artifact hash mismatch: {path}")
            elif epoch != state["epoch"]:
                raise FileNotFoundError(f"Earlier validation artifact missing: {path}")
    best_path = OUT / "best_checkpoint.pt"
    if state["best_epoch"] is None:
        if best_path.exists():
            raise ValueError("Unexpected best checkpoint without eligible epoch")
    else:
        if not best_path.is_file():
            raise FileNotFoundError("Selected best checkpoint is missing; refusing reconstruction")
        best = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
        if (best.get("epoch") != state["best_epoch"] or
                best.get("best_validation_loss") != state["best_validation_loss"] or
                not finite_tensors(best["model_state"].items())):
            raise ValueError("Existing selected checkpoint is inconsistent/corrupt")


def recover_saved_epoch(state: dict, model, optimizer, scaler) -> None:
    epoch = state["epoch"]
    predictions_path, metrics_path = epoch_paths(epoch)
    prediction_bytes, metrics_bytes = validation_bytes(
        state["latest_validation_payload"]["predictions"],
        state["latest_validation_payload"]["metrics"])
    if validation_hashes(prediction_bytes, metrics_bytes) != state["validation_artifacts"][epoch]:
        raise ValueError("Saved validation payload hash mismatch")
    write_once(predictions_path, prediction_bytes)
    write_once(metrics_path, metrics_bytes)


def write_validation_comparison(best_epoch: int | None) -> dict:
    reference15_path = ROOT / "experiments/stage15_single_candidate_exp1/run/validation_epochs/epoch_016_metrics.json"
    reference23_path = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_metrics.json"
    if (base.sha256(reference15_path) != STAGE15_VALIDATION_METRICS_SHA or
            base.sha256(reference23_path) != STAGE23_VALIDATION_SHA):
        raise ValueError("Frozen validation comparison metrics changed")
    stage15 = base.read_json(reference15_path)
    stage23 = base.read_json(reference23_path)
    stage25 = None
    if best_epoch is not None:
        _, selected_metrics = epoch_paths(best_epoch)
        stage25 = base.read_json(selected_metrics)
    result = compare_stage25(stage25, stage23, stage15)
    encoded = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
    write_once(OUT / "stage15_stage23_stage25_validation_comparison.json", encoded)
    return result


def execute(root: Path, cfg: dict, rule: dict, numerical: dict, *, resume: bool) -> None:
    saved = None
    if resume:
        if (OUT / "experiment_manifest.json").exists():
            raise FileExistsError("Run is already finalized; resume is unnecessary")
        path = OUT / "last_checkpoint.pt"
        if not path.is_file():
            raise FileNotFoundError("--resume requires existing last_checkpoint.pt; no fresh fallback")
        report_path = OUT / "initialization_report.json"
        if not report_path.is_file():
            raise FileNotFoundError("--resume requires the saved initialization report")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if base.sha256(report_path) != saved.get("initialization_report_sha256"):
            raise ValueError("Initialization report hash differs from the saved checkpoint")
        if base.read_json(report_path) != saved.get("initialization_report"):
            raise ValueError("Initialization report content differs from the saved checkpoint")
        validate_resume(saved, cfg, rule)
        verify_run_files_for_resume(saved)
        if len(saved["cuda_rng_states"]) != torch.cuda.device_count():
            raise ValueError("Saved CUDA RNG device count differs")
    else:
        if OUT.exists():
            raise FileExistsError("Fresh run refuses existing output directory")
        OUT.mkdir(parents=True)
        for filename in ("config.json", "numerical_protocol.json", "selection_rule.json", "initialization_spec.json", "lr_policy.json"):
            write_once(OUT / filename, (HERE / filename).read_bytes())
    model, optimizer, scheduler, scaler, train_loader, val_loader, generator, initialization_report, initialization_report_sha = \
        build_stage25_components(root, cfg, base=base, output_dir=OUT, initial_scale=INITIAL_SCALE)
    restore_policy = install_selective_qk_policy(model)
    history, best_epoch, best_loss, start_epoch = [], None, math.inf, 1
    events = []
    validation_artifacts = {}
    ready = not resume
    try:
        if resume:
            if (initialization_report_sha != saved["initialization_report_sha256"] or
                    initialization_report != saved["initialization_report"]):
                raise ValueError("Reconstructed initialization provenance differs from checkpoint")
            events = copy.deepcopy(saved["numerical_events"])
            validation_artifacts = copy.deepcopy(saved["validation_artifacts"])
            start_epoch, history, best_epoch, best_loss = restore_resume(
                saved, model, optimizer, scheduler, scaler, generator)
            guard.require_finite_states(model, optimizer, {"experiment": NAME, "epoch": saved["epoch"]})
            recover_saved_epoch(saved, model, optimizer, scaler)
            ready = True
            print(f"Stage25 resume saved_epoch={saved['epoch']} next_epoch={start_epoch} "
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
            val_loss, val_metrics, predictions = guard.guarded_validation_epoch(
                model, val_loader, NAME, epoch, predictions=True)
            guard.require_finite_states(model, optimizer, {"experiment": NAME, "epoch": epoch})
            guard.require_finite_epoch_losses(train_loss, val_loss, NAME, epoch)
            evidence = validation_evidence(epoch, val_loss, val_metrics, predictions, val_loader, rule)
            prediction_bytes, metrics_bytes = validation_bytes(predictions, evidence)
            validation_artifacts[epoch] = validation_hashes(prediction_bytes, metrics_bytes)
            scheduler_step_preserving_ratio(scheduler, optimizer, val_loss)
            is_eligible = eligible(val_metrics, rule)
            if evidence["eligible"] is not is_eligible:
                raise ValueError("Per-epoch evidence eligibility differs from frozen rule")
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                   "val_accuracy": val_metrics["accuracy"],
                   "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                     "val_macro_precision": evidence["macro_precision"],
                     "val_macro_recall": evidence["macro_recall"],
                   "val_macro_f1": val_metrics["macro_f1"],
                     "val_weighted_f1": evidence["weighted_f1"],
                   "val_mel_precision": val_metrics["per_class"]["mel"]["precision"],
                   "val_mel_recall": val_metrics["per_class"]["mel"]["recall"],
                   "val_mel_f1": val_metrics["per_class"]["mel"]["f1"],
                   "val_nv_recall": val_metrics["per_class"]["nv"]["recall"],
                   "eligible": is_eligible, "lr_new_modules": optimizer.param_groups[0]["lr"],
                   "lr_pretrained_resnet": optimizer.param_groups[1]["lr"],
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
                     "lr_policy_sha256": LR_POLICY_SHA,
                     "runner_sha256": base.sha256(Path(__file__)),
                     "initialization_sha256": base.sha256(HERE / "initialization.py"),
                     "comparison_sha256": base.sha256(HERE / "comparison.py"),
                     "class_order": list(base.CLASSES), "architecture": cfg["architecture"],
                     "architecture_metadata": copy.deepcopy(ARCHITECTURE_METADATA),
                     "numerical_protocol": copy.deepcopy(numerical),
                     "initialization_report": initialization_report,
                     "initialization_report_sha256": initialization_report_sha,
                     "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                     "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                     "history": history, "best_epoch": best_epoch,
                     "best_validation_loss": best_loss,
                     "numerical_events": copy.deepcopy(events),
                     "validation_artifacts": copy.deepcopy(validation_artifacts),
                     "latest_validation_payload": {"predictions": predictions, "metrics": evidence},
                     "sampler_generator_state": generator.get_state(),
                     "python_rng_state": random.getstate(),
                     "numpy_rng_state": np.random.get_state(),
                     "torch_rng_state": torch.get_rng_state(),
                     "cuda_rng_states": torch.cuda.get_rng_state_all()}
            guard.checked_checkpoint(OUT / "last_checkpoint.pt", state, model, optimizer, scaler, NAME, epoch)
            if improved:
                guard.checked_checkpoint(OUT / "best_checkpoint.pt", state, model, optimizer, scaler, NAME, epoch)
            predictions_path, metrics_path = epoch_paths(epoch)
            write_once(predictions_path, prediction_bytes)
            write_once(metrics_path, metrics_bytes)
            write_atomic(OUT / "training_history.csv", history_bytes(state["history"]))
            write_atomic(OUT / "numerical_events.json",
                         (json.dumps(state["numerical_events"], indent=2, allow_nan=False) + "\n").encode("utf-8"))
            write_atomic(OUT / "training_history.csv", history_bytes(history))
            write_atomic(OUT / "numerical_events.json",
                         (json.dumps(events, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            print(f"Stage25 epoch={epoch}/25 train_loss={train_loss:.6f} "
                  f"val_loss={val_loss:.6f} eligible={is_eligible} scale={scaler.get_scale():.0f}",
                  flush=True)
        best_checkpoint_for_finalization(best_epoch, OUT / "best_checkpoint.pt")
        if best_epoch is None:
            status = "NO_CANDIDATE_SELECTED"
        else:
            require_existing_artifacts(["best_checkpoint.pt"], context=f"Eligible epoch {best_epoch} finalization")
            selected_predictions, selected_metrics = epoch_paths(best_epoch)
            recorded = base.read_json(selected_metrics)
            if (not recorded["eligible"] or recorded["validation_loss"] != best_loss or
                    recorded["failed_gates"] or
                    base.sha256(selected_predictions) != validation_artifacts[best_epoch]["predictions_sha256"] or
                    base.sha256(selected_metrics) != validation_artifacts[best_epoch]["metrics_sha256"]):
                raise ValueError("Selected validation evidence differs from frozen rule/history")
            write_once(OUT / "validation_predictions.csv", selected_predictions.read_bytes())
            write_once(OUT / "validation_metrics.json", selected_metrics.read_bytes())
            status = "CANDIDATE_SELECTED_VALIDATION_ONLY"
        comparison = write_validation_comparison(best_epoch)
        artifacts = ["config.json", "selection_rule.json", "numerical_protocol.json",
                 "lr_policy.json", "initialization_spec.json", "initialization_report.json",
                     "training_history.csv", "last_checkpoint.pt",
                     "stage15_stage23_stage25_validation_comparison.json"]
        artifacts += (["best_checkpoint.pt", "validation_predictions.csv", "validation_metrics.json"]
                      if best_epoch is not None else [])
        artifacts.append("numerical_events.json")
        for epoch in range(1, len(history) + 1):
            for path in epoch_paths(epoch):
                artifacts.append(str(path.relative_to(OUT)).replace("\\", "/"))
        require_existing_artifacts(artifacts, context="Manifest finalization")
        manifest = {
            "experiment": NAME, "status": status, "epochs": len(history),
            "validation_decision": comparison["decision"],
            "selected_epoch": best_epoch,
            "best_validation_loss": best_loss if best_epoch is not None else None,
            "config_sha256": CONFIG_SHA, "selection_rule_sha256": RULE_SHA,
            "numerical_protocol_sha256": NUMERICAL_SHA,
            "lr_policy_sha256": LR_POLICY_SHA,
            "split_sha256": base.SPLIT_SHA, "stage9_reference_checkpoint_sha256": base.STAGE9_SHA,
            "numerical_gate_sha256": GATE_SHA, "runner_sha256": base.sha256(Path(__file__)),
            "initialization_sha256": base.sha256(HERE / "initialization.py"),
            "comparison_sha256": base.sha256(HERE / "comparison.py"),
            "initialization_report_sha256": initialization_report_sha,
            "resnet_weight_enum": f"ResNet50_Weights.{RESNET_WEIGHTS.name}",
            "artifact_sha256": {name: base.sha256(OUT / name) for name in artifacts},
            "runtime": runtime(), "ham_test_accessed": False, "ph2_accessed": False}
        write_atomic(OUT / "experiment_manifest.json",
                     (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    except Exception as exc:
        if ready:
            failure = (exc.record if isinstance(exc, guard.NumericalFailure) else
                       {"type": type(exc).__name__, "message": str(exc)})
            failure_record = {
                "experiment": NAME, "status": "STOPPED", "failure": failure,
                "validation_decision": "EXPERIMENT_INVALID",
                "completed_epochs": len(history), "ham_test_accessed": False,
                "ph2_accessed": False}
            write_atomic(OUT / "failure.json",
                         (json.dumps(failure_record, indent=2, allow_nan=False) + "\n").encode("utf-8"))
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
    print(json.dumps({"registered_selection_rule": rule,
                      "max_epochs": cfg["epochs"],
                      "optimizer_groups": cfg["optimizer"]["parameter_groups"],
                      "all_resnet_layers_trainable_from_epoch": 1,
                      "resnet_weight_enum": f"ResNet50_Weights.{RESNET_WEIGHTS.name}",
                      "resnet_weight_url": RESNET_WEIGHTS.url,
                      "training_started": False}, indent=2, allow_nan=False), flush=True)
    if args.check:
        print(json.dumps({"status": "PREFLIGHT_PASS", "experiment": NAME,
                          "config_sha256": CONFIG_SHA, "selection_rule_sha256": RULE_SHA,
                          "numerical_protocol_sha256": NUMERICAL_SHA,
                          "lr_policy_sha256": LR_POLICY_SHA,
                          "bounded_B_gate_sha256": GATE_SHA,
                          "full_training_started": False,
                          "ham_test_accessed": False, "ph2_accessed": False}, indent=2))
        return
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Full experiment requires Tesla T4")
    execute(root, cfg, rule, numerical, resume=args.resume)


if __name__ == "__main__":
    main()
