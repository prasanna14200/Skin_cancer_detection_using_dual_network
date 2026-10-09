"""Stage24 staged ResNet50 fine-tuning; --check is read-only and training is never automatic."""
from __future__ import annotations

import argparse
import copy
import csv
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
STAGE13 = ROOT / "experiments/egvan_melanoma_ablation_exp"
STAGE13_ANALYSIS = ROOT / "analysis/stage13_melanoma_ablation"
STAGE23 = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1"
_IMPORT_PATHS = (SRC, HERE, STAGE13, STAGE13_ANALYSIS, STAGE23)
for _path in _IMPORT_PATHS:
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
sys.path[0:0] = [str(path) for path in _IMPORT_PATHS]

import train_ablation as base  # noqa: E402
import stage13c_guarded as guard  # noqa: E402
from stage13c_selective_qk_fp32_train import (  # noqa: E402
    checked_train_batch, validate_saved_scaler,
)
from stage13c_abc_selective_qk_fp32_gate import install_selective_qk_policy  # noqa: E402
from comparison import compare_stage24  # noqa: E402
from finetuning import (  # noqa: E402
    LR_MULTIPLIERS, Stage24EGVAN, apply_scheduler_step, assert_optimizer_layout,
    build_optimizer_scheduler, group_lrs, maintain_lr_ratios,
)
from torchvision.models import ResNet50_Weights  # noqa: E402

NAME = "stage24_staged_finetuning_exp1"
OUT = HERE / "run"
CONFIG_SHA = "777cf9387f2395f85177ef6e8192ce880ffffaacba07a013531d0e1275478e2b"
RULE_SHA = "5f1db2609d4aba3dec188b95f23feffb6900a8159d283a182188a22fd9635059"
NUMERICAL_SHA = "8103dc1f24692ff581be45376934fbc0e1d3a78974b3a242c3fe9d52df90dc27"
FINETUNING_POLICY_SHA = "9ffa7e6adf60f5c9534236196935ce1da7880511ca0b3f32f30cbfda596ebc41"
INITIALIZATION_SPEC_SHA = "726f4eb45c5f1ce8197feeab528f4b00465a5d37e5a87f4f7f254a8ab4c804a4"
STAGE24_BASE_LR = 0.001
STAGE24_MAX_EPOCHS = 25
STAGE23_CHECKPOINT_SHA = "42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab"
STAGE23_SELECTED_METRICS_SHA = "beaebb91d9b7043698b5b4580d4af606eac477a985512589e4eb8e99e16af058"
STAGE23_MANIFEST_SHA = "48e7fe9d75ac9b4b35750c8d632e3cced5939574e0080b3f12026ea959d44076"
STAGE23_INITIALIZATION_REPORT_SHA = "c86113181d9a84ecb7f3dd38e2e43f1e2926b564a682ed3162a16e7f0b4e5fb9"
STAGE23_COMPARISON_SHA = "c25443d695b072e54d0581c8c2ad827fb5c3b6bbee4617eb4bea489c90f329b5"
STAGE15_CONFIG_SHA = "3ad8c205e06cde8203a869be0a6cc9643ed0ca5b6beb553b57d53d4dce9d25f2"
STAGE15_RULE_SHA = "daf07a1bac3538b5a35ed59121bd5561aba4797a6889e852fd0ca07ad88be735"
STAGE15_NUMERICAL_SHA = "f59010b7e2509c0bf1633b7126ff1d8d629afab39b515f57e95012fb78eb0196"
STAGE15_RUNNER_SHA = "391769f3993e1593064cf9f6460feeb8846e0ca2cc955e3fc469700420d9fdb9"
STAGE15_HISTORY_SHA = "2df7edd11a23658e56dc723ee57638476579cb3620d9c5d10039eefa07941fd0"
STAGE15_METRICS_SHA = "630bb7a41c41cd5b125576ca0d14beef7d5b523dba7919680e526501f52c0571"
STAGE15_CHECKPOINT_SHA = "85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5"
STAGE23_CONFIG_SHA = "67a9d77d8464ef51d94f5861004084b179c3c42159e172c680d85952514d7afa"
STAGE23_RULE_SHA = "a03213d5db62af968a425f3557773aa14c85adeabdbaeb8e6d46bf4d5481111d"
STAGE23_NUMERICAL_SHA = "06222c374635f15ad99642e1bdbeaf4c6c69959dd753d6b332cdb753b87abdfb"
STAGE23_INITIALIZATION_SPEC_SHA = "145758267bb655d21f70d69c8c384259e4b4074a7596dcf99d2dcb0a2f7a9d76"
STAGE23_RUNNER_SHA = "d009803461a52818014f82a0d791650a86abf3737e98fb0d198eb4224136e967"
STAGE23_INITIALIZATION_HELPER_SHA = "56a5b767c4c204eac64fbbc8a4c91c1ee7694fdcc3e87ed50bdaf9e0d7b483be"
SPLIT_SHA = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
STAGE13_CONFIG_B_SHA = "9f69e862c8d63a2375291343f732c73f5f4bbc4339029b4afced844c52cf7562"
STAGE13_SELECTIVE_RUNNER_SHA = "d5cdfbade60e834e56acf9d67fbb3f83d90f0c6f96473dff9b9d3dd4c399ef1a"
STAGE13_GUARD_SCRIPT_SHA = "198769909e0572b0d5959c849731bb7c7613d578e6c231fd0040580b3519fcff"
STAGE13_GATE_ARTIFACT_SHA = "721eeaf6c64740d7dd8d7cdb06890a2ffa1c1be49c336f4d07033e4d172cdb00"
FROZEN_REFERENCE_HASHES = {
    "experiments/stage15_single_candidate_exp1/config.json": STAGE15_CONFIG_SHA,
    "experiments/stage15_single_candidate_exp1/selection_rule.json": STAGE15_RULE_SHA,
    "experiments/stage15_single_candidate_exp1/numerical_protocol.json": STAGE15_NUMERICAL_SHA,
    "experiments/stage15_single_candidate_exp1/train.py": STAGE15_RUNNER_SHA,
    "experiments/stage15_single_candidate_exp1/run/training_history.csv": STAGE15_HISTORY_SHA,
    "experiments/stage15_single_candidate_exp1/run/validation_epochs/epoch_016_metrics.json": STAGE15_METRICS_SHA,
    "experiments/stage15_single_candidate_exp1/stage15_single_candidate_exp1/recovery_epoch16/best_checkpoint.pt": STAGE15_CHECKPOINT_SHA,
    "experiments/stage23_resnet50_imagenet_init_exp1/config.json": STAGE23_CONFIG_SHA,
    "experiments/stage23_resnet50_imagenet_init_exp1/selection_rule.json": STAGE23_RULE_SHA,
    "experiments/stage23_resnet50_imagenet_init_exp1/numerical_protocol.json": STAGE23_NUMERICAL_SHA,
    "experiments/stage23_resnet50_imagenet_init_exp1/initialization_spec.json": STAGE23_INITIALIZATION_SPEC_SHA,
    "experiments/stage23_resnet50_imagenet_init_exp1/train.py": STAGE23_RUNNER_SHA,
    "experiments/stage23_resnet50_imagenet_init_exp1/initialization.py": STAGE23_INITIALIZATION_HELPER_SHA,
    "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt": STAGE23_CHECKPOINT_SHA,
    "experiments/stage23_resnet50_imagenet_init_exp1/run/experiment_manifest.json": STAGE23_MANIFEST_SHA,
    "experiments/stage23_resnet50_imagenet_init_exp1/run/initialization_report.json": STAGE23_INITIALIZATION_REPORT_SHA,
    "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_metrics.json": STAGE23_SELECTED_METRICS_SHA,
    "experiments/stage23_resnet50_imagenet_init_exp1/run/stage15_vs_stage23_validation_comparison.json": STAGE23_COMPARISON_SHA,
    "data/splits/split_leakage_aware.csv": SPLIT_SHA,
}
INITIAL_SCALE = 8192.0
WEIGHTS = ResNet50_Weights.IMAGENET1K_V2

FROZEN_SOURCE_HASHES = {
    "experiments/egvan_melanoma_ablation_exp/train_ablation.py": "56a71821c1f944028874a76febf00ea68b3b152e481e6450ddd5bd94606e5389",
    "experiments/egvan_melanoma_ablation_exp/stage13c_guarded.py": "93e2318afafff8ad1c6907752781d61856cdf3bbde3a614b29686686dac0340d",
    "experiments/egvan_melanoma_ablation_exp/stage13c_selective_qk_fp32_train.py": STAGE13_SELECTIVE_RUNNER_SHA,
    "experiments/egvan_melanoma_ablation_exp/config_B.json": STAGE13_CONFIG_B_SHA,
    "analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.py": STAGE13_GUARD_SCRIPT_SHA,
    "analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.json": STAGE13_GATE_ARTIFACT_SHA,
    "src/models/egvan/egvan.py": "f5f5217d6cced6ccc9c1a2d17a5691701bfe454eb46bf52bf97468030cc7b44f",
    "src/models/egvan/modified_resnet50.py": "5f9a79d4de988f0e2e046bc5b38d90d62dc7c28273c7c922edad6e64f76caeba",
    "src/models/egvan/efficientnet_branch.py": "e9f25b66247349ff0597ff76a1e4d3dc86a80d8358ad9f63ef1bf4e96cb98b0e",
    "src/models/egvan/scga.py": "6d8a72871c7ae46de12d52f3b0fe973cf75b022434678161b6ab801ffa900d11",
    "src/models/egvan/non_local.py": "d1795018519ae6650a130c44a17aac88fcc58b2ece4336879caf703c24c45c3a",
    "src/models/egvan/mff.py": "ad97108386f6b749bf30d6624d3db88663635393519d0cb6a519e65f96bbc b06".replace(" ", ""),
    "src/models/egvan/spatial_attention.py": "34600fbd2207040de06baff88532d1df684852dcf9086e49789856352cf52ae7",
    "src/models/egvan/__init__.py": "e858a0996e67168ab50bd5524a2ee3609d90db2fec86109ce5ce0273a3e61449",
    "src/preprocessing.py": "cf0a176a610b52ba45b0e91761b027a560252234f81f7b6fd572f02d8baab6ed",
    "src/dataset.py": "02c6d280e468a8c83719bffd7b6bc042b138893d7badd794d0ecfa9291ec8cad",
    "src/train.py": "c195142d2f924b83374c0d6a8d4d93061ecf2e77e947710e6a3a8eb8264686bc",
}


def sha256(path: Path) -> str:
    return base.sha256(path)


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".stage24_", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_bytes(contents)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_once(path: Path, contents: bytes) -> None:
    if path.exists():
        if sha256(path) != hashlib.sha256(contents).hexdigest():
            raise ValueError(f"Existing Stage24 artifact differs; refusing overwrite: {path}")
        return
    atomic_write(path, contents)


def read_protocol() -> tuple[dict, dict, dict, dict, dict]:
    expected = {
        "config.json": CONFIG_SHA,
        "selection_rule.json": RULE_SHA,
        "numerical_protocol.json": NUMERICAL_SHA,
        "finetuning_policy.json": FINETUNING_POLICY_SHA,
        "initialization_spec.json": INITIALIZATION_SPEC_SHA,
    }
    for filename, digest in expected.items():
        if sha256(HERE / filename) != digest:
            raise ValueError(f"Stage24 preregistered file hash mismatch: {filename}")
    return tuple(base.read_json(HERE / name) for name in (
        "config.json", "selection_rule.json", "numerical_protocol.json",
        "finetuning_policy.json", "initialization_spec.json"))


def check_frozen_references(root: Path) -> tuple[dict, dict, dict, dict, dict]:
    if root.resolve() != ROOT.resolve():
        raise ValueError("Use the pinned EG-VAN project root")
    cfg, rule, numerical, policy, init_spec = read_protocol()
    stage15 = root / "experiments/stage15_single_candidate_exp1"
    stage23 = root / "experiments/stage23_resnet50_imagenet_init_exp1"
    for relative, digest in FROZEN_REFERENCE_HASHES.items():
        if sha256(root / relative) != digest:
            raise ValueError(f"Frozen reference hash mismatch: {relative}")
    for relative, digest in FROZEN_SOURCE_HASHES.items():
        if sha256(root / relative) != digest:
            raise ValueError(f"Frozen Stage23 source hash mismatch: {relative}")
    stage15_config = base.read_json(stage15 / "config.json")
    stage23_config = base.read_json(stage23 / "config.json")
    stage15_rule = base.read_json(stage15 / "selection_rule.json")
    stage23_rule = base.read_json(stage23 / "selection_rule.json")
    candidate = copy.deepcopy(cfg)
    reference = copy.deepcopy(stage23_config)
    for value in (candidate, reference):
        value.pop("experiment", None)
        value.pop("description", None)
    if candidate != reference:
        raise ValueError("Stage24 changes a Stage23 scientific config field outside finetuning policy")
    for source_config in (stage15_config, stage23_config):
        if (source_config["classes"] != cfg["classes"] or
                source_config["image_size"] != cfg["image_size"] or
                source_config["epochs"] != cfg["epochs"] or
                source_config["sampler"] != cfg["sampler"] or
                source_config["loss"] != cfg["loss"]):
            raise ValueError("Stage24 config differs from frozen Stage15/Stage23 training factors")
    for reference_rule in (stage15_rule, stage23_rule):
        for key in ("melanoma_support", "stage9_correct_melanomas", "eligible_if_all",
                    "checkpoint_selection", "if_none_eligible"):
            if rule.get(key) != reference_rule.get(key):
                raise ValueError(f"Stage24 selection gate differs from frozen reference: {key}")
    if {k: v for k, v in numerical.items() if k != "name"} != {
            k: v for k, v in base.read_json(stage23 / "numerical_protocol.json").items() if k != "name"}:
        raise ValueError("Stage24 numerical policy differs from Stage23")
    if (cfg["data"]["split_sha256"] != SPLIT_SHA or cfg["epochs"] != STAGE24_MAX_EPOCHS or
            cfg["optimizer"]["learning_rate"] != STAGE24_BASE_LR or
            WEIGHTS.name != "IMAGENET1K_V2" or WEIGHTS.url != init_spec["official_weight_url"] or
            init_spec["stage23_trained_checkpoint_loaded"] is not False or
            init_spec["loader"] != "ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True, check_hash=True)" or
            policy["phase_a"]["epochs"] != [1, 5] or policy["phase_b"]["epochs"] != [6, 25] or
            [g["lr_multiplier"] for g in policy["optimizer_groups"]] != [1.0, 0.1, 0.1, 0.05] or
            [g["initial_learning_rate"] for g in policy["optimizer_groups"]] !=
            [0.001, 0.0001, 0.0001, 0.00005]):
        raise ValueError("Stage24 fine-tuning policy or source is inconsistent")
    if ([g["name"] for g in policy["optimizer_groups"]] !=
            ["non_resnet_and_new_modules", "resnet.layer3", "resnet.layer4", "resnet.layer2"] or
            policy["phase_a"]["requires_grad_false"] !=
            ["resnet.conv1", "resnet.bn1", "resnet.layer1", "resnet.layer2"] or
            policy["phase_b"]["requires_grad_false"] !=
            ["resnet.conv1", "resnet.bn1", "resnet.layer1"] or
            policy["phase_a"]["frozen_batchnorm_eval"] !=
            ["resnet.bn1", "all BatchNorm modules under resnet.layer1",
             "all BatchNorm modules under resnet.layer2"] or
            policy["phase_b"]["frozen_batchnorm_eval"] !=
            ["resnet.bn1", "all BatchNorm modules under resnet.layer1"] or
            policy["scheduler"]["class"] != "ReduceLROnPlateau" or
            policy["scheduler"]["factor"] != cfg["scheduler"]["factor"] or
            policy["scheduler"]["patience"] != cfg["scheduler"]["patience"]):
        raise ValueError("Stage24 registered phase, BN, LR group, or scheduler policy mismatch")
    return cfg, rule, numerical, policy, init_spec


def finetuning_policy() -> dict:
    return base.read_json(HERE / "finetuning_policy.json")


def preflight(root: Path) -> dict:
    cfg, rule, numerical, policy, init_spec = check_frozen_references(root)
    if OUT.exists():
        raise FileExistsError("--check requires no existing Stage24 run directory")
    integrity = base.preflight(root)
    return {
        "status": "PREFLIGHT_PASS",
        "experiment": NAME,
        "stage23_finetuning_policy": "all ResNet trunk layers trainable from epoch 1; one global Adamax group",
        "stage24_phases": {"A": [1, 5], "B": [6, 25]},
        "optimizer_groups": [
            {"name": g["name"], "multiplier": g["lr_multiplier"], "initial_lr": g["initial_learning_rate"]}
            for g in policy["optimizer_groups"]
        ],
        "selection_gates": rule["eligible_if_all"],
        "initialization": init_spec["resnet_weight_enum"],
        "split_sha256": cfg["data"]["split_sha256"],
        "train_val_preflight": integrity,
        "pretrained_weights_downloaded": False,
        "validation_inference_performed": False,
        "ham_test_accessed": False,
        "ph2_accessed": False,
        "training_started": False,
        "run_directory_created": False,
    }


def eligible(metrics: dict, rule: dict) -> bool:
    return not failed_gates(metrics, rule)


def validate_metrics(metrics: dict, rule: dict) -> bool:
    gates = rule["eligible_if_all"]
    mel = metrics["per_class"]["mel"]
    if mel["support"] != rule["melanoma_support"]:
        raise ValueError("HAM validation MEL support changed")
    return (mel["tp"] >= gates["melanoma_correct_minimum"] and
            mel["recall"] + 1e-12 >= gates["melanoma_recall_minimum"] and
            mel["f1"] + 1e-12 >= gates["melanoma_f1_minimum"] and
            metrics["macro_f1"] + 1e-12 >= gates["macro_f1_minimum"] and
            metrics["accuracy"] + 1e-12 >= gates["accuracy_minimum"] and
            metrics["per_class"]["nv"]["recall"] + 1e-12 >= gates["nevus_recall_minimum"])


def failed_gates(metrics: dict, rule: dict) -> list[str]:
    gates = rule["eligible_if_all"]
    mel = metrics["per_class"]["mel"]
    checks = (
        ("melanoma_support", mel["support"] == rule["melanoma_support"]),
        ("melanoma_correct_minimum", mel["tp"] >= gates["melanoma_correct_minimum"]),
        ("melanoma_recall", mel["recall"] + 1e-12 >= gates["melanoma_recall_minimum"]),
        ("melanoma_f1", mel["f1"] + 1e-12 >= gates["melanoma_f1_minimum"]),
        ("macro_f1", metrics["macro_f1"] + 1e-12 >= gates["macro_f1_minimum"]),
        ("accuracy", metrics["accuracy"] + 1e-12 >= gates["accuracy_minimum"]),
        ("nevus_recall", metrics["per_class"]["nv"]["recall"] + 1e-12 >= gates["nevus_recall_minimum"]),
    )
    return [name for name, passed in checks if not passed]


def validate_epoch_evidence(epoch: int, val_loss: float, metrics: dict,
                            predictions: list[dict], loader, rule: dict) -> dict:
    rows = loader.dataset.rows
    if len(rows) != 986 or len(predictions) != len(rows):
        raise ValueError("Validation prediction count differs from frozen Stage23 validation partition")
    matrix = [[0] * 7 for _ in range(7)]
    seen = set()
    for expected, record in zip(rows, predictions):
        if (record["image_id"] != expected["image_id"] or
                record["true_label"] != expected["dx"] or
                record["image_id"] in seen or
                record["true_label"] not in base.CLASSES or
                record["predicted_label"] not in base.CLASSES or
                record["correct"] != str(record["true_label"] == record["predicted_label"])):
            raise ValueError("Validation predictions differ from frozen validation IDs/classes")
        seen.add(record["image_id"])
        ti = base.CLASSES.index(record["true_label"])
        pi = base.CLASSES.index(record["predicted_label"])
        matrix[ti][pi] += 1
        probabilities = json.loads(record["probabilities"])
        if (len(probabilities) != 7 or
                not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities) or
                abs(sum(probabilities) - 1.0) > 1e-3 or
                base.CLASSES[max(range(7), key=lambda i: probabilities[i])] != record["predicted_label"]):
            raise ValueError("Validation probabilities are malformed or disagree with argmax")
    rebuilt = base.metrics(matrix)
    if rebuilt != metrics:
        raise ValueError("Validation metrics do not match the prediction-derived confusion matrix")
    classes = {}
    for index, name in enumerate(base.CLASSES):
        tp = matrix[index][index]
        classes[name] = {
            **metrics["per_class"][name],
            "tp": tp,
            "fp": sum(row[index] for row in matrix) - tp,
            "fn": sum(matrix[index]) - tp,
        }
    aggregates = macro_metrics({**metrics, "per_class": classes})
    report = {
        "experiment": NAME,
        "epoch": epoch,
        "sample_count": len(predictions),
        "class_order": list(base.CLASSES),
        "validation_loss": val_loss,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        **aggregates,
        "macro_f1": metrics["macro_f1"],
        "per_class": classes,
        "confusion_matrix": matrix,
    }
    report["failed_gates"] = failed_gates(report, rule)
    report["eligible"] = not report["failed_gates"]
    mel = classes["mel"]
    nv = classes["nv"]
    mel_index, nv_index = base.CLASSES.index("mel"), base.CLASSES.index("nv")
    report.update({
        "mel_tp": mel["tp"],
        "mel_fn": mel["fn"],
        "mel_precision": mel["precision"],
        "mel_recall": mel["recall"],
        "mel_f1": mel["f1"],
        "nv_recall": nv["recall"],
        "mel_to_nv": matrix[mel_index][nv_index],
        "nv_to_mel": matrix[nv_index][mel_index],
    })
    return report


def macro_metrics(metrics: dict) -> dict:
    per_class = metrics["per_class"]
    total = sum(v["support"] for v in per_class.values())
    return {
        "macro_precision": sum(v["precision"] for v in per_class.values()) / len(per_class),
        "macro_recall": sum(v["recall"] for v in per_class.values()) / len(per_class),
        "weighted_f1": sum(v["f1"] * v["support"] for v in per_class.values()) / total,
    }


def write_csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def validation_bytes(predictions: list[dict], metrics: dict) -> tuple[bytes, bytes]:
    return write_csv_bytes(predictions), json_bytes(metrics)


def epoch_paths(epoch: int) -> tuple[Path, Path]:
    folder = OUT / "validation_epochs"
    return folder / f"epoch_{epoch:03d}_predictions.csv", folder / f"epoch_{epoch:03d}_metrics.json"


def checkpoint_best(history: list[dict], rule: dict) -> tuple[int | None, float]:
    eligible_rows = [row for row in history if row["eligible"]]
    if not eligible_rows:
        return None, math.inf
    selected = min(eligible_rows, key=lambda row: (row["val_loss"], row["epoch"]))
    return selected["epoch"], selected["val_loss"]


def best_checkpoint_for_finalization(best_epoch: int | None, best_path: Path) -> Path | None:
    if best_epoch is None:
        if best_path.exists():
            raise ValueError("Best checkpoint exists although no epoch qualified")
        return None
    if not best_path.is_file():
        raise FileNotFoundError(f"Selected best checkpoint is missing: {best_path}")
    return best_path


def atomic_checkpoint(path: Path, state: dict, model, optimizer, scaler, epoch: int) -> None:
    guard.checked_checkpoint(path, state, model, optimizer, scaler, NAME, epoch)


def validate_checkpoint_state(state: dict, cfg: dict, rule: dict) -> dict:
    required = {
        "experiment", "epoch", "configuration", "config_sha256", "selection_rule_sha256",
        "numerical_protocol_sha256", "finetuning_policy_sha256", "initialization_spec_sha256",
        "reference_hashes", "architecture", "architecture_metadata", "class_order",
        "model_state", "optimizer_state", "scheduler_state", "scaler_state", "history",
        "best_epoch", "best_validation_loss", "python_rng_state", "numpy_rng_state",
        "validation_artifacts", "latest_validation_payload", "fine_tuning_phase",
        "trainability_state", "frozen_batchnorm_state", "optimizer_group_lrs",
        "optimizer_group_multipliers", "initialization_report", "initialization_report_file_sha256",
        "runner_sha256", "numerical_events", "sampler_generator_state",
        "torch_rng_state", "cuda_rng_states", "numerical_protocol", "finetuning_policy",
        "initialization_spec",
    }
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError("Stage24 checkpoint is missing required state")
    if (state["experiment"] != NAME or state["configuration"] != cfg or
            state["config_sha256"] != CONFIG_SHA or state["selection_rule_sha256"] != RULE_SHA or
            state["numerical_protocol_sha256"] != NUMERICAL_SHA or
            state["finetuning_policy_sha256"] != FINETUNING_POLICY_SHA or
            state["initialization_spec_sha256"] != INITIALIZATION_SPEC_SHA or
            state["reference_hashes"] != FROZEN_REFERENCE_HASHES or
            state["runner_sha256"] != sha256(Path(__file__)) or
            state["numerical_protocol"] != base.read_json(HERE / "numerical_protocol.json") or
            state["finetuning_policy"] != base.read_json(HERE / "finetuning_policy.json") or
            state["initialization_spec"] != base.read_json(HERE / "initialization_spec.json") or
            state["class_order"] != cfg["classes"] or state["architecture"] != cfg["architecture"]):
        raise ValueError("Stage24 checkpoint provenance/configuration mismatch")
    epoch = state["epoch"]
    history = state["history"]
    if type(epoch) is not int or not 1 <= epoch <= STAGE24_MAX_EPOCHS or len(history) != epoch:
        raise ValueError("Stage24 checkpoint epoch/history mismatch")
    if [r["epoch"] for r in history] != list(range(1, epoch + 1)):
        raise ValueError("Stage24 checkpoint history numbering mismatch")
    for row in history:
        expected_eligible = bool(row["eligible"])
        if expected_eligible != validate_metrics({
                "accuracy": row["val_accuracy"], "macro_f1": row["val_macro_f1"],
                "per_class": {"mel": {"support": rule["melanoma_support"], "tp": row["val_mel_tp"],
                                      "recall": row["val_mel_recall"], "f1": row["val_mel_f1"]},
                              "nv": {"recall": row["val_nv_recall"]}},
            }, rule):
            raise ValueError("Checkpoint history eligibility is inconsistent")
        if not math.isfinite(float(row["train_loss"])) or not math.isfinite(float(row["val_loss"])):
            raise ValueError("Stage24 checkpoint history has non-finite loss")
        if row["fine_tuning_phase"] != Stage24EGVAN.phase_for_epoch(row["epoch"]):
            raise ValueError("Checkpoint history phase label is inconsistent")
    selected_epoch, selected_loss = checkpoint_best(history, rule)
    if state["best_epoch"] != selected_epoch or state["best_validation_loss"] != selected_loss:
        raise ValueError("Checkpoint best selection violates registered validation rule")
    if state["fine_tuning_phase"] != Stage24EGVAN.phase_for_epoch(epoch):
        raise ValueError("Checkpoint active phase does not match its epoch")
    expected_phase = Stage24EGVAN.phase_for_epoch(epoch)
    expected_trainability = {
        "fine_tuning_phase": expected_phase,
        "conv1_trainable": False,
        "bn1_trainable": False,
        "layer1_trainable": False,
        "layer2_trainable": expected_phase == "B",
        "layer3_trainable": True,
        "layer4_trainable": True,
    }
    if state["trainability_state"] != expected_trainability:
        raise ValueError("Checkpoint trainability metadata differs from registered phase")
    if ("resnet.bn1" not in state["frozen_batchnorm_state"] or
            not all(state["frozen_batchnorm_state"].values())):
        raise ValueError("Checkpoint records a frozen BatchNorm in training mode")
    if not state["model_state"] or not finite_tree(state["model_state"]):
        raise ValueError("Stage24 checkpoint model state is missing or non-finite")
    if not finite_tree(state["optimizer_state"]) or not finite_tree(state["scaler_state"]):
        raise ValueError("Stage24 optimizer/scaler state is non-finite")
    validate_saved_scaler(state["scaler_state"])
    if not isinstance(state["numerical_events"], list) or not isinstance(state["validation_artifacts"], dict):
        raise ValueError("Stage24 checkpoint event/validation history is malformed")
    if set(state["validation_artifacts"]) != set(range(1, epoch + 1)):
        raise ValueError("Stage24 checkpoint validation artifact epochs are incomplete")
    if not isinstance(state["sampler_generator_state"], torch.Tensor) or not isinstance(state["torch_rng_state"], torch.Tensor):
        raise ValueError("Stage24 checkpoint sampler/torch RNG state is missing")
    if not isinstance(state["cuda_rng_states"], list) or not state["cuda_rng_states"]:
        raise ValueError("Stage24 checkpoint CUDA RNG state is missing")
    try:
        probe = random.Random()
        probe.setstate(state["python_rng_state"])
        probe = np.random.RandomState()
        probe.set_state(state["numpy_rng_state"])
        probe = torch.Generator(device="cpu")
        probe.set_state(state["sampler_generator_state"])
        probe.set_state(state["torch_rng_state"])
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Stage24 checkpoint RNG state is invalid") from error
    if state["initialization_report"]["weight_enum"] != "ResNet50_Weights.IMAGENET1K_V2":
        raise ValueError("Checkpoint initialization source differs from preregistration")
    if state["initialization_report_file_sha256"] != sha256(OUT / "initialization_report.json"):
        raise ValueError("Checkpoint initialization report hash mismatch")
    groups = state["optimizer_state"]["param_groups"]
    multipliers = [float(group["lr_multiplier"]) for group in groups]
    if multipliers != [1.0, 0.1, 0.1, 0.05] or len(groups) != 4:
        raise ValueError("Checkpoint optimizer group multipliers changed")
    if ([group.get("group_name") for group in groups] !=
            ["non_resnet_and_new_modules", "resnet.layer3", "resnet.layer4", "resnet.layer2"] or
            state["optimizer_group_multipliers"] != multipliers):
        raise ValueError("Checkpoint optimizer group names/metadata changed")
    group_lrs_saved = [float(group["lr"]) for group in groups]
    if len(state["optimizer_group_lrs"]) != 4 or any(
            not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
            for a, b in zip(group_lrs_saved, state["optimizer_group_lrs"])):
        raise ValueError("Checkpoint optimizer group LR record mismatch")
    if any(not math.isclose(lr, group_lrs_saved[0] * mult, rel_tol=1e-12, abs_tol=1e-15)
           for lr, mult in zip(group_lrs_saved, multipliers)):
        raise ValueError("Checkpoint optimizer group LR ratios changed")
    scheduler_state = state["scheduler_state"]
    # ReduceLROnPlateau legitimately stores +inf as mode_worse for mode=min.
    if (not finite_tree({k: v for k, v in scheduler_state.items() if k != "mode_worse"}) or
            scheduler_state.get("mode_worse") != math.inf or
            scheduler_state.get("last_epoch") != epoch or
            len(scheduler_state.get("_last_lr", [])) != 4 or
            any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)
                for a, b in zip(scheduler_state["_last_lr"], group_lrs_saved))):
        raise ValueError("Stage24 scheduler epoch differs from checkpoint epoch")
    expected_next_phase = Stage24EGVAN.phase_for_next_epoch(epoch) if epoch < STAGE24_MAX_EPOCHS else None
    return {"saved_epoch": epoch, "next_epoch": epoch + 1,
            "saved_phase": state["fine_tuning_phase"], "next_phase": expected_next_phase,
            "history_rows": len(history), "best_epoch": selected_epoch}


def finite_tree(value) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def reconcile_sidecars(state: dict, out_dir: Path) -> dict:
    """Checkpoint is authoritative: repair missing/behind sidecars, reject ahead/conflicting ones."""
    rows = state["history"]
    history_path = out_dir / "training_history.csv"
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            sidecar = list(reader)
        if len(sidecar) > len(rows):
            raise ValueError("Training history sidecar is ahead of authoritative checkpoint")
        if any(any(record.get(key) != str(value) for key, value in saved.items())
               for record, saved in zip(sidecar, rows)):
            raise ValueError("Training history sidecar conflicts with checkpoint prefix")
    atomic_write(history_path, write_csv_bytes(rows))
    events_path = out_dir / "numerical_events.json"
    if events_path.exists():
        current_events = json.loads(events_path.read_text(encoding="utf-8"))
        if len(current_events) > len(state["numerical_events"]):
            raise ValueError("Numerical event sidecar is ahead of authoritative checkpoint")
        if current_events != state["numerical_events"][:len(current_events)]:
            raise ValueError("Numerical event sidecar conflicts with checkpoint prefix")
    atomic_write(events_path, json_bytes(state["numerical_events"]))
    for epoch, digests in state["validation_artifacts"].items():
        predictions_path = out_dir / "validation_epochs" / f"epoch_{int(epoch):03d}_predictions.csv"
        metrics_path = out_dir / "validation_epochs" / f"epoch_{int(epoch):03d}_metrics.json"
        if predictions_path.exists() and sha256(predictions_path) != digests["predictions_sha256"]:
            raise ValueError(f"Validation prediction sidecar conflicts with checkpoint: {epoch}")
        if metrics_path.exists() and sha256(metrics_path) != digests["metrics_sha256"]:
            raise ValueError(f"Validation metric sidecar conflicts with checkpoint: {epoch}")
        if not predictions_path.exists() or not metrics_path.exists():
            if int(epoch) != state["epoch"]:
                raise FileNotFoundError(f"Earlier validation sidecar missing: epoch {epoch}")
            pred_bytes, metric_bytes = validation_bytes(
                state["latest_validation_payload"]["predictions"],
                state["latest_validation_payload"]["metrics"])
            atomic_write(predictions_path, pred_bytes)
            atomic_write(metrics_path, metric_bytes)
    best_path = out_dir / "best_checkpoint.pt"
    if state["best_epoch"] is not None:
        if best_path.is_file():
            best_state = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
            best_saved_epoch = best_state.get("epoch")
            if best_saved_epoch > state["epoch"]:
                raise ValueError("Best checkpoint is ahead of authoritative last checkpoint")
            if best_saved_epoch == state["best_epoch"]:
                if best_state.get("best_validation_loss") != state["best_validation_loss"]:
                    raise ValueError("Best checkpoint loss differs from authoritative checkpoint")
            elif state["best_epoch"] == state["epoch"]:
                base.save_checkpoint(best_path, state)
            else:
                raise ValueError("Best checkpoint does not match selected epoch in authoritative checkpoint")
        elif state["best_epoch"] == state["epoch"]:
            base.save_checkpoint(best_path, state)
        else:
            raise FileNotFoundError("Earlier selected best checkpoint missing; cannot recover from last epoch")
    elif best_path.exists():
        raise ValueError("Best checkpoint exists despite no eligible epoch")
    return {"history_reconciled_to_epoch": state["epoch"], "event_count": len(state["numerical_events"])}


def verify_sidecars_before_resume(state: dict) -> None:
    for filename, expected in (
        ("config.json", state["configuration"]),
        ("selection_rule.json", base.read_json(HERE / "selection_rule.json")),
        ("numerical_protocol.json", base.read_json(HERE / "numerical_protocol.json")),
        ("finetuning_policy.json", base.read_json(HERE / "finetuning_policy.json")),
        ("initialization_spec.json", base.read_json(HERE / "initialization_spec.json")),
    ):
        path = OUT / filename
        if not path.is_file() or base.read_json(path) != expected:
            raise ValueError(f"Stage24 run snapshot changed: {filename}")
    init_report = OUT / "initialization_report.json"
    if (not init_report.is_file() or
            sha256(init_report) != state["initialization_report_file_sha256"] or
            base.read_json(init_report) != state["initialization_report"]):
        raise ValueError("Stage24 initialization provenance changed")
    history_path = OUT / "training_history.csv"
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if (len(rows) > len(state["history"]) or
                    reader.fieldnames != list(state["history"][0]) or
                    any(any(row[key] != str(saved[key]) for key in reader.fieldnames)
                        for row, saved in zip(rows, state["history"]))):
                raise ValueError("Training history sidecar is ahead of or conflicts with checkpoint")
    events_path = OUT / "numerical_events.json"
    if events_path.exists():
        sidecar = base.read_json(events_path)
        if (len(sidecar) > len(state["numerical_events"]) or
                sidecar != state["numerical_events"][:len(sidecar)]):
            raise ValueError("Numerical event sidecar is ahead of or conflicts with checkpoint")
    for epoch, digests in state["validation_artifacts"].items():
        prediction_path = OUT / "validation_epochs" / f"epoch_{int(epoch):03d}_predictions.csv"
        metrics_path = OUT / "validation_epochs" / f"epoch_{int(epoch):03d}_metrics.json"
        if prediction_path.exists() and sha256(prediction_path) != digests["predictions_sha256"]:
            raise ValueError(f"Validation predictions conflict with checkpoint epoch {epoch}")
        if metrics_path.exists() and sha256(metrics_path) != digests["metrics_sha256"]:
            raise ValueError(f"Validation metrics conflict with checkpoint epoch {epoch}")
        if (int(epoch) != state["epoch"] and
                (not prediction_path.is_file() or not metrics_path.is_file())):
            raise FileNotFoundError(f"Earlier validation sidecar missing at epoch {epoch}")
    validation_dir = OUT / "validation_epochs"
    if validation_dir.exists():
        for path in validation_dir.iterdir():
            if not path.is_file() or not path.name.startswith("epoch_"):
                continue
            try:
                recorded_epoch = int(path.name.split("_")[1])
            except (IndexError, ValueError) as error:
                raise ValueError(f"Unrecognized validation sidecar: {path.name}") from error
            if recorded_epoch > state["epoch"]:
                raise ValueError(f"Validation sidecar is ahead of authoritative checkpoint: {path.name}")
    latest_prediction_bytes, latest_metrics_bytes = validation_bytes(
        state["latest_validation_payload"]["predictions"],
        state["latest_validation_payload"]["metrics"])
    latest_hashes = state["validation_artifacts"][state["epoch"]]
    if (hashlib.sha256(latest_prediction_bytes).hexdigest() != latest_hashes["predictions_sha256"] or
            hashlib.sha256(latest_metrics_bytes).hexdigest() != latest_hashes["metrics_sha256"]):
        raise ValueError("Checkpoint latest validation payload differs from saved artifact hashes")
    best_path = OUT / "best_checkpoint.pt"
    if state["best_epoch"] is None:
        if best_path.exists():
            raise ValueError("Best checkpoint exists without an eligible epoch")
    elif best_path.is_file():
        best = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
        if (best.get("epoch") != state["best_epoch"] or
                best.get("best_validation_loss") != state["best_validation_loss"] or
                not finite_tree(best.get("model_state", {}))):
            raise ValueError("Existing best checkpoint conflicts with authoritative checkpoint")
    elif state["best_epoch"] != state["epoch"]:
        raise FileNotFoundError("Earlier best checkpoint missing and cannot be recovered from last checkpoint")


def prepare_resume_state(state: dict, model, optimizer, scheduler, scaler, sampler_generator, cfg: dict, rule: dict) -> dict:
    resume = validate_checkpoint_state(state, cfg, rule)
    model.load_state_dict(state["model_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["scaler_state"])
    sampler_generator.set_state(state["sampler_generator_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    phase = resume["next_phase"]
    if phase is not None:
        model.set_phase(resume["next_epoch"])
    expected_lrs = maintain_lr_ratios(optimizer)
    recorded = [float(x) for x in state["optimizer_group_lrs"]]
    if any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15) for a, b in zip(expected_lrs, recorded)):
        raise ValueError("Restored optimizer group LRs differ from checkpoint")
    assert_optimizer_layout(model, optimizer)
    if phase is not None and model.trainability_state()["layer2_trainable"] != (phase == "B"):
        raise ValueError("Resume phase does not match layer2 trainability")
    if float(scaler.get_scale()) != float(state["scaler_state"]["scale"]):
        raise ValueError("Resume reset the GradScaler scale")
    return resume


def build_model_and_optimizer(root: Path, cfg: dict, *, output_dir: Path, resume_state: dict | None = None):
    _, Dataset, _, _, _, make_transforms, set_seed = base.imports(root)
    set_seed(cfg["seed"])
    efficient_weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    train_tf, eval_tf = make_transforms(efficient_weights)
    split_path = root / cfg["data"]["split_csv"]
    images_path = root / "data/processed/images"
    train_dataset = Dataset(images_path, split_path, "train", train_tf)
    val_dataset = Dataset(images_path, split_path, "val", eval_tf)
    if len(train_dataset) != cfg["data"]["train_count"] or len(val_dataset) != cfg["data"]["validation_count"]:
        raise ValueError("Stage24 train/validation counts differ from preregistration")
    generator = torch.Generator(device="cpu").manual_seed(cfg["sampler"]["seed"])
    stage23_cfg = base.variant_config("B")
    if stage23_cfg["sampler"] != cfg["sampler"]:
        raise ValueError("Stage24 sampler differs from frozen Stage23 sampler")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["physical_batch_size"],
        sampler=base.make_sampler(train_dataset, cfg["sampler"]["mel_weight"], generator),
        num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["physical_batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True)
    fresh = resume_state is None
    model = Stage24EGVAN(
        num_classes=len(cfg["classes"]),
        pretrained_efficient=fresh,
        pretrained_resnet=False,
    ).cuda()
    if fresh:
        source_state = WEIGHTS.get_state_dict(progress=True, check_hash=True)
        from initialization import apply_resnet_state
        loaded = apply_resnet_state(model.resnet, source_state)
        init_report = {
            **loaded,
            "experiment": NAME,
            "weight_enum": f"ResNet50_Weights.{WEIGHTS.name}",
            "torchvision_version": torchvision.__version__,
            "weight_url": WEIGHTS.url,
            "download_hash_check": True,
            "seed": cfg["seed"],
            "project_specific_dermoscopy_pretraining": False,
            "stage23_trained_checkpoint_loaded": False,
            "stage23_selected_checkpoint_hash_reference_only": STAGE23_CHECKPOINT_SHA,
            "ham_test_accessed": False,
            "ph2_accessed": False,
        }
        init_report["sha256"] = hashlib.sha256(
            json.dumps(init_report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        write_once(output_dir / "initialization_report.json", json_bytes(init_report))
    else:
        init_report = resume_state["initialization_report"]
        if not (output_dir / "initialization_report.json").is_file():
            raise FileNotFoundError("Resume requires the saved Stage24 initialization report")
        if sha256(output_dir / "initialization_report.json") != resume_state["initialization_report_file_sha256"]:
            raise ValueError("Stage24 initialization report hash mismatch on resume")
    model.set_phase(1)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)
    assert_optimizer_layout(model, optimizer)
    scaler = torch.amp.GradScaler("cuda", init_scale=INITIAL_SCALE)
    return model, optimizer, scheduler, scaler, train_loader, val_loader, generator, init_report


def make_checkpoint_state(epoch: int, cfg: dict, rule: dict, numerical: dict,
                          policy: dict, init_spec: dict, model, optimizer,
                          scheduler, scaler, history: list[dict], best_epoch: int | None,
                          best_loss: float, events: list[dict], validation_artifacts: dict,
                          predictions: list[dict], metrics: dict, sampler_generator,
                          init_report: dict, init_report_file_sha: str) -> dict:
    phase_state = model.trainability_state()
    if phase_state["fine_tuning_phase"] != Stage24EGVAN.phase_for_epoch(epoch):
        raise ValueError("Model phase does not match checkpoint epoch")
    return {
        "experiment": NAME,
        "epoch": epoch,
        "configuration": copy.deepcopy(cfg),
        "config_sha256": CONFIG_SHA,
        "selection_rule_sha256": RULE_SHA,
        "numerical_protocol_sha256": NUMERICAL_SHA,
        "finetuning_policy_sha256": FINETUNING_POLICY_SHA,
        "initialization_spec_sha256": INITIALIZATION_SPEC_SHA,
        "reference_hashes": copy.deepcopy(FROZEN_REFERENCE_HASHES),
        "runner_sha256": sha256(Path(__file__)),
        "architecture": cfg["architecture"],
        "architecture_metadata": {
            "model_class": "Stage24EGVAN( EG-VAN EGVAN subclass; unchanged module/state-dict topology )",
            "resnet_stage_modules": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
            "scga_modules": ["scga1", "scga2"],
            "nonlocal_modules": ["nonlocal3", "nonlocal4"],
            "mff_count": 4,
            "class_order": list(cfg["classes"]),
        },
        "class_order": list(cfg["classes"]),
        "numerical_protocol": copy.deepcopy(numerical),
        "finetuning_policy": copy.deepcopy(policy),
        "initialization_spec": copy.deepcopy(init_spec),
        "initialization_report": copy.deepcopy(init_report),
        "initialization_report_file_sha256": init_report_file_sha,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": copy.deepcopy(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "numerical_events": copy.deepcopy(events),
        "validation_artifacts": copy.deepcopy(validation_artifacts),
        "latest_validation_payload": {"predictions": copy.deepcopy(predictions),
                                       "metrics": copy.deepcopy(metrics)},
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
        "sampler_generator_state": sampler_generator.get_state(),
        "fine_tuning_phase": phase_state["fine_tuning_phase"],
        "trainability_state": phase_state,
        "frozen_batchnorm_state": model.frozen_batchnorm_state(),
        "optimizer_group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "optimizer_group_multipliers": [float(group["lr_multiplier"]) for group in optimizer.param_groups],
    }


def verify_sidecars_before_resume(state: dict) -> None:
    snapshots = {
        "config.json": state["configuration"],
        "selection_rule.json": base.read_json(HERE / "selection_rule.json"),
        "numerical_protocol.json": base.read_json(HERE / "numerical_protocol.json"),
        "finetuning_policy.json": base.read_json(HERE / "finetuning_policy.json"),
        "initialization_spec.json": base.read_json(HERE / "initialization_spec.json"),
    }
    for name, expected in snapshots.items():
        path = OUT / name
        if not path.is_file() or base.read_json(path) != expected:
            raise ValueError(f"Stage24 run snapshot changed: {name}")
    report_path = OUT / "initialization_report.json"
    if (not report_path.is_file() or
            sha256(report_path) != state["initialization_report_file_sha256"] or
            base.read_json(report_path) != state["initialization_report"]):
        raise ValueError("Stage24 initialization provenance changed")
    history_path = OUT / "training_history.csv"
    if history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            if (len(rows) > len(state["history"]) or
                    reader.fieldnames != list(state["history"][0]) or
                    any(any(row[key] != str(saved[key]) for key in reader.fieldnames)
                        for row, saved in zip(rows, state["history"]))):
                raise ValueError("History sidecar is ahead of or conflicts with checkpoint")
    event_path = OUT / "numerical_events.json"
    if event_path.exists():
        sidecar = base.read_json(event_path)
        if (len(sidecar) > len(state["numerical_events"]) or
                sidecar != state["numerical_events"][:len(sidecar)]):
            raise ValueError("Numerical event sidecar is ahead of or conflicts with checkpoint")
    for epoch, digest in state["validation_artifacts"].items():
        predictions_path, metrics_path = epoch_paths(int(epoch))
        for path, key in ((predictions_path, "predictions_sha256"),
                          (metrics_path, "metrics_sha256")):
            if path.exists() and sha256(path) != digest[key]:
                raise ValueError(f"Validation sidecar conflicts with checkpoint epoch {epoch}")
            if int(epoch) != state["epoch"] and not path.is_file():
                raise FileNotFoundError(f"Earlier validation sidecar missing at epoch {epoch}")


def stage24_reference_metrics(root: Path) -> tuple[dict, dict]:
    stage15_path = root / "experiments/stage15_single_candidate_exp1/run/validation_epochs/epoch_016_metrics.json"
    stage23_path = root / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_metrics.json"
    if sha256(stage15_path) != STAGE15_METRICS_SHA or sha256(stage23_path) != STAGE23_SELECTED_METRICS_SHA:
        raise ValueError("Frozen Stage15/Stage23 validation comparison artifact changed")
    return base.read_json(stage15_path), base.read_json(stage23_path)


def finalize_run(root: Path, cfg: dict, rule: dict, numerical: dict, policy: dict,
                 init_spec: dict, history: list[dict], best_epoch: int | None,
                 best_loss: float, validation_artifacts: dict, init_report: dict,
                 init_report_file_sha: str) -> dict:
    if len(history) != STAGE24_MAX_EPOCHS or [r["epoch"] for r in history] != list(range(1, 26)):
        raise ValueError("Cannot finalize an incomplete Stage24 history")
    if best_epoch is None:
        best_checkpoint_for_finalization(None, OUT / "best_checkpoint.pt")
        selected_metrics = None
    else:
        best_checkpoint_for_finalization(best_epoch, OUT / "best_checkpoint.pt")
        best_state = torch.load(OUT / "best_checkpoint.pt", map_location="cpu", weights_only=False, mmap=True)
        if (best_state["epoch"] != best_epoch or best_state["best_epoch"] != best_epoch or
                best_state["best_validation_loss"] != best_loss):
            raise ValueError("Best checkpoint differs from selected epoch/history")
        pred_path, metrics_path = epoch_paths(best_epoch)
        selected_metrics = base.read_json(metrics_path)
        if (not selected_metrics["eligible"] or selected_metrics["validation_loss"] != best_loss or
                selected_metrics["failed_gates"]):
            raise ValueError("Selected validation record fails preregistered gates")
        write_once(OUT / "validation_predictions.csv", pred_path.read_bytes())
        write_once(OUT / "validation_metrics.json", metrics_path.read_bytes())
    stage15, stage23 = stage24_reference_metrics(root)
    comparison = compare_stage24(selected_metrics, stage23, stage15)
    write_once(OUT / "comparison.json", json_bytes(comparison))
    write_once(OUT / "stage15_stage23_stage24_comparison.json", json_bytes(comparison))
    status = "NO_CANDIDATE_SELECTED" if best_epoch is None else "CANDIDATE_SELECTED_VALIDATION_ONLY"
    artifacts = [
        "config.json", "selection_rule.json", "numerical_protocol.json",
        "finetuning_policy.json", "initialization_spec.json", "initialization_report.json",
        "training_history.csv", "numerical_events.json", "last_checkpoint.pt",
        "comparison.json", "stage15_stage23_stage24_comparison.json",
    ]
    if best_epoch is not None:
        artifacts += ["best_checkpoint.pt", "validation_predictions.csv", "validation_metrics.json"]
    for epoch in range(1, 26):
        pred_path, metrics_path = epoch_paths(epoch)
        artifacts += [str(pred_path.relative_to(OUT)).replace("\\", "/"),
                      str(metrics_path.relative_to(OUT)).replace("\\", "/")]
    missing = [name for name in artifacts if not (OUT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Finalization artifacts missing: {missing}")
    manifest = {
        "experiment": NAME,
        "status": status,
        "validation_decision": comparison["decision"],
        "epochs": len(history),
        "selected_epoch": best_epoch,
        "best_validation_loss": best_loss if best_epoch is not None else None,
        "config_sha256": CONFIG_SHA,
        "selection_rule_sha256": RULE_SHA,
        "numerical_protocol_sha256": NUMERICAL_SHA,
        "finetuning_policy_sha256": FINETUNING_POLICY_SHA,
        "initialization_spec_sha256": INITIALIZATION_SPEC_SHA,
        "initialization_report_file_sha256": init_report_file_sha,
        "reference_hashes": FROZEN_REFERENCE_HASHES,
        "split_sha256": SPLIT_SHA,
        "resnet_weight_enum": f"ResNet50_Weights.{WEIGHTS.name}",
        "initialization_report": init_report,
        "artifact_sha256": {name: sha256(OUT / name) for name in artifacts},
        "runtime": runtime(),
        "ham_test_accessed": False,
        "ph2_accessed": False,
    }
    atomic_write(OUT / "experiment_manifest.json", json_bytes(manifest))
    return manifest


def runtime() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true", help="read-only frozen-reference/data preflight")
    modes.add_argument("--train", action="store_true", help="start or resume the one Stage24 experiment")
    parser.add_argument("--resume", action="store_true", help="resume from authoritative last_checkpoint.pt")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if root != ROOT.resolve():
        raise ValueError("Use the pinned EG-VAN project root")
    if args.resume and not args.train:
        parser.error("--resume requires --train")
    cfg, rule, numerical, policy, init_spec = check_frozen_references(root)
    if args.check:
        print(json.dumps(preflight(root), indent=2, allow_nan=False))
        return
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Stage24 training requires a Tesla T4")
    if args.resume:
        if not (OUT / "last_checkpoint.pt").is_file():
            raise FileNotFoundError("--resume requires an existing last_checkpoint.pt; no fresh fallback")
        execute(root, cfg, rule, numerical, policy, init_spec, resume=True)
    else:
        if OUT.exists():
            raise FileExistsError("Fresh Stage24 run refuses an existing output directory")
        execute(root, cfg, rule, numerical, policy, init_spec, resume=False)


def execute(root: Path, cfg: dict, rule: dict, numerical: dict, policy: dict,
            init_spec: dict, *, resume: bool) -> None:
    saved = None
    if resume:
        if (OUT / "experiment_manifest.json").exists():
            raise FileExistsError("Stage24 run is finalized; resume is not permitted")
        saved = torch.load(OUT / "last_checkpoint.pt", map_location="cpu", weights_only=False)
        validate_checkpoint_state(saved, cfg, rule)
        verify_sidecars_before_resume(saved)
    else:
        OUT.mkdir(parents=True)
        for name in ("config.json", "selection_rule.json", "numerical_protocol.json",
                     "finetuning_policy.json", "initialization_spec.json"):
            write_once(OUT / name, (HERE / name).read_bytes())
    model, optimizer, scheduler, scaler, train_loader, val_loader, sampler_generator, init_report = \
        build_model_and_optimizer(root, cfg, output_dir=OUT, resume_state=saved)
    restore_policy = install_selective_qk_policy(model)
    history, best_epoch, best_loss = [], None, math.inf
    events, validation_artifacts = [], {}
    initialization_report_file_sha = sha256(OUT / "initialization_report.json")
    ready = not resume
    try:
        if resume:
            resume_info = prepare_resume_state(saved, model, optimizer, scheduler, scaler,
                                               sampler_generator, cfg, rule)
            history = copy.deepcopy(saved["history"])
            events = copy.deepcopy(saved["numerical_events"])
            validation_artifacts = copy.deepcopy(saved["validation_artifacts"])
            best_epoch, best_loss = saved["best_epoch"], saved["best_validation_loss"]
            reconcile_sidecars(saved, OUT)
            ready = True
            print(f"Stage24 resume saved_epoch={saved['epoch']} next_epoch={resume_info['next_epoch']} "
                  f"phase={resume_info['next_phase']}", flush=True)
            del saved
        start_epoch = len(history) + 1
        for epoch in range(start_epoch, STAGE24_MAX_EPOCHS + 1):
            trainability = model.set_phase(epoch)
            train_sum = train_count = optimizer_steps = 0
            for batch, (images_cpu, labels_cpu) in enumerate(train_loader, start=1):
                value, stepped, event = checked_train_batch(
                    model, optimizer, scaler,
                    images_cpu.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True),
                    NAME, epoch, batch, cfg["loss"]["mel_multiplier"])
                train_sum += value * len(labels_cpu)
                train_count += len(labels_cpu)
                optimizer_steps += int(stepped)
                if event is not None:
                    events.append(event)
            if optimizer_steps == 0:
                raise RuntimeError(f"No optimizer step occurred in epoch {epoch}")
            train_loss = train_sum / train_count
            val_loss, val_metrics, predictions = guard.guarded_validation_epoch(
                model, val_loader, NAME, epoch, predictions=True)
            guard.require_finite_states(model, optimizer, {"experiment": NAME, "epoch": epoch})
            guard.require_finite_epoch_losses(train_loss, val_loss, NAME, epoch)
            evidence = validate_epoch_evidence(epoch, val_loss, val_metrics, predictions, val_loader, rule)
            prediction_bytes, metric_bytes = validation_bytes(predictions, evidence)
            validation_artifacts[epoch] = {
                "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
                "metrics_sha256": hashlib.sha256(metric_bytes).hexdigest(),
            }
            learning_rates = apply_scheduler_step(scheduler, optimizer, val_loss)
            if evidence["eligible"] != validate_metrics({
                    "accuracy": val_metrics["accuracy"], "macro_f1": val_metrics["macro_f1"],
                    "per_class": evidence["per_class"]}, rule):
                raise ValueError("Saved eligibility differs from frozen Stage23 gates")
            phase = trainability["fine_tuning_phase"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_precision": evidence["macro_precision"],
                "val_macro_recall": evidence["macro_recall"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": evidence["weighted_f1"],
                "val_mel_precision": evidence["per_class"]["mel"]["precision"],
                "val_mel_recall": evidence["per_class"]["mel"]["recall"],
                "val_mel_f1": evidence["per_class"]["mel"]["f1"],
                "val_mel_tp": evidence["per_class"]["mel"]["tp"],
                "val_mel_fn": evidence["per_class"]["mel"]["fn"],
                "val_nv_recall": evidence["per_class"]["nv"]["recall"],
                "eligible": evidence["eligible"],
                "fine_tuning_phase": phase,
                "layer2_trainable": trainability["layer2_trainable"],
                "layer3_trainable": trainability["layer3_trainable"],
                "layer4_trainable": trainability["layer4_trainable"],
                "conv1_trainable": trainability["conv1_trainable"],
                "layer1_trainable": trainability["layer1_trainable"],
                "lr_non_resnet": learning_rates[0],
                "lr_layer3": learning_rates[1],
                "lr_layer4": learning_rates[2],
                "lr_layer2": learning_rates[3],
                "optimizer_steps": optimizer_steps,
                "amp_skips": sum(event["epoch"] == epoch for event in events),
            }
            history.append(row)
            best_epoch, best_loss = checkpoint_best(history, rule)
            state = make_checkpoint_state(
                epoch, cfg, rule, numerical, policy, init_spec, model, optimizer,
                scheduler, scaler, history, best_epoch, best_loss, events,
                validation_artifacts, predictions, evidence, sampler_generator,
                init_report, initialization_report_file_sha)
            atomic_checkpoint(OUT / "last_checkpoint.pt", state, model, optimizer, scaler, epoch)
            if evidence["eligible"] and best_epoch == epoch:
                atomic_checkpoint(OUT / "best_checkpoint.pt", state, model, optimizer, scaler, epoch)
            pred_path, metrics_path = epoch_paths(epoch)
            write_once(pred_path, prediction_bytes)
            write_once(metrics_path, metric_bytes)
            atomic_write(OUT / "training_history.csv", write_csv_bytes(history))
            atomic_write(OUT / "numerical_events.json", json_bytes(events))
            print(f"Stage24 epoch={epoch}/{STAGE24_MAX_EPOCHS} phase={phase} loss={val_loss:.6f} "
                  f"eligible={evidence['eligible']} group_lrs={learning_rates}", flush=True)
        finalize_run(root, cfg, rule, numerical, policy, init_spec, history, best_epoch, best_loss,
                     validation_artifacts, init_report, initialization_report_file_sha)
    except Exception as exc:
        if ready:
            record = {"experiment": NAME, "status": "STOPPED", "failure": str(exc),
                      "completed_epochs": len(history), "validation_decision": "EXPERIMENT_INVALID",
                      "ham_test_accessed": False, "ph2_accessed": False}
            atomic_write(OUT / "failure.json", json_bytes(record))
        raise
    finally:
        restore_policy()
        del model, optimizer, scheduler, scaler, train_loader, val_loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
