"""Frozen Stage 16 evaluation contracts. Preflight does not decode images."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_REL = "experiments/stage15_single_candidate_exp1/stage15_single_candidate_exp1/recovery_epoch16/best_checkpoint.pt"
CHECKPOINT_SHA = "85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5"
SPLIT_REL = "data/splits/split_leakage_aware.csv"
SPLIT_SHA = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
PH2_REL = "analysis/ph2_external_validation_exp5/ph2_manifest.csv"
PH2_SHA = "3c130b3b44dcc1052ae29656510a95a4e91df4b8f65c121838941efb375d3842"
PH2_PROTOCOL_SOURCE_HASHES = {
    "analysis/egvan_ph2_external_followup_exp1/evaluate_stage11.py":
        "d154ba3e02feef7c8ad1547f2f72a5c66ff91c883db32820149730e543fadb3a",
    "analysis/egvan_ph2_external_followup_exp1/stage11_config.json":
        "f750c47a441703ba49b96fc2a0faa7ff35f77f6f031361cb04ed2a0543826d9f",
    "analysis/ph2_external_validation_exp5/external_validation_config.json":
        "410162f26af0ff11d54ba943436c3573d43f91cdba767d790f9236aa7540dcf8",
    "src/preprocessing.py": "cf0a176a610b52ba45b0e91761b027a560252234f81f7b6fd572f02d8baab6ed",
}
CONFIG_SHA = "3ad8c205e06cde8203a869be0a6cc9643ed0ca5b6beb553b57d53d4dce9d25f2"
RULE_SHA = "daf07a1bac3538b5a35ed59121bd5561aba4797a6889e852fd0ca07ad88be735"
NUMERICAL_SHA = "f59010b7e2509c0bf1633b7126ff1d8d629afab39b515f57e95012fb78eb0196"
GATE_SHA = "721eeaf6c64740d7dd8d7cdb06890a2ffa1c1be49c336f4d07033e4d172cdb00"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
ARCHITECTURE = "stage8b_egvan_four_mff_serial_terminal_branches"
TEST_COUNTS = {"akiec": 40, "bcc": 58, "bkl": 104, "df": 11, "mel": 107, "nv": 676, "vasc": 18}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def preflight(root: Path) -> dict:
    import torch
    root = root.resolve()
    checkpoint = root / CHECKPOINT_REL
    split = root / SPLIT_REL
    if sha256(checkpoint) != CHECKPOINT_SHA or sha256(split) != SPLIT_SHA:
        raise ValueError("Frozen checkpoint or HAM split SHA256 mismatch")
    prereg = root / "experiments/stage15_single_candidate_exp1"
    for filename, expected in (("config.json", CONFIG_SHA), ("selection_rule.json", RULE_SHA),
                               ("numerical_protocol.json", NUMERICAL_SHA)):
        if sha256(prereg / filename) != expected:
            raise ValueError(f"Frozen Stage 15 {filename} changed")
    gate_path = root / "analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.json"
    if sha256(gate_path) != GATE_SHA:
        raise ValueError("Frozen selective-FP32 bounded gate changed")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    policy_source = root / "analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.py"
    if sha256(policy_source) != gate["gate_script_sha256"]:
        raise ValueError("Selective q@k FP32 policy source changed")
    for relative, digest in gate["source_sha256"].items():
        if sha256(root / relative) != digest:
            raise ValueError(f"Frozen model or preprocessing source changed: {relative}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if (state["epoch"] != 16 or state["best_epoch"] != 16 or
        state["experiment"] != "stage15_single_candidate_exp1" or
        state["architecture"] != ARCHITECTURE or tuple(state["class_order"]) != CLASSES or
        state["config_sha256"] != CONFIG_SHA or state["selection_rule_sha256"] != RULE_SHA or
        state["numerical_protocol_sha256"] != NUMERICAL_SHA or
        state["configuration"] != json.loads((prereg / "config.json").read_text(encoding="utf-8"))):
        raise ValueError("Frozen checkpoint metadata/configuration mismatch")
    if any(not bool(torch.isfinite(value).all()) for value in state["model_state"].values()
           if value.is_floating_point() or value.is_complex()):
        raise ValueError("Non-finite model parameter or buffer")
    del state
    rows = read_csv(split)
    counts = Counter(row["split"] for row in rows)
    if len(rows) != 10015 or counts != {"train": 8015, "val": 986, "test": 1014}:
        raise ValueError("HAM split counts changed")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate HAM image ID")
    lesions = defaultdict(set)
    for row in rows:
        if row["dx"] not in CLASSES:
            raise ValueError("Unknown HAM class")
        lesions[row["lesion_id"]].add(row["split"])
    if any(len(parts) != 1 for parts in lesions.values()):
        raise ValueError("HAM lesion crosses partitions")
    test = [row for row in rows if row["split"] == "test"]
    if dict(Counter(row["dx"] for row in test)) != TEST_COUNTS:
        raise ValueError("HAM test class counts changed")
    if any(not (root / "data/processed/images" / f"{row['image_id']}.jpg").is_file() for row in test):
        raise FileNotFoundError("Frozen HAM test image missing")
    return {"checkpoint_sha256": CHECKPOINT_SHA, "checkpoint_epoch": 16,
            "split_sha256": SPLIT_SHA, "split_counts": dict(counts),
            "test_class_counts": TEST_COUNTS, "class_order": list(CLASSES),
            "numerical_protocol_sha256": NUMERICAL_SHA, "status": "PASS"}


def ph2_preflight(root: Path) -> dict:
    root = root.resolve()
    common = preflight(root)
    manifest = root / PH2_REL
    if sha256(manifest) != PH2_SHA:
        raise ValueError("Frozen PH2 cohort manifest changed")
    for relative, digest in PH2_PROTOCOL_SOURCE_HASHES.items():
        if sha256(root / relative) != digest:
            raise ValueError(f"Frozen PH2 protocol source changed: {relative}")
    protocol = json.loads((root / "analysis/egvan_ph2_external_followup_exp1/stage11_config.json").read_text(encoding="utf-8"))
    if (protocol["ph2_cohort_manifest_sha256"] != PH2_SHA or protocol["included"] != 120 or
        protocol["excluded_atypical"] != 80 or protocol["true_ham_counts"] != {"nv": 80, "mel": 40} or
        tuple(protocol["class_order"]) != CLASSES):
        raise ValueError("Frozen PH2 mapping protocol changed")
    rows = read_csv(manifest)
    if len(rows) != 200 or len({r["image_id"] for r in rows}) != 200:
        raise ValueError("PH2 cohort count or IDs changed")
    mapping = {"common nevus": ("True", "nv"), "melanoma": ("True", "mel"),
               "atypical nevus": ("False", "")}
    for row in rows:
        if (row["true_external_label"] not in mapping or
            (row["included"], row["true_ham_label"]) != mapping[row["true_external_label"]]):
            raise ValueError("PH2 frozen mapping changed")
    included = [r for r in rows if r["included"] == "True"]
    if (dict(Counter(r["true_ham_label"] for r in included)) != {"nv": 80, "mel": 40} or
        any(not (root / r["relative_image_path"]).is_file() for r in included)):
        raise ValueError("PH2 mapped cohort or source images changed")
    return {**common, "ph2_manifest_sha256": PH2_SHA, "ph2_total": 200,
            "ph2_included": 120, "ph2_excluded_atypical": 80,
            "ph2_mapping": {"common nevus": "nv", "melanoma": "mel", "atypical nevus": None},
            "ph2_prior_project_use": True}


def load_model(root: Path):
    import torch
    from torchvision.models import EfficientNet_V2_S_Weights
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "analysis/stage13_melanoma_ablation"))
    from models.egvan import EGVAN
    from train import make_transforms
    from stage13c_abc_selective_qk_fp32_gate import install_selective_qk_policy
    state = torch.load(root / CHECKPOINT_REL, map_location="cpu", weights_only=False, mmap=True)
    model = EGVAN(num_classes=7, pretrained_efficient=False, pretrained_resnet=False)
    model.load_state_dict(state["model_state"], strict=True)
    del state
    install_selective_qk_policy(model)
    model.cuda().eval()
    _, transform = make_transforms(EfficientNet_V2_S_Weights.DEFAULT)
    if ([type(t).__name__ for t in transform.transforms] != ["Resize", "ToTensor", "Normalize"] or
        tuple(transform.transforms[0].size) != (384, 384)):
        raise ValueError("Frozen evaluation transform drift")
    return model, transform


def runtime() -> dict:
    import torch
    import torchvision
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def require_t4() -> None:
    import torch
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Frozen Stage 16 inference requires Tesla T4")


def source_hashes(root: Path, script: Path) -> dict[str, str]:
    paths = [script, Path(__file__), root / CHECKPOINT_REL, root / SPLIT_REL,
             root / "experiments/stage15_single_candidate_exp1/config.json",
             root / "experiments/stage15_single_candidate_exp1/selection_rule.json",
             root / "experiments/stage15_single_candidate_exp1/numerical_protocol.json",
             root / "src/models/egvan/egvan.py", root / "src/dataset.py",
             root / "src/train.py",
             root / "analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.py"]
    return {str(path.relative_to(root)): sha256(path) for path in paths}
