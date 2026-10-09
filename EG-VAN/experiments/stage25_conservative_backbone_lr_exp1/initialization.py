"""Stage25 independent torchvision ResNet50 ImageNet V2 initialization helpers."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
import torchvision
from torchvision.models import ResNet50_Weights

RESNET_WEIGHTS = ResNet50_Weights.IMAGENET1K_V2
RESNET_LOADER_NAME = "ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True, check_hash=True)"
BACKBONE_PREFIXES = ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")
MODEL_TRUNK_PREFIXES = tuple("resnet." + prefix for prefix in BACKBONE_PREFIXES)
CLASSIFIER_KEYS = {"fc.weight", "fc.bias"}


def apply_resnet_state(model, source_state: dict) -> dict:
    target_state = model.state_dict()
    expected_backbone = {key for key in target_state if key.startswith(BACKBONE_PREFIXES)}
    source_keys = set(source_state)
    missing = sorted(expected_backbone - source_keys)
    unexpected_backbone = sorted(key for key in source_keys
                                 if key.startswith(BACKBONE_PREFIXES) and key not in expected_backbone)
    intentionally_ignored = sorted(source_keys - expected_backbone)
    shape_mismatches = [
        {"key": key, "source": list(source_state[key].shape), "target": list(target_state[key].shape)}
        for key in sorted(expected_backbone & source_keys)
        if tuple(source_state[key].shape) != tuple(target_state[key].shape)
    ]
    unexpected_source = sorted(set(intentionally_ignored) - CLASSIFIER_KEYS)
    if (missing or unexpected_backbone or unexpected_source or
            set(intentionally_ignored) != CLASSIFIER_KEYS or shape_mismatches):
        raise ValueError(json.dumps({"missing_backbone_keys": missing,
                                     "unexpected_backbone_keys": unexpected_backbone,
                                     "unexpected_source_keys": unexpected_source,
                                     "intentionally_ignored_source_keys": intentionally_ignored,
                                     "shape_mismatches": shape_mismatches}, sort_keys=True))
    incompatible = model.load_state_dict(source_state, strict=False)
    expected_new = sorted(set(target_state) - expected_backbone)
    if (sorted(incompatible.missing_keys) != expected_new or
            set(incompatible.unexpected_keys) != CLASSIFIER_KEYS):
        raise ValueError("torchvision state-dict load returned unexpected missing/unexpected keys")
    parameters = dict(model.named_parameters())
    loaded_parameter_keys = sorted(key for key in expected_backbone if key in parameters)
    loaded_buffer_keys = sorted(expected_backbone - set(parameters))
    return {
        "status": "LOADED",
        "loaded_backbone_keys": sorted(expected_backbone),
        "loaded_parameter_keys": loaded_parameter_keys,
        "loaded_parameter_tensor_count": len(loaded_parameter_keys),
        "loaded_parameter_numel": sum(parameters[key].numel() for key in loaded_parameter_keys),
        "loaded_buffer_keys": loaded_buffer_keys,
        "loaded_buffer_tensor_count": len(loaded_buffer_keys),
        "intentionally_not_loaded_source_keys": sorted(CLASSIFIER_KEYS),
        "new_module_keys": expected_new,
        "missing_backbone_keys": [],
        "unexpected_backbone_keys": [],
        "unexpected_source_keys": [],
        "shape_mismatches": [],
    }


def _write_once(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(contents).hexdigest()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing initialization report differs: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".stage25_init_", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_bytes(contents)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_initialization_report(path: Path, report: dict) -> tuple[dict, str]:
    report = copy.deepcopy(report)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_once(path, encoded)
    return report, hashlib.sha256(encoded).hexdigest()


def build_two_group_optimizer(model, cfg: dict):
    """Assign every trainable tensor exactly once, without altering trainability."""
    named = list(model.named_parameters())
    if not named or any(not parameter.requires_grad for _, parameter in named):
        raise ValueError("Stage25 requires every model parameter trainable from epoch 1")
    trunk = [(name, parameter) for name, parameter in named
             if name.startswith(MODEL_TRUNK_PREFIXES)]
    other = [(name, parameter) for name, parameter in named
             if not name.startswith(MODEL_TRUNK_PREFIXES)]
    if not trunk or not other:
        raise ValueError("Stage25 optimizer group is empty")
    ids = [id(parameter) for _, parameter in trunk + other]
    if len(ids) != len(set(ids)) or set(ids) != {id(parameter) for _, parameter in named}:
        raise ValueError("Stage25 optimizer groups duplicate or omit model parameters")
    for prefix in MODEL_TRUNK_PREFIXES:
        if not any(name.startswith(prefix) for name, _ in trunk):
            raise ValueError(f"Stage25 pretrained trunk prefix missing: {prefix}")
    settings = cfg["optimizer"]
    declared = settings["parameter_groups"]
    if ([(group["name"], group["initial_learning_rate"], group["lr_multiplier"])
         for group in declared] !=
            [("new_and_non_resnet", 0.001, 1.0),
             ("pretrained_resnet_trunk", 0.0001, 0.1)] or
            tuple(declared[1]["prefixes"]) != MODEL_TRUNK_PREFIXES):
        raise ValueError("Stage25 registered optimizer groups changed")
    optimizer = torch.optim.Adamax([
        {"params": [parameter for _, parameter in other], "lr": 0.001,
         "group_name": "new_and_non_resnet"},
        {"params": [parameter for _, parameter in trunk], "lr": 0.0001,
         "group_name": "pretrained_resnet_trunk"},
    ], betas=tuple(settings["betas"]), eps=settings["eps"],
       weight_decay=settings["weight_decay"])
    scheduler_cfg = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_cfg["mode"], factor=scheduler_cfg["factor"],
        patience=scheduler_cfg["patience"])
    return optimizer, scheduler


def scheduler_step_preserving_ratio(scheduler, optimizer, validation_loss: float) -> None:
    scheduler.step(validation_loss)
    group_new, group_trunk = optimizer.param_groups
    group_trunk["lr"] = group_new["lr"] * 0.1
    scheduler._last_lr = [group_new["lr"], group_trunk["lr"]]
    if not (group_new["lr"] > 0 and group_trunk["lr"] > 0 and
            group_trunk["lr"] == group_new["lr"] * 0.1):
        raise ValueError("Stage25 scheduler broke registered 10:1 LR ratio")


def build_stage25_components(root: Path, cfg: dict, *, base, output_dir: Path, initial_scale: float):
    _, Dataset, _, EGVAN, _, make_transforms, set_seed = base.imports(root)
    set_seed(cfg["seed"])
    efficient_weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    train_tf, eval_tf = make_transforms(efficient_weights)
    split = root / cfg["data"]["split_csv"]
    images = root / "data/processed/images"
    train_ds = Dataset(images, split, "train", train_tf)
    val_ds = Dataset(images, split, "val", eval_tf)
    generator = torch.Generator(device="cpu").manual_seed(cfg["sampler"]["seed"])
    source_cfg = base.variant_config("B")
    if source_cfg["sampler"] != cfg["sampler"]:
        raise ValueError("Stage23 sampler differs from the frozen Stage15 variant-B sampler")
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["physical_batch_size"],
        sampler=base.make_sampler(train_ds, cfg["sampler"]["mel_weight"], generator),
        num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["physical_batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True)
    model = EGVAN(num_classes=len(cfg["classes"]), pretrained_efficient=True,
                  pretrained_resnet=False).cuda()
    source_state = RESNET_WEIGHTS.get_state_dict(progress=True, check_hash=True)
    loaded = apply_resnet_state(model.resnet, source_state)
    report, report_sha = write_initialization_report(output_dir / "initialization_report.json", {
        **loaded,
        "experiment": cfg["experiment"],
        "weight_enum": f"ResNet50_Weights.{RESNET_WEIGHTS.name}",
        "torchvision_version": torchvision.__version__,
        "weight_url": RESNET_WEIGHTS.url,
        "download_hash_check": True,
        "initialization_seed": cfg["seed"],
        "efficientnet_initialization": cfg["initialization"]["efficientnet"],
        "architecture": cfg["architecture"],
        "class_order": cfg["classes"],
        "project_specific_dermoscopy_pretraining": False,
        "ham_test_accessed": False,
        "ph2_accessed": False,
    })
    optimizer, scheduler = build_two_group_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", init_scale=initial_scale)
    return model, optimizer, scheduler, scaler, train_loader, val_loader, generator, report, report_sha


def best_checkpoint_for_finalization(best_epoch: int | None, path: Path) -> Path | None:
    if best_epoch is None:
        if path.exists():
            raise ValueError("Unexpected best checkpoint when no epoch qualified")
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Selected best checkpoint is missing: {path}")
    return path
