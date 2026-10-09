"""Stage24-only frozen-BN and staged ResNet50 optimizer helpers."""
from __future__ import annotations

from torch import nn
from torch.optim import Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models.egvan.egvan import EGVAN

PHASE_A_LAST_EPOCH = 5
MAX_EPOCHS = 25
LR_MULTIPLIERS = {
    "non_resnet_and_new_modules": 1.0,
    "resnet.layer3": 0.1,
    "resnet.layer4": 0.1,
    "resnet.layer2": 0.05,
}


class Stage24EGVAN(EGVAN):
    """Unchanged EGVAN topology with phase-aware trainability and frozen BN modes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage24_epoch = 1
        self.set_phase(1)

    @staticmethod
    def phase_for_epoch(epoch: int) -> str:
        if type(epoch) is not int or not 1 <= epoch <= MAX_EPOCHS:
            raise ValueError(f"Epoch must be in [1, {MAX_EPOCHS}]")
        return "A" if epoch <= PHASE_A_LAST_EPOCH else "B"

    @classmethod
    def phase_for_next_epoch(cls, saved_epoch: int) -> str:
        if type(saved_epoch) is not int or not 0 <= saved_epoch <= MAX_EPOCHS:
            raise ValueError("Saved epoch is outside the Stage24 range")
        if saved_epoch == MAX_EPOCHS:
            raise ValueError("No epoch remains after a completed Stage24 checkpoint")
        return cls.phase_for_epoch(saved_epoch + 1)

    def set_phase(self, epoch: int) -> dict:
        phase = self.phase_for_epoch(epoch)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        frozen_modules = [self.resnet.conv1, self.resnet.bn1, self.resnet.layer1]
        if phase == "A":
            frozen_modules.append(self.resnet.layer2)
        for module in frozen_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.stage24_epoch = epoch
        if self.training:
            self._freeze_running_statistics()
        return self.trainability_state()

    def _freeze_running_statistics(self) -> None:
        self.resnet.bn1.eval()
        for module in (self.resnet.layer1,):
            for child in module.modules():
                if isinstance(child, nn.modules.batchnorm._BatchNorm):
                    child.eval()
        if self.phase_for_epoch(self.stage24_epoch) == "A":
            for child in self.resnet.layer2.modules():
                if isinstance(child, nn.modules.batchnorm._BatchNorm):
                    child.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._freeze_running_statistics()
        return self

    def trainability_state(self) -> dict:
        return {
            "fine_tuning_phase": self.phase_for_epoch(self.stage24_epoch),
            "conv1_trainable": any(p.requires_grad for p in self.resnet.conv1.parameters()),
            "bn1_trainable": any(p.requires_grad for p in self.resnet.bn1.parameters()),
            "layer1_trainable": any(p.requires_grad for p in self.resnet.layer1.parameters()),
            "layer2_trainable": any(p.requires_grad for p in self.resnet.layer2.parameters()),
            "layer3_trainable": all(p.requires_grad for p in self.resnet.layer3.parameters()),
            "layer4_trainable": all(p.requires_grad for p in self.resnet.layer4.parameters()),
        }

    def frozen_batchnorm_state(self) -> dict:
        frozen = {"resnet.bn1": self.resnet.bn1}
        for root_name, root_module in (("resnet.layer1", self.resnet.layer1),):
            for name, module in root_module.named_modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    frozen[f"{root_name}.{name}".rstrip(".")] = module
        if self.phase_for_epoch(self.stage24_epoch) == "A":
            for name, module in self.resnet.layer2.named_modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    frozen[f"resnet.layer2.{name}".rstrip(".")] = module
        return {name: not module.training for name, module in frozen.items()}


def optimizer_parameter_groups(model: Stage24EGVAN, base_lr: float) -> list[dict]:
    if base_lr <= 0:
        raise ValueError("base_lr must be positive")
    named = dict(model.named_parameters())
    layer_prefixes = {
        "resnet.layer2": "resnet.layer2.",
        "resnet.layer3": "resnet.layer3.",
        "resnet.layer4": "resnet.layer4.",
    }
    assigned = set()
    groups = []
    for group_name, prefixes, multiplier in (
        ("non_resnet_and_new_modules", tuple(), LR_MULTIPLIERS["non_resnet_and_new_modules"]),
        ("resnet.layer3", (layer_prefixes["resnet.layer3"],), LR_MULTIPLIERS["resnet.layer3"]),
        ("resnet.layer4", (layer_prefixes["resnet.layer4"],), LR_MULTIPLIERS["resnet.layer4"]),
        ("resnet.layer2", (layer_prefixes["resnet.layer2"],), LR_MULTIPLIERS["resnet.layer2"]),
    ):
        if group_name == "non_resnet_and_new_modules":
            selected = [(name, parameter) for name, parameter in named.items()
                        if not name.startswith("resnet.conv1.")
                        and not name.startswith("resnet.bn1.")
                        and not name.startswith("resnet.layer1.")
                        and not name.startswith("resnet.layer2.")
                        and not name.startswith("resnet.layer3.")
                        and not name.startswith("resnet.layer4.")]
        else:
            selected = [(name, parameter) for name, parameter in named.items()
                        if any(name.startswith(prefix) for prefix in prefixes)]
        if not selected:
            raise ValueError(f"Optimizer group {group_name} is empty")
        overlap = assigned.intersection(name for name, _ in selected)
        if overlap:
            raise ValueError(f"Parameters assigned to multiple optimizer groups: {sorted(overlap)}")
        assigned.update(name for name, _ in selected)
        groups.append({
            "params": [parameter for _, parameter in selected],
            "lr": base_lr * multiplier,
            "lr_multiplier": multiplier,
            "group_name": group_name,
        })
    active = {name for name, parameter in named.items() if parameter.requires_grad}
    if assigned != active | {name for name in named if name.startswith("resnet.layer2.")}:
        raise ValueError("Optimizer groups do not match trainable modules plus reserved layer2")
    return groups


def build_optimizer_scheduler(model: Stage24EGVAN, config: dict):
    optimizer = Adamax(
        optimizer_parameter_groups(model, config["optimizer"]["learning_rate"]),
        lr=config["optimizer"]["learning_rate"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=config["scheduler"]["mode"],
        factor=config["scheduler"]["factor"],
        patience=config["scheduler"]["patience"],
    )
    maintain_lr_ratios(optimizer)
    return optimizer, scheduler


def maintain_lr_ratios(optimizer) -> list[float]:
    base_lr = float(optimizer.param_groups[0]["lr"])
    learning_rates = []
    for group in optimizer.param_groups:
        group["lr"] = base_lr * float(group["lr_multiplier"])
        learning_rates.append(float(group["lr"]))
    return learning_rates


def assert_optimizer_layout(model: Stage24EGVAN, optimizer) -> None:
    expected_names = ["non_resnet_and_new_modules", "resnet.layer3", "resnet.layer4", "resnet.layer2"]
    if [group.get("group_name") for group in optimizer.param_groups] != expected_names:
        raise ValueError("Stage24 optimizer group order/name mismatch")
    group_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(group_parameters) != len({id(parameter) for parameter in group_parameters}):
        raise ValueError("A parameter appears in more than one Stage24 optimizer group")
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    included_names = {parameter_names[id(parameter)] for parameter in group_parameters}
    excluded_prefixes = ("resnet.conv1.", "resnet.bn1.", "resnet.layer1.")
    if any(any(name.startswith(prefix) for prefix in excluded_prefixes) for name in included_names):
        raise ValueError("Permanently frozen ResNet parameters are present in the optimizer")
    layer2_names = {name for name in included_names if name.startswith("resnet.layer2.")}
    if not layer2_names or layer2_names != {name for name in parameter_names.values()
                                         if name.startswith("resnet.layer2.")}:
        raise ValueError("Reserved layer2 optimizer group is incomplete")
    grouped_active = {id(parameter) for parameter in group_parameters if parameter.requires_grad}
    model_active = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if grouped_active != model_active:
        raise ValueError("An active trainable parameter is missing from the optimizer groups")


def apply_scheduler_step(scheduler, optimizer, validation_loss: float) -> list[float]:
    scheduler.step(validation_loss)
    learning_rates = maintain_lr_ratios(optimizer)
    scheduler._last_lr = list(learning_rates)
    return learning_rates


def group_lrs(optimizer) -> dict:
    return {group["group_name"]: float(group["lr"]) for group in optimizer.param_groups}
