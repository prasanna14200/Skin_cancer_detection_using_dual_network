"""Training entry point for approved Phase 4C EfficientNetV2S runs.

This module defines the frozen training configuration but does not run on import.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import HAM10000Dataset
from models.baseline_effnet import build_model

SEED = 42
IMAGE_SIZE = 384
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.001
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible baseline runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def focal_loss(logits: Tensor, targets: Tensor, alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA) -> Tensor:
    """Compute focal loss to down-weight already confident examples."""
    probabilities = torch.softmax(logits, dim=1)
    # Gather the predicted probability assigned to the true class for each item;
    # this is the term focal loss uses to reduce easy-example influence.
    target_probability = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    cross_entropy = nn.functional.cross_entropy(logits, targets, reduction="none")
    return (alpha * (1.0 - target_probability).pow(gamma) * cross_entropy).mean()

def make_transforms(weights):
    """Use fixed 384px model input; augmentation remains train-only."""

    # The mean/std come from the EfficientNetV2S ImageNet weights so the input
    # distribution matches the pretrained backbone's expected preprocessing.
    preprocess = weights.transforms()

    # Random augmentation is restricted to training. Validation and test are
    # deterministic so reported metrics are not affected by sampling noise.
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=preprocess.mean,
            std=preprocess.std
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=preprocess.mean,
            std=preprocess.std
        ),
    ])

    return train_transform, eval_transform

def build_run_components(project_root: str | Path, split_name: str):
    """Create the model and datasets for one frozen split file."""
    project_root = Path(project_root)
    split_csv = project_root / "data" / "splits" / split_name
    images_dir = project_root / "data" / "processed" / "images"
    # Build the model before transforms so we can reuse the normalization tied
    # to the selected pretrained weights.
    model, weights = build_model(pretrained=True)
    train_transform, eval_transform = make_transforms(weights)
    # All three datasets read the same frozen split CSV; only the split filter
    # and transform differ.
    train_dataset = HAM10000Dataset(images_dir, split_csv, "train", train_transform)
    val_dataset = HAM10000Dataset(images_dir, split_csv, "val", eval_transform)
    test_dataset = HAM10000Dataset(images_dir, split_csv, "test", eval_transform)
    return model, train_dataset, val_dataset, test_dataset


def configuration(device: torch.device) -> dict:
    return {
        "seed": SEED,
        "model": "EfficientNetV2S",
        "pretrained_weight_source": "torchvision EfficientNet_V2_S_Weights.DEFAULT",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": "Adamax",
        "learning_rate": LEARNING_RATE,
        "focal_alpha": FOCAL_ALPHA,
        "focal_gamma": FOCAL_GAMMA,
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 1},
        "mixed_precision": device.type == "cuda",
        "device": str(device),
    }


def make_optimizer_and_scheduler(model: nn.Module):
    """Use the approved optimizer and validation-loss scheduler."""
    optimizer = Adamax(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    return optimizer, scheduler


def save_configuration(path: str | Path, device: torch.device) -> None:
    Path(path).write_text(json.dumps(configuration(device), indent=2), encoding="utf-8")
