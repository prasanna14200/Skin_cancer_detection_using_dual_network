"""Run one approved EfficientNetV2S baseline experiment.

This command is intentionally separate from the smoke-test notebook. It is not
executed during preparation; approval is required before launching it.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, HAM10000Dataset, validate_split_integrity
from evaluate import evaluate_checkpoint
from train import (
    BATCH_SIZE,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    SEED,
    build_run_components,
    configuration,
    focal_loss,
    make_optimizer_and_scheduler,
    set_seed,
)


def metrics_from_confusion(matrix: torch.Tensor) -> dict:
    """Derive accuracy, macro-F1, recalls, and the serializable matrix."""
    # Rows are true labels and columns are predicted labels.
    total = int(matrix.sum().item())
    correct = int(torch.diag(matrix).sum().item())
    recalls = {}
    f1_values = []
    for index, name in enumerate(CLASS_NAMES):
        true_positive = float(matrix[index, index].item())
        actual = float(matrix[index, :].sum().item())
        predicted = float(matrix[:, index].sum().item())
        # Guard against classes with no predictions so metric calculation never
        # crashes on rare-class edge cases.
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        recalls[name] = recall
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class_recall": recalls,
        "confusion_matrix": matrix.tolist(),
    }


def run_epoch(model, loader, optimizer, scaler, device, training: bool) -> tuple[float, dict]:
    """Run one train or evaluation pass over a DataLoader."""
    model.train(training)
    total_loss = 0.0
    total_items = 0
    # Confusion matrix is accumulated on CPU so it stays small and serializable.
    matrix = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.long)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            # Clear gradients only for training; validation/test use the same
            # metric path without modifying model parameters.
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = focal_loss(logits, labels)
        if training:
            # Mixed precision is enabled on CUDA to match the Colab baseline
            # configuration while keeping the loss scaling numerically stable.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # The baseline uses the highest-logit class as the predicted diagnosis.
        predictions = logits.argmax(dim=1)
        for target, prediction in zip(labels.detach().cpu(), predictions.detach().cpu()):
            matrix[int(target), int(prediction)] += 1
        count = labels.size(0)
        total_loss += float(loss.detach().cpu()) * count
        total_items += count
    return total_loss / total_items, metrics_from_confusion(matrix)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-csv", default="split_naive.csv")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        # The approved experiment was designed for Colab GPU execution; local
        # CPU training would create a different runtime condition.
        raise RuntimeError("Phase 4C baseline requires CUDA; CPU training is not permitted.")
    set_seed(SEED)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.project_root / "data" / "splits" / args.split_csv
    # Only leakage-aware runs must enforce zero lesion crossing. Naive runs are
    # allowed to report crossing lesions because that is the comparison target.
    require_isolation = "leakage_aware" in args.split_csv
    split_stats = validate_split_integrity(split_path, require_isolation)
    write_json(args.run_dir / "split_integrity.json", split_stats)

    model, train_dataset, val_dataset, test_dataset = build_run_components(args.project_root, args.split_csv)
    # Training shuffles batches; validation and test preserve deterministic
    # ordering because their metrics do not require randomization.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers, pin_memory=True)
    model.to(device)
    optimizer, scheduler = make_optimizer_and_scheduler(model)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    config = configuration(device) | {
        "split_csv": args.split_csv,
        "classes": list(CLASS_NAMES),
        "software": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()},
        "early_stopping": False,
    }
    write_json(args.run_dir / "config.json", config)

    history = []
    best_val_loss = float("inf")
    best_checkpoint = args.run_dir / "best_checkpoint.pt"
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, True)
        with torch.no_grad():
            val_loss, val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, False)
        # The scheduler reacts to validation loss only, matching the checkpoint
        # selection signal.
        scheduler.step(val_loss)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train": train_metrics, "validation": val_metrics, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(record)
        write_json(args.run_dir / "training_history.json", {"epochs": history})
        if val_loss < best_val_loss:
            # The best checkpoint is selected only by HAM10000 validation loss,
            # never by test or external-validation performance.
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "config": config}, best_checkpoint)
    torch.save({"model_state": model.state_dict(), "epoch": EPOCHS, "config": config}, args.run_dir / "final_checkpoint.pt")
    test_loss, test_metrics = run_epoch(model, test_loader, optimizer, scaler, device, False)
    write_json(args.run_dir / "test_metrics.json", {"loss": test_loss, **test_metrics})
    print(json.dumps({"run_dir": str(args.run_dir), "best_checkpoint": str(best_checkpoint), "test_metrics": test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
