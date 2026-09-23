#!/usr/bin/env python3
"""Create naive and leakage-aware splits for HAM10000 metadata.

The script is deterministic using the frozen random seed 42.
It validates the result and stops if leakage-aware splitting fails.
"""

from __future__ import annotations

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METADATA_FILE = ROOT / "data" / "processed" / "metadata_clean.csv"
SPLITS_DIR = ROOT / "data" / "splits"
# Frozen seed: changing this changes every split assignment and invalidates
# already reported baseline metrics.
SEED = 42


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed metadata not found: {path}\nRun prepare_metadata.py first and ensure the official HAM10000 dataset is available."
        )
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    required = {"lesion_id", "image_id", "dx"}
    missing = sorted(required - set(reader.fieldnames or []))
    if missing:
        raise ValueError(f"Missing required columns for split creation: {missing}")
    return rows


def assign_split_counts(total: int) -> tuple[int, int, int]:
    """Convert an item count into deterministic 80/10/10 split counts."""
    train = int(total * 0.8)
    val = int(total * 0.1)
    test = total - train - val
    if test < 0:
        test = 0
    return train, val, test


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    split_counts = Counter(row["split"] for row in rows)
    class_counts = defaultdict(Counter)
    for row in rows:
        class_counts[row["split"]][row["dx"]] += 1

    summary = {
        "total": len(rows),
        "split_counts": dict(split_counts),
        "class_counts": {split: dict(counts) for split, counts in class_counts.items()},
    }
    return summary


def write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "lesion_id", "dx", "split"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)


def build_naive(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build the image-level split where lesion IDs may cross partitions."""
    by_dx = defaultdict(list)
    for row in rows:
        by_dx[row["dx"]].append(row)

    output = []
    rng = random.Random(SEED)
    for dx, items in by_dx.items():
        # Stratify by diagnosis so each split roughly preserves class balance.
        rng.shuffle(items)
        total = len(items)
        train_n, val_n, test_n = assign_split_counts(total)
        splits = ["train"] * train_n + ["val"] * val_n + ["test"] * test_n
        if len(splits) < total:
            splits += ["test"] * (total - len(splits))
        for item, split in zip(items, splits):
            output.append({**item, "split": split})
    return output


def build_leakage_aware(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Build the lesion-level split where each lesion stays in one partition."""
    lesion_groups = defaultdict(list)
    for row in rows:
        lesion_groups[row["lesion_id"]].append(row)

    lesions_by_dx = defaultdict(list)
    for lesion_id, items in lesion_groups.items():
        dx = items[0]["dx"]
        lesions_by_dx[dx].append(lesion_id)

    rng = random.Random(SEED)
    final_rows = []
    for dx, lesion_ids in lesions_by_dx.items():
        # Split lesion IDs, not images, so related images cannot leak between
        # train/validation/test partitions.
        rng.shuffle(lesion_ids)
        total = len(lesion_ids)
        train_n, val_n, test_n = assign_split_counts(total)
        assign_map = {}
        for lesion_id in lesion_ids[:train_n]:
            assign_map[lesion_id] = "train"
        for lesion_id in lesion_ids[train_n:train_n + val_n]:
            assign_map[lesion_id] = "val"
        for lesion_id in lesion_ids[train_n + val_n:train_n + val_n + test_n]:
            assign_map[lesion_id] = "test"
        if len(lesion_ids) - (train_n + val_n + test_n) > 0:
            remaining = lesion_ids[train_n + val_n + test_n:]
            for lesion_id in remaining:
                assign_map[lesion_id] = "test"

        for lesion_id in lesion_ids:
            for row in lesion_groups[lesion_id]:
                row_copy = {**row, "split": assign_map[lesion_id]}
                final_rows.append(row_copy)

    return final_rows


def count_lesion_crossing(rows: list[dict[str, str]], mode: str) -> dict[str, int]:
    """Count lesions whose images appear in more than one split partition."""
    lesion_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        lesion_to_splits[row["lesion_id"]].add(row["split"])

    counts = {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
        "any_partition": 0,
    }

    for lesion_id, splits in lesion_to_splits.items():
        if len(splits) > 1:
            counts["any_partition"] += 1
        if {"train", "val"} <= splits:
            counts["train_val"] += 1
        if {"train", "test"} <= splits:
            counts["train_test"] += 1
        if {"val", "test"} <= splits:
            counts["val_test"] += 1

    if mode == "leakage_aware":
        # Leakage-aware output is only acceptable if the crossing count is zero.
        if counts["any_partition"] != 0:
            raise ValueError(f"Leakage-aware split failed: {counts['any_partition']} lesion_id values cross partitions.")
    return counts


def print_summary(label: str, rows: list[dict[str, str]]) -> None:
    print(f"\n=== {label} summary ===")
    summary = summarize(rows)
    print(f"train: {summary['split_counts'].get('train', 0)}")
    print(f"val: {summary['split_counts'].get('val', 0)}")
    print(f"test: {summary['split_counts'].get('test', 0)}")
    print("class counts per split:")
    for split in ["train", "val", "test"]:
        counts = summary["class_counts"].get(split, {})
        print(f"  {split}: {dict(sorted(counts.items()))}")


def main() -> int:
    print("=== Build HAM10000 splits ===")
    try:
        rows = load_rows(METADATA_FILE)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[BLOCKED] {exc}")
        return 1

    naive_rows = build_naive(rows)
    leakage_rows = build_leakage_aware(rows)

    naive_path = SPLITS_DIR / "split_naive.csv"
    leakage_path = SPLITS_DIR / "split_leakage_aware.csv"

    write_split_csv(naive_path, naive_rows)
    write_split_csv(leakage_path, leakage_rows)

    print_summary("Naive", naive_rows)
    print_summary("Leakage-aware", leakage_rows)

    naive_cross = count_lesion_crossing(naive_rows, mode="naive")
    print("\nNaive split lesion crossing counts:")
    print(f"  train/val: {naive_cross['train_val']}")
    print(f"  train/test: {naive_cross['train_test']}")
    print(f"  val/test: {naive_cross['val_test']}")

    try:
        leakage_cross = count_lesion_crossing(leakage_rows, mode="leakage_aware")
    except ValueError as exc:
        print(f"[BLOCKED] {exc}")
        return 1

    print("\nLeakage-aware split lesion crossing counts:")
    print(f"  any partition: {leakage_cross['any_partition']}")
    print(f"split_naive.csv saved to: {naive_path}")
    print(f"split_leakage_aware.csv saved to: {leakage_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
    except Exception as exc:  # pragma: no cover
        print(f"[UNEXPECTED ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
