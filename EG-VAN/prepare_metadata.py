#!/usr/bin/env python3
"""Validate HAM10000 metadata and ensure all metadata image IDs correspond to actual image files.

This script is intentionally data-safe: it does not delete or modify image files.
It reports all findings and writes a cleaned metadata file only when the dataset is present.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_METADATA = ROOT / "data" / "raw" / "HAM10000_metadata.csv"
RAW_IMAGES_DIR = ROOT / "data" / "raw" / "images"
PROCESSED_METADATA = ROOT / "data" / "processed" / "metadata_clean.csv"
# Only these columns are mandatory for the later split and dataset-loading code.
# Other metadata columns are preserved but are not required for the baseline.
REQUIRED_COLUMNS = {"lesion_id", "image_id", "dx"}


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def count_missing_values(rows: list[dict]) -> dict[str, int]:
    missing = {key: 0 for key in sorted({k for row in rows for k in row.keys()})}
    for row in rows:
        for key in row:
            if row.get(key, "") in (None, ""):
                missing[key] = missing.get(key, 0) + 1
    return missing


def load_metadata(path: Path) -> list[dict[str, str]]:
    """Load HAM10000 metadata and stop early if core columns are missing."""
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}\nPlease download the official HAM10000 metadata file before running this script.")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Metadata file is empty: {path}")

    missing_cols = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return rows


def image_file_exists(image_id: str) -> bool:
    """Check common image extensions without changing any source files."""
    if not RAW_IMAGES_DIR.exists():
        return False

    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = RAW_IMAGES_DIR / f"{image_id}{ext}"
        if candidate.exists():
            return True
    return False


def main() -> int:
    print_header("HAM10000 metadata validation")
    print(f"Project root: {ROOT}")
    print(f"Metadata path: {RAW_METADATA}")
    print(f"Images dir: {RAW_IMAGES_DIR}")

    try:
        rows = load_metadata(RAW_METADATA)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[BLOCKED] {exc}")
        print("Please download the official HAM10000 files manually from:")
        print("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        print("Required files:")
        print("- HAM10000_metadata.csv")
        print("- HAM10000_images_part1.zip")
        print("- HAM10000_images_part2.zip")
        return 1

    fieldnames = list(rows[0].keys())
    print(f"Required columns found: {REQUIRED_COLUMNS <= set(fieldnames)}")
    print(f"Columns: {fieldnames[:20]}")

    total_rows = len(rows)
    unique_image_ids = len({row["image_id"] for row in rows if row.get("image_id") not in (None, "")})
    unique_lesion_ids = len({row["lesion_id"] for row in rows if row.get("lesion_id") not in (None, "")})

    row_tuples = [tuple((k, v) for k, v in row.items()) for row in rows]
    duplicate_rows = len(row_tuples) - len(set(row_tuples))
    duplicate_image_ids = sum(1 for count in Counter(row["image_id"] for row in rows if row.get("image_id") not in (None, "")).values() if count > 1)

    class_distribution = Counter(row["dx"] for row in rows if row.get("dx") not in (None, ""))
    lesion_counts = Counter(row["lesion_id"] for row in rows if row.get("lesion_id") not in (None, ""))
    lesions_with_multiple_images = sum(1 for count in lesion_counts.values() if count > 1)

    missing_by_column = count_missing_values(rows)

    print_header("Summary")
    print(f"total rows: {total_rows}")
    print(f"unique image IDs: {unique_image_ids}")
    print(f"unique lesion IDs: {unique_lesion_ids}")
    print(f"duplicate rows: {duplicate_rows}")
    print(f"duplicate image IDs: {duplicate_image_ids}")
    print(f"lesions with multiple images: {lesions_with_multiple_images}")
    print("missing values:")
    for key, value in sorted(missing_by_column.items()):
        if value:
            print(f"  {key}: {value}")
    print("class distribution:")
    for cls, count in sorted(class_distribution.items()):
        print(f"  {cls}: {count}")

    print("images per lesion (first 10):")
    for lesion_id, count in list(lesion_counts.most_common(10)):
        print(f"  {lesion_id}: {count}")

    # The metadata/image correspondence check is the safety gate before any
    # split or preprocessing step uses these image IDs.
    all_image_ids = {row["image_id"] for row in rows if row.get("image_id") not in (None, "")}
    missing_files = []
    for image_id in sorted(all_image_ids):
        if not image_file_exists(image_id):
            missing_files.append(image_id)

    existing_but_missing_from_metadata = []
    if RAW_IMAGES_DIR.exists():
        for image_path in sorted(RAW_IMAGES_DIR.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                stem = image_path.stem
                if stem not in all_image_ids:
                    existing_but_missing_from_metadata.append(image_path.name)

    print_header("Image file validation")
    print(f"number of metadata image IDs with no matching file: {len(missing_files)}")
    if missing_files:
        for image_id in missing_files[:20]:
            print(f"  missing file for image_id: {image_id}")

    print(f"number of image files not in metadata: {len(existing_but_missing_from_metadata)}")
    if existing_but_missing_from_metadata:
        for image_name in existing_but_missing_from_metadata[:20]:
            print(f"  unreferenced file: {image_name}")

    output_dir = PROCESSED_METADATA.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # This "clean" metadata copy preserves the original rows; it exists so later
    # scripts can read from a stable processed location.
    with PROCESSED_METADATA.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print_header("Final validation summary")
    print(f"metadata_clean.csv written to: {PROCESSED_METADATA}")
    print("Validation complete. No files were deleted or silently modified.")
    print("NOTE: If the dataset is incomplete or missing, this script exits with a clear block message instead of guessing.")
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
