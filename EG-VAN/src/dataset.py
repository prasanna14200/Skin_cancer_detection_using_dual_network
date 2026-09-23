"""HAM10000 dataset loading for the EfficientNetV2S baseline."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def validate_split_integrity(split_csv: str | Path, require_lesion_isolation: bool) -> dict[str, int]:
    """Validate row uniqueness and optionally enforce lesion-level isolation."""
    with Path(split_csv).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "lesion_id", "dx", "split"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Split CSV is missing required columns: {sorted(required)}")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate image_id values found in {split_csv}")
    # Lesion IDs are the leakage-sensitive grouping unit. A leakage-aware split
    # must keep every lesion entirely inside one partition.
    lesion_partitions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        lesion_partitions[row["lesion_id"]].add(row["split"])
    crossing = sum(len(partitions) > 1 for partitions in lesion_partitions.values())
    if require_lesion_isolation and crossing:
        raise ValueError(f"{crossing} lesion_id values cross partitions in {split_csv}")
    return {"rows": len(rows), "unique_lesions": len(lesion_partitions), "crossing_lesions": crossing}


class HAM10000Dataset(Dataset[tuple[Tensor, int]]):
    """Load processed images using an existing frozen split CSV."""

    def __init__(
        self,
        images_dir: str | Path,
        split_csv: str | Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transform = transform
        with Path(split_csv).open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        # The split CSV is treated as frozen evidence; this dataset only filters
        # by the requested partition and never reshuffles or reassigns samples.
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows found for split={split!r} in {split_csv}")
        missing = [key for key in ("image_id", "lesion_id", "dx", "split") if key not in rows[0]]
        if missing:
            raise ValueError(f"Split CSV is missing required columns: {missing}")
        unknown = sorted({row["dx"] for row in self.rows} - set(CLASS_TO_INDEX))
        if unknown:
            raise ValueError(f"Unknown class labels: {unknown}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.rows[index]
        image_path = self.images_dir / f"{row['image_id']}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"Processed image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            # Training/evaluation policy lives in train.py; this class simply
            # applies the transform supplied by the caller.
            image = self.transform(image)
        return image, CLASS_TO_INDEX[row["dx"]]
