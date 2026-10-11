"""Second-pass perceptual screening of fixed Stage 4B dHash candidates only.

Read-only images and source candidate CSV; writes new derived files exclusively.
Scores are candidates for human adjudication, never automatic duplicate labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUT = HERE / "stage4b_near_duplicate_candidates_v1.csv"
EXPECTED_SHA = "10edd9c8d80fece258691761c135e36b3b3b81088deda3fbc5f1e07a2518907e"
OUTPUT = HERE / "stage4b_phash_refined_candidates_v1.csv"
SUMMARY = HERE / "stage4b_phash_refinement_v1.json"
FIELDS = ("development_train_image_id", "internal_selection_image_id",
          "development_train_lesion_id", "internal_selection_lesion_id",
          "dhash_hamming_distance", "phash_hamming_distance")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def phash64(image_id: str) -> int:
    path = ROOT / "data/raw/images" / f"{image_id}.jpg"
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Undecodable raw image {image_id}")
    dct = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))
    low = dct[:8, :8].reshape(-1)[1:]  # omit DC; 63 comparison bits
    median = np.median(low)
    result = 0
    for value in low:
        result = (result << 1) | int(value > median)
    return result


def main() -> None:
    if OUTPUT.exists() or SUMMARY.exists():
        raise FileExistsError("Refusing to overwrite Stage 4B refinement evidence")
    if sha(INPUT) != EXPECTED_SHA:
        raise ValueError("Fixed first-pass candidate list changed")
    with INPUT.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 6192:
        raise ValueError("First-pass candidate count changed")
    ids = {r[key] for r in rows for key in ("development_train_image_id", "internal_selection_image_id")}
    hashes = {image_id: phash64(image_id) for image_id in sorted(ids)}
    distances = Counter()
    refined = []
    for row in rows:
        distance = (hashes[row["development_train_image_id"]] ^
                    hashes[row["internal_selection_image_id"]]).bit_count()
        distances[distance] += 1
        if distance <= 8:
            refined.append({**row, "phash_hamming_distance": distance})
    refined.sort(key=lambda r: (r["phash_hamming_distance"], r["dhash_hamming_distance"],
                                r["development_train_image_id"], r["internal_selection_image_id"]))
    with OUTPUT.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(refined)
    result = {
        "status": "SECOND_PASS_CANDIDATE_SCREEN_ONLY", "source_dhash_candidates": len(rows),
        "unique_source_images": len(ids), "phash_hamming_le8_candidates": len(refined),
        "phash_hamming_le4_candidates": sum(v for d, v in distances.items() if d <= 4),
        "minimum_phash_hamming": min(distances),
        "method": "OpenCV grayscale raw JPEG, INTER_AREA resize 32x32, float32 DCT, top-left 8x8 excluding DC, median threshold, Hamming <=8 among first-pass candidates",
        "candidate_status": "requires human review; absence is not proof of patient independence",
        "first_pass_sha256": EXPECTED_SHA, "refined_csv_sha256": sha(OUTPUT),
        "script_sha256": sha(Path(__file__)),
    }
    with SUMMARY.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
