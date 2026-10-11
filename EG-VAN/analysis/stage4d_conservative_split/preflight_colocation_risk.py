"""Read-only comparison of co-location and quarantine containment options."""

import csv
import io
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "analysis/stage4b_development_split"
RAW = ROOT / "data/raw/images"


def rows(name):
    with (BASE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def dhash(image_id):
    with Image.open(io.BytesIO((RAW / f"{image_id}.jpg").read_bytes())) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def phash(image_id):
    gray = cv2.imread(str(RAW / f"{image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(image_id)
    low = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].reshape(-1)[1:]
    median = np.median(low)
    value = 0
    for coefficient in low:
        value = (value << 1) | int(coefficient > median)
    return value


def main():
    selected = rows("stage4b_internal_selection_v1.csv")
    pairs = rows("stage4b_phash_refined_candidates_v1.csv")
    affected = {p["internal_selection_lesion_id"] for p in pairs}
    moved = [r for r in selected if r["lesion_id"] in affected]
    kept = [r for r in selected if r["lesion_id"] not in affected]
    hashes = {r["image_id"]: dhash(r["image_id"]) for r in selected}
    low = []
    for a in moved:
        for b in kept:
            distance = (hashes[a["image_id"]] ^ hashes[b["image_id"]]).bit_count()
            if distance <= 4:
                low.append((a["image_id"], b["image_id"], distance))
    phashes = {image_id: phash(image_id) for pair in low for image_id in pair[:2]}
    refined = [(a, b, d, (phashes[a] ^ phashes[b]).bit_count()) for a, b, d in low
               if (phashes[a] ^ phashes[b]).bit_count() <= 8]
    # Any two original-selection lesions that satisfy the same frozen screen
    # must be co-located if either one moves into training. Compute the closure.
    within_dhash = []
    for i, a in enumerate(selected):
        for b in selected[i + 1:]:
            if a["lesion_id"] == b["lesion_id"]:
                continue
            d = (hashes[a["image_id"]] ^ hashes[b["image_id"]]).bit_count()
            if d <= 4:
                within_dhash.append((a, b, d))
    for a, b, _ in within_dhash:
        for row in (a, b):
            image_id = row["image_id"]
            if image_id not in phashes:
                phashes[image_id] = phash(image_id)
    adjacency = {}
    within_refined = []
    for a, b, d in within_dhash:
        distance = (phashes[a["image_id"]] ^ phashes[b["image_id"]]).bit_count()
        if distance <= 8:
            within_refined.append((a["image_id"], b["image_id"], d, distance))
            adjacency.setdefault(a["lesion_id"], set()).add(b["lesion_id"])
            adjacency.setdefault(b["lesion_id"], set()).add(a["lesion_id"])
    closure = set(affected)
    frontier = list(affected)
    while frontier:
        for neighbor in sorted(adjacency.get(frontier.pop(), ())):
            if neighbor not in closure:
                closure.add(neighbor)
                frontier.append(neighbor)
    after = [r for r in selected if r["lesion_id"] not in closure]
    print(json.dumps({"moved_images": len(moved), "moved_lesions": len(affected),
                      "remaining_selection_images": len(kept),
                      "remaining_selection_mel_images": sum(r["dx"] == "mel" for r in kept),
                      "new_cross_dhash_le4_pairs": len(low),
                      "new_cross_refined_phash_le8_pairs": len(refined),
                      "refined_pairs": refined[:20],
                      "within_selection_dhash_le4_pairs": len(within_dhash),
                      "within_selection_refined_phash_le8_pairs": len(within_refined),
                      "closure_moved_lesions": len(closure),
                      "closure_moved_images": len(selected) - len(after),
                      "closure_selection_images": len(after),
                      "closure_selection_mel_images": sum(r["dx"] == "mel" for r in after),
                      "closure_selection_mel_lesions": len({r["lesion_id"] for r in after if r["dx"] == "mel"}),
                      "additional_lesions": sorted(closure - affected)}, indent=2))


if __name__ == "__main__":
    main()
