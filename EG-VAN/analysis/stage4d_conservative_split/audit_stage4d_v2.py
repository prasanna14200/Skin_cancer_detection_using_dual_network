"""Independent, read-only audit of the Stage 4D v2 proposed split."""

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V1 = ROOT / "analysis/stage4b_development_split"
STAGE4C = ROOT / "analysis/stage4c_near_duplicate_adjudication"
RAW = ROOT / "data/raw/images"
PROCESSED = ROOT / "data/processed/images"
CLASSES = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def dhash(path):
    with Image.open(io.BytesIO(path.read_bytes())) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[9 * y + x] > pixels[9 * y + x + 1])
    return bits


def phash(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Undecodable raw image: {path}")
    coefficients = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].reshape(-1)[1:]
    median = np.median(coefficients)
    bits = 0
    for value in coefficients:
        bits = (bits << 1) | int(value > median)
    return bits


def audit():
    manifest_path = HERE / "stage4d_split_manifest_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["artifact_sha256"].items():
        assert sha(HERE / name) == expected, f"v2 artifact hash: {name}"
    source_map = {
        "stage4b_split_manifest_v1.json": V1 / "stage4b_split_manifest_v1.json",
        "stage4b_development_train_v1.csv": V1 / "stage4b_development_train_v1.csv",
        "stage4b_internal_selection_v1.csv": V1 / "stage4b_internal_selection_v1.csv",
        "stage4b_phash_refined_candidates_v1.csv": V1 / "stage4b_phash_refined_candidates_v1.csv",
        "stage4c_review_manifest.json": STAGE4C / "stage4c_review_manifest.json",
        "NEAR_DUPLICATE_PAIR_REVIEW.csv": STAGE4C / "NEAR_DUPLICATE_PAIR_REVIEW.csv",
        "split_leakage_aware.csv": ROOT / "data/splits/split_leakage_aware.csv",
        "STAGE4D_CONSERVATIVE_SPLIT_PROTOCOL.md": HERE / "STAGE4D_CONSERVATIVE_SPLIT_PROTOCOL.md",
        "build_stage4d_v2.py": HERE / "build_stage4d_v2.py",
    }
    for name, path in source_map.items():
        assert sha(path) == manifest["source_sha256"][name], f"v2 source hash: {name}"
    v1_manifest = json.loads(source_map["stage4b_split_manifest_v1.json"].read_text(encoding="utf-8"))
    for name, expected in v1_manifest["artifact_sha256"].items():
        assert sha(V1 / name) == expected, f"immutable v1 artifact: {name}"
    c4 = json.loads(source_map["stage4c_review_manifest.json"].read_text(encoding="utf-8"))
    assert sha(source_map["NEAR_DUPLICATE_PAIR_REVIEW.csv"]) == c4["review_csv_sha256"]
    old_train = csv_rows(source_map["stage4b_development_train_v1.csv"])
    old_selection = csv_rows(source_map["stage4b_internal_selection_v1.csv"])
    new_train = csv_rows(HERE / "stage4d_development_train_v2.csv")
    new_selection = csv_rows(HERE / "stage4d_internal_selection_v2.csv")
    edges = csv_rows(source_map["stage4b_phash_refined_candidates_v1.csv"])
    reviews = csv_rows(source_map["NEAR_DUPLICATE_PAIR_REVIEW.csv"])
    assert len(edges) == len(reviews) == 40
    assert all(not r["reviewer_decision"] and not r["reviewer_notes"] for r in reviews)
    assert len(old_train) == 6425 and len(old_selection) == 1590
    assert len(new_train) + len(new_selection) == 8015
    original = csv_rows(source_map["split_leakage_aware.csv"])
    original_train = {r["image_id"]: r for r in original if r["split"] == "train"}
    valtest = [r for r in original if r["split"] != "train"]
    all_new = new_train + new_selection
    new_by_id = {r["image_id"]: r for r in all_new}
    old_by_id = {r["image_id"]: r for r in old_train + old_selection}
    assert len(new_by_id) == len(old_by_id) == len(original_train) == 8015
    assert set(new_by_id) == set(old_by_id) == set(original_train)
    assert not ({r["lesion_id"] for r in all_new} & {r["lesion_id"] for r in valtest})
    assert not ({r["image_id"] for r in all_new} & {r["image_id"] for r in valtest})
    train_lesions = {r["lesion_id"] for r in new_train}
    selection_lesions = {r["lesion_id"] for r in new_selection}
    assert not (train_lesions & selection_lesions)
    assert {r["dx"] for r in new_train} == {r["dx"] for r in new_selection} == CLASSES
    d = {}
    for image_id, row in new_by_id.items():
        before = old_by_id[image_id]
        original_row = original_train[image_id]
        for name in ("lesion_id", "dx", "original_split", "raw_image_sha256", "processed_image_sha256"):
            assert row[name] == before[name], (image_id, name)
        assert row["stage4b_role"] == before["stage4b_role"]
        assert (row["lesion_id"], row["dx"]) == (original_row["lesion_id"], original_row["dx"])
        assert sha(RAW / f"{image_id}.jpg") == row["raw_image_sha256"]
        assert sha(PROCESSED / f"{image_id}.jpg") == row["processed_image_sha256"]
        d[image_id] = dhash(RAW / f"{image_id}.jpg")
    assert all(r["stage4d_role"] == "development_train" for r in new_train)
    assert all(r["stage4d_role"] == "internal_selection" for r in new_selection)
    for field in ("raw_image_sha256", "processed_image_sha256"):
        assert not ({r[field] for r in new_train} & {r[field] for r in new_selection})
    assert all(r["stage4d_role"] == "development_train" for r in new_train if r["stage4b_role"] == "development_train")
    p = {}
    def ph(image_id):
        if image_id not in p:
            p[image_id] = phash(RAW / f"{image_id}.jpg")
        return p[image_id]
    # Recompute selection-side closure from all original selection image pairs.
    adjacency = defaultdict(set)
    refined_within_count = 0
    for index, a in enumerate(old_selection):
        for b in old_selection[index + 1:]:
            if a["lesion_id"] == b["lesion_id"]:
                continue
            if (d[a["image_id"]] ^ d[b["image_id"]]).bit_count() <= 4:
                if (ph(a["image_id"]) ^ ph(b["image_id"])).bit_count() <= 8:
                    refined_within_count += 1
                    adjacency[a["lesion_id"]].add(b["lesion_id"])
                    adjacency[b["lesion_id"]].add(a["lesion_id"])
    closure = {r["internal_selection_lesion_id"] for r in edges}
    queue = list(closure)
    while queue:
        for neighbor in adjacency[queue.pop()]:
            if neighbor not in closure:
                closure.add(neighbor)
                queue.append(neighbor)
    assert len(closure) == manifest["closure_moved_selection_lesions"]
    assert sorted(closure) == manifest["closure_moved_selection_lesion_ids"]
    assert refined_within_count == manifest["within_selection_refined_phash_le8_pairs"]
    for image_id, row in new_by_id.items():
        expected = "development_train" if old_by_id[image_id]["stage4b_role"] == "development_train" or row["lesion_id"] in closure else "internal_selection"
        assert row["stage4d_role"] == expected, image_id
    assert all(new_by_id[r["development_train_image_id"]]["stage4d_role"] ==
               new_by_id[r["internal_selection_image_id"]]["stage4d_role"] == "development_train" for r in edges)
    # Independent brute-force full-boundary screen, rather than the builder's buckets.
    expected_screen = []
    for a in new_train:
        for b in new_selection:
            distance = (d[a["image_id"]] ^ d[b["image_id"]]).bit_count()
            if distance <= 4:
                pd = (ph(a["image_id"]) ^ ph(b["image_id"])).bit_count()
                expected_screen.append((a["image_id"], b["image_id"], a["lesion_id"], b["lesion_id"], distance, pd))
    actual_screen = csv_rows(HERE / "stage4d_boundary_screen_v2.csv")
    recorded = [(r["development_train_image_id"], r["internal_selection_image_id"],
                 r["development_train_lesion_id"], r["internal_selection_lesion_id"],
                 int(r["dhash_hamming_distance"]), int(r["phash_hamming_distance"])) for r in actual_screen]
    assert sorted(expected_screen) == sorted(recorded)
    assert len(expected_screen) == manifest["cross_role_dhash_le4_pairs"]
    assert not any(r[-1] <= 8 for r in expected_screen)
    class_table = csv_rows(HERE / "STAGE4D_CLASS_DISTRIBUTION.csv")
    assert len(class_table) == 14
    for role, members in (("development_train", new_train), ("internal_selection", new_selection)):
        counts = manifest["counts"][role]
        assert counts["images"] == len(members)
        assert counts["lesions"] == len({r["lesion_id"] for r in members})
        for dx in CLASSES:
            subset = [r for r in members if r["dx"] == dx]
            assert counts["images_by_class"][dx] == len(subset)
            assert counts["lesions_by_class"][dx] == len({r["lesion_id"] for r in subset})
            table = next(r for r in class_table if r["class"] == dx and r["stage4d_role"] == role)
            assert int(table["images"]) == len(subset)
            assert int(table["lesions"]) == len({r["lesion_id"] for r in subset})
    return {
        "status": "PASS_STAGE4D_V2_INTEGRITY",
        "v2_manifest_sha256": sha(manifest_path),
        "train_images": len(new_train), "selection_images": len(new_selection),
        "train_lesions": len(train_lesions), "selection_lesions": len(selection_lesions),
        "selection_melanoma_images": sum(r["dx"] == "mel" for r in new_selection),
        "selection_melanoma_lesions": len({r["lesion_id"] for r in new_selection if r["dx"] == "mel"}),
        "frozen_edges_contained": len(edges),
        "v2_dhash_only_cross_pairs": len(expected_screen),
        "v2_refined_cross_pairs": 0,
        "human_adjudication": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
