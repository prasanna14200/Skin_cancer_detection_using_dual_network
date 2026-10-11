"""Build the Stage 4D v2 lesion-level co-location split, without model access."""

from __future__ import annotations

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
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
MAIN_COLUMNS = ("image_id", "lesion_id", "dx", "original_split", "stage4b_role", "stage4d_role",
                "raw_image_sha256", "processed_image_sha256")
SCREEN_COLUMNS = ("development_train_image_id", "internal_selection_image_id",
                  "development_train_lesion_id", "internal_selection_lesion_id",
                  "dhash_hamming_distance", "phash_hamming_distance")
OUTPUTS = ("stage4d_development_train_v2.csv", "stage4d_internal_selection_v2.csv",
           "STAGE4D_CLASS_DISTRIBUTION.csv", "stage4d_boundary_screen_v2.csv",
           "stage4d_split_manifest_v2.json")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows: list[dict], columns: tuple[str, ...]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def dhash64(raw: bytes) -> int:
    with Image.open(io.BytesIO(raw)) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        pixels = list(gray.getdata())
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return result


def phash63(image_id: str) -> int:
    gray = cv2.imread(str(RAW / f"{image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Undecodable raw image: {image_id}")
    low = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].reshape(-1)[1:]
    median = np.median(low)
    result = 0
    for value in low:
        result = (result << 1) | int(value > median)
    return result


def dhash_pairs(left: list[dict], right: list[dict], hashes: dict[str, int], same_list: bool = False):
    """All distinct-lesion pairs with dHash Hamming distance <= 4."""
    # Any <=4-bit pair shares at least one of eight nonoverlapping bytes.
    buckets = defaultdict(list)
    for row in left:
        value = hashes[row["image_id"]]
        for pos in range(8):
            buckets[(pos, (value >> (8 * pos)) & 255)].append(row)
    position = {row["image_id"]: i for i, row in enumerate(left)} if same_list else None
    for selected in right:
        value = hashes[selected["image_id"]]
        pool = {row["image_id"]: row for pos in range(8)
                for row in buckets[(pos, (value >> (8 * pos)) & 255)]}
        for other in pool.values():
            if other["lesion_id"] == selected["lesion_id"]:
                continue
            if same_list and position[other["image_id"]] >= position[selected["image_id"]]:
                continue
            distance = (value ^ hashes[other["image_id"]]).bit_count()
            if distance <= 4:
                yield other, selected, distance


def source_check():
    v1_manifest_path = V1 / "stage4b_split_manifest_v1.json"
    v1_manifest = json.loads(v1_manifest_path.read_text(encoding="utf-8"))
    stage4c_manifest_path = STAGE4C / "stage4c_review_manifest.json"
    stage4c_manifest = json.loads(stage4c_manifest_path.read_text(encoding="utf-8"))
    if sha_file(v1_manifest_path) != "dd2ce3f2d698ff5cbf392ccc2a8889338fe4cb7b1017dcff677214eec392e649":
        raise ValueError("Stage 4B v1 manifest changed")
    if stage4c_manifest["stage4b_manifest_sha256"] != sha_file(v1_manifest_path):
        raise ValueError("Stage 4C provenance does not match Stage 4B v1")
    if sha_file(STAGE4C / "NEAR_DUPLICATE_PAIR_REVIEW.csv") != stage4c_manifest["review_csv_sha256"]:
        raise ValueError("Stage 4C pair table changed")
    if sha_file(V1 / "stage4b_phash_refined_candidates_v1.csv") != stage4c_manifest["refined_candidate_sha256"]:
        raise ValueError("Stage 4B refined candidates changed")
    for relative, expected in v1_manifest["source_sha256"].items():
        if sha_file(ROOT / relative) != expected:
            raise ValueError(f"Frozen Stage 4B source changed: {relative}")
    for name, expected in v1_manifest["artifact_sha256"].items():
        if sha_file(V1 / name) != expected:
            raise ValueError(f"Frozen Stage 4B artifact changed: {name}")
    for name in ("stage4b_development_train_v1.csv", "stage4b_internal_selection_v1.csv"):
        if not (V1 / name).is_file():
            raise ValueError(f"Missing Stage 4B membership: {name}")
    train = read_csv(V1 / "stage4b_development_train_v1.csv")
    selection = read_csv(V1 / "stage4b_internal_selection_v1.csv")
    pairs = read_csv(V1 / "stage4b_phash_refined_candidates_v1.csv")
    review = read_csv(STAGE4C / "NEAR_DUPLICATE_PAIR_REVIEW.csv")
    if len(train) != 6425 or len(selection) != 1590 or len(pairs) != 40 or len(review) != 40:
        raise ValueError("Unexpected Stage 4B/C source counts")
    if any(r["reviewer_decision"] or r["reviewer_notes"] for r in review):
        raise ValueError("This unadjudicated containment protocol cannot consume filled decisions")
    train_by_id = {r["image_id"]: r for r in train}
    selection_by_id = {r["image_id"]: r for r in selection}
    if len(train_by_id) != len(train) or len(selection_by_id) != len(selection) or set(train_by_id) & set(selection_by_id):
        raise ValueError("Stage 4B v1 image membership invalid")
    v1_lesions = {r["lesion_id"] for r in train} & {r["lesion_id"] for r in selection}
    if v1_lesions:
        raise ValueError("Stage 4B v1 crosses lesions")
    original = read_csv(ROOT / "data/splits/split_leakage_aware.csv")
    original_train = {r["image_id"]: r for r in original if r["split"] == "train"}
    valtest = [r for r in original if r["split"] != "train"]
    if (len(original_train) != 8015 or set(original_train) != set(train_by_id) | set(selection_by_id)
            or ({r["lesion_id"] for r in train + selection} & {r["lesion_id"] for r in valtest})):
        raise ValueError("Stage 4B pool differs from original leakage-aware train")
    for image_id, row in {**train_by_id, **selection_by_id}.items():
        source = original_train[image_id]
        if (row["lesion_id"], row["dx"], row["original_split"]) != (source["lesion_id"], source["dx"], "train"):
            raise ValueError(f"Stage 4B source row differs: {image_id}")
    seen_pairs = set()
    for r, c in zip(pairs, review):
        key = (r["development_train_image_id"], r["internal_selection_image_id"])
        if key in seen_pairs or key != (c["train_image_id"], c["selection_image_id"]):
            raise ValueError("Stage 4B/C pair identity mismatch")
        seen_pairs.add(key)
        if (r["development_train_lesion_id"] != train_by_id[key[0]]["lesion_id"]
                or r["internal_selection_lesion_id"] != selection_by_id[key[1]]["lesion_id"]):
            raise ValueError("Stage 4B/C pair lesion mismatch")
    return v1_manifest_path, stage4c_manifest_path, train, selection, pairs, original_train


def build():
    if any((HERE / name).exists() for name in OUTPUTS):
        raise FileExistsError("Stage 4D v2 output already exists; refusing overwrite")
    v1_manifest_path, stage4c_manifest_path, train, selection, pairs, original_train = source_check()
    all_rows = sorted(train + selection, key=lambda r: r["image_id"])
    dhashes = {}
    for row in all_rows:
        image_id = row["image_id"]
        raw = (RAW / f"{image_id}.jpg").read_bytes()
        processed = (PROCESSED / f"{image_id}.jpg").read_bytes()
        if sha_bytes(raw) != row["raw_image_sha256"] or sha_bytes(processed) != row["processed_image_sha256"]:
            raise ValueError(f"Raw/processed source changed: {image_id}")
        dhashes[image_id] = dhash64(raw)
    phashes = {}
    def p(image_id):
        if image_id not in phashes:
            phashes[image_id] = phash63(image_id)
        return phashes[image_id]
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for r in pairs:
        a, b = r["development_train_lesion_id"], r["internal_selection_lesion_id"]
        parent[find(a)] = find(b)
        left, right = r["development_train_image_id"], r["internal_selection_image_id"]
        if ((dhashes[left] ^ dhashes[right]).bit_count() != int(r["dhash_hamming_distance"])
                or (p(left) ^ p(right)).bit_count() != int(r["phash_hamming_distance"])):
            raise ValueError(f"Stage 4B similarity score mismatch: {left}, {right}")
    components = defaultdict(set)
    for lesion in parent:
        components[find(lesion)].add(lesion)
    initial_selection_lesions = {r["internal_selection_lesion_id"] for r in pairs}
    if len(components) != 20 or len(parent) != 59 or len(initial_selection_lesions) != 23:
        raise ValueError("Unexpected frozen risk graph structure")
    # Construct all refined within-selection edges, not only edges incident to
    # the initial 23 lesions. This makes closure order-independent.
    adjacency = defaultdict(set)
    within_dhash_count = within_refined_count = 0
    for a, b, _ in dhash_pairs(selection, selection, dhashes, same_list=True):
        within_dhash_count += 1
        if (p(a["image_id"]) ^ p(b["image_id"])).bit_count() <= 8:
            within_refined_count += 1
            adjacency[a["lesion_id"]].add(b["lesion_id"])
            adjacency[b["lesion_id"]].add(a["lesion_id"])
    moved_lesions = set(initial_selection_lesions)
    queue = sorted(moved_lesions)
    while queue:
        for neighbor in sorted(adjacency[queue.pop()]):
            if neighbor not in moved_lesions:
                moved_lesions.add(neighbor)
                queue.append(neighbor)
    new_train = []
    new_selection = []
    for row in all_rows:
        role = "development_train" if (row["stage4b_role"] == "development_train" or row["lesion_id"] in moved_lesions) else "internal_selection"
        output = {key: row[key] for key in row}
        output["stage4d_role"] = role
        (new_train if role == "development_train" else new_selection).append(output)
    new_train.sort(key=lambda r: r["image_id"])
    new_selection.sort(key=lambda r: r["image_id"])
    train_lesions = {r["lesion_id"] for r in new_train}
    selection_lesions = {r["lesion_id"] for r in new_selection}
    if train_lesions & selection_lesions or set(r["image_id"] for r in new_train + new_selection) != set(original_train):
        raise ValueError("Stage 4D v2 lesion or source membership invalid")
    if {r["dx"] for r in new_train} != set(CLASSES) or {r["dx"] for r in new_selection} != set(CLASSES):
        raise ValueError("A v2 arm lacks a class")
    for r in pairs:
        if r["development_train_lesion_id"] not in train_lesions or r["internal_selection_lesion_id"] not in train_lesions:
            raise ValueError("A frozen Stage 4C edge still crosses roles")
    for column in ("raw_image_sha256", "processed_image_sha256"):
        by_hash = defaultdict(set)
        for row in new_train + new_selection:
            by_hash[row[column]].add(row["stage4d_role"])
        if any(len(roles) > 1 for roles in by_hash.values()):
            raise ValueError(f"Exact-byte {column} duplicate crosses roles")
    boundary = []
    for a, b, distance in dhash_pairs(new_train, new_selection, dhashes):
        pd = (p(a["image_id"]) ^ p(b["image_id"])).bit_count()
        boundary.append({"development_train_image_id": a["image_id"], "internal_selection_image_id": b["image_id"],
                         "development_train_lesion_id": a["lesion_id"], "internal_selection_lesion_id": b["lesion_id"],
                         "dhash_hamming_distance": distance, "phash_hamming_distance": pd})
    boundary.sort(key=lambda r: (r["internal_selection_image_id"], r["development_train_image_id"]))
    remaining_refined = [r for r in boundary if r["phash_hamming_distance"] <= 8]
    if remaining_refined:
        raise ValueError(f"Refined cross-role pairs remain after closure: {remaining_refined[:3]}")
    counts = []
    original_by_class = Counter(r["dx"] for r in all_rows)
    for role, members in (("development_train", new_train), ("internal_selection", new_selection)):
        for dx in CLASSES:
            subset = [r for r in members if r["dx"] == dx]
            counts.append({"stage4d_role": role, "class": dx, "images": len(subset),
                           "lesions": len({r["lesion_id"] for r in subset}),
                           "fraction_of_original_train_class_images": f"{len(subset) / original_by_class[dx]:.8f}"})
    count_columns = ("stage4d_role", "class", "images", "lesions", "fraction_of_original_train_class_images")
    payloads = {
        OUTPUTS[0]: csv_bytes(new_train, MAIN_COLUMNS),
        OUTPUTS[1]: csv_bytes(new_selection, MAIN_COLUMNS),
        OUTPUTS[2]: csv_bytes(counts, count_columns),
        OUTPUTS[3]: csv_bytes(boundary, SCREEN_COLUMNS),
    }
    manifest = {
        "schema": "stage4d_conservative_split_v2", "status": "PROPOSED_CONTAINED_INTERNAL_DEVELOPMENT_BOUNDARY",
        "policy": "whole-lesion co-location in development training; full closure over refined within-selection edges",
        "source_sha256": {
            "stage4b_split_manifest_v1.json": sha_file(v1_manifest_path),
            "stage4b_development_train_v1.csv": sha_file(V1 / "stage4b_development_train_v1.csv"),
            "stage4b_internal_selection_v1.csv": sha_file(V1 / "stage4b_internal_selection_v1.csv"),
            "stage4b_phash_refined_candidates_v1.csv": sha_file(V1 / "stage4b_phash_refined_candidates_v1.csv"),
            "stage4c_review_manifest.json": sha_file(stage4c_manifest_path),
            "NEAR_DUPLICATE_PAIR_REVIEW.csv": sha_file(STAGE4C / "NEAR_DUPLICATE_PAIR_REVIEW.csv"),
            "split_leakage_aware.csv": sha_file(ROOT / "data/splits/split_leakage_aware.csv"),
            "STAGE4D_CONSERVATIVE_SPLIT_PROTOCOL.md": sha_file(HERE / "STAGE4D_CONSERVATIVE_SPLIT_PROTOCOL.md"),
            "build_stage4d_v2.py": sha_file(Path(__file__)),
        },
        "artifact_sha256": {name: sha_bytes(data) for name, data in payloads.items()},
        "original_train_images": len(all_rows), "original_train_lesions": len({r["lesion_id"] for r in all_rows}),
        "original_refined_edges": len(pairs), "original_edge_components": len(components),
        "original_edge_lesions": len(parent), "initial_selection_lesions": len(initial_selection_lesions),
        "within_selection_dhash_le4_pairs": within_dhash_count,
        "within_selection_refined_phash_le8_pairs": within_refined_count,
        "closure_moved_selection_lesions": len(moved_lesions),
        "closure_moved_selection_lesion_ids": sorted(moved_lesions),
        "closure_moved_selection_images": sum(r["lesion_id"] in moved_lesions for r in selection),
        "counts": {role: {"images": len(members), "lesions": len({r["lesion_id"] for r in members}),
                          "images_by_class": dict(sorted(Counter(r["dx"] for r in members).items())),
                          "lesions_by_class": {dx: len({r["lesion_id"] for r in members if r["dx"] == dx}) for dx in CLASSES}}
                   for role, members in (("development_train", new_train), ("internal_selection", new_selection))},
        "cross_role_dhash_le4_pairs": len(boundary),
        "cross_role_refined_phash_le8_pairs": 0,
        "cross_role_lesion_ids": 0, "cross_role_exact_raw_or_processed_groups": 0,
        "all_original_train_images_once": True,
        "human_adjudication_performed": False, "patient_ids_available": False,
        "model_training_or_inference_performed": False,
        "original_ham_validation_test_or_ph2_used_for_membership": False,
    }
    for name, data in payloads.items():
        with (HERE / name).open("xb") as stream:
            stream.write(data)
    with (HERE / OUTPUTS[4]).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    result = build()
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "closure_moved_selection_lesions": result["closure_moved_selection_lesions"],
                      "closure_moved_selection_images": result["closure_moved_selection_images"],
                      "cross_role_dhash_le4_pairs": result["cross_role_dhash_le4_pairs"],
                      "cross_role_refined_phash_le8_pairs": result["cross_role_refined_phash_le8_pairs"]}, indent=2))
