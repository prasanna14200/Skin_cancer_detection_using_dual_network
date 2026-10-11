"""Independent read-only verification of Stage 4B proposed manifests."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit() -> dict:
    m = json.loads((HERE / "stage4b_split_manifest_v1.json").read_text(encoding="utf-8"))
    if (m["schema"] != "stage4b_development_split_v1" or
            m["status"] != "PROPOSED_INTERNAL_SELECTION_BOUNDARY" or
            m["seed"] != "20261010" or tuple(m["class_order"]) != CLASS_ORDER or
            m["training_performed"] or m["inference_performed"]):
        raise ValueError("Proposed split manifest status/config invalid")
    source = ROOT / "data/splits/split_leakage_aware.csv"
    if sha(source) != "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883":
        raise ValueError("Original split changed")
    for path, expected in m["source_sha256"].items():
        if sha(ROOT/path) != expected:
            raise ValueError(f"Source hash changed: {path}")
    for name, expected in m["artifact_sha256"].items():
        if sha(HERE/name) != expected:
            raise ValueError(f"Output hash changed: {name}")
    original = read(source)
    orig_train = {r["image_id"]: r for r in original if r["split"] == "train"}
    valtest_ids = {r["image_id"] for r in original if r["split"] != "train"}
    valtest_lesions = {r["lesion_id"] for r in original if r["split"] != "train"}
    a = read(HERE / "stage4b_development_train_v1.csv")
    b = read(HERE / "stage4b_internal_selection_v1.csv")
    rows = a+b
    ids = [r["image_id"] for r in rows]
    if len(ids) != 8015 or len(set(ids)) != len(ids) or set(ids) != set(orig_train) or set(ids) & valtest_ids:
        raise ValueError("Manifest image IDs do not exactly partition original train")
    roles: dict[str, set] = defaultdict(set)
    diagnoses: dict[str, str] = {}
    raw_hash_groups: dict[str, set] = defaultdict(set)
    processed_hash_groups: dict[str, set] = defaultdict(set)
    for r in rows:
        image_id = r["image_id"]
        src = orig_train[image_id]
        if (r["lesion_id"] != src["lesion_id"] or r["dx"] != src["dx"] or
                r["original_split"] != "train" or r["lesion_id"] in valtest_lesions):
            raise ValueError(f"Original mapping mismatch {image_id}")
        role = r["stage4b_role"]
        if role not in ("development_train", "internal_selection"):
            raise ValueError(f"Unknown role {role}")
        roles[r["lesion_id"]].add(role)
        diagnoses[r["lesion_id"]] = r["dx"]
        for directory, column, groups in (("data/raw/images", "raw_image_sha256", raw_hash_groups),
                                          ("data/processed/images", "processed_image_sha256", processed_hash_groups)):
            digest = sha(ROOT / directory / f"{image_id}.jpg")
            if digest != r[column]:
                raise ValueError(f"Image content changed {directory}/{image_id}")
            groups[digest].add(role)
    if any(len(x) != 1 for x in roles.values()) or len(roles) != 5973:
        raise ValueError("Lesion grouping violated")
    if any(len(x) > 1 for x in raw_hash_groups.values()) or any(len(x) > 1 for x in processed_hash_groups.values()):
        raise ValueError("Cross-role exact-byte duplicate found")
    for label in CLASS_ORDER:
        lesions = [lesion for lesion, dx in diagnoses.items() if dx == label]
        lesions.sort(key=lambda x: (hashlib.sha256(f"stage4b-v1|20261010|{label}|{x}".encode()).hexdigest(),x))
        expected = set(lesions[:math.floor(.2*len(lesions)+.5)])
        actual = {lesion for lesion in lesions if "internal_selection" in roles[lesion]}
        if actual != expected:
            raise ValueError(f"Assignment algorithm mismatch for {label}")
    for role, subset in (("development_train",a),("internal_selection",b)):
        if any(r["stage4b_role"] != role for r in subset) or [r["image_id"] for r in subset] != sorted(r["image_id"] for r in subset):
            raise ValueError(f"Role/sort failure: {role}")
        expected = m["counts"][role]
        if (len(subset) != expected["images"] or len({r["lesion_id"] for r in subset}) != expected["lesions"] or
                dict(Counter(r["dx"] for r in subset)) != expected["images_by_class"] or
                dict(Counter({r["lesion_id"]:r["dx"] for r in subset}.values())) != expected["lesions_by_class"]):
            raise ValueError(f"Class/count mismatch: {role}")
    first = read(HERE / "stage4b_near_duplicate_candidates_v1.csv")
    refine = json.loads((HERE / "stage4b_phash_refinement_v1.json").read_text(encoding="utf-8"))
    if (len(first) != m["integrity"]["cross_role_dhash_le4_different_lesion_candidates"] or
            sha(HERE / "stage4b_phash_refined_candidates_v1.csv") != refine["refined_csv_sha256"] or
            len(read(HERE / "stage4b_phash_refined_candidates_v1.csv")) != refine["phash_hamming_le8_candidates"]):
        raise ValueError("Near-duplicate screen evidence mismatch")
    return {"status": "PASS_PROPOSED_SPLIT_INTEGRITY", "images": len(rows),
            "development_train": len(a), "internal_selection": len(b),
            "lesions": len(roles), "cross_role_lesions": 0,
            "cross_role_exact_duplicates": 0,
            "dhash_candidates": len(first),
            "phash_refined_candidates": refine["phash_hamming_le8_candidates"],
            "manifest_sha256": sha(HERE/"stage4b_split_manifest_v1.json")}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
