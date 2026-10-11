"""Create immutable Stage 4B HAM-train-only development manifests; no inference."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/splits/split_leakage_aware.csv"
SOURCE_SHA = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
SEED = "20261010"
RAW_DIR = ROOT / "data/raw/images"
PROCESSED_DIR = ROOT / "data/processed/images"
FIELDS = ("image_id", "lesion_id", "dx", "original_split", "stage4b_role",
          "raw_image_sha256", "processed_image_sha256")
NAMES = ("stage4b_development_train_v1.csv", "stage4b_internal_selection_v1.csv",
         "stage4b_near_duplicate_candidates_v1.csv", "stage4b_split_manifest_v1.json")
NEAR_FIELDS = ("development_train_image_id", "internal_selection_image_id",
               "development_train_lesion_id", "internal_selection_lesion_id",
               "dhash_hamming_distance")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_bytes(rows: list[dict], fields: tuple[str, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def read_source() -> tuple[list[dict], set[str], set[str]]:
    if sha_file(SOURCE) != SOURCE_SHA:
        raise ValueError("Frozen leakage-aware split hash mismatch")
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    if (len(all_rows) != 10015 or len({r["image_id"] for r in all_rows}) != 10015 or
            set(r["dx"] for r in all_rows) != set(CLASSES)):
        raise ValueError("Frozen split completeness or class order changed")
    original_train = [r for r in all_rows if r["split"] == "train"]
    valtest = [r for r in all_rows if r["split"] in ("val", "test")]
    if len(original_train) != 8015 or len(valtest) != 2000:
        raise ValueError("Frozen split partition sizes changed")
    train_lesions = {r["lesion_id"] for r in original_train}
    valtest_lesions = {r["lesion_id"] for r in valtest}
    if len(train_lesions) != 5973 or train_lesions & valtest_lesions:
        raise ValueError("Original lesion boundary changed")
    return original_train, {r["image_id"] for r in valtest}, valtest_lesions


def assign_lesions(rows: list[dict]) -> dict[str, str]:
    diagnoses: dict[str, str] = {}
    for row in rows:
        lesion, label = row["lesion_id"], row["dx"]
        if lesion in diagnoses and diagnoses[lesion] != label:
            raise ValueError(f"Conflicting diagnosis within lesion {lesion}")
        diagnoses[lesion] = label
    if len(rows) != 8015 or len(diagnoses) != 5973:
        raise ValueError("Only the full original HAM train partition is eligible")
    assignments: dict[str, str] = {}
    for label in CLASSES:
        lesions = [lesion for lesion, dx in diagnoses.items() if dx == label]
        lesions.sort(key=lambda lesion: (sha_bytes(f"stage4b-v1|{SEED}|{label}|{lesion}".encode()), lesion))
        n_selection = math.floor(.20 * len(lesions) + .5)
        if n_selection < 1 or n_selection >= len(lesions):
            raise ValueError(f"Insufficient lesion groups for {label}")
        for i, lesion in enumerate(lesions):
            assignments[lesion] = "internal_selection" if i < n_selection else "development_train"
    if len(assignments) != len(diagnoses):
        raise ValueError("Incomplete lesion assignment")
    return assignments


def dhash64(raw: bytes) -> int:
    """64-bit horizontal grayscale difference hash; a candidate screen only."""
    with Image.open(io.BytesIO(raw)) as img:
        gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        pixels = list(gray.getdata())
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | int(pixels[y*9+x] > pixels[y*9+x+1])
    return result


def near_candidates(rows: list[dict], hashes: dict[str, int]) -> list[dict]:
    train = [r for r in rows if r["stage4b_role"] == "development_train"]
    selection = [r for r in rows if r["stage4b_role"] == "internal_selection"]
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in train:
        value = hashes[row["image_id"]]
        for pos in range(8):
            buckets[(pos, (value >> (8*pos)) & 255)].append(row)
    found = []
    for selected in selection:
        value = hashes[selected["image_id"]]
        pool = {r["image_id"]: r for pos in range(8)
                for r in buckets[(pos, (value >> (8*pos)) & 255)]}
        for other in pool.values():
            if other["lesion_id"] == selected["lesion_id"]:
                continue
            distance = (value ^ hashes[other["image_id"]]).bit_count()
            if distance <= 4:
                found.append({"development_train_image_id": other["image_id"],
                              "internal_selection_image_id": selected["image_id"],
                              "development_train_lesion_id": other["lesion_id"],
                              "internal_selection_lesion_id": selected["lesion_id"],
                              "dhash_hamming_distance": distance})
    return sorted(found, key=lambda r: (int(r["dhash_hamming_distance"]),
                                       r["development_train_image_id"], r["internal_selection_image_id"]))


def make_outputs() -> tuple[dict[str, bytes], dict]:
    original, valtest_ids, valtest_lesions = read_source()
    assignments = assign_lesions(original)
    rows: list[dict] = []
    dhashes: dict[str, int] = {}
    exact: dict[str, list[dict]] = defaultdict(list)
    exact_processed: dict[str, list[dict]] = defaultdict(list)
    for source in sorted(original, key=lambda r: r["image_id"]):
        image_id = source["image_id"]
        raw_path = RAW_DIR / f"{image_id}.jpg"
        processed_path = PROCESSED_DIR / f"{image_id}.jpg"
        raw = raw_path.read_bytes()
        processed = processed_path.read_bytes()
        row = {"image_id": image_id, "lesion_id": source["lesion_id"],
               "dx": source["dx"], "original_split": "train",
               "stage4b_role": assignments[source["lesion_id"]],
               "raw_image_sha256": sha_bytes(raw), "processed_image_sha256": sha_bytes(processed)}
        rows.append(row)
        exact[row["raw_image_sha256"]].append(row)
        exact_processed[row["processed_image_sha256"]].append(row)
        dhashes[image_id] = dhash64(raw)
    train = [r for r in rows if r["stage4b_role"] == "development_train"]
    selection = [r for r in rows if r["stage4b_role"] == "internal_selection"]
    if (len(rows) != 8015 or len({r["image_id"] for r in rows}) != 8015 or
            {r["image_id"] for r in rows} & valtest_ids or
            {r["lesion_id"] for r in rows} & valtest_lesions or
            {r["lesion_id"] for r in train} & {r["lesion_id"] for r in selection} or
            set(r["dx"] for r in train) != set(CLASSES) or
            set(r["dx"] for r in selection) != set(CLASSES)):
        raise ValueError("Stage 4B partition integrity failed")
    dup = [group for group in exact.values() if len(group) > 1]
    dup_processed = [group for group in exact_processed.values() if len(group) > 1]
    cross = [group for group in dup if len({r["stage4b_role"] for r in group}) > 1]
    cross_processed = [group for group in dup_processed if len({r["stage4b_role"] for r in group}) > 1]
    if cross or cross_processed:
        raise ValueError("Exact-byte raw/processed duplicate crosses proposed boundary")
    possible_near = near_candidates(rows, dhashes)
    artifacts = {NAMES[0]: csv_bytes(train, FIELDS),
                 NAMES[1]: csv_bytes(selection, FIELDS),
                 NAMES[2]: csv_bytes(possible_near, NEAR_FIELDS)}
    source_paths = [SOURCE,
                    ROOT / "data/processed/metadata_clean.csv",
                    ROOT / "analysis/stage4a_data_governance/CONTROLLED_DEVELOPMENT_PROTOCOL.md",
                    ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/config.json",
                    ROOT / "src/train.py", ROOT / "src/dataset.py", ROOT / "src/preprocessing.py",
                    Path(__file__), OUT / "STAGE4B_SPLIT_PROTOCOL.md"]
    counts = {}
    for role, sub in (("development_train", train), ("internal_selection", selection)):
        counts[role] = {"images": len(sub), "lesions": len({r["lesion_id"] for r in sub}),
                        "images_by_class": dict(sorted(Counter(r["dx"] for r in sub).items())),
                        "lesions_by_class": dict(sorted(Counter({r["lesion_id"]:r["dx"] for r in sub}.values()).items()))}
    manifest = {
        "schema": "stage4b_development_split_v1", "status": "PROPOSED_INTERNAL_SELECTION_BOUNDARY",
        "seed": SEED, "algorithm": "per-class SHA256 sort of stage4b-v1|20261010|class|lesion_id; first floor(0.2*n+0.5) lesion IDs to internal_selection",
        "original_pool": "HAM leakage-aware train only", "original_pool_images": 8015,
        "original_pool_lesions": 5973, "class_order": CLASSES,
        "counts": counts,
        "integrity": {"all_train_images_once": True, "original_validation_or_test_images_used": False,
                      "cross_role_lesion_ids": 0, "cross_role_exact_raw_duplicate_groups": 0,
                      "cross_role_exact_processed_duplicate_groups": 0,
                      "within_pool_exact_raw_duplicate_groups": len(dup),
                      "within_pool_exact_processed_duplicate_groups": len(dup_processed),
                      "cross_role_dhash_le4_different_lesion_candidates": len(possible_near)},
        "near_duplicate_method": "64-bit raw grayscale horizontal dHash, Hamming <=4, different lesion IDs across roles; candidate screen only",
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha_file(p) for p in source_paths},
        "artifact_sha256": {name: sha_bytes(data) for name, data in artifacts.items()},
        "patient_identifiers_available": False,
        "frozen_stage23_checkpoint_training_overlap": "all original HAM train images; this split cannot evaluate that checkpoint",
        "training_performed": False, "inference_performed": False,
        "ham_validation_test_or_ph2_outcomes_used_for_assignment": False,
    }
    artifacts[NAMES[3]] = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return artifacts, manifest


def main() -> None:
    for name in NAMES:
        if (OUT / name).exists():
            raise FileExistsError(f"Refusing to overwrite {name}")
    artifacts, manifest = make_outputs()
    for name in NAMES:
        with (OUT / name).open("xb") as stream:
            stream.write(artifacts[name])
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"],
                      "integrity": manifest["integrity"],
                      "manifest_sha256": sha_bytes(artifacts[NAMES[3]])}, indent=2))


if __name__ == "__main__":
    main()
