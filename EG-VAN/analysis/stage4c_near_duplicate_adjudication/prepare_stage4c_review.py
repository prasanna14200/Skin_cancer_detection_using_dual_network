"""Prepare read-only Stage 4C near-duplicate review evidence from Stage 4B v1.

No model predictions, evaluation outcomes, or inference are used. Outputs are
created exclusively; this script will refuse to overwrite existing evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
STAGE4B = ROOT / "analysis" / "stage4b_development_split"
HERE = Path(__file__).resolve().parent
SHEETS = HERE / "NEAR_DUPLICATE_CONTACT_SHEETS"
SOURCE_CSV = STAGE4B / "stage4b_phash_refined_candidates_v1.csv"
TRAIN_CSV = STAGE4B / "stage4b_development_train_v1.csv"
SELECT_CSV = STAGE4B / "stage4b_internal_selection_v1.csv"
METADATA_CSV = ROOT / "data" / "raw" / "HAM10000_metadata.csv"
RAW_DIR = ROOT / "data" / "raw" / "images"
EXPECTED = {
    SOURCE_CSV: "66b9256855efa525c9b493fdad3b2707d125b6a0f324d3a1b1b920e91c8f2b51",
    TRAIN_CSV: "8fbbea0afc03bef91f9398b7208292d01bf30dd699433edddf3aabe3ce8d1058",
    SELECT_CSV: "a1b63582e069a45fffc71eeaa08af6c89b9fa829030266e621918bc76c28f8c1",
}
DECISIONS = (
    "Confirmed duplicate",
    "Same lesion / different capture",
    "Likely same patient",
    "Unrelated lookalike",
    "Unresolved",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def fit_uncropped(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    """Keep all pixels, aspect ratio, and color; add neutral margins only."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(box[0] / image.width, box[1] / image.height, 1.0)
    dimensions = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    if dimensions != image.size:
        image = image.resize(dimensions, Image.Resampling.LANCZOS)
    return image


def draw_sheet(pair: dict[str, str], train_meta: dict[str, str], select_meta: dict[str, str], path: Path) -> None:
    canvas = Image.new("RGB", (2000, 1100), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, meta_font = font(31), font(22)
    draw.text((50, 22), f"{pair['pair_id']}  |  dHash distance {pair['dhash_hamming_distance']}  |  pHash distance {pair['phash_hamming_distance']}", fill="black", font=title_font)
    draw.text((50, 68), "Lower hash distance means more similar; reviewer decision is not prefilled.", fill="black", font=meta_font)
    for x, side, label, meta in ((50, "train", "IMAGE A", train_meta), (1020, "selection", "IMAGE B", select_meta)):
        image_id = pair[f"{side}_image_id"]
        source = RAW_DIR / f"{image_id}.jpg"
        with Image.open(source) as original:
            original.load()
            source_size = original.size
            shown = fit_uncropped(original, (930, 730))
        y = 122
        canvas.paste(shown, (x + (930 - shown.width) // 2, y + (730 - shown.height) // 2))
        draw.rectangle((x, y, x + 930, y + 730), outline="#666666", width=2)
        info = [
            f"{label}: {image_id}  /  {pair[side + '_lesion_id']}",
            f"class={meta['dx']}  source={meta['dx_type']}  age={meta['age'] or 'missing'}  sex={meta['sex'] or 'missing'}",
            f"site={meta['localization'] or 'missing'}  original pixels={source_size[0]}x{source_size[1]}",
            f"raw: data/raw/images/{image_id}.jpg",
        ]
        for j, line in enumerate(info):
            draw.text((x, 875 + j * 47), line, fill="black", font=meta_font)
    canvas.save(path, format="PNG", optimize=True)


def prepare() -> dict:
    manifest = json.loads((STAGE4B / "stage4b_split_manifest_v1.json").read_text(encoding="utf-8"))
    # Raw metadata and cleaned metadata are checked for row agreement below;
    # their byte hashes need not be equal.
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise ValueError(f"Stage 4B input hash mismatch: {path}")
    if sha256(ROOT / "data" / "processed" / "metadata_clean.csv") != manifest["source_sha256"]["data/processed/metadata_clean.csv"]:
        raise ValueError("Cleaned metadata differs from Stage 4B source")
    candidates = rows(SOURCE_CSV)
    if len(candidates) != 40 or len({(r["development_train_image_id"], r["internal_selection_image_id"]) for r in candidates}) != 40:
        raise ValueError("Expected exactly 40 unique refined candidate pairs")
    train = {r["image_id"]: r for r in rows(TRAIN_CSV)}
    select = {r["image_id"]: r for r in rows(SELECT_CSV)}
    metadata = {r["image_id"]: r for r in rows(METADATA_CSV)}
    cleaned = {r["image_id"]: r for r in rows(ROOT / "data" / "processed" / "metadata_clean.csv")}
    out = []
    source_hashes = {}
    lesion_edges = []
    for number, row in enumerate(candidates, 1):
        train_id, select_id = row["development_train_image_id"], row["internal_selection_image_id"]
        a, b = train[train_id], select[select_id]
        if (a["lesion_id"], b["lesion_id"]) != (row["development_train_lesion_id"], row["internal_selection_lesion_id"]):
            raise ValueError(f"Candidate lesion mismatch: {train_id}, {select_id}")
        if a["lesion_id"] == b["lesion_id"] or a["stage4b_role"] != "development_train" or b["stage4b_role"] != "internal_selection":
            raise ValueError(f"Candidate does not cross different lesions/roles: {train_id}, {select_id}")
        for image_id, split_row in ((train_id, a), (select_id, b)):
            meta = metadata[image_id]
            clean = cleaned[image_id]
            if any(meta[key] != clean[key] for key in ("lesion_id", "dx", "dx_type", "age", "sex", "localization")):
                raise ValueError(f"Raw/cleaned metadata mismatch: {image_id}")
            if meta["lesion_id"] != split_row["lesion_id"] or meta["dx"] != split_row["dx"]:
                raise ValueError(f"Metadata/split mismatch: {image_id}")
            path = RAW_DIR / f"{image_id}.jpg"
            digest = sha256(path)
            if digest != split_row["raw_image_sha256"]:
                raise ValueError(f"Raw image differs from Stage 4B split: {image_id}")
            source_hashes[image_id] = digest
        if not 0 <= int(row["dhash_hamming_distance"]) <= 4 or not 0 <= int(row["phash_hamming_distance"]) <= 8:
            raise ValueError(f"Similarity screen value out of prespecified range: {train_id}, {select_id}")
        pair_id = f"P{number:03d}"
        out_row = {
            "pair_id": pair_id,
            "train_image_id": train_id, "selection_image_id": select_id,
            "train_lesion_id": a["lesion_id"], "selection_lesion_id": b["lesion_id"],
            "dhash_hamming_distance": row["dhash_hamming_distance"],
            "phash_hamming_distance": row["phash_hamming_distance"],
            "train_dx": metadata[train_id]["dx"], "selection_dx": metadata[select_id]["dx"],
            "train_dx_type": metadata[train_id]["dx_type"], "selection_dx_type": metadata[select_id]["dx_type"],
            "train_age": metadata[train_id]["age"], "selection_age": metadata[select_id]["age"],
            "train_sex": metadata[train_id]["sex"], "selection_sex": metadata[select_id]["sex"],
            "train_localization": metadata[train_id]["localization"],
            "selection_localization": metadata[select_id]["localization"],
            "train_raw_path": f"data/raw/images/{train_id}.jpg",
            "selection_raw_path": f"data/raw/images/{select_id}.jpg",
            "train_raw_sha256": source_hashes[train_id], "selection_raw_sha256": source_hashes[select_id],
            "evidence_reference": f"NEAR_DUPLICATE_CONTACT_SHEETS/{pair_id}.png",
            "reviewer_decision": "", "reviewer_notes": "",
        }
        out.append(out_row)
        lesion_edges.append((a["lesion_id"], b["lesion_id"]))
    if (HERE.joinpath("NEAR_DUPLICATE_PAIR_REVIEW.csv").exists()
            or HERE.joinpath("BLINDED_REVIEW_FORM.csv").exists()
            or HERE.joinpath("stage4c_review_manifest.json").exists()
            or SHEETS.exists()):
        raise FileExistsError("Stage 4C review evidence already exists; refusing to overwrite")
    SHEETS.mkdir(parents=False, exist_ok=False)
    sheet_hashes = {}
    for row in out:
        path = SHEETS / f"{row['pair_id']}.png"
        draw_sheet(row, metadata[row["train_image_id"]], metadata[row["selection_image_id"]], path)
        sheet_hashes[path.name] = sha256(path)
    csv_path = HERE / "NEAR_DUPLICATE_PAIR_REVIEW.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    blinded_path = HERE / "BLINDED_REVIEW_FORM.csv"
    with blinded_path.open("x", newline="", encoding="utf-8") as stream:
        blinded_fields = ("pair_id", "image_a_id", "image_b_id", "evidence_reference", "reviewer_decision", "reviewer_notes")
        writer = csv.DictWriter(stream, fieldnames=blinded_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({
            "pair_id": row["pair_id"],
            "image_a_id": row["train_image_id"],
            "image_b_id": row["selection_image_id"],
            "evidence_reference": row["evidence_reference"],
            "reviewer_decision": "", "reviewer_notes": "",
        } for row in out)
    # Connected lesion groups quantify the scope of any conservative quarantine.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    for a, b in lesion_edges:
        parent[find(a)] = find(b)
    components = defaultdict(set)
    for lesion_id in parent:
        components[find(lesion_id)].add(lesion_id)
    involved = set(parent)
    train_affected = [r for r in train.values() if r["lesion_id"] in involved]
    selection_affected = [r for r in select.values() if r["lesion_id"] in involved]
    result = {
        "status": "REVIEW_PREPARED_NO_DECISIONS",
        "stage4b_manifest_sha256": sha256(STAGE4B / "stage4b_split_manifest_v1.json"),
        "refined_candidate_sha256": sha256(SOURCE_CSV),
        "raw_metadata_sha256": sha256(METADATA_CSV),
        "cleaned_metadata_sha256": sha256(ROOT / "data" / "processed" / "metadata_clean.csv"),
        "review_csv_sha256": sha256(csv_path),
        "blinded_review_form_sha256": sha256(blinded_path),
        "contact_sheet_sha256": sheet_hashes,
        "distinct_source_images": len(source_hashes),
        "source_image_sha256": dict(sorted(source_hashes.items())),
        "candidate_pairs": len(out),
        "connected_components": len(components),
        "component_lesion_counts": sorted((len(v) for v in components.values()), reverse=True),
        "affected_lesions": len(involved),
        "affected_train_images": len(train_affected),
        "affected_selection_images": len(selection_affected),
        "affected_selection_melanoma_images": sum(r["dx"] == "mel" for r in selection_affected),
        "affected_selection_melanoma_lesions": len({r["lesion_id"] for r in selection_affected if r["dx"] == "mel"}),
        "affected_train_by_class": dict(sorted(Counter(r["dx"] for r in train_affected).items())),
        "affected_selection_by_class": dict(sorted(Counter(r["dx"] for r in selection_affected).items())),
        "patient_identifiers_available": False,
        "reviewer_decisions_filled": 0,
        "training_or_inference_performed": False,
    }
    with HERE.joinpath("stage4c_review_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


if __name__ == "__main__":
    info = prepare()
    print(json.dumps({k: v for k, v in info.items() if k not in ("source_image_sha256", "contact_sheet_sha256")}, indent=2))
