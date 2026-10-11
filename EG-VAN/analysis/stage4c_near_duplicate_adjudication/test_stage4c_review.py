"""Focused read-only integrity checks for Stage 4C review preparation."""

import csv
import hashlib
import io
import json
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from prepare_stage4c_review import HERE, RAW_DIR, SHEETS, SOURCE_CSV, fit_uncropped


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class ReviewEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((HERE / "stage4c_review_manifest.json").read_text(encoding="utf-8"))
        cls.audit = read_csv(HERE / "NEAR_DUPLICATE_PAIR_REVIEW.csv")
        cls.blinded = read_csv(HERE / "BLINDED_REVIEW_FORM.csv")

    def test_source_and_record_identity(self):
        self.assertEqual(sha(SOURCE_CSV), self.manifest["refined_candidate_sha256"])
        self.assertEqual(len(self.audit), 40)
        self.assertEqual(len(self.blinded), 40)
        self.assertEqual([r["pair_id"] for r in self.audit], [f"P{i:03d}" for i in range(1, 41)])
        self.assertEqual([r["pair_id"] for r in self.audit], [r["pair_id"] for r in self.blinded])
        self.assertTrue(all(r["train_lesion_id"] != r["selection_lesion_id"] for r in self.audit))

    def test_no_decisions_or_role_information_in_blinded_form(self):
        for row in self.audit + self.blinded:
            self.assertEqual(row["reviewer_decision"], "")
            self.assertEqual(row["reviewer_notes"], "")
        self.assertFalse(any("train" in name or "selection" in name for name in self.blinded[0]))
        for full, blind in zip(self.audit, self.blinded):
            self.assertEqual(full["train_image_id"], blind["image_a_id"])
            self.assertEqual(full["selection_image_id"], blind["image_b_id"])

    def test_all_sheets_and_hashes(self):
        self.assertEqual(sha(HERE / "NEAR_DUPLICATE_PAIR_REVIEW.csv"), self.manifest["review_csv_sha256"])
        self.assertEqual(sha(HERE / "BLINDED_REVIEW_FORM.csv"), self.manifest["blinded_review_form_sha256"])
        self.assertEqual(set(self.manifest["contact_sheet_sha256"]), {f"P{i:03d}.png" for i in range(1, 41)})
        self.assertEqual({p.name for p in SHEETS.iterdir()}, set(self.manifest["contact_sheet_sha256"]))
        for name, expected in self.manifest["contact_sheet_sha256"].items():
            path = SHEETS / name
            self.assertEqual(sha(path), expected)
            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (2000, 1100))

    def test_source_images_and_similarity_scores(self):
        dhashes, phashes = {}, {}
        for image_id, expected in self.manifest["source_image_sha256"].items():
            path = RAW_DIR / f"{image_id}.jpg"
            self.assertEqual(sha(path), expected)
            with Image.open(io.BytesIO(path.read_bytes())) as image:
                pixels = list(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
            bits = 0
            for y in range(8):
                for x in range(8):
                    bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
            dhashes[image_id] = bits
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(gray)
            dct = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))
            low = dct[:8, :8].reshape(-1)[1:]
            median = np.median(low)
            bits = 0
            for value in low:
                bits = (bits << 1) | int(value > median)
            phashes[image_id] = bits
        for row in self.audit:
            a, b = row["train_image_id"], row["selection_image_id"]
            self.assertEqual((dhashes[a] ^ dhashes[b]).bit_count(), int(row["dhash_hamming_distance"]))
            self.assertEqual((phashes[a] ^ phashes[b]).bit_count(), int(row["phash_hamming_distance"]))

    def test_display_preserves_aspect_ratio_without_cropping(self):
        with Image.new("RGB", (800, 400), "red") as original:
            shown = fit_uncropped(original, (400, 400))
            self.assertEqual(shown.size, (400, 200))
            self.assertEqual(shown.getpixel((0, 0)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
