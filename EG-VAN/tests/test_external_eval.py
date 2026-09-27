"""Synthetic unit tests for PH2 mapping and external evaluation policy."""

from __future__ import annotations

import sys
import tempfile
import unittest
import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from external_eval import (  # noqa: E402
    AuditError,
    audit_manifest_rows,
    binary_confusion_matrix,
    binary_metrics,
    build_eval_transform,
    map_ph2_label,
    provenance_status,
)


def sample_rows():
    return [
        {"image_id": "IMD001", "external_label": "Common Nevus", "ham_label": "nv", "included": "true"},
        {"image_id": "IMD002", "external_label": "Melanoma", "ham_label": "mel", "included": "true"},
        {"image_id": "IMD003", "external_label": "Atypical Nevus", "ham_label": "", "included": "false"},
    ]


class MappingTests(unittest.TestCase):
    def test_common_nevus_maps_to_nv(self):
        self.assertEqual(map_ph2_label("Common Nevus"), "nv")

    def test_melanoma_maps_to_mel(self):
        self.assertEqual(map_ph2_label("Melanoma"), "mel")

    def test_atypical_nevus_is_excluded(self):
        self.assertIsNone(map_ph2_label("Atypical Nevus"))
        audit = audit_manifest_rows(
            sample_rows(),
            {"IMD001": "common nevus", "IMD002": "melanoma", "IMD003": "atypical nevus"},
        )
        self.assertEqual(audit["included_count"], 2)
        self.assertEqual(audit["excluded_atypical_count"], 1)
        self.assertEqual({r["ham_label"] for r in audit["included_rows"]}, {"nv", "mel"})

    def test_unknown_label_rejected(self):
        with self.assertRaises(AuditError):
            map_ph2_label("dysplastic nevus")

    def test_label_mapping_must_match_source_metadata(self):
        rows = sample_rows()
        rows[0]["ham_label"] = "bkl"
        with self.assertRaises(AuditError):
            audit_manifest_rows(rows, {"IMD001": "common nevus", "IMD002": "melanoma", "IMD003": "atypical nevus"})

    def test_missing_metadata_row_rejected(self):
        with self.assertRaises(AuditError):
            audit_manifest_rows(sample_rows(), {"IMD001": "common nevus", "IMD002": "melanoma"})

    def test_duplicate_image_id_rejected(self):
        rows = sample_rows()
        rows[1]["image_id"] = rows[0]["image_id"]
        with self.assertRaises(AuditError):
            audit_manifest_rows(rows, {"IMD001": "common nevus", "IMD003": "atypical nevus"})

    def test_missing_image_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            image_dir = Path(temp)
            (image_dir / "IMD001.bmp").touch()
            (image_dir / "IMD003.bmp").touch()
            with self.assertRaises(AuditError):
                audit_manifest_rows(
                    sample_rows(),
                    {"IMD001": "common nevus", "IMD002": "melanoma", "IMD003": "atypical nevus"},
                    images_dir=image_dir,
                )


class BinaryMetricTests(unittest.TestCase):
    def test_out_of_overlap_predictions_are_retained_as_other_and_errors(self):
        matrix = binary_confusion_matrix(
            ["nv", "nv", "mel", "mel"],
            ["nv", "bkl", "mel", "vasc"],
        )
        self.assertEqual(matrix, [[1, 0, 1], [0, 1, 1]])
        metrics = binary_metrics(matrix)
        self.assertEqual(metrics["total"], 4)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["other_prediction_policy"].startswith("Out-of-overlap"), True)

    def test_unknown_actual_or_prediction_rejected(self):
        with self.assertRaises(AuditError):
            binary_confusion_matrix(["bkl"], ["nv"])
        with self.assertRaises(AuditError):
            binary_confusion_matrix(["nv"], ["unknown"])
        with self.assertRaises(AuditError):
            binary_confusion_matrix(["nv"], ["other"])


class TransformTests(unittest.TestCase):
    def test_eval_transform_is_deterministic_and_384_square(self):
        transform = build_eval_transform()
        image = Image.fromarray(np.full((570, 760, 3), 128, dtype=np.uint8), mode="RGB")
        first = transform(image)
        second = transform(image)
        self.assertEqual(tuple(first.shape), (3, 384, 384))
        self.assertTrue(np.array_equal(first.numpy(), second.numpy()))


class ProvenanceTests(unittest.TestCase):
    def valid_record(self):
        return {
            "status": "VERIFIED",
            "dataset_url": "https://www.kaggle.com/datasets/spacesurfer/ph2-dataset",
            "dataset_name": "PH2 Dataset",
            "uploader": "Dmitrii K (spacesurfer)",
            "revision": "Kaggle dataset version 2",
            "download_date": None,
            "explicit_permission_verified": False,
            "kaggle_secondary_source": {
                "url": "https://www.kaggle.com/datasets/spacesurfer/ph2-dataset",
                "dataset_id": 6220095,
                "ref": "spacesurfer/ph2-dataset",
                "uploader": "Dmitrii K (spacesurfer)",
                "version": 2,
                "license_field": "Other (specified in description)",
                "license_url": None,
            },
            "original_dataset_provenance": {
                "status": "DOCUMENTED",
                "source_url": "https://www.fc.up.pt/addi/ph2%20database.html",
                "access_reuse_evidence": {
                    "quotations": [
                        "The data included in the PH2 database can be used for research and educational purposes.",
                        "It is important to note that redistribution and commercial use is not allowed.",
                    ],
                    "commercial_use_permitted": False,
                    "redistribution_permitted": False,
                    "citation_required": "Cite the PH2 database paper in publications.",
                },
            },
            "source_traceability": {
                "status": "VERIFIED",
                "evidence": ["Official identity and local package concordance are recorded."],
                "integrity_report": "verification_report.json",
            },
            "intended_use": {
                "scope": "non-commercial academic research and external validation",
                "commercial_use": False,
                "redistribution": False,
            },
        }

    def check_record(self, record):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "provenance.json"
            report_path = path.parent / "verification_report.json"
            report_path.write_text(json.dumps({
                "C_local_file_integrity": {
                    "status": "VERIFIED",
                    "source_traceability_status": "VERIFIED FOR PH2 SOURCE IDENTITY",
                },
                "D_metadata_label_integrity": {"status": "VERIFIED"},
            }), encoding="utf-8")
            record["source_traceability"]["integrity_report_sha256"] = hashlib.sha256(
                report_path.read_bytes()
            ).hexdigest()
            path.write_text(json.dumps(record), encoding="utf-8")
            return provenance_status(path)

    def test_documented_research_use_passes_without_claiming_license(self):
        result = self.check_record(self.valid_record())
        self.assertEqual(result["status"], "VERIFIED")
        self.assertFalse(result["record"]["explicit_permission_verified"])
        self.assertEqual(
            result["record"]["kaggle_secondary_source"]["license_field"],
            "Other (specified in description)",
        )
        self.assertTrue(result["warnings"])

    def test_missing_authoritative_terms_remains_partial(self):
        record = self.valid_record()
        record["original_dataset_provenance"]["access_reuse_evidence"]["quotations"] = []
        self.assertEqual(self.check_record(record)["status"], "PARTIALLY VERIFIED")

    def test_commercial_or_redistribution_use_remains_partial(self):
        for field in ("commercial_use", "redistribution"):
            with self.subTest(field=field):
                record = self.valid_record()
                record["intended_use"][field] = True
                self.assertEqual(self.check_record(record)["status"], "PARTIALLY VERIFIED")

    def test_kaggle_other_field_is_not_a_license_grant(self):
        record = self.valid_record()
        record["kaggle_secondary_source"]["license_field"] = "CC0"
        self.assertEqual(self.check_record(record)["status"], "PARTIALLY VERIFIED")

    def test_unverified_source_traceability_remains_partial(self):
        record = self.valid_record()
        record["source_traceability"]["status"] = "PARTIALLY VERIFIED"
        self.assertEqual(self.check_record(record)["status"], "PARTIALLY VERIFIED")


if __name__ == "__main__":
    unittest.main()
