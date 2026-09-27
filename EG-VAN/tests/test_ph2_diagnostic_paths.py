"""Synthetic tests for cross-platform PH2 manifest path validation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

DIAGNOSTIC_DIR = Path(__file__).resolve().parents[1] / "experiments" / "ph2_preprocessing_diagnostic"
sys.path.insert(0, str(DIAGNOSTIC_DIR))

from run_diagnostic import (  # noqa: E402
    validate_manifest_image_path,
    validate_unique_image_ids,
)


class ManifestImagePathTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_dir = Path(self.temp_dir.name) / "images"
        self.image_dir.mkdir()
        self.image_path = self.image_dir / "IMD003.bmp"
        self.image_path.write_bytes(b"synthetic image")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_posix_manifest_path_passes(self):
        validate_manifest_image_path(
            "/content/drive/MyDrive/EG-VAN/data/external/ph2/images/IMD003.bmp",
            "IMD003",
            self.image_path,
        )

    def test_windows_backslash_relative_path_passes(self):
        validate_manifest_image_path(
            r"data\external\ph2\images\IMD003.bmp",
            "IMD003",
            self.image_path,
        )

    def test_windows_absolute_path_passes(self):
        validate_manifest_image_path(
            r"d:\Cancerdetection\EG-VAN\data\external\ph2\images\IMD003.bmp",
            "IMD003",
            self.image_path,
        )

    def test_wrong_basename_fails(self):
        with self.assertRaisesRegex(ValueError, "path/name mismatch"):
            validate_manifest_image_path(
                r"d:\data\ph2\images\IMD004.bmp",
                "IMD003",
                self.image_path,
            )

    def test_unexpected_extension_fails(self):
        with self.assertRaisesRegex(ValueError, "extension"):
            validate_manifest_image_path(
                r"d:\data\ph2\images\IMD003.jpg",
                "IMD003",
                self.image_path,
            )

    def test_missing_local_image_fails(self):
        missing = self.image_dir / "IMD004.bmp"
        with self.assertRaisesRegex(FileNotFoundError, "image missing"):
            validate_manifest_image_path(
                r"d:\data\ph2\images\IMD004.bmp",
                "IMD004",
                missing,
            )

    def test_duplicate_manifest_id_fails(self):
        rows = [{"image_id": "IMD003"}, {"image_id": "IMD003"}]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_unique_image_ids(rows)


if __name__ == "__main__":
    unittest.main()
