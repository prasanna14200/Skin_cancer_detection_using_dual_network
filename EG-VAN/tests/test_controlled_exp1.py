import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "efficientnetv2s_controlled_exp1"
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_experiment import (  # noqa: E402
    EXPECTED_CLASS_NAMES,
    class_balanced_focal_loss,
    compute_class_weights_from_train,
)
from train import focal_loss  # noqa: E402


class ControlledExperimentOneTests(unittest.TestCase):
    def write_split(self, path: Path, rows: list[tuple[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("image_id", "lesion_id", "dx", "split"))
            for index, (label, split) in enumerate(rows):
                writer.writerow((f"image-{index}", f"lesion-{index}", label, split))

    def test_weights_use_train_counts_only_and_match_frozen_class_order(self) -> None:
        rows = [
            ("akiec", "train"),
            ("bcc", "train"),
            ("bcc", "train"),
            ("bkl", "train"),
            ("df", "train"),
            ("mel", "train"),
            ("mel", "train"),
            ("mel", "train"),
            ("nv", "train"),
            ("vasc", "train"),
            ("nv", "val"),
            ("nv", "val"),
            ("nv", "test"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "split-first.csv"
            second = Path(directory) / "split-second.csv"
            self.write_split(first, rows)
            changed_nontrain = rows + [
                ("mel", "val"),
                ("mel", "val"),
                ("akiec", "test"),
                ("akiec", "test"),
                ("akiec", "test"),
            ]
            self.write_split(second, changed_nontrain)
            weights = compute_class_weights_from_train(first)
            weights_after_nontrain_changes = compute_class_weights_from_train(second)

        self.assertEqual(tuple(EXPECTED_CLASS_NAMES), ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"))
        self.assertTrue(torch.equal(weights, weights_after_nontrain_changes))
        expected_counts = (1, 2, 1, 1, 3, 1, 1)
        total = sum(expected_counts)
        expected = torch.tensor(
            [total / (len(EXPECTED_CLASS_NAMES) * count) for count in expected_counts],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(weights, expected))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights > 0).all())

    def test_missing_train_class_is_rejected(self) -> None:
        rows = [(label, "train") for label in EXPECTED_CLASS_NAMES[:-1]]
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "split.csv"
            self.write_split(split, rows)
            with self.assertRaisesRegex(ValueError, "no samples"):
                compute_class_weights_from_train(split)

    def test_noncanonical_class_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "split.csv"
            self.write_split(split, [(label, "train") for label in EXPECTED_CLASS_NAMES])
            with self.assertRaisesRegex(ValueError, "Class order"):
                compute_class_weights_from_train(split, tuple(reversed(EXPECTED_CLASS_NAMES)))

    def test_unweighted_path_matches_baseline_focal_loss(self) -> None:
        logits = torch.tensor(
            [[1.0, -0.5, 0.3, 0.0, 0.7, -0.4, 0.1], [0.0, 0.4, -0.3, 0.2, 1.1, 0.6, -0.2]],
            dtype=torch.float32,
        )
        targets = torch.tensor([4, 1], dtype=torch.long)
        self.assertTrue(
            torch.equal(
                class_balanced_focal_loss(logits, targets, None),
                focal_loss(logits, targets),
            )
        )
        self.assertTrue(
            torch.equal(
                class_balanced_focal_loss(logits, targets, torch.ones(7)),
                focal_loss(logits, targets),
            )
        )

    def test_class_weight_is_applied_to_the_matching_target_loss(self) -> None:
        logits = torch.tensor(
            [[1.0, -0.5, 0.3, 0.0, 0.7, -0.4, 0.1], [0.0, 0.4, -0.3, 0.2, 1.1, 0.6, -0.2]],
            dtype=torch.float32,
        )
        targets = torch.tensor([4, 1], dtype=torch.long)
        weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0])
        probabilities = torch.softmax(logits, dim=1)
        pt = probabilities.gather(1, targets[:, None]).squeeze(1)
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        expected = (weights[targets] * 0.25 * (1.0 - pt).pow(2.0) * ce).mean()
        actual = class_balanced_focal_loss(logits, targets, weights)
        self.assertTrue(torch.allclose(actual, expected))

    def test_invalid_class_weights_are_rejected(self) -> None:
        logits = torch.zeros((1, len(EXPECTED_CLASS_NAMES)))
        targets = torch.tensor([0])
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            class_balanced_focal_loss(logits, targets, torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, float("nan")]))


if __name__ == "__main__":
    unittest.main()
