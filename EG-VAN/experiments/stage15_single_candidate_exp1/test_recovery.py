"""CPU-only checks for strict saved-evidence comparison; no training or inference."""
from __future__ import annotations

import json
import inspect
import unittest

import recover_epoch16 as recovery
import train as current_runner


class RecoveryComparatorTests(unittest.TestCase):
    def test_archived_resume_restoration_matches_tested_runner(self):
        self.assertEqual(inspect.getsource(recovery.original.restore_resume),
                         inspect.getsource(current_runner.restore_resume))
        self.assertEqual(inspect.getsource(recovery.original.checked_train_batch),
                         inspect.getsource(current_runner.checked_train_batch))

    def test_saved_epochs_compare_byte_for_byte_to_themselves(self):
        for epoch in (14, 15, 16):
            with self.subTest(epoch=epoch):
                predictions_path, metrics_path = recovery.reference_paths(epoch)
                result = recovery.compare_validation(
                    epoch, predictions_path.read_bytes(), metrics_path.read_bytes())
                self.assertTrue(result["prediction_bytes_exact"])
                self.assertTrue(result["metrics_bytes_exact"])
                self.assertTrue(result["ids_true_predictions_exact"])
                self.assertTrue(result["per_class_exact"])
                self.assertTrue(result["confusion_matrix_exact"])
                self.assertTrue(result["eligibility_exact"])
                self.assertEqual(result["maximum_probability_difference"], 0.0)
                self.assertTrue(all(value == 0 for value in
                                    result["aggregate_metric_differences"].values()))

    def test_metric_change_is_not_exact(self):
        predictions_path, metrics_path = recovery.reference_paths(16)
        metric = json.loads(metrics_path.read_text(encoding="utf-8"))
        metric["validation_loss"] += 1e-7
        changed = (json.dumps(metric, indent=2) + "\n").encode("utf-8")
        result = recovery.compare_validation(16, predictions_path.read_bytes(), changed)
        self.assertFalse(result["metrics_bytes_exact"])
        self.assertGreater(result["aggregate_metric_differences"]["validation_loss"], 0)
        self.assertTrue(result["ids_true_predictions_exact"])

    def test_history_comparison_distinguishes_exact_from_close(self):
        saved = {"epoch": "16", "train_loss": "0.01897857934213287",
                 "val_loss": "0.08387391282241738", "eligible": "True"}
        exact = {"epoch": 16, "train_loss": 0.01897857934213287,
                 "val_loss": 0.08387391282241738, "eligible": True}
        self.assertTrue(recovery.compare_row(exact, saved)["exact"])
        close = {**exact, "val_loss": exact["val_loss"] + 1e-8}
        result = recovery.compare_row(close, saved)
        self.assertFalse(result["exact"])
        self.assertTrue(result["numerically_close"])


if __name__ == "__main__":
    unittest.main()
