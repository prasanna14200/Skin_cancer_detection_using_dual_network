"""Read-only integrity and statistical-boundary checks for Stage 23 Phase A."""
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis/stage23_image_quality_final/phase_a/analyze_phase_a.py"
spec = importlib.util.spec_from_file_location("stage23_quality_phase_a", SCRIPT)
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)


class PhaseAIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = phase.load_data()
        cls.output = SCRIPT.parent

    def test_frozen_inputs_and_validation_only(self):
        hashes = phase.check_sources()
        self.assertEqual(hashes["checkpoint"], "42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab")
        self.assertEqual(len(self.data), 986)
        self.assertEqual(self.data.image_id.nunique(), 986)
        self.assertEqual(self.data.error.sum(), 168)
        self.assertEqual(self.data.groupby("lesion_id").fold.nunique().max(), 1)

    def test_group_bootstrap_preserves_lesions(self):
        sample = next(phase.cluster_samples(self.data, reps=1, seed=44))
        counts = pd.Series(sample).value_counts()
        for _, members in self.data.groupby("lesion_id").indices.items():
            multiplicity = {int(counts.get(i, 0)) for i in members}
            self.assertEqual(len(multiplicity), 1)

    def test_reproduced_oof_and_melanoma_counts(self):
        summary = json.loads((self.output / "quality_entropy_comparison.json").read_text())
        e = summary["methods"]["entropy_direct"]
        c = summary["methods"]["entropy_plus_quality_oof"]
        self.assertAlmostEqual(e["auroc"], 0.8348687274420771)
        self.assertAlmostEqual(c["auroc"], 0.8403117359413204)
        self.assertEqual(e["at_80pct_coverage"]["errors_captured"], 96)
        self.assertEqual(c["at_80pct_coverage"]["errors_captured"], 93)
        mel = self.data[self.data.true_class == "mel"]
        self.assertEqual((len(mel), int(mel.error.sum()), int(mel.mel_to_nv.sum())), (107, 37, 26))

    def test_all_features_and_finite_results(self):
        rows = pd.read_csv(self.output / "class_adjusted_quality_analysis.csv")
        self.assertEqual(len(rows), 8 * 8)
        self.assertEqual(set(rows.feature), set(phase.FEATURES))
        assert np.isfinite(rows.feature_median_all).all()
        assert rows.loc[rows.true_class == "ALL", "adjusted_error_OR_per_SD"].gt(0).all()


if __name__ == "__main__":
    unittest.main()
