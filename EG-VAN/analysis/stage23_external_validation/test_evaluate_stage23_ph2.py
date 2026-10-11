"""CPU-only tests; synthetic records never represent PH2 outcomes."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import evaluate_stage23_ph2 as study


class ExternalPreparationTests(unittest.TestCase):
    def test_frozen_protocol_identity(self):
        self.assertEqual(study.sha256(study.PROTOCOL_PATH), study.PROTOCOL_SHA)
        protocol = json.loads(study.PROTOCOL_PATH.read_text())
        self.assertEqual(protocol["inference"]["batch_size"], 16)
        self.assertEqual(protocol["cohort"]["included"], 120)
        self.assertIn("used earlier", protocol["independence_limitation"].lower())

    def test_metadata_parser_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clinical.txt"
            path.write_text("|| IMD001 || || 2 ||\n|| IMD001 || || 2 ||\n")
            with self.assertRaisesRegex(ValueError, "Invalid PH2 metadata"):
                study.metadata_codes(path)

    def test_mapping_and_binary_other_policy(self):
        # Synthetic records ensure OTHER is incorrect in mapped accuracy but
        # negative in binary MEL specificity, exactly as preregistered.
        probs = {"nv": [0,0,0,0,.1,.9,0],
                 "mel": [0,0,0,0,.9,.1,0],
                 "bkl": [0,0,.9,0,.05,.05,0]}
        rows = []
        for i in range(120):
            truth = "nv" if i < 80 else "mel"
            pred = "bkl" if i == 0 else ("mel" if i >= 80 else "nv")
            p = probs[pred]
            rows.append({"image_id": f"SYN{i:03}", "true_ham_label": truth,
                "predicted_class": pred, "confidence": max(p),
                "entropy_nats": -sum(v*math.log(v) for v in p if v),
                **{f"p_{name}": p[j] for j,name in enumerate(study.CLASSES)}})
        result = study.summarize(rows)
        self.assertEqual(result["confusion_matrix"], [[79,0,1],[0,40,0]])
        self.assertEqual(result["mapped_accuracy"], 119/120)
        self.assertEqual(result["mapped_balanced_accuracy"], (79/80+1)/2)
        self.assertEqual(result["binary_mel_specificity"], 1.0)
        rows[1]["image_id"] = rows[0]["image_id"]
        with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
            study.summarize(rows)

    def test_wilson_bounds(self):
        self.assertIsNone(study.wilson(0,0))
        lower, upper = study.wilson(20,40)
        self.assertLess(lower, .5)
        self.assertGreater(upper, .5)


if __name__ == "__main__":
    unittest.main()
