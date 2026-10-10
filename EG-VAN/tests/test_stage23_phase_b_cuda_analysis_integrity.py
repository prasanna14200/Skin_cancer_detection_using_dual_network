"""Focused read-only checks for the completed CUDA table comparison."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis/stage23_image_quality_final/phase_b_full_validation_cuda/analyze_cuda_full_validation.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("stage23_cuda_analysis_integrity", SCRIPT)
assert spec and spec.loader
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def records():
    committed = pd.DataFrame([{
        "image_id": "ISIC_0025339", "lesion_id": "HAM_0006755", "true_class": "bkl",
        "condition": "baseline", "batch_index": 0, "batch_offset": 0,
        "predicted_class": "akiec", "correct": 0,
        "confidence": 0.49667981266975403,
        "entropy_nats": 0.99611453270002581,
        "probabilities": json.dumps([0.49667981266975403, 0.010111348703503609,
                                    0.4006554186344147, 0.00119326775893569,
                                    0.08966060727834702, 0.0015472222585231066,
                                    0.00015230073768179864]),
        "reference_predicted_class": "akiec", "baseline_max_abs_probability_delta": 0.0,
    }], columns=analysis.CSV_COLUMNS)
    final = committed.copy(deep=True)
    final.loc[0, "confidence"] = 0.49667981266975397
    return final, committed


def test_observed_csv_round_trip_difference_is_accepted():
    final, committed = records()
    assert final.loc[0, "confidence"] != committed.loc[0, "confidence"]
    analysis.assert_same_records(final, committed, "baseline")


@pytest.mark.parametrize("column,corruption", [
    ("predicted_class", "bkl"),
    ("image_id", "ISIC_CORRUPTED"),
    ("confidence", 0.49),
    ("entropy_nats", 0.9),
])
def test_substantive_record_corruption_is_rejected(column, corruption):
    final, committed = records()
    final.loc[0, column] = corruption
    with pytest.raises(ValueError, match=f"column={column}"):
        analysis.assert_same_records(final, committed, "baseline")


def test_any_of_seven_class_probabilities_changed_is_rejected():
    final, committed = records()
    for class_index in range(7):
        damaged = final.copy(deep=True)
        p = json.loads(damaged.loc[0, "probabilities"])
        p[class_index] += 1e-12
        damaged.loc[0, "probabilities"] = json.dumps(p)
        with pytest.raises(ValueError, match="column=probabilities"):
            analysis.assert_same_records(damaged, committed, "baseline")


def test_missing_or_reordered_rows_are_rejected():
    final, committed = records()
    with pytest.raises(ValueError, match="row count"):
        analysis.assert_same_records(final.iloc[0:0], committed, "baseline")
    doubled = pd.concat([committed, committed.assign(image_id="ISIC_NEXT")], ignore_index=True)
    with pytest.raises(ValueError, match="column=image_id"):
        analysis.assert_same_records(doubled.iloc[::-1].reset_index(drop=True), doubled, "baseline")


def test_column_order_is_not_silently_accepted():
    final, committed = records()
    reversed_columns = final[list(reversed(final.columns))]
    with pytest.raises(ValueError, match="column schema"):
        analysis.assert_same_records(reversed_columns, committed, "baseline")
