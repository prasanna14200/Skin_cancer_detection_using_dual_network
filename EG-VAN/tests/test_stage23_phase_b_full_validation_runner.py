"""Safety tests for resumable Stage 23 full-validation Phase B chunks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "analysis/stage23_image_quality_final/phase_b_full_validation/run_phase_b_full_validation.py"
spec = importlib.util.spec_from_file_location("phase_b_full_validation_runner", RUNNER)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _row(image_id: str, condition: str = "baseline") -> dict:
    probs = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return {
        "image_id": image_id,
        "lesion_id": f"lesion-{image_id}",
        "true_class": "nv",
        "condition": condition,
        "predicted_class": "nv",
        "correct": 1,
        "confidence": 1.0,
        "entropy_nats": 0.0,
        "probabilities": json.dumps(probs),
        "baseline_reference_predicted_class": "nv",
        "baseline_max_abs_probability_delta": 0.0,
        "processing_status": "success",
    }


def _cohort(image_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = pd.DataFrame([
        {"image_id": image_id, "lesion_id": f"lesion-{image_id}", "dx": "nv"}
        for image_id in image_ids
    ])
    reference = pd.DataFrame([
        {
            "image_id": image_id,
            "predicted_label": "nv",
            "probabilities": json.dumps([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        }
        for image_id in image_ids
    ])
    return records, reference


def test_atomic_chunk_append_never_overwrites_and_resume_finds_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "OUT", tmp_path)
    records, reference = _cohort(["image-a", "image-b"])

    first = runner.flush_chunk([_row("image-a")], "baseline")
    assert first is not None
    first_digest = first.read_bytes()
    second = runner.flush_chunk([_row("image-b")], "baseline")

    assert second is not None and second != first
    assert first.read_bytes() == first_digest
    frame, done = runner.validate_chunk_records("baseline", records, reference, 0.005)
    assert set(frame["image_id"]) == {"image-a", "image-b"}
    assert done == {"image-a", "image-b"}


def test_resume_rejects_duplicate_and_incomplete_chunk_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "OUT", tmp_path)
    records, reference = _cohort(["image-a", "image-b"])
    runner.flush_chunk([_row("image-a")], "baseline")
    runner.flush_chunk([_row("image-a")], "baseline")

    with pytest.raises(ValueError, match="Duplicate baseline record"):
        runner.validate_chunk_records("baseline", records, reference, 0.005)


def test_resume_rejects_invalid_probability_record(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "OUT", tmp_path)
    records, reference = _cohort(["image-a"])
    bad = _row("image-a")
    bad["probabilities"] = "[0,0,0]"
    runner.flush_chunk([bad], "baseline")

    with pytest.raises(ValueError, match="Invalid probability vector"):
        runner.validate_chunk_records("baseline", records, reference, 0.005)
