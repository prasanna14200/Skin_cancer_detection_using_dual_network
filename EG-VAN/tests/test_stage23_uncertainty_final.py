"""CPU-only checks for saved-probability research calculations."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis/stage23_uncertainty_final/analyze_stage23_uncertainty.py"
spec = importlib.util.spec_from_file_location("stage23_uncertainty", SOURCE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_tie_breaking_and_discrete_aurc():
    order = module.ranked_indices(np.array([.2, .1, .2]), np.array(["z", "a", "b"]))
    assert order.tolist() == [1, 2, 0]
    risks = module.risk_curve(np.array([True, False, True]), order)
    assert risks.tolist() == pytest.approx([0, .5, 2 / 3])
    assert risks.mean() == pytest.approx((0 + .5 + 2 / 3) / 3)


def test_saved_metrics_independent_probability_calculation():
    base = ROOT / "analysis/stage23_uncertainty_final"
    result = json.loads((base / "uncertainty_metrics.json").read_text(encoding="utf-8"))
    with (ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/validation_predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    probs = np.asarray([json.loads(row["probabilities"]) for row in rows], dtype=np.float64)
    labels = np.array([module.CLASSES.index(row["true_label"]) for row in rows])
    target = np.eye(7)[labels]
    assert len(rows) == 986
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    assert result["metrics"]["classification_accuracy"] == pytest.approx(np.mean(np.argmax(probs, axis=1) == labels))
    assert result["metrics"]["multiclass_brier_sum"] == pytest.approx(np.mean(np.sum((probs - target) ** 2, axis=1)))
    assert result["metrics"]["negative_log_likelihood_nats"] == pytest.approx(log_loss(labels, probs / probs.sum(axis=1, keepdims=True), labels=np.arange(7)))
    assert result["melanoma"]["mel_total"] == 107
    assert result["melanoma"]["mel_false_negatives"] == 37
    assert result["melanoma"]["mel_to_nv"] == 26


def test_coverage_accounting_and_figures():
    base = ROOT / "analysis/stage23_uncertainty_final"
    manifest = json.loads((base / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_sha256"]["checkpoint"] == module.CHECKPOINT_SHA
    assert manifest["analysis_source_sha256"] == module.sha256(SOURCE)
    assert all(module.sha256(base / name) == digest for name, digest in manifest["output_sha256"].items())
    per_image = pd.read_csv(base / "validation_uncertainty.csv")
    assert len(per_image) == 986 and per_image.image_id.nunique() == 986
    assert np.isfinite(per_image[["maximum_softmax_probability", "predictive_entropy_nats", "normalized_predictive_entropy"]].to_numpy()).all()
    rows = pd.read_csv(base / "risk_coverage.csv")
    assert len(rows) == 18
    assert set(rows.method) == {"entropy", "maximum_softmax_confidence", "random_review_mean"}
    assert (rows.retained + rows.reviewed == 986).all()
    deterministic = rows[rows.method != "random_review_mean"]
    assert (deterministic.mel_false_negatives_reviewed + deterministic.mel_false_negatives_retained == 37).all()
    point = deterministic[(deterministic.method == "entropy") & np.isclose(deterministic.target_coverage, .8)].iloc[0]
    assert point.retained == 789 and point.reviewed == 197
    assert point.retained_mel_true_count == 78 and point.retained_mel_true_positives == 56
    assert point.mel_false_negatives_reviewed == 15 and point.mel_false_negatives_retained == 22
    for name in ("reliability_diagram.png", "risk_coverage.png", "uncertainty_distributions.png"):
        with Image.open(base / name) as image:
            assert image.width > 100 and image.height > 100
            assert image.info.get("dpi", (0,))[0] >= 300
