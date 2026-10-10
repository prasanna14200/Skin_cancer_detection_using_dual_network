"""Read-only integrity checks for bounded Stage 23 Phase B results."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/stage23_image_quality_final/phase_b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_protocol_precedes_output_and_uses_distinct_lesions():
    p = json.loads((OUT / "phase_b_protocol.json").read_text())
    assert p["status"] == "FROZEN_BEFORE_INFERENCE"
    assert len(p["cases"]) == 21 and len({x["lesion_id"] for x in p["cases"]}) == 21
    assert pd.Series([x["true_class"] for x in p["cases"]]).value_counts().eq(3).all()
    assert (OUT / "phase_b_protocol.json").stat().st_mtime < (OUT / "phase_b_predictions.csv").stat().st_mtime
    assert p["primary_application"] == "raw RGB before frozen paper preprocessing"


def test_manifest_and_checkpoint_hashes():
    m = json.loads((OUT / "phase_b_manifest.json").read_text())
    assert m["status"] == "COMPLETE_BOUNDED_SAMPLE" and m["predictions"] == 147
    assert all(sha(OUT / name) == digest for name, digest in m["output_sha256"].items())
    assert m["input_and_source_sha256"]["checkpoint"] == sha(
        ROOT / "experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt"
    ) == "42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab"


def test_complete_predictions_baseline_and_pairing():
    t = pd.read_csv(OUT / "phase_b_predictions.csv")
    assert len(t) == 147 and t.image_id.nunique() == t.lesion_id.nunique() == 21
    assert not t.duplicated(["image_id", "condition"]).any()
    assert t.groupby("image_id").condition.nunique().eq(7).all()
    baseline = t[t.condition == "baseline"]
    assert (baseline.predicted_class == baseline.saved_stage23_predicted_class).all()
    assert baseline.baseline_max_abs_probability_delta.max() < .005
    assert int(baseline.correct.sum()) == 15
    assert (t.prediction_changed == (t.predicted_class != t.baseline_predicted_class).astype(int)).all()
    assert (t.correct_to_incorrect == ((t.baseline_correct == 1) & (t.correct == 0)).astype(int)).all()
    assert (t.incorrect_to_correct == ((t.baseline_correct == 0) & (t.correct == 1)).astype(int)).all()
    probabilities = np.asarray([json.loads(x) for x in t.probabilities], dtype=float)
    assert probabilities.shape == (147, 7) and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)


def test_reported_summary_and_intervals_recompute():
    t = pd.read_csv(OUT / "phase_b_predictions.csv")
    s = pd.read_csv(OUT / "phase_b_summary.csv")
    stats = json.loads((OUT / "phase_b_statistical_results.json").read_text())
    assert len(s) == 7 and stats["bootstrap_repetitions"] == 1000
    for r in s.itertuples():
        sub = t[t.condition == r.condition]
        mel = sub[sub.true_class == "mel"]
        assert len(sub) == 21 and len(mel) == 3
        assert int(sub.correct.sum()) == r.correct
        assert int(sub.prediction_changed.sum()) == r.prediction_flips
        assert int(mel.correct.sum()) == r.mel_correct
        assert np.isclose(sub.entropy_delta.mean(), r.mean_entropy_delta)
        assert r.condition in stats["paired_lesion_bootstrap_95"]


def test_figures_exist_at_publication_resolution():
    for name in ("phase_b_robustness_curves.png", "phase_b_prediction_transitions.png", "phase_b_melanoma_robustness.png"):
        with Image.open(OUT / name) as image:
            # PNG stores pixels per metre, so nominal 300 DPI round-trips as 299.9994.
            assert image.width > 100 and image.info.get("dpi", (0,))[0] >= 299.9
