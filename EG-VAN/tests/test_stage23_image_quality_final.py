"""CPU-only checks of Stage 23 validation quality research artifacts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis/stage23_image_quality_final"
SOURCE = BASE / "analyze_quality.py"
spec = importlib.util.spec_from_file_location("stage23_quality", SOURCE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_wilson_and_risk_ranking_direction():
    assert module.wilson(0, 0) is None
    low, high = module.wilson(5, 10)
    assert 0 < low < .5 < high < 1
    error = np.array([0, 1, 0, 1])
    score = np.array([.1, .9, .2, .8])
    result = module.rank_metrics(error, score, np.array(["a", "b", "c", "d"]))
    assert result["risk_coverage"]["0.5"]["retained_error_rate"] == 0
    assert result["aurc"] < error.mean()


def test_complete_join_and_output_hashes():
    table = pd.read_csv(BASE / "image_quality_metrics.csv")
    manifest = json.loads((BASE / "quality_analysis_manifest.json").read_text(encoding="utf-8"))
    assert len(table) == 986 and table.image_id.nunique() == 986
    assert set(table.true_class).issubset(module.CLASSES)
    assert int(table.error.sum()) == 168
    assert int(table.mel_false_negative.sum()) == 37
    assert int(table.mel_to_nv.sum()) == 26
    assert (table.raw_width > 0).all() and (table.processed_width > 0).all()
    assert np.isfinite(table[list(module.PRIMARY_FEATURES)].to_numpy()).all()
    assert manifest["analysis_source_sha256"] == module.sha256(SOURCE)
    assert all(module.sha256(BASE / name) == digest for name, digest in manifest["output_sha256"].items())
    for name in ("quality_error_distributions.png", "quality_vs_entropy.png", "quality_error_rate_analysis.png"):
        with Image.open(BASE / name) as image:
            assert image.width > 100 and image.info.get("dpi", (0,))[0] >= 300


def test_grouped_oof_and_negative_or_uncertain_added_value():
    table = pd.read_csv(BASE / "quality_ranking_oof.csv")
    comparison = json.loads((BASE / "quality_uncertainty_comparison.json").read_text(encoding="utf-8"))
    assert len(table) == 986 and table.image_id.nunique() == 986
    assert set(table.fold) == set(range(5))
    assert table.groupby("lesion_id").fold.nunique().eq(1).all()
    assert all(fold["group_overlap"] == 0 for fold in comparison["folds"])
    methods = comparison["methods"]
    assert methods["entropy_direct"]["aurc"] < methods["quality_only_oof"]["aurc"]
    assert methods["entropy_direct"]["risk_coverage"]["0.8"]["errors_reviewed"] == 96
    assert comparison["bootstrap"]["combined_minus_entropy_auroc_delta_95"][0] < 0 < comparison["bootstrap"]["combined_minus_entropy_auroc_delta_95"][1]


def test_degradation_protocol_is_fixed_and_validation_only():
    protocol = json.loads((BASE / "degradation_protocol.json").read_text(encoding="utf-8"))
    assert protocol["pilot_predictions_expected"] == 49
    assert len(protocol["cases"]) == 7 and len(protocol["variants"]) == 7
    assert "data/processed/images" in protocol["input"]
    assert protocol["checkpoint_sha256"] == module.day1.CHECKPOINT_SHA
