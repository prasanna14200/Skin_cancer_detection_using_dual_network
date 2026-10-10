"""CPU-only tests for the unexecuted CUDA full-validation runner."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis/stage23_image_quality_final/phase_b_full_validation_cuda/run_cuda_full_validation.py"
spec = importlib.util.spec_from_file_location("stage23_phase_b_cuda", SCRIPT)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(scope="module")
def frozen():
    return runner.frozen_inputs()


def frame_for(rows, reference, condition="baseline", batch_index=0):
    result = []
    for offset, row in enumerate(rows.itertuples(index=False)):
        ref = reference.loc[row.image_id]
        p = np.asarray(json.loads(ref.probabilities), dtype=float)
        positive = p[p > 0]
        result.append({"image_id": row.image_id, "lesion_id": row.lesion_id, "true_class": row.dx,
                       "condition": condition, "batch_index": batch_index, "batch_offset": offset,
                       "predicted_class": ref.predicted_label,
                       "correct": int(ref.predicted_label == row.dx),
                       "confidence": float(p.max()),
                       "entropy_nats": float(-(positive*np.log(positive)).sum()),
                       "probabilities": json.dumps(p.tolist()),
                       "reference_predicted_class": ref.predicted_label,
                       "baseline_max_abs_probability_delta": 0.0 if condition == "baseline" else None})
    return pd.DataFrame(result, columns=runner.CSV_COLUMNS)


def test_original_validation_batch_order_and_no_cpu_chunks(frozen):
    protocol, validation, reference, hashes = frozen
    assert len(validation) == 986 and len(runner.batches(validation)) == 62
    assert len(runner.batches(validation)[-1][1]) == 10
    assert validation.index[validation.image_id.eq("ISIC_0029026")].tolist() == [130]
    assert runner.batches(validation)[8][1].iloc[2].image_id == "ISIC_0029026"
    assert protocol["degradation_conditions"][0]["name"] == "baseline"
    assert hashes["checkpoint"] == runner.CHECKPOINT_SHA
    assert "phase_b_full_validation_cuda" in str(runner.HERE)


def test_baseline_gate_requires_exact_label_and_existing_tolerance(frozen):
    _, validation, reference, _ = frozen
    rows = runner.batches(validation)[0][1]
    frame = frame_for(rows, reference)
    runner.validate_batch(frame, rows, reference, "baseline", 0)
    wrong_label = frame.copy()
    wrong_label.loc[0, "predicted_class"] = next(c for c in runner.CLASSES if c != frame.loc[0,"predicted_class"])
    with pytest.raises(ValueError):
        runner.validate_batch(wrong_label, rows, reference, "baseline", 0)
    wrong_delta = frame.copy()
    wrong_delta.loc[0, "baseline_max_abs_probability_delta"] = .006
    with pytest.raises(ValueError):
        runner.validate_batch(wrong_delta, rows, reference, "baseline", 0)
    wrong_order = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="ordering"):
        runner.validate_batch(wrong_order, rows, reference, "baseline", 0)


def test_atomic_commit_resume_integrity_and_nonoverwrite(tmp_path, monkeypatch, frozen):
    _, validation, reference, _ = frozen
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"gpu":"Tesla T4","torch_version":torch.__version__}))
    monkeypatch.setattr(runner,"HERE",tmp_path)
    monkeypatch.setattr(runner,"PROBE",probe)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda: "Tesla T4")
    rows = runner.batches(validation)[0][1]
    frame = frame_for(rows, reference)
    protocol_sha = runner.sha(runner.FROZEN)
    runner.commit_batch(frame,"baseline",0,protocol_sha)
    loaded = runner.existing_batches("baseline",validation,reference,protocol_sha)
    assert len(loaded)==1 and loaded[0].image_id.tolist()==rows.image_id.tolist()
    with pytest.raises(FileExistsError):
        runner.commit_batch(frame,"baseline",0,protocol_sha)
    metadata = runner.batch_dir("baseline",0)/"metadata.json"
    value=json.loads(metadata.read_text())
    value["records_sha256"]="0"*64
    metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="provenance"):
        runner.existing_batches("baseline",validation,reference,protocol_sha)


def test_incomplete_baseline_blocks_degradation_and_final_analysis(tmp_path, monkeypatch, frozen):
    _, validation, reference, _ = frozen
    monkeypatch.setattr(runner,"HERE",tmp_path)
    with pytest.raises(ValueError,match="Baseline incomplete"):
        runner.complete_baseline_gate(validation,reference,runner.sha(runner.FROZEN))
    assert not (tmp_path/"baseline_gate.json").exists()
    assert not (tmp_path/"cuda_full_validation_predictions.csv").exists()
    with pytest.raises(FileNotFoundError, match="requires existing"):
        runner.execute(SimpleNamespace(check=False, run=True, resume=True, baseline_only=True))
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        analysis_path = SCRIPT.parent / "analyze_cuda_full_validation.py"
        spec2 = importlib.util.spec_from_file_location("stage23_phase_b_cuda_analysis",analysis_path)
        analysis = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(analysis)
        monkeypatch.setattr(analysis,"PREDICTIONS",tmp_path/"missing_predictions.csv")
        with pytest.raises(FileNotFoundError, match="Complete 6,902-row"):
            analysis.main()
    finally:
        sys.path.remove(str(SCRIPT.parent))


def test_condition_and_output_guards_are_fixed(frozen):
    protocol, _, _, _ = frozen
    assert [c["name"] for c in protocol["degradation_conditions"]] == [
        "baseline","blur_r1","blur_r2","underexposure_070","overexposure_130","contrast_070","jpeg_q40"]
    assert protocol["baseline_max_abs_probability_delta_tolerance"] == .005
    assert protocol["expected_predictions"] == 6902


@pytest.mark.parametrize("image_id", ["ISIC_0025339", "ISIC_0029026"])
def test_baseline_replay_matches_saved_pil_image(image_id, monkeypatch):
    baseline = {"name":"baseline","type":"identity"}
    # The baseline must use the frozen processed JPEG, with no raw replay or
    # degradation. The historical hash and pixel checks still run.
    monkeypatch.setattr(runner, "preprocess_image", lambda *_: pytest.fail("baseline reprocessed raw image"))
    monkeypatch.setattr(runner, "degraded_rgb", lambda *_: pytest.fail("baseline applied degradation"))
    replay = runner.processed_image(image_id,baseline)
    with Image.open(ROOT/"data/processed/images"/f"{image_id}.jpg") as saved:
        assert np.array_equal(np.asarray(replay),np.asarray(saved.convert("RGB")))
    assert runner.sha(ROOT/"data/processed/images"/f"{image_id}.jpg") == runner.frozen_processed_hashes()[image_id]


def test_baseline_rejects_changed_processed_jpeg(tmp_path, monkeypatch):
    image_id = "ISIC_0025339"
    target = tmp_path/"data/processed/images"/f"{image_id}.jpg"
    target.parent.mkdir(parents=True)
    original = ROOT/"data/processed/images"/f"{image_id}.jpg"
    target.write_bytes(original.read_bytes() + b"changed")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Frozen processed JPEG identity changed"):
        runner.processed_image(image_id, {"name":"baseline","type":"identity"})


def test_baseline_rejects_non_identity_condition():
    with pytest.raises(ValueError, match="Baseline condition changed"):
        runner.processed_image("ISIC_0025339", {"name":"baseline","type":"gaussian_blur"})


def test_baseline_pixel_assertion_rejects_changed_pixels():
    image_id = "ISIC_0025339"
    saved_path = ROOT/"data/processed/images"/f"{image_id}.jpg"
    with Image.open(saved_path) as saved:
        changed = np.asarray(saved.convert("RGB")).copy()
    changed[0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="Baseline preprocessing is not pixel-identical"):
        runner.assert_baseline_pixel_identity(Image.fromarray(changed), saved_path, image_id)


def test_paired_analysis_uses_same_images_and_lesions(frozen):
    _, validation, reference, _ = frozen
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        analysis_path = SCRIPT.parent / "analyze_cuda_full_validation.py"
        spec2 = importlib.util.spec_from_file_location("stage23_phase_b_cuda_analysis_paired",analysis_path)
        analysis = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(analysis)
        baseline = validation.rename(columns={"dx":"true_class"}).copy()
        baseline["predicted_class"] = [reference.loc[x,"predicted_label"] for x in baseline.image_id]
        baseline["correct"] = (baseline.predicted_class == baseline.true_class).astype(int)
        baseline["entropy_nats"] = .5
        degraded = baseline.copy()
        old = degraded.loc[0,"predicted_class"]
        degraded.loc[0,"predicted_class"] = next(c for c in runner.CLASSES if c != old)
        degraded.loc[0,"correct"] = int(degraded.loc[0,"predicted_class"] == degraded.loc[0,"true_class"])
        degraded["entropy_nats"] = .6
        result, paired = analysis.paired_statistics(None,degraded,baseline)
        assert result["images"] == 986 and result["prediction_flips"] == 1
        assert result["lesions"] == validation.lesion_id.nunique()
        assert result["mean_entropy_delta"] == pytest.approx(.1)
        assert len(paired) == 986
    finally:
        sys.path.remove(str(SCRIPT.parent))
