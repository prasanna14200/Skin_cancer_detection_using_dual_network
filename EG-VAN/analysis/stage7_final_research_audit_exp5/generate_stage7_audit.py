"""Read-only evidence audit; writes only this Stage 7 directory."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PAPER = "https://doi.org/10.1109/ACCESS.2025.3561240"
CP = "experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt"
SPLIT = "data/splits/split_leakage_aware.csv"
CP_HASH = "81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a"
SPLIT_HASH = "f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883"
F = {
    "model": "src/models/baseline_effnet.py", "train": "src/train.py", "runner": "experiments/efficientnetv2s_controlled_exp5/run_experiment.py",
    "preprocess": "src/preprocessing.py", "split_code": "src/build_splits.py", "dataset": "src/dataset.py", "gradcam_code": "src/explainability/gradcam.py",
    "project_status": "docs/PROJECT_STATUS.md", "deviations": "docs/DEVIATIONS.md", "data_provenance": "docs/PHASE_4A_DATASET_REPORT.md",
    "ph2_provenance": "docs/PHASE_8_PH2_PROVENANCE_AUDIT.md", "preprocessing_report": "docs/PHASE_4B_PREPROCESSING_REPORT.md",
    "config": "experiments/efficientnetv2s_controlled_exp5/config.json", "plan": "experiments/efficientnetv2s_controlled_exp5/experiment_plan.json",
    "history": "experiments/efficientnetv2s_controlled_exp5/training_history.json", "manifest": "experiments/efficientnetv2s_controlled_exp5/experiment_manifest.json",
    "validation_metrics": "experiments/efficientnetv2s_controlled_exp5/validation_metrics.json", "validation_predictions": "experiments/efficientnetv2s_controlled_exp5/validation_predictions.csv",
    "test_metrics": "experiments/efficientnetv2s_controlled_exp5/test_metrics.json", "test_predictions": "experiments/efficientnetv2s_controlled_exp5/test_predictions.csv",
    "ph2_config": "analysis/ph2_external_validation_exp5/external_validation_config.json", "ph2_manifest": "analysis/ph2_external_validation_exp5/ph2_manifest.csv",
    "ph2_metrics": "analysis/ph2_external_validation_exp5/ph2_metrics.json", "ph2_predictions": "analysis/ph2_external_validation_exp5/ph2_predictions.csv",
    "quality": "experiments/image_quality/image_quality.csv", "domain_metrics": "analysis/domain_shift_exp5/domain_shift_metrics.json",
    "domain_features": "analysis/domain_shift_exp5/domain_features.csv", "domain_summary": "analysis/domain_shift_exp5/domain_feature_summary.csv",
    "gradcam_manifest": "analysis/domain_shift_exp5/gradcam_comparison/ph2_gradcam_manifest.csv", "gradcam_review": "analysis/domain_shift_exp5/gradcam_comparison/gradcam_review.csv",
    "human_review": "analysis/domain_shift_exp5/stage3_synthesis/stage3_case_comparison_reviewed.csv",
    "human_summary": "analysis/domain_shift_exp5/stage3_synthesis/human_review_summary.json",
    "ham_unc": "analysis/uncertainty_calibration_exp5/ham_uncertainty_predictions.csv", "ph2_unc": "analysis/uncertainty_calibration_exp5/ph2_uncertainty_predictions.csv",
    "cal_metrics": "analysis/uncertainty_calibration_exp5/calibration_metrics.json", "cal_config": "analysis/uncertainty_calibration_exp5/uncertainty_calibration_config.json",
    "cal_manifest": "analysis/uncertainty_calibration_exp5/experiment_manifest.json", "stage6_master": "analysis/stage6_failure_mode_synthesis_exp5/ph2_master_evidence.csv",
    "stage6_summary": "analysis/stage6_failure_mode_synthesis_exp5/stage6_summary.json", "stage6_report": "analysis/stage6_failure_mode_synthesis_exp5/stage6_failure_mode_synthesis_report.md",
    "stage6_manifest": "analysis/stage6_failure_mode_synthesis_exp5/experiment_manifest.json", "stage6_taxonomy": "analysis/stage6_failure_mode_synthesis_exp5/failure_mode_taxonomy.csv",
    "stage6_context": "analysis/stage6_failure_mode_synthesis_exp5/ham_vs_ph2_failure_context.csv",
    "preprocess_comparison": "experiments/ph2_preprocessing_diagnostic/preprocessing_comparison.csv",
    "preprocess_comparison_metrics": "experiments/ph2_preprocessing_diagnostic/summary_metrics.json",
}
FIXED = set("IMD003 IMD009 IMD010 IMD020 IMD035 IMD045 IMD058 IMD061 IMD063 IMD065 IMD085 IMD168".split())
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def sha(path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            d.update(block)
    return d.hexdigest()

def j(key):
    return json.loads((ROOT / F[key]).read_text(encoding="utf-8"))

def rows(key):
    with (ROOT / F[key]).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def save_csv(name, records, fields):
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(records)

def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def cross(component, purpose, present, key, fidelity, result, required, severity, action):
    return {"Original EG-VAN component/claim": component, "Original paper purpose": purpose, "Present in our implementation?": present,
            "Evidence path": F.get(key, ""), "Reproduced exactly / approximated / modified / absent": fidelity,
            "Relevant result": result, "Required for our extension?": required, "Gap severity": severity, "Recommended action": action}

def contribution(part, classification, evidence, reason):
    return {"project_part": part, "classification": classification, "evidence_path": F.get(evidence, ""), "rationale": reason}

def claim(cid, text, kind, key, metric, strength, wording, overclaim, note=""):
    return {"claim_id": cid, "proposed_claim": text, "claim_type": kind, "supporting_artifact": F[key], "supporting_metric_or_field": metric,
            "evidence_strength": strength, "allowed_wording": wording, "prohibited_overclaim": overclaim, "notes": note}

def arrow(src, dst, status, key, reason):
    return {"from_node": src, "to_node": dst, "status": status, "evidence_path": F[key], "rationale": reason}

def gap(candidate, rank, reason, inputs, outputs, saved, inference, retraining):
    return {"candidate": candidate, "classification": rank, "reason": reason, "required_inputs": inputs, "expected_outputs": outputs,
            "frozen_predictions_sufficient": saved, "new_inference_required": inference, "retraining_required": retraining}

def main():
    for key, path in F.items():
        assert (ROOT / path).is_file(), f"Missing {key}: {path}"
    assert sha(ROOT / CP) == CP_HASH and sha(ROOT / SPLIT) == SPLIT_HASH
    source_hashes = {p: sha(ROOT / p) for p in [CP, SPLIT, *F.values()]}
    stage6_dir = ROOT / "analysis/stage6_failure_mode_synthesis_exp5"
    stage6_all = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in stage6_dir.iterdir() if p.is_file()}
    config, manifest, ham, ph, s6, cal = (j(k) for k in ("config", "manifest", "test_metrics", "ph2_metrics", "stage6_summary", "cal_metrics"))
    phrows, master, unc, grad, human = (rows(k) for k in ("ph2_predictions", "stage6_master", "ph2_unc", "gradcam_manifest", "human_review"))
    assert config["classes"] == CLASSES and ham["class_order"] == CLASSES and s6["class_order"] == CLASSES
    assert manifest["checkpoint_sha256"] == CP_HASH and manifest["split_sha256"] == SPLIT_HASH
    assert len(rows("test_predictions")) == ham["sample_count"] == 1014
    assert len(phrows) == ph["total_samples"] == len(master) == len(unc) == s6["ph2_n"] == 120
    assert len({x["image_id"] for x in phrows}) == 120
    assert {c: sum(x["true_ham_label"] == c for x in phrows) for c in ("nv", "mel")} == {"nv": 80, "mel": 40}
    correct = sum(x["true_ham_label"] == x["predicted_class"] for x in phrows)
    assert correct == 75 and len(phrows)-correct == 45 and ph["accuracy"] == correct/120 == s6["ph2_accuracy"]
    hc = sum(x["correct"] == "False" and x["confidence_band"] in ("HIGH", "VERY_HIGH") for x in unc)
    assert hc == 11 == s6["ph2_high_confidence_error_count"]
    assert {x["image_id"] for x in grad} == {x["image_id"] for x in human} == FIXED
    assert s6["fixed_gradcam_join_count"] == 12
    assert abs(ham["accuracy"] - cal["HAM10000"]["accuracy"]) < 1e-12
    assert abs(ph["accuracy"] - cal["PH2"]["accuracy"]) < 1e-12
    status_doc = (ROOT / F["project_status"]).read_text(encoding="utf-8")
    model_code = (ROOT / F["model"]).read_text(encoding="utf-8")
    assert "full EG-VAN architecture has not yet been implemented" in status_doc
    assert "only its classifier replaced" in model_code and "efficientnet_v2_s(weights=weights)" in model_code
    assert not any(p.name.lower().startswith(("scga", "nonlocal", "non_local", "mff", "fusion")) for p in (ROOT / "src").rglob("*.py"))

    crosswalk = [
        cross("EfficientNetV2S branch", "Efficient multiscale feature backbone", "Yes, standalone", "model", "modified", "Frozen seven-class single-branch Experiment #5", "Yes", "NONE for baseline; CRITICAL for full EG-VAN", "Name model EfficientNetV2S baseline"),
        cross("Modified ResNet50 branch", "Complementary global/local features", "No", "project_status", "absent", "No trained dual-branch checkpoint", "Yes for full EG-VAN claims", "CRITICAL", "Implement and evaluate only if retaining full EG-VAN claim"),
        cross("SCGA", "Local contextual attention", "No", "project_status", "absent", "No module or ablation", "Yes for full EG-VAN claims", "CRITICAL", "Implement in full architecture or remove architecture claim"),
        cross("Non-Local Block", "Long-range relationships", "No", "project_status", "absent", "No module or ablation", "Yes for full EG-VAN claims", "CRITICAL", "Implement in full architecture or remove architecture claim"),
        cross("Multi-Scale Feature Fusion (MFF)", "Aggregate features across scales", "No", "project_status", "absent", "No module or ablation", "Yes for full EG-VAN claims", "CRITICAL", "Implement in full architecture or remove architecture claim"),
        cross("Dual-branch fusion", "Combine EfficientNet and ResNet representations", "No", "model", "absent", "Classifier head replaced on one backbone", "Yes for full EG-VAN claims", "CRITICAL", "Train/evaluate actual dual-branch model"),
        cross("Hair removal", "Reduce hair artifacts", "Yes", "preprocess", "approximated", "Blackhat mask and Telea inpainting; numeric choices documented", "Yes for preprocessing description", "LOW", "Report implementation choices"),
        cross("Gray World / Retinex", "Color balancing", "Yes", "preprocess", "approximated", "Deterministic per-image correction; scale/normalization choices documented", "Yes for preprocessing description", "LOW", "Report deviations"),
        cross("Training augmentation", "Increase training diversity", "Yes", "train", "approximated", "Horizontal/vertical flips and 15-degree rotation", "Yes for baseline method", "LOW", "Do not claim paper-exact augmentation"),
        cross("Focal loss", "Downweight easy samples", "Yes", "train", "modified", "Alpha 0.25, gamma 2.0; seven-class baseline", "Yes for baseline method", "LOW", "Report exact equation and parameters"),
        cross("Class imbalance handling", "Mitigate class skew", "Yes", "config", "modified", "WeightedRandomSampler with MEL weight; no class-specific loss weights", "Yes for baseline method", "LOW", "Report sampler derivation"),
        cross("Internal evaluation", "Measure held-out classification", "Yes", "test_metrics", "modified", "HAM test n=1014; accuracy 0.8333", "Yes", "NONE", "Use exact split provenance"),
        cross("Class-wise metrics", "Show class-level behavior", "Yes", "test_metrics", "modified", "Seven HAM class metrics; PH² mapped NV/MEL metrics", "Yes", "NONE", "Distinguish different class scopes"),
        cross("Confusion matrix", "Describe error types", "Yes", "test_metrics", "modified", "HAM 7x7; PH² NV/MEL/other matrix", "Yes", "NONE", "Report PH² other-prediction policy"),
        cross("ROC/AUC", "Threshold-free discrimination", "Partial", "ph2_metrics", "modified", "PH² melanoma AUROC 0.6047; no full seven-class AUC claim", "Optional", "LOW", "Limit wording to saved melanoma AUROC"),
        cross("Architecture ablations", "Attribute gains to components", "No", "project_status", "absent", "No SCGA/NLB/MFF component comparison", "Only for architecture contribution claim", "HIGH", "Do not make component-effect claims"),
        cross("Preprocessing comparison", "Compare input processing choices", "Partial", "preprocess_comparison_metrics", "modified", "PH² raw versus HAM-processed diagnostic exists; prior PH² use", "Optional", "MEDIUM", "Do not treat diagnostic as independent causal ablation"),
        cross("Parameter count", "Measure model size", "No verified Experiment #5 count", "project_status", "absent", "No frozen parameter-count artifact found", "No for reliability claim", "LOW", "Optional technical appendix measurement"),
        cross("Computational cost / latency", "Evaluate efficiency", "Partial", "preprocess", "modified", "Preprocessing log timing exists; no controlled Experiment #5 FLOPs/latency benchmark", "No for reliability claim", "LOW", "Avoid efficiency claims or benchmark separately"),
        cross("Grad-CAM", "Visualize class activation", "Yes", "human_review", "modified", "Frozen 12 PH² cases with qualitative human review", "Yes for exploratory synthesis", "NONE", "Limit interpretation to selected cases"),
        cross("External validation", "Assess distribution transfer (beyond original paper)", "Yes for baseline", "ph2_metrics", "modified", "PH² n=120, accuracy 0.625; baseline checkpoint only", "Yes", "CRITICAL for EG-VAN-specific claim", "Name evaluated model accurately"),
        cross("Calibration", "Assess confidence reliability (beyond original paper)", "Yes for baseline", "cal_metrics", "modified", "PH² ECE 0.0741 vs HAM 0.0390", "Yes", "NONE for baseline", "Do not imply fitted calibration"),
        cross("Uncertainty", "Characterize predictive spread (beyond original paper)", "Yes for baseline", "ph2_unc", "modified", "Saved entropy, confidence, margin", "Yes", "NONE for baseline", "Use descriptive wording"),
        cross("Domain-shift / image-quality analysis", "Describe domain differences (beyond original paper)", "Yes for baseline", "domain_metrics", "modified", "HAM NV/MEL and PH² descriptors; case-level PH² joins", "Yes", "NONE for descriptive claim", "Do not claim cause"),
    ]
    save_csv("original_paper_project_crosswalk.csv", crosswalk, list(crosswalk[0]))

    contribution_rows = [
        contribution("EfficientNetV2S backbone", "A. ORIGINAL EG-VAN REPRODUCTION", "model", "Backbone family exists in paper, but only standalone branch is implemented."),
        contribution("Standalone seven-class classifier and frozen Experiment #5", "B. IMPLEMENTATION ADAPTATION", "manifest", "Paper architecture is not reproduced; model, split and selection protocol are project-specific."),
        contribution("Hair removal / Gray World / Retinex implementation", "B. IMPLEMENTATION ADAPTATION", "deviations", "Paper-informed sequence with explicitly chosen numerical parameters."),
        contribution("Training augmentation, focal loss, class sampler", "B. IMPLEMENTATION ADAPTATION", "config", "Concrete project settings differ from or extend underspecified paper procedures."),
        contribution("Internal HAM held-out evaluation", "D. SUPPORTING ANALYSIS", "test_metrics", "Establishes the frozen baseline's internal context."),
        contribution("PH² follow-up evaluation", "C. NEW EXTENSION", "ph2_metrics", "External evidence for the baseline, subject to prior PH² use and mapped subset."),
        contribution("Cross-domain image descriptors", "C. NEW EXTENSION", "domain_metrics", "Saved comparative domain and image-quality evidence."),
        contribution("Uncertainty and calibration", "C. NEW EXTENSION", "cal_metrics", "Saved-probability reliability analysis."),
        contribution("High-confidence error and Stage 6 synthesis", "C. NEW EXTENSION", "stage6_summary", "Joint descriptive audit across frozen artifacts."),
        contribution("Grad-CAM and fixed human review", "D. SUPPORTING ANALYSIS", "human_review", "Qualitative, selected-case evidence rather than population inference."),
        contribution("Modified ResNet50, SCGA, Non-Local Block, MFF and fusion", "E. NOT IMPLEMENTED", "project_status", "The defining full EG-VAN architecture is absent."),
        contribution("Original component ablations and efficiency benchmark", "E. NOT IMPLEMENTED", "project_status", "No matched component experiments or controlled Experiment #5 cost measure."),
    ]
    save_csv("contribution_crosswalk.csv", contribution_rows, list(contribution_rows[0]))

    chain = [
        arrow("Frozen Experiment #5", "Internal HAM evaluation", "SUPPORTED", "test_metrics", "Manifest ties selected checkpoint and frozen split to 1,014 held-out HAM predictions."),
        arrow("Internal HAM evaluation", "External PH² evaluation", "SUPPORTED", "ph2_config", "Same frozen Experiment #5 checkpoint and seven-class mapping; cohorts differ and PH² had prior project use."),
        arrow("External PH² evaluation", "Observed performance degradation", "SUPPORTED", "stage6_context", "Saved accuracy is 0.8333 HAM versus 0.6250 PH²; this arrow denotes comparison, not cause."),
        arrow("Observed performance degradation", "Domain/image-quality shift", "PARTIALLY SUPPORTED", "domain_metrics", "Measured domain descriptors differ, but cohort composition and acquisition confound causal attribution."),
        arrow("Domain/image-quality shift", "Uncertainty/calibration behavior", "PARTIALLY SUPPORTED", "cal_metrics", "Metrics differ across cohorts; no controlled test shows descriptors caused reliability changes."),
        arrow("Uncertainty/calibration behavior", "High-confidence errors", "SUPPORTED", "stage6_summary", "Eleven PH² errors occur in prespecified HIGH/VERY_HIGH bands."),
        arrow("High-confidence errors", "Grad-CAM + fixed human review", "PARTIALLY SUPPORTED", "human_review", "Twelve frozen stratified cases have reviews; they are not a representative sample of all 11 high-confidence errors."),
        arrow("Grad-CAM + fixed human review", "Stage 6 failure-mode synthesis", "SUPPORTED", "stage6_master", "Twelve cases join across reviewed, Grad-CAM, uncertainty and prediction artifacts; interpretations remain descriptive."),
    ]
    save_csv("evidence_chain_audit.csv", chain, list(chain[0]))

    claims = [
        claim("C01", "Experiment #5 internal and external accuracy differ", "observational", "stage6_context", "accuracy: 0.833333 vs 0.625", "DIRECT", "The frozen baseline scored 83.3% on HAM and 62.5% on PH².", "EG-VAN's full dual-branch model lost 20.8 points.", "Different cohorts and class scopes."),
        claim("C02", "PH² accuracy is lower than HAM accuracy", "observational", "stage6_summary", "ham_accuracy; ph2_accuracy", "DIRECT", "PH² observed accuracy is 20.83 percentage points lower.", "Domain shift caused the entire gap."),
        claim("C03", "Saved PH² calibration metrics are worse", "observational", "cal_metrics", "HAM10000.ece/brier_score/nll; PH2.ece/brier_score/nll", "DIRECT", "PH² ECE 0.0741 versus HAM 0.0390; Brier and NLL are also higher.", "The model is inherently miscalibrated on all external populations."),
        claim("C04", "Predictive uncertainty behavior differs", "observational", "cal_metrics", "mean_entropy; mean_confidence", "DIRECT", "PH² mean entropy is higher and mean confidence lower in saved predictions.", "Uncertainty change is caused by image quality."),
        claim("C05", "High-confidence PH² errors exist", "observational", "stage6_summary", "ph2_high_confidence_error_count=11", "DIRECT", "Eleven of 45 PH² errors are HIGH/VERY_HIGH by prespecified bands.", "High confidence implies clinical harm."),
        claim("C06", "Measurable image/domain differences exist", "observational", "domain_metrics", "feature_comparison", "DIRECT", "Saved NV/MEL cohort image descriptors differ descriptively.", "One descriptor explains performance degradation."),
        claim("C07", "Some reviewed errors show peripheral/background attention", "qualitative", "human_review", "correct; qualitative_border_attention; qualitative_background_attention", "DIRECT FOR SELECTED CASES", "Some fixed misclassified cases were reviewed as showing border/background activation.", "All PH² errors have mislocalized attention.", "Human summary reports 11/12 selected cases with noticeable border/background activation, including correct cases."),
        claim("C08", "Confidence alone does not separate all external outcomes", "descriptive inference", "stage6_master", "correct; confidence; confidence_band", "SUPPORTED", "Correct and incorrect PH² cases overlap in confidence; 11 errors fall in high bands.", "Confidence is useless or causally explains errors."),
        claim("C09", "No single quality descriptor has been established as cause", "limit", "stage6_report", "Domain-shift and image-quality evidence", "SUPPORTED AS LIMIT", "The saved descriptive analyses do not identify a causal quality determinant.", "A specific quality variable has been ruled out as a cause."),
        claim("C10", "Grad-CAM is descriptive rather than causal evidence", "methodological limit", "human_summary", "limitations", "DIRECT", "Reviewed maps illustrate selected activations; they do not establish decision mechanism or clinical validity.", "Heatmaps prove model reasoning."),
        claim("C11", "Full EG-VAN external reliability has been measured", "architecture-specific", "model", "build_model", "UNSUPPORTED", "External reliability of the EfficientNetV2S baseline was measured.", "The full EG-VAN dual-branch model was externally validated."),
        claim("C12", "PH² melanoma recall is lower", "observational", "ph2_metrics", "class_metrics.mel.recall=0.3", "DIRECT", "Mapped PH² melanoma recall is 0.30 (12/40).", "This estimates sensitivity across all clinical settings."),
    ]
    save_csv("claim_evidence_matrix.csv", claims, list(claims[0]))

    gaps = [
        gap("Full EG-VAN architecture external evaluation", "CRITICAL", "Required only if the central paper claim remains an external-validity extension of the original dual-branch EG-VAN; current checkpoint is single-branch baseline.", "Implemented ResNet50+SCGA+Non-Local+MFF/fusion; frozen HAM split; original preprocessing; PH² mapping", "Frozen dual-branch checkpoint, internal/external predictions, metrics and provenance", "NO", "YES", "YES"),
        gap("Architecture/component ablations", "USEFUL BUT OPTIONAL", "Needed for claims about individual component benefit, which current manuscript should omit.", "Trained full model and controlled component variants", "Matched ablation metrics", "NO", "YES", "YES"),
        gap("Preprocessing ablation", "USEFUL BUT OPTIONAL", "Prior PH² diagnostic is not a controlled independent causal test; no preprocessing-causal claim is needed.", "Frozen model and alternative input pipelines", "Matched performance by pipeline", "NO", "YES", "NO unless retraining is part of claim"),
        gap("Calibration method comparison", "UNNECESSARY FOR CURRENT CLAIMS", "Existing claim concerns observed reliability, not improvement after recalibration.", "Saved probabilities", "Comparative calibration metrics", "YES", "NO", "NO"),
        gap("Confidence intervals / bootstrap uncertainty", "USEFUL BUT OPTIONAL", "Adds precision estimates for modest PH² n; descriptive point estimates remain traceable.", "Saved image-level predictions; lesion IDs if cluster-aware", "Interval estimates", "YES", "NO", "NO"),
        gap("Statistical significance testing", "UNNECESSARY FOR CURRENT CLAIMS", "Descriptive cross-cohort comparison avoids an unsupported causal or population inference.", "Saved predictions and defensible sampling model", "Test statistic/p-value", "YES", "NO", "NO"),
        gap("Sensitivity/specificity, balanced accuracy, class-wise external evaluation", "UNNECESSARY FOR CURRENT CLAIMS", "Saved PH² metrics already include mapped class recall/precision and balanced accuracy; specificity can be derived under explicit other-prediction policy.", "Saved PH² predictions", "Additional derived metrics", "YES", "NO", "NO"),
        gap("MCC", "USEFUL BUT OPTIONAL", "Not needed to establish observed baseline external weakness.", "Saved PH² predictions", "MCC with defined treatment of other predictions", "YES", "NO", "NO"),
        gap("Parameter count / FLOPs / inference latency", "USEFUL BUT OPTIONAL", "Only necessary if paper makes computational-efficiency claims; current evidence does not support such claims.", "Exact model and controlled hardware/profiler", "Parameter/FLOP/latency measurements", "NO", "YES for latency", "NO"),
        gap("External dataset expansion", "USEFUL BUT OPTIONAL", "Would broaden generality beyond one previously used PH² follow-up cohort.", "Independent external dataset and mapping", "External predictions and metrics", "NO", "YES", "NO for existing baseline"),
        gap("Subgroup analysis", "USEFUL BUT OPTIONAL", "Could reveal heterogeneity if covariates and adequate sample sizes are available.", "Saved predictions plus reliable subgroup labels", "Stratified descriptive metrics", "POSSIBLY", "NO if labels exist", "NO"),
        gap("Robustness testing", "OUT OF SCOPE", "Perturbation robustness is a distinct experimental question.", "Images and prespecified perturbations", "Robustness curves", "NO", "YES", "NO"),
    ]
    save_csv("missing_experiment_audit.csv", gaps, list(gaps[0]))

    readiness = {
        "status": "REQUIRES_CRITICAL_EXPERIMENT", "central_claim_scope": "External validity and reliability of the original full dual-branch EG-VAN",
        "critical_gaps": [g for g in gaps if g["classification"] == "CRITICAL"],
        "baseline_only_alternative": "Existing evidence is adequate to begin a narrower paper explicitly about a paper-informed EfficientNetV2S baseline, without claiming full EG-VAN reproduction or validation.",
        "optional_improvements": [g["candidate"] for g in gaps if g["classification"] == "USEFUL BUT OPTIONAL"],
        "central_paper_contribution_supported_now": "A frozen, reproducible EfficientNetV2S baseline evaluated across internal HAM and external PH² cohorts, with saved domain descriptors, uncertainty/calibration, high-confidence errors, and fixed-case human-reviewed Grad-CAM synthesis.",
        "unresolved_claim": "Whether the original dual-branch EG-VAN architecture behaves similarly under external shift is untested.",
        "frozen_counts": {"ham_test_n": 1014, "ph2_n": 120, "ph2_nv_n": 80, "ph2_mel_n": 40, "ph2_correct": 75, "ph2_errors": 45, "ph2_high_confidence_errors": 11, "fixed_gradcam_cases": 12},
    }
    save_json("paper_readiness.json", readiness)

    report = [
        "# Stage 7 — final research gap and evidence audit", "",
        "## Decision", "", "**REQUIRES_CRITICAL_EXPERIMENT** for a paper claiming to extend or externally validate the *full EG-VAN architecture*. The frozen Experiment #5 is a plain EfficientNetV2S classifier with a replaced seven-class head. The modified ResNet50, SCGA, Non-Local Block, MFF, and dual-branch fusion are absent. Therefore the current PH² findings cannot be attributed to the original EG-VAN architecture. A narrower paper about the paper-informed EfficientNetV2S baseline can be drafted now if all full-architecture claims are removed and the title, abstract and methods name the evaluated model precisely.", "",
        "## Original paper and project scope", "", f"The original paper ([Saeed et al., IEEE Access 2025]({PAPER})) proposes a dual-path EfficientNetV2S/modified-ResNet50 architecture with attention and fusion, plus color balancing. Its reported nine-class results are not directly comparable to this project's seven-class HAM test or mapped NV/MEL PH² follow-up. The repository explicitly documents incomplete architecture reconstruction in `docs/PROJECT_STATUS.md`; `src/models/baseline_effnet.py` replaces only the torchvision EfficientNetV2S classifier. The current study's real addition is a well-provenanced external-validity and reliability audit **of that baseline**.", "",
        "## Repository inventory and provenance", "", f"Frozen checkpoint `{CP}`: `{CP_HASH}`. Frozen leakage-aware split `{SPLIT}`: `{SPLIT_HASH}`. Experiment #5 config, 25-epoch training history, selected epoch 15, validation predictions/metrics, and HAM test predictions/metrics are present. HAM test has 1,014 images with seven-class accuracy 0.8333. PH² manifest and predictions include 120 mapped cases (80 NV, 40 MEL); PH² accuracy is 0.6250, balanced accuracy 0.54375, and melanoma recall 0.30. The original PH² manifest includes 80 excluded atypical nevi. PH² was previously used in diagnostics/earlier evaluation, so this is follow-up external evidence, not a pristine untouched external set.", "",
        "Image quality exists for HAM in `experiments/image_quality/` and for the NV/MEL cross-domain cohorts in `analysis/domain_shift_exp5/`. Grad-CAM manifests, frozen maps, reviewed 12-case table, and human-review summary exist. Stage 5 uncertainty/calibration and Stage 6 case synthesis exist; Stage 6 source hashes and joined counts were checked. Split/data provenance is documented in Phase 4A and PH² provenance documents. Preprocessing implementation and numerical deviations are recorded in `src/preprocessing.py` and `docs/DEVIATIONS.md`. The loss, augmentation and weighted sampler are in `src/train.py` and Experiment #5 config. There are baseline and controlled EfficientNetV2S experiments, but no full architecture or component ablations. A preprocessing timing log exists, but no controlled Experiment #5 parameter/FLOP/latency benchmark was found. `docs/PROJECT_STATUS.md` predates later PH² and Stage 6 results; its statements about those later stages are stale, while its architecture-incomplete statement is consistent with the inspected model code.", "",
        "## Paper-to-project crosswalk", "", "The full row-level audit is in `original_paper_project_crosswalk.csv`. The distinction that controls interpretation is that EfficientNetV2S is present as a standalone baseline; the four defining ResNet/fusion components are absent. Hair removal and color balancing are paper-informed approximations with documented choices. Internal metrics, PH² external metrics, descriptive domain comparisons, uncertainty/calibration, and reviewed Grad-CAM are supported for the baseline.", "",
        "## Evidence chain and claim boundaries", "", "`evidence_chain_audit.csv` marks each link. Frozen checkpoint → HAM evaluation → PH² evaluation → observed lower PH² accuracy is supported by saved artifacts. Domain/image-quality differences and changed uncertainty metrics coexist with degradation, but no causal intervention isolates their effects. Eleven PH² errors occupy the saved HIGH/VERY_HIGH confidence bands. The 12 fixed Grad-CAM cases are deliberately stratified; their reviews cannot characterize all PH² errors or quantify population attention prevalence. `claim_evidence_matrix.csv` gives permitted wording for each central claim.", "",
        "## Sample sizes and legitimate conclusions", "", "HAM test n=1,014 supports held-out seven-class performance for this split. PH² n=120, with 80 NV and 40 MEL cases, supports descriptive follow-up results on those mapped classes. There are 45 PH² errors and 11 HIGH/VERY_HIGH errors, which support explicit counts and case inspection. Twelve fixed Grad-CAM cases support selected-case, human-reviewed observations only. Cross-domain accuracy, calibration and descriptor differences are descriptive; attention and image-quality explanations remain exploratory and hypothesis-generating. No current artifact supports broad clinical generalization, causation, or claims about the full original architecture.", "",
        "## Missing experiments", "", "The only critical experiment **for full EG-VAN external-validity claims** is to implement and train the actual dual-branch architecture with the documented frozen HAM split, then evaluate its frozen checkpoint internally and on the mapped PH² cohort under a prespecified protocol. Saved EfficientNetV2S predictions are insufficient; this would require training and new inference, which Stage 7 does not perform. Component ablations, confidence intervals, additional untouched external cohorts and controlled efficiency measurements are useful but optional for a narrowly descriptive baseline paper. Do not claim component gains or computational efficiency without their respective measurements.", "",
        "## Paper novelty statement", "", "Original EG-VAN contributes the proposed dual-branch attention/fusion architecture and color balancing. This repository contributes a reproducible, paper-informed **single-branch EfficientNetV2S baseline** and a linked external follow-up reliability audit: HAM/PH² performance, image/domain descriptors, uncertainty and calibration, high-confidence failures, and fixed human-reviewed Grad-CAM cases. This does not establish how the original dual-branch model would perform externally.", "",
        "## Reproducibility and audit limits", "", "All Stage 7 work used saved artifacts and read-only inspection. No model training, inference, prediction generation, Grad-CAM regeneration, or calibration fitting occurred. The source SHA256 inventory and generated-file hashes appear in `stage7_experiment_manifest.json`. The manifest excludes its own hash to avoid recursion. The original paper was consulted via its DOI; repository evidence determines implementation status.", "",
    ]
    (OUT / "stage7_final_audit_report.md").write_text("\n".join(report), encoding="utf-8")

    for name in ("original_paper_project_crosswalk.csv", "contribution_crosswalk.csv", "claim_evidence_matrix.csv", "evidence_chain_audit.csv", "missing_experiment_audit.csv"):
        with (OUT / name).open(encoding="utf-8", newline="") as f:
            assert len(list(csv.DictReader(f))) > 0
    for name in ("original_paper_project_crosswalk.csv", "contribution_crosswalk.csv", "claim_evidence_matrix.csv", "evidence_chain_audit.csv"):
        for row in csv.DictReader((OUT / name).open(encoding="utf-8", newline="")):
            for col in ("Evidence path", "evidence_path", "supporting_artifact"):
                if row.get(col):
                    assert (ROOT / row[col]).is_file(), f"Missing referenced path: {row[col]}"
    assert {p: sha(ROOT / p) for p in source_hashes} == source_hashes
    assert {p: sha(ROOT / p) for p in stage6_all} == stage6_all
    try:
        commit = subprocess.check_output(["git", "-c", "safe.directory=D:/Cancerdetection", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    generated = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "stage7_experiment_manifest.json"}
    save_json("stage7_experiment_manifest.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "git_commit_if_available": commit,
        "frozen_checkpoint_path": CP, "frozen_checkpoint_sha256": CP_HASH, "frozen_split_path": SPLIT, "frozen_split_sha256": SPLIT_HASH,
        "relevant_source_artifact_paths": list(source_hashes), "source_artifact_sha256": source_hashes,
        "generated_artifact_paths": list(generated), "generated_artifact_sha256": generated,
        "manifest_self_hash_policy": "Excluded to avoid recursive digest", "training_performed": False, "inference_performed": False,
        "gradcam_regenerated": False, "calibration_fitted": False, "stage1_to_6_artifacts_modified": False,
        "original_paper_doi": PAPER})
    print("STAGE 7 STATUS: REQUIRES_CRITICAL_EXPERIMENT")
    print("Frozen counts: HAM 1014; PH2 120, correct 75, errors 45, high-confidence errors 11; fixed Grad-CAM 12/12")
    print("Critical gap: actual dual-branch EG-VAN architecture has not been implemented or externally evaluated")
    print("Training performed: NO; inference performed: NO; frozen checkpoint modified: NO")
    print("Artifacts:", ", ".join(sorted(p.name for p in OUT.iterdir() if p.is_file())))

if __name__ == "__main__":
    main()
