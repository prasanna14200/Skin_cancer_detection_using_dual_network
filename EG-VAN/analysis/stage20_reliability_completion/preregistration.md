# Stage 20 reliability analysis protocol (frozen before Stage 20 calculations)

Date: 2026-10-06. This is a post-Stage-16, **exploratory reliability extension** around the frozen recovered Stage 15 epoch-16 EG-VAN checkpoint (SHA256 `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`). HAM test and PH² outcomes have already been viewed in the project. These data are not newly untouched confirmation sets. Nothing here changes model weights, preprocessing, seven-class argmax predictions, the Stage 16 evaluation, or the original selection rule.

## Questions and data

1. How well do the final model's saved probabilities discriminate correct from incorrect predictions on **selected epoch-16 validation**?
2. At a validation-set target of **80% retained coverage**, can one simple uncertainty rule identify cases for human review? What happens when the frozen rule is applied once to saved HAM test and, separately, mapped PH² follow-up predictions?
3. Are technical image-quality proxies associated with validation error, confidence or entropy? This is association, not causal evidence or a clinical image-quality diagnosis.

Primary rule-selection data: `experiments/stage15_single_candidate_exp1/stage15_single_candidate_exp1/run/validation_epochs/epoch_016_predictions.csv` (986 rows, seven saved probabilities). The recovered copy may be used only if hash/provenance requires it. Evaluation data: `analysis/stage16_final_evaluation/final_ham_test_predictions.csv` (1,014 rows) and `final_ph2_predictions.csv` (120 included mapped rows). All saved probabilities are used as-is; no inference. Class order: `akiec,bcc,bkl,df,mel,nv,vasc`. Join image quality using `experiments/image_quality/image_quality.csv`, whose measures are from **processed**, not raw, HAM images.

## Fixed metrics

- Validate unique IDs, known labels, seven finite probabilities in [0,1], row sum within `1e-5` of one, and saved argmax/correctness. Stop on failure.
- Maximum softmax probability (MSP) `max(p)`; predictive entropy `-sum(p log p)` with `0 log 0 := 0`; uncertainty scores `1-MSP` and entropy. These are scores, not established calibrated probabilities.
- Seven-class Brier score: mean over images of `sum_c (p_c - 1[y=c])²` (no division by seven). NLL: mean `-log(max(p_true,1e-12))`. ECE: 10 equal-width confidence bins `[0,.1),...,[.9,1]`, weighted mean absolute gap between bin mean MSP and bin accuracy. Report bin counts and a reliability plot.
- Error-detection AUROC uses incorrectness as the positive class and each uncertainty score above; tie handling by standard rank/AUC. Also show confidence distributions for correct and incorrect predictions.
- Risk–coverage curve: sort by ascending selected uncertainty, retain top `k` for each `k=1..N`, plot retained error rate against `k/N`. This curve is descriptive and does not select the rule.

## Review-rule selection — validation only

Compare validation error-detection AUROC for `1-MSP` and entropy. Choose the higher; an exact tie chooses `1-MSP`. This metric choice is validation-only. Set the operational threshold to retain **at least 80%** of validation cases by choosing the uncertainty value of the `ceil(0.8*N)`-th least-uncertain validation image; accept when score `<= threshold`, review when `> threshold`. Ties at the boundary may yield more than 80% coverage. No search for the highest validation accuracy and no HAM/PH² adjustment. Freeze the metric, threshold, source hashes, and validation operating point to `uncertainty_protocol.json` **before** computing HAM or PH² review outcomes. The rule does not claim clinical safety or a guaranteed test accuracy.

On each frozen evaluation set report coverage, review rate, retained accuracy/error rate, count and fraction of all original errors reviewed, MEL cases reviewed, and MEL false negatives reviewed. PH² OTHER predictions remain incorrect. Analyze PH² calibration for the **mapped included cohort** using the original seven-class probabilities and NV/MEL mapped truth; interpret cautiously because atypical nevi are excluded and the source/domain differ. Do not fit a calibration transform. ECE/Brier are descriptive and no temperature scaling is performed.

## Quality protocol and interpretation

Use the existing eight proxies: brightness, contrast, Laplacian-variance sharpness, saturation, dark-pixel ratio (`gray<30`), bright-pixel ratio (`gray>225`), grayscale histogram entropy, and 16×16-grid illumination variation. Thresholds 30/225 define descriptive pixel intensities, **not** accept/reject limits. On validation only, join by image ID, compare correctly/incorrectly classified distributions, report rank correlations with confidence/entropy and error rates by prespecified equal-count terciles for brightness, contrast and sharpness. Describe MEL false-negative counts by tercile if nonempty; make no significance or causal claim. No binary quality gate or quality acceptability label is asserted. A future prospective gate requires a separate label/cutoff protocol and evaluation; this analysis must not derive a cutoff from HAM test or PH².

## Success criteria and outputs

The analysis is technically complete if all three saved probability sets validate, the rule freezes from validation before held-out application, and reproducible uncertainty/calibration/quality-association outputs are saved with source hashes. This is **not** a prespecified performance-improvement criterion; a poor ECE, weak error detection, or poor transfer is reported honestly. A clinical-quality gate and final-model Grad-CAM remain partial until separately evaluated.

Outputs: `uncertainty_protocol.json`, `reliability_analysis.json`, validation/HAM/PH² reliability figures and tables, `quality_protocol.md`, `quality_analysis.json`, `final_egvan_plus_pipeline.md`, `objective_scorecard.md`, `viva_explanation.md`, `manuscript_update_plan.md`. A bounded final-checkpoint Grad-CAM script/instructions may be prepared for GPU but is not locally executed. Stage 16 artifacts remain immutable.
