# Stage 23 validation-only uncertainty analysis

**Status: COMPLETE, EXPLORATORY VALIDATION EVIDENCE.** This analysis uses the saved Stage 23 selected epoch-14 validation predictions only. It did not load the model, run inference, train, inspect HAM test outcomes, or access PH2. It does not establish a clinical accept/review threshold. The Stage 15/20 entropy rule was not applied.

## Frozen input audit

The original `run/validation_predictions.csv` SHA256 is `b3f5d2e43b26fb63916d4e29ab597fde4183b3c76e75d4d0a7caeaccc88a368a`; the frozen split SHA256 is `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. The Stage 23 checkpoint SHA256 was independently verified as `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab` without loading it. The freeze registry and manifest agree with these hashes and class order `akiec, bcc, bkl, df, mel, nv, vasc`.

All **986** saved prediction IDs are unique and equal the frozen validation ID set. True labels match the split's `dx` values. Each probability vector has seven finite elements in `[0,1]`, sums to one within absolute tolerance `1e-5`, and its argmax and correctness flag match the saved fields. Recomputed accuracy is **0.8296146044624746**, equal to the frozen selected validation metric. There are 818 correct and 168 incorrect predictions; 70/107 melanoma cases are correct. The script fails before writing outputs if any of these checks fails.

## Probability quality and error detection

| Measure | Stage 23 validation result | Convention |
|---|---:|---|
| Classification accuracy | 0.829615 | 818/986; classification metric |
| Seven-class Brier | 0.257445 | Mean per-image sum of seven squared probability errors; not divided by seven |
| Negative log likelihood | 0.523876 nats | Saved probability assigned to the true class; logarithm clipped at smallest positive float if needed |
| Top-label ECE | 0.039768 | Ten equal-width confidence bins; left closed/right open, final bin includes 1; weighted absolute confidence–accuracy gap |
| Error-detection AUROC, entropy | 0.834869 | Misclassification is positive; larger entropy means more likely error |
| Error-detection average precision, entropy | 0.466027 | Error prevalence baseline 0.170385 |
| Error-detection AUROC, 1−max-softmax | 0.838180 | Misclassification is positive |
| Error-detection average precision, 1−max-softmax | 0.475007 | Same error prevalence baseline |

Mean top softmax score was **0.8300** for correct cases and **0.5976** for incorrect cases. Mean predictive entropy was **0.4680** versus **0.9364 nats**; normalized entropy divides by `ln(7)`. These are discrimination and calibration descriptions, not evidence that the scores are calibrated clinical probabilities. [Reliability diagram](reliability_diagram.png) and [uncertainty distributions](uncertainty_distributions.png) show the saved-score patterns.

## Selective prediction, without a deployed cutoff

At each requested coverage `c`, retain `ceil(c×986)` predictions with lowest uncertainty; exact score ties use ascending image ID. For confidence, uncertainty is `1−max-softmax`. The random-review baseline averages **500** independent fixed-seed (`42`) uniform permutations of the saved row order. Its fractional counts are means, not actual patients. Retained macro F1 uses the fixed seven classes with zero F1 for absent classes. **Retained MEL recall uses the number of true MEL cases in the retained subset as its denominator**, so it is not directly comparable to full-cohort 70/107 MEL recall. Values below are descriptive operating points selected after viewing the same validation set.

| Method | Target coverage | Retained / reviewed | Retained accuracy | Retained macro F1 | Retained MEL recall | MEL FN sent to review / 37 | Total errors sent to review / 168 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Entropy | 100% | 986 / 0 | 0.8296 | 0.6713 | 70/107 = 0.6542 | 0 | 0 |
| Entropy | 90% | 888 / 98 | 0.8694 | 0.7385 | 65/96 = 0.6771 | 6 | 52 |
| Entropy | 80% | 789 / 197 | 0.9087 | 0.7998 | 56/78 = 0.7179 | 15 | 96 |
| Entropy | 70% | 691 / 295 | 0.9291 | 0.8275 | 45/64 = 0.7031 | 18 | 119 |
| Entropy | 60% | 592 / 394 | 0.9561 | 0.8561 | 27/38 = 0.7105 | 26 | 142 |
| Entropy | 50% | 493 / 493 | 0.9675 | 0.8697 | 21/28 = 0.7500 | 30 | 152 |
| 1−max-softmax | 80% | 789 / 197 | 0.9037 | 0.7581 | 47/70 = 0.6714 | 14 | 92 |
| Random mean | 80% | 789 / 197 | 0.8297 | 0.6697 | 0.6549 mean | 7.396 mean | 33.660 mean |

The complete **18-row** table, including all three methods at 100%, 90%, 80%, 70%, 60% and 50%, retained MEL denominators, remaining MEL false negatives and error fractions, is [risk_coverage.csv](risk_coverage.csv). At 80% entropy coverage, **22/37 melanoma false negatives remain in the retained subset**; 96/168 total errors (57.14%) are sent to review. The retained 90.87% accuracy applies only to the 789 retained images, not the whole cohort or an improved classifier.

Area under the risk–coverage curve (AURC) is the discrete mean of retained error rate over every prefix `k=1,…,986`, with coverage `k/986`; no zero-coverage point or trapezoidal interpolation is used. AURC is **0.049940** for entropy, **0.049692** for `1−max-softmax`, and **0.170773** for the mean of 500 random permutations. The random expected risk at every coverage equals the full-cohort error rate **0.170385**; the small Monte Carlo difference is sampling variation. Lower AURC is better. The two model-score rankings are close on this validation set. See [risk–coverage figure](risk_coverage.png).

## Melanoma-specific findings

Among **107** validation melanomas, **70** were classified as MEL and **37** were false negatives; **26/37** false negatives were classified as NV. Mean confidence of correct MEL cases was **0.7103** versus **0.6708** for melanoma false negatives; medians were **0.7197** and **0.6892**. Mean entropy was **0.6814** versus **0.7828 nats**; medians were **0.6811** and **0.7158**. The full quartiles and ranges are in the metrics JSON. Within true MEL cases, entropy's AUROC for distinguishing false negatives from correct MEL is only **0.5927** and average precision is **0.4342**. This is considerably weaker than the all-class error AUROC and does not establish dependable melanoma miss detection.

There is substantial overlap: **15/37** melanoma false negatives had entropy at or below the median entropy of correctly classified MEL cases (`0.681053` nats). This is a descriptive comparison, not an operational threshold. The most confident melanoma false negative had a top softmax score of **0.989733**, illustrating that a confidently wrong melanoma can escape uncertainty review. At the exploratory 80% entropy-coverage point, 15 melanoma false negatives are reviewed but 22 remain retained. The per-case, non-sensitive [melanoma analysis CSV](melanoma_error_analysis.csv) records IDs, labels, confidence, entropy and MEL score only; it contains no image pixels or patient data.

## Limits and reproducibility

Stage 23 was selected on this same validation partition; analyzing many coverages and both uncertainty scores on it risks validation overfitting. No cutoff is promoted to a validated rule. Selective review changes the composition of the retained subset and cannot be described as increasing the full classifier's accuracy. ECE is a top-label summary and can hide class-specific miscalibration. The melanoma subgroup has 107 cases and 37 false negatives, so small count changes can alter subgroup conclusions. HAM test and PH2 were previously exposed elsewhere in the project and were not used here.

Reproduce using `python analysis/stage23_uncertainty_final/analyze_stage23_uncertainty.py --project-root D:\Cancerdetection\EG-VAN`. The script checks source hashes, split membership, probabilities, argmax, correctness and selected accuracy before writing outputs. It reads CSV/JSON and hashes checkpoint bytes; it never deserializes the model. Full-precision results and calculation conventions are in [uncertainty_metrics.json](uncertainty_metrics.json); all per-image confidence/entropy values are in [validation_uncertainty.csv](validation_uncertainty.csv). The [analysis manifest](analysis_manifest.json) records the analysis-script, input and generated-output hashes.
