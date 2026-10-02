# Stage 6: Unified external-validation and failure-mode synthesis

## Executive summary

Frozen Experiment #5 achieved 83.3% on the 1,014-image HAM internal test and 62.5% on the 120-image PH² external follow-up. PH² produced 45 errors, including 11 in the saved HIGH/VERY_HIGH bands. These observations describe different cohorts and do not isolate domain shift as the cause.

## Research motivation and provenance

This is saved-artifact analysis. Checkpoint SHA256 `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a` and frozen split SHA256 `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` matched before writing outputs. Frozen class order: akiec, bcc, bkl, df, mel, nv, vasc. No training, inference, map regeneration, calibration fitting, or threshold selection occurred.

## Evidence sources and cohort definitions

Sources include Experiment #5 test probabilities, PH² external probabilities, domain descriptors, HAM image-quality assessment, saved uncertainty/calibration, and the frozen 12-case human Grad-CAM review. HAM is an internal held-out seven-class test. PH² is external follow-up evidence restricted to 80 mapped NV and 40 mapped MEL images; 80 atypical nevi were excluded by the existing mapping. PH² had prior project use in diagnostics and earlier evaluation.

Groups were fixed before analysis: A correct, B incorrect, C incorrect HIGH/VERY_HIGH, D incorrect LOW/MODERATE, and E correct HIGH/VERY_HIGH. HIGH means 0.75–<0.90 and VERY_HIGH 0.90–1.00, as specified by Stage 5. The 12 Grad-CAM cases retain their recorded human-review categories.

## Internal HAM and PH² context (RQ1, RQ2)

| Metric | HAM internal | PH² follow-up |
|---|---:|---:|
| N | 1014 | 120 |
| Accuracy | 0.8333 | 0.6250 |
| Mean confidence | 0.8099 | 0.6899 |
| Mean entropy, nats | 0.5140 | 0.7737 |
| ECE, 10 bins | 0.0390 | 0.0741 |
| Seven-class Brier | 0.2408 | 0.5171 |
| NLL | 0.4757 | 0.9838 |

Observed accuracy difference is -20.83 percentage points. PH² has higher ECE, Brier, NLL and entropy, and lower mean confidence in these saved predictions. Accuracy is not directly attributable to image domain: class composition and cohort selection differ. The pre-existing NV/MEL-only HAM comparison (n=783) reports accuracy 0.8799; this is a more comparable label subset, but still differs in prevalence and acquisition.

## Domain-shift and image-quality evidence (RQ4)

The saved domain analysis provides per-image brightness, contrast, sharpness, saturation, dark/bright pixel ratios, image entropy, and illumination variation for all PH² cases. The earlier `image_quality.csv` itself covers HAM only. Stage 6 joins PH² descriptors by exact image ID from `domain_features.csv` and reports descriptive outcome groups in `image_quality_failure_summary.csv`. These are image statistics, not clinical quality ratings. No broad hypothesis-testing search or causal attribution was performed.

In the saved NV/MEL domain-feature cohorts, HAM has 783 images and PH² has 120; their mean brightness is 55.20 versus 65.91, and mean contrast is 48.87 versus 46.67. All saved descriptor means are in `ham_vs_ph2_failure_context.csv`.

Incorrect minus correct PH² mean descriptor differences: brightness +21.311; contrast +2.157; sharpness +86.971; saturation -0.027; dark_pixel_ratio -0.168; bright_pixel_ratio +0.002; entropy +0.424; illumination_variation -0.188.

## External error transitions (RQ3)

| True → predicted | Count | Fraction of errors | Mean confidence |
|---|---:|---:|---:|
| MEL → NV | 15 | 33.3% | 0.689 |
| MEL → BKL | 11 | 24.4% | 0.585 |
| NV → BKL | 9 | 20.0% | 0.608 |
| NV → MEL | 8 | 17.8% | 0.616 |
| MEL → DF | 1 | 2.2% | 0.740 |
| MEL → VASC | 1 | 2.2% | 0.631 |

All observed transitions are included. MEL→NV and NV→MEL are interpreted descriptively alongside out-of-subset predictions.

## High-confidence external failures (RQ6)

There are 11 incorrect PH² predictions in HIGH/VERY_HIGH bands: IMD044, IMD020, IMD399, IMD061, IMD211, IMD285, IMD404, IMD418, IMD406, IMD411, IMD423. This is a saved-band count, not a threshold selected on PH². Full probabilities and case descriptors appear in `high_confidence_external_failures.csv`.

## Fixed 12-case Grad-CAM, uncertainty and human review (RQ5)

All 12 frozen cases join exactly. The human-reviewed set contains 4 correct and 8 incorrect cases. The saved human summary records 11 cases with noticeable border/background activation and 5 misclassified cases with visibly different predicted and true maps. These counts are confined to this stratified set. IMD085 remains an uncertain reviewed case. `gradcam_review.csv` contains pending `UNCERTAIN` placeholders; Stage 6 uses the later `stage3_case_comparison_reviewed.csv` for qualitative interpretation.

## Failure-mode taxonomy (RQ7)

Tags are nonexclusive. Direct tags encode observed errors, confidence bands, and recorded human categories. Attention-localization concern tags are tentative textual associations, not localization measurements or explanations of model behavior.

| Tag | Unique cases |
|---|---:|
| class-confusion failure | 45 |
| reviewed Grad-CAM case category | 12 |
| high-confidence misclassification | 11 |
| low-confidence misclassification | 10 |
| attention-localization concern | 7 |
| low-confidence correct prediction | 7 |

## Integrated interpretation: observation, inference, hypothesis

**Observation:** PH² accuracy and calibration metrics are worse than HAM metrics in the saved predictions; PH² has 45 errors and 11 HIGH/VERY_HIGH errors. Some selected errors co-occur with reviewed attention concerns and measured image descriptors.

**Supported inference:** This external follow-up exposes prediction and calibration weaknesses for the mapped NV/MEL subset; the combined table allows exact case-level audit.

**Unsupported hypotheses (RQ8):** Domain shift alone caused the difference; image quality caused particular errors; Grad-CAM localization caused class confusion; the model is clinically ready or generalizes to other external datasets. These require controlled evidence.

## Limitations

PH² previously contributed to project diagnostics and an earlier evaluation. Cohort composition differs from HAM. The 12 Grad-CAM cases were fixed and stratified, not a random sample. Human visual lesion boundaries have no masks or localization accuracy measure. Image statistics are imperfect proxies. No uncertainty intervals or significance claims are made.

## Recommended next experiment

Evaluate a separately prespecified, untouched external cohort with matched NV/MEL composition and independently assessed image quality and lesion localization.

## Reproducibility and figures

Run `python analysis/stage6_failure_mode_synthesis_exp5/generate_stage6_failure_synthesis.py` from any working directory. It verifies source hashes and schemas before writing. Figures show saved distributions, transitions, confidence bands, calibration context, quality descriptors, fixed-case confidence, and overlapping tag counts. No requested optional figure was skipped. `experiment_manifest.json` records source and generated SHA256 values; it excludes its own hash to avoid a recursive digest.
