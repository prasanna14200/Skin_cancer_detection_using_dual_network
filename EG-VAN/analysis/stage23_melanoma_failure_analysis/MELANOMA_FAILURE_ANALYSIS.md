# Frozen Stage 23 melanoma failure analysis

**Scope:** existing Stage 23 HAM validation predictions (986 images; 107 true MEL) and completed, qualified PH2 follow-up predictions (120 included images; 40 true MEL). This is saved-record analysis only. The PH2 cohort was previously used in the project and excludes 80 atypical nevi under a frozen mapping; it is not untouched independent validation. No inference, training, threshold optimization, checkpoint selection or preprocessing change was made.

## Evidence integrity and denominators

The [reproducible audit script](build_failure_analysis.py) checked checkpoint SHA256 `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`, split SHA256 `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`, the model's seven-class order, prediction/uncertainty/quality source hashes, and PH2's fixed manifest. HAM has 986 unique saved validation IDs, matching the frozen `val` split and 986 saved uncertainty/quality IDs with labels and lesion IDs agreeing. PH2 has 120 unique IDs matching the fixed included manifest in order, with 80 NV and 40 MEL. Every seven-class vector is finite, nonnegative, normalized within `1e-5`, and has a matching argmax; saved confidence and entropy reproduce from probabilities. PH2 image hashes match the inference manifest. No HAM test outcomes were read. Exact source hashes appear in [failure_summary.json](failure_summary.json).

| Cohort | True MEL | Correctly predicted MEL | False negatives | MEL→NV | MEL→other HAM class |
|---|---:|---:|---:|---:|---:|
| HAM validation | 107 | 70 | **37** | 26 | 11 |
| PH2 mapped follow-up | 40 | 13 | **27** | 13 | 14 |

HAM's 26 MEL→NV errors are **70.27% of its 37 MEL misses** and 24.30% of all 107 true melanomas. Its other misses are 3 BKL, 3 BCC, 4 VASC and 1 DF. PH2's 13 MEL→NV errors are **48.15% of its 27 misses** and 32.5% of all 40 true melanomas. PH2's OTHER misses are 9 BKL, 4 DF and 1 AKIEC. Because PH2 has only NV/MEL mapped truth categories while the model retains seven possible predictions, the PH2 OTHER column is substantively important and must not be merged silently with NV.

## Saved-score pattern

All figures below are descriptive within *true melanoma* cases; confidence means maximum seven-class softmax output, not clinical probability.

| Dataset and MEL subgroup | n | Median confidence | Median `p_mel` | Median predictive entropy (nats) |
|---|---:|---:|---:|---:|
| HAM correct MEL | 70 | 0.7197 | 0.7197 | 0.6811 |
| HAM MEL false negative | 37 | 0.6892 | 0.2179 | 0.7158 |
| PH2 correct MEL | 13 | 0.5504 | 0.5504 | 1.0616 |
| PH2 MEL false negative | 27 | 0.4909 | 0.1677 | 1.1525 |

The lower `p_mel` of misses largely follows **by construction** from the argmax error definition; it is not evidence of a deployable threshold. Entropy does not reliably separate true-MEL errors: its within-MEL error-detection AUROC is **0.5927** on HAM and **0.5755** on PH2. On HAM, 9/37 melanoma false negatives have top-label confidence ≥0.8 and 2/37 ≥0.9; the highest is **0.9897** (ISIC_0026296, predicted NV). On PH2, 1/27 has confidence ≥0.8 and none ≥0.9; the highest is **0.8093** (IMD284, predicted DF). These cutoffs are descriptive bins, **not** selected referral thresholds. See [uncertainty analysis](UNCERTAINTY_FAILURE_ANALYSIS.md) and the case-level error CSVs.

## Available image characteristics

HAM's eight raw-image proxies are reused from its [frozen Phase A table](../stage23_image_quality_final/phase_a/final_phase_a_report.md). For the 40 PH2 melanoma BMPs, the same deterministic `src/image_quality.py` function was applied **read-only to raw BMP pixels** so that [PH2_MELANOMA_ERRORS.csv](PH2_MELANOMA_ERRORS.csv) has available indicators for each miss. They are **image characteristics**, not human technical-quality ratings or proof of blur/exposure defects. HAM source JPEGs are uniformly 600×450; PH2 melanoma BMPs range around 764–768×575–576. Compression and pixel size differ, so absolute Laplacian-variance sharpness should not be compared as a quality score across datasets.

| Within-dataset raw proxy | Correct MEL median | MEL false-negative median | Interpretation limit |
|---|---:|---:|---|
| HAM sharpness | 91.89 | 53.75 | Phase A lesion-bootstrap rank-biserial effect for higher sharpness among misses −0.432 (95% interval −0.661 to −0.215); high-frequency texture/hair/noise also affects the proxy. |
| HAM brightness | 150.43 | 152.01 | Similar medians; darkness is not demonstrated as a HAM miss cause. |
| PH2 sharpness | 680.16 | 486.52 | Lower median among misses **within PH2**, but only 13 correct and 27 misses; different resolution/format and no technical-adequacy labels. |
| PH2 brightness | 156.28 | 118.87 | Descriptive within-PH2 difference; could reflect lesion pigment, framing or acquisition, not necessarily underexposure. |
| PH2 saturation | 0.314 | 0.360 | Descriptive difference; not a quality defect or causal effect. |

The completed [full-validation Phase B study](../stage23_image_quality_final/phase_b_full_validation_cuda/final_research_results/PHASE_B_RESULTS_SUMMARY.md) provides a **controlled within-HAM perturbation**: melanoma recall changed from 70/107 baseline to 16/107 after blur radius 1 and 2/107 after blur radius 2. Underexposure, contrast reduction and JPEG compression also reduced MEL recall. These synthetic changes support robustness vulnerability under those exact perturbations, but do not show that the PH2 errors were caused by blur, exposure, color or compression. Entropy sometimes *decreased* when degradation increased errors, so it is not a validated quality alarm.

## Interpretation

The most important observed patterns are (1) incomplete melanoma sensitivity in both datasets, (2) MEL→NV as the largest individual HAM error route, (3) a substantial PH2 OTHER route under the restricted mapping, and (4) confidently wrong HAM melanoma predictions that entropy can miss. The PH2 melanoma recall of 13/40 (32.5%) is descriptively below HAM validation's 70/107 (65.42%), but the source population, label space, class composition and image pipeline differ. No single cause is identified. Read [domain-shift hypotheses](DOMAIN_SHIFT_HYPOTHESES.md) before treating any proxy or error route as a mechanism.

The case-level CSVs retain only IDs, labels, probabilities/uncertainty, raw numeric proxies and repository-relative original-image paths. They support blinded technical review without claiming clinical lesion localization or appearance. The restricted PH2 follow-up cannot establish clinical readiness or patient-level generalization.
