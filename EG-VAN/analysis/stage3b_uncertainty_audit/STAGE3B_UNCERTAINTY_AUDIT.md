# Stage 3B: frozen Stage 23 uncertainty and calibration audit

**Scope:** read-only analysis of 986 saved HAM validation predictions and 120 saved PH2 external-follow-up predictions. Stage 23 epoch-14 checkpoint SHA256 `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`; class order `akiec,bcc,bkl,df,mel,nv,vasc`. No model weights, scores, cutoffs or original prediction files were changed. PH2 was previously used in this project; its restricted NV/MEL cohort is not untouched independent validation.

## Input verification and method

The [reproducible saved-score script](analyze_saved_uncertainty.py) calls the historical Stage 23 HAM [input audit](../stage23_uncertainty_final/analyze_stage23_uncertainty.py) and PH2 [follow-up audit](../stage23_external_validation/audit_saved_followup.py). Those checks verify the frozen checkpoint and source hashes, the HAM lesion-disjoint validation split, all 986 unique HAM IDs and truth labels, all 120 fixed PH2 included IDs (80 NV, 40 MEL), seven finite normalized probabilities, saved argmax and confidence/entropy, PH2 source image hashes, and saved metrics/confusion. The prior HAM uncertainty output hashes also match its manifest. Recomputed HAM accuracy is 818/986; PH2 mapped exact-label accuracy is 81/120. Calibration uses **top-label correctness**: `predicted_class == mapped true label`, so PH2 predictions into the five other HAM classes count wrong.

Predictive entropy is `−Σ p_k ln(p_k)` in nats; confidence is maximum seven-class softmax and margin is top probability minus second highest. Misclassification is the positive class for error detection. Uncertainty scores are entropy, `1 − confidence`, and `−margin`. AUROC and average precision (AUPRC) are reported; AP's no-skill reference is error prevalence, **not 0.5**. These are research model scores, not calibrated clinical probabilities.

Top-label ECE uses **10 fixed equal-width confidence bins** `[0,.1), …, [.9,1]`; empty bins contribute zero. ECE is `Σ (n_bin/n) |mean confidence − observed top-label accuracy|`. Multiclass Brier is the mean per-image **sum over seven classes** `(p_k−1[y=k])²`, without division by seven. NLL is mean `−ln(p_true)` with only zero probabilities clipped to the smallest positive float. The [HAM](ham_reliability_diagram.png) and [PH2](ph2_reliability_diagram.png) diagrams plot nonempty bin means, observed accuracy and bin counts; they do not establish calibration at the individual-patient level.

| Cohort | ECE | Seven-class Brier sum | NLL (nats) | Interpretation |
|---|---:|---:|---:|---|
| HAM validation, 986 images | 0.03977 | 0.25745 | 0.52388 | Same validation partition used to select Stage 23; exploratory. |
| PH2 mapped follow-up, 120 images | 0.09483 | 0.44661 | 0.86310 | Truth restricted to NV/MEL while prediction remains seven-class; not directly interchangeable with HAM's seven-class distribution. |

Full precision and cluster-resampling intervals are in [CALIBRATION_METRICS.csv](CALIBRATION_METRICS.csv). A low ECE alone does not mean the seven-class outputs are calibrated for each class, under source shift, or for clinical risk. PH2's label-space restriction and prior use constrain interpretation. ECE is bin-sensitive and sparse PH2 bins are noisy.

## Error-detection discrimination

| Cohort and scope | Errors / n | Entropy AUROC; AP | `1−confidence` AUROC; AP | `−margin` AUROC; AP |
|---|---:|---:|---:|---:|
| HAM all validation | 168/986 | 0.8349; 0.4660 | 0.8382; 0.4750 | 0.8269; 0.4343 |
| HAM true MEL only | 37/107 | **0.5927; 0.4342** | 0.5718; 0.4078 | 0.5591; 0.3658 |
| PH2 all mapped cases | 39/120 | 0.7610; 0.5629 | 0.7901; 0.6295 | 0.7775; 0.5845 |
| PH2 true MEL only | 27/40 | **0.5755; 0.7547** | 0.6296; 0.7753 | 0.6154; 0.7870 |

Full precision, 95% intervals and prevalence baselines appear in [ERROR_DETECTION_METRICS.csv](ERROR_DETECTION_METRICS.csv). HAM true-MEL entropy AUROC 95% lesion-cluster interval is **0.4761–0.7071**; PH2 true-MEL entropy AUROC 95% image-ID interval is **0.3708–0.7649**. PH2 true-MEL AP appears numerically high mainly because **27/40 = 67.5%** are errors; its no-skill AP reference is 0.675. The much stronger all-class AUROCs cannot be applied to melanoma-miss detection. Cross-cohort metric gaps do not isolate a source effect because case mix and truth label spaces differ.

## Prespecified review-policy context

The existing Stage 23 [validation uncertainty report](../stage23_uncertainty_final/uncertainty_report.md) includes descriptive **rank-based** 100%, 90%, 80%, 70%, 60% and 50% retained-coverage scenarios. These were evaluated on the same reused validation data; no Stage 23 scalar threshold was independently validated. At its exploratory **80% entropy coverage** point, 789/986 HAM images were retained, retained-subset accuracy was 0.9087, 96/168 errors and 15/37 MEL false negatives were referred, and **22/37 MEL false negatives remained retained**. This is a property of a reviewed subset, not a new classifier accuracy. The historical Stage 15/20 entropy cutoff belongs to a different checkpoint and is **not** a Stage 23 policy. No cutoff from either dataset was fitted or applied to PH2 here.

Intervals in the new tables use 1,000 fixed-seed percentile resamples: whole lesions for HAM (743 clusters), image IDs for PH2 because patient/lesion linkage is unavailable. For the MEL-only rows, grouping is restricted to true-MEL cases. They describe within-cohort sampling uncertainty; PH2 image-ID intervals may be too narrow if cases share patients. No independent generalization or deployed review benefit follows. See [limitations](STAGE3B_LIMITATIONS.md) and [melanoma subgroup analysis](MELANOMA_UNCERTAINTY_ANALYSIS.md).
