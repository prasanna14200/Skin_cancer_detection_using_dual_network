# Stage 23 Phase B: full-validation results summary

All numbers below come from the hash-verified **986-image Stage 23 validation** prediction table. The same 743 lesions and 107 melanoma images appear in each condition. Changes are paired to the unchanged baseline; confidence intervals are 2.5/97.5 percentiles from 1,000 lesion-cluster bootstrap resamples in the frozen analysis. Percentage-point (pp) changes are not relative percentages. These are exploratory synthetic-degradation comparisons, without multiplicity adjustment.

| Condition | Accuracy | Δ accuracy (pp) | Macro F1 | Δ macro F1 | MEL recall | Δ MEL recall (pp) | Flips / 986 | Mean entropy (nats) | Δ entropy (nats) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 82.96% | 0 | 0.6713 | 0 | 70/107 (65.42%) | 0 | 0 | 0.5478 | 0 |
| Blur radius 1 | 72.41% | −10.55 | 0.3940 | −0.2773 | 16/107 (14.95%) | −50.47 | 212 (21.50%) | 0.4064 | −0.1414 |
| Blur radius 2 | 69.17% | −13.79 | 0.2722 | −0.3990 | 2/107 (1.87%) | −63.55 | 273 (27.69%) | 0.5678 | +0.0201 |
| Underexposure ×0.70 | 79.61% | −3.35 | 0.5929 | −0.0784 | 33/107 (30.84%) | −34.58 | 133 (13.49%) | 0.4838 | −0.0640 |
| Overexposure ×1.30 | 68.46% | −14.50 | 0.4395 | −0.2318 | 67/107 (62.62%) | −2.80 | 217 (22.01%) | 0.7515 | +0.2037 |
| Contrast ×0.70 | 78.70% | −4.26 | 0.5472 | −0.1241 | 27/107 (25.23%) | −40.19 | 152 (15.42%) | 0.4637 | −0.0841 |
| JPEG quality 40 | 76.17% | −6.80 | 0.5097 | −0.1616 | 38/107 (35.51%) | −29.91 | 177 (17.95%) | 0.6563 | +0.1085 |

## Cluster-bootstrap intervals for paired changes

Intervals are shown in the native units of the change. Accuracy and MEL recall entries below are percentage points; macro F1 is on its 0–1 scale; entropy is in nats.

| Condition | Δ accuracy 95% CI (pp) | Δ macro F1 95% CI | Δ MEL recall 95% CI (pp) | Δ mean entropy 95% CI (nats) |
|---|---:|---:|---:|---:|
| Blur radius 1 | [−13.43, −7.81] | [−0.3475, −0.2088] | [−60.58, −40.70] | [−0.1633, −0.1182] |
| Blur radius 2 | [−17.03, −10.44] | [−0.4908, −0.3178] | [−74.77, −51.56] | [−0.0072, +0.0476] |
| Underexposure ×0.70 | [−5.63, −1.11] | [−0.1243, −0.0318] | [−43.75, −25.27] | [−0.0795, −0.0486] |
| Overexposure ×1.30 | [−17.19, −11.77] | [−0.2860, −0.1758] | [−9.80, +3.74] | [+0.1816, +0.2267] |
| Contrast ×0.70 | [−6.70, −1.78] | [−0.1743, −0.0745] | [−50.01, −30.43] | [−0.1030, −0.0664] |
| JPEG quality 40 | [−9.49, −4.20] | [−0.2373, −0.0893] | [−40.22, −20.00] | [+0.0847, +0.1304] |

The machine-readable `condition_comparison.csv` also contains the lesion-cluster intervals for prediction-flip rate. All six accuracy and macro-F1 change intervals are below zero. The overexposure MEL-recall interval includes zero; it does not establish a change in MEL recall despite substantial aggregate degradation.

## Melanoma and uncertainty

At baseline, 70/107 melanomas were correctly classified and 37 were false negatives, including 26 predicted as NV. Blur radius 1 caused 54 previously correct melanomas to become incorrect; blur radius 2 caused 68. The resulting false-negative counts were 91 and 105, respectively, with 74 and 95 MEL→NV cases. Underexposure caused 37 previously correct MEL cases to become incorrect; contrast reduction caused 43; JPEG quality 40 caused 33 losses and one recovery. Overexposure caused 10 losses and seven recoveries, for a net three fewer correctly identified melanomas. These are paired **image-level** transitions; the bootstrap intervals above resample at the **lesion** level.

Mean entropy **decreased** under radius-1 blur, underexposure and contrast reduction, even while errors increased. Entropy increased under overexposure and JPEG compression. Radius-2 blur's mean entropy change was +0.0201 nats, with an interval spanning zero. Thus entropy does not rise reliably as these synthetic images degrade. Within each condition, higher entropy nevertheless ranks incorrect predictions above correct ones reasonably well: descriptive error-detection AUROC ranges from 0.7418 to 0.8440 (baseline 0.8349). These AUROCs were calculated on the same validation records, have no reported uncertainty intervals, and do **not** make entropy a clinical image-quality detector.

## Relation to the 21-image pilot

The pilot's 21 cases are included in the full cohort. All 147 matched case-condition **predicted labels** agree between the CPU pilot and CUDA full run. The pilot correctly suggested lower accuracy under all six changes and severe blur-associated MEL loss. Its small, class-balanced denominator produced much larger apparent accuracy losses for blur (−38.10 and −47.62 pp versus −10.55 and −13.79 pp in the full cohort). Pilot entropy-change signs agree with the full cohort, but the magnitude and precision differ; radius-2 blur's full-cohort interval spans zero. The pilot's overexposure MEL recall remained 3/3, while the full cohort was 67/107 versus 70/107 baseline, with a paired interval including zero. There is no robust contradiction about overexposure MEL behavior; the pilot had only three melanomas.

Full precision and all frozen statistical intervals are in `condition_comparison.csv` and the original `cuda_statistical_results.json`. No classification threshold, review cutoff, or quality gate was selected from these results.
