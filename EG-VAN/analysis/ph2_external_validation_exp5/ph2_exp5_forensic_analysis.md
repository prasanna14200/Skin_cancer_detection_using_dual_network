# PH² Experiment #5 Forensic Analysis

## Evaluation scope and limitation

This is a **PH² external follow-up evaluation** of the frozen Controlled Experiment #5 checkpoint. It is not a completely untouched independent external validation.

PH² had previously been used for preprocessing diagnostics and an earlier model evaluation before Experiment #5. Therefore this analysis is reported as an external follow-up evaluation rather than a completely untouched independent external validation.

Experiment #5 remains frozen. These PH² results must not be used to change its checkpoint, preprocessing, thresholds, or selection decisions.

## Integrity review

The output directory contained the six expected evaluation artifacts before this report was created: `ph2_manifest.csv`, `ph2_predictions.csv`, `ph2_metrics.json`, `confusion_matrix.csv`, `external_validation_config.json`, and `external_validation_report.md`.

Independent checks passed:

- 120 prediction rows and 120 unique included IDs; every included manifest ID appears exactly once.
- True-label counts are 80 `nv` and 40 `mel`; no atypical-nevus case appears in inference predictions.
- All seven probability columns are present, finite, and sum to 1 within `1e-5` for every case.
- Every `predicted_class` agrees with the seven-probability argmax.
- Prediction labels agree with both the PH² evaluation manifest and the source PH² manifest.
- The standalone confusion matrix agrees with the independently reconstructed matrix and `ph2_metrics.json`.
- Current Experiment #5 checkpoint SHA256 is `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`; it matches the run config and frozen value.
- Current HAM split SHA256 is `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`; it matches the run config and frozen value.
- Selected epoch is 15 and class order is `akiec, bcc, bkl, df, mel, nv, vasc`.
- Recorded preprocessing matches the previous diagnostic's HAM-matched contract. It reuses `src/preprocessing.py`, includes the same in-memory JPEG encode/decode, and records no masks/ROI, augmentation, or test-time augmentation.

No model inference was run for this forensic analysis. The existing prediction, metric, configuration, and preprocessing-diagnostic artifacts were not modified.

## Independent metric recomputation

Metrics below were recomputed from `ph2_predictions.csv`, not copied from `ph2_metrics.json`. They agree with the saved metrics. Class precision uses the predicted `nv` or `mel` columns; predictions mapped to `other` remain errors for accuracy and true-class recall and are retained in the confusion matrix.

| Metric | PH² follow-up |
|---|---:|
| Total samples | 120 |
| Correct predictions | 75 |
| Accuracy | 0.625000 |
| Balanced accuracy | 0.543750 |
| `nv` precision | 0.807692 |
| `nv` recall | 0.787500 |
| `nv` F1 | 0.797468 |
| `mel` precision | 0.600000 |
| `mel` sensitivity/recall | 0.300000 |
| `mel` F1 | 0.400000 |
| Macro precision over `nv`, `mel` | 0.703846 |
| Macro recall over `nv`, `mel` | 0.543750 |
| Macro F1 over `nv`, `mel` | 0.598734 |
| Melanoma AUROC using `p_mel` | 0.604688 |

The confusion matrix uses true rows `nv`, `mel` and predicted columns `nv`, `mel`, `other`:

| True \ Predicted | nv | mel | other |
|---|---:|---:|---:|
| nv | 63 | 8 | 9 |
| mel | 15 | 12 | 13 |

Seven-class prediction distribution:

| Predicted HAM class | Count |
|---|---:|
| akiec | 0 |
| bcc | 0 |
| bkl | 20 |
| df | 1 |
| mel | 20 |
| nv | 78 |
| vasc | 1 |

No seven-class macro-F1 is calculated for PH² because five HAM classes are absent from the true-label cohort.

## Error analysis

True melanoma cases (40):

- Predicted `mel`: 12.
- Predicted `nv`: 15.
- Predicted another HAM class (`other`): 13, comprising 11 `bkl`, 1 `df`, and 1 `vasc`.

True common-nevus cases (80):

- Predicted `nv`: 63.
- Predicted `mel`: 8.
- Predicted another HAM class (`other`): 9, all `bkl`.

Wrong-prediction counts by emitted class were `bkl=20`, `nv=15`, `mel=8`, `df=1`, and `vasc=1`. Thus melanoma misses are split between `nv` (15) and other classes (13), while `bkl` is the most common wrong output across both true groups.

### Ten melanoma cases with lowest `p_mel`

| image_id | p_mel | predicted class |
|---|---:|---|
| IMD091 | 0.002087 | df |
| IMD418 | 0.010847 | nv |
| IMD211 | 0.011544 | nv |
| IMD085 | 0.019816 | bkl |
| IMD411 | 0.024111 | nv |
| IMD403 | 0.033010 | bkl |
| IMD406 | 0.033567 | nv |
| IMD413 | 0.039775 | bkl |
| IMD424 | 0.045951 | vasc |
| IMD404 | 0.047788 | bkl |

### Ten correctly classified melanoma cases with lowest confidence

For these correctly predicted melanoma cases, confidence is the maximum of the seven saved probabilities; because `mel` is the predicted class, it equals `p_mel`.

| image_id | confidence |
|---|---:|
| IMD168 | 0.399038 |
| IMD065 | 0.416335 |
| IMD429 | 0.560809 |
| IMD426 | 0.586298 |
| IMD242 | 0.668079 |
| IMD284 | 0.681643 |
| IMD405 | 0.696819 |
| IMD435 | 0.700286 |
| IMD408 | 0.745891 |
| IMD421 | 0.750293 |

### Ten nevus cases with highest `p_mel`

| image_id | p_mel | predicted class |
|---|---:|---|
| IMD399 | 0.816671 | mel |
| IMD379 | 0.706934 | mel |
| IMD375 | 0.701469 | mel |
| IMD367 | 0.627470 | mel |
| IMD035 | 0.585192 | mel |
| IMD374 | 0.535623 | mel |
| IMD045 | 0.533242 | mel |
| IMD182 | 0.428799 | nv |
| IMD112 | 0.421608 | mel |
| IMD384 | 0.418080 | nv |

## Internal-test comparison

The selected Experiment #5 checkpoint's saved HAM10000 internal-test metrics report 1,014 cases: accuracy 0.833333, seven-class macro-F1 0.699414, melanoma recall 0.551402, melanoma F1 0.559242, and nevus recall 0.931953.

| Metric | Internal HAM test | PH² follow-up | Difference (PH² − internal) |
|---|---:|---:|---:|
| Accuracy | 0.833333 | 0.625000 | -0.208333 |
| Melanoma recall | 0.551402 | 0.300000 | -0.251402 |
| Melanoma F1 | 0.559242 | 0.400000 | -0.159242 |
| Nevus recall | 0.931953 | 0.787500 | -0.144453 |
| Macro-F1 | 0.699414 (7 classes) | 0.598734 (2 represented classes) | -0.100680, numeric only |

These are different populations and label supports. Internal accuracy/macro-F1 span all seven HAM classes; PH² contains only `nv` and `mel`, with other model outputs counted as errors. The differences are descriptive evidence of a generalization shift, not a controlled causal comparison. In particular, the macro-F1 values have different class scopes and should not be interpreted as directly comparable estimates.

## Historical preprocessing-diagnostic comparison

The previous HAM-preprocessed PH² condition used the leakage-aware epoch-6 checkpoint (`f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848`), not the frozen Experiment #5 epoch-15 checkpoint. It is a historical, descriptive comparison only.

The previous diagnostic's stored HAM-preprocessed result reported accuracy 0.741667, melanoma sensitivity 0.475000, binary melanoma F1 0.550725, and melanoma AUROC 0.632500. Its binary metric explicitly counts `other` predictions on true nevi as false positives and on true melanomas as false negatives. The current Experiment #5 metrics use class-column precision/F1 from the 2x3 confusion matrix, so those F1 definitions are not identical.

For a same-formula descriptive comparison, recalculating classwise metrics from the old diagnostic's saved matrix `[[70, 4, 6], [14, 19, 7]]` gives melanoma precision 0.826087, recall 0.475000, F1 0.603175, nevus recall 0.875000, accuracy 0.741667, and two-class macro-F1 0.728417. Compared numerically with Experiment #5's follow-up: accuracy -0.116667, melanoma recall -0.175000, melanoma classwise F1 -0.203175, nevus recall -0.087500, and melanoma AUROC -0.0278125. These numbers must not be attributed to a causal Experiment #5 effect: the checkpoint differs, and the diagnostic was not a controlled comparison of checkpoints.

## Scientific interpretation

1. **Melanoma sensitivity:** The model retains nonzero sensitivity: it identifies 12/40 PH² melanomas (30%). That is limited sensitivity on this cohort, not evidence of clinically useful performance.
2. **Internal-to-external change:** Melanoma recall is 25.14 percentage points lower and melanoma F1 is 15.92 points lower than the internal test. Nevus recall is 14.45 points lower. These cross-dataset differences are consistent with a generalization gap, but are not causal estimates.
3. **Error pattern:** Among the 28 missed melanomas, 15 are predicted `nv` and 13 are predicted as other classes. Nevus confusion is the single largest melanoma-miss destination, but out-of-overlap classes are nearly as frequent. Across all errors, `bkl` is the most common wrong output.
4. **Domain/generalization gap:** The lower follow-up metrics despite the approved HAM-matched preprocessing are a substantial descriptive indication that internal performance does not transfer unchanged to this PH² cohort. The limited sample and prior PH² exposure qualify this conclusion.
5. **AUROC versus argmax sensitivity:** AUROC 0.604688 indicates some ranking separation in saved melanoma probabilities at the cohort level, but it is modest and does not override the 0.30 argmax sensitivity. AUROC assesses ranking over thresholds; no threshold was changed or selected using PH².
6. **Limitations:** The cohort has only 40 melanomas; PH² is a secondary mirror with the provenance limitations recorded in its audit; PH² was used in earlier evaluation and preprocessing diagnostics; only direct-overlap labels are scored; class distributions and capture conditions differ from HAM10000; and no confidence intervals or independent replication are included here. No clinical-validity claim is supported.

## Conclusion and next research step

On this PH² external follow-up cohort, the frozen Experiment #5 model produces 0.625 accuracy, 0.300 melanoma sensitivity, and 0.604688 melanoma AUROC. Most melanoma errors go to `nv` or `bkl`/other classes. The results support reporting a cross-dataset generalization concern, not changing or retuning the frozen model from PH² outcomes.

**Recommended next research step:** evaluate the still-frozen Experiment #5 checkpoint once on a genuinely untouched, traceable external cohort with a preregistered overlap mapping and preprocessing protocol. Do not use the current PH² results to tune the checkpoint or threshold.