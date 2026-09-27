# PH² Preprocessing Sensitivity Diagnostic Report

## 1. Executive summary

This paired frozen-checkpoint experiment shows that applying HAM10000's
preprocessing to the same PH² images materially changes model outputs:
**48/120 hard predictions changed**, and mean melanoma probability increased
by **0.13749**. Under the existing binary-overlap policy, the HAM-preprocessed
condition predicted melanoma for 23 cases, including 19 of the 40 actual
melanomas. The RAW condition predicted melanoma for none.

This is a material improvement in argmax melanoma detection for this sample:
melanoma recall increased from **0% to 47.5%**, with accuracy increasing from
**54.17% to 74.17%**. However, melanoma ROC-AUC decreased from **0.6606 to
0.6325**. Melanoma probabilities rose in both true classes; their mean increase
was larger for true melanomas (+0.16389) than for true common nevi (+0.12428),
but the lower ROC-AUC means the paired outputs do **not** demonstrate improved
overall ranking/discrimination of melanoma versus nevus.

The result supports **preprocessing sensitivity** and shows that the chosen
HAM-matched preprocessing condition substantially changes this checkpoint's
predictions. It does not prove preprocessing caused the original PH² failure,
nor does it establish a general performance improvement beyond these 120
cases.

## 2. Experiment integrity

The local diagnostic artifacts report:

- Preflight: PASS.
- RAW reproduction: PASS for all 120 IDs.
- HAM diagnostic: PASS.
- Runtime: Google Colab, PyTorch 2.11.0+cu128, CUDA 12.8, Tesla T4.
- Checkpoint SHA-256: `f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848`.
- Checkpoint: EfficientNetV2S, epoch 6, classes
  `akiec, bcc, bkl, df, mel, nv, vasc`.
- Samples: 120 unique IDs, 80 true `nv`, 40 true `mel`.

All seven generated diagnostic artifacts were present and read:

1. `preprocessing_comparison.csv`
2. `summary_metrics.json`
3. `vasc_case_analysis.csv`
4. `diagnostic_config.json`
5. `README.md`
6. `confusion_matrix_raw.csv`
7. `confusion_matrix_ham_preprocessed.csv`

Independent read-only checks confirmed 120 comparison rows with no duplicate
IDs; their ID set exactly matches the prior PH² predictions. The RAW predicted
class agrees with the prior prediction for all 120 rows. The full probability
columns sum to 1 within floating-point tolerance. The confusion CSVs and
aggregate JSON agree with metrics recomputed from the per-image table.

The paired comparison holds the case IDs, checkpoint, model, class order,
resize, tensor conversion, normalization, and evaluation mode constant. The
changed condition is applying the frozen HAM preprocessing operations to PH².
No inference was rerun during this local artifact review.

## 3. RAW reproduction

The RAW distribution exactly reproduces the previously verified external
evaluation:

| Predicted class | RAW count |
|---|---:|
| `akiec` | 0 |
| `bcc` | 0 |
| `bkl` | 0 |
| `df` | 0 |
| `mel` | 0 |
| `nv` | 102 |
| `vasc` | 18 |

The RAW 2×3 binary-overlap confusion matrix (actual rows `nv`, `mel`;
predicted columns `nv`, `mel`, `other`) is:

| Actual \ Predicted | `nv` | `mel` | `other` |
|---|---:|---:|---:|
| `nv` | 65 | 0 | 15 |
| `mel` | 37 | 0 | 3 |

The prior evaluator's explicit policy is retained: any class outside `{nv,
mel}` is `other` and counts as an error for the binary metrics. Thus the 15
`vasc` predictions on actual nevi count as false-positive errors; the three
`vasc` predictions on actual melanomas count as false-negative errors.

## 4. HAM-preprocessed results

The HAM-preprocessed distribution is:

| Predicted class | Count |
|---|---:|
| `akiec` | 0 |
| `bcc` | 0 |
| `bkl` | 12 |
| `df` | 0 |
| `mel` | 23 |
| `nv` | 84 |
| `vasc` | 1 |

The HAM-preprocessed 2×3 binary-overlap confusion matrix is:

| Actual \ Predicted | `nv` | `mel` | `other` |
|---|---:|---:|---:|
| `nv` | 70 | 4 | 6 |
| `mel` | 14 | 19 | 7 |

Exact binary counts under the evaluator's policy:

- **True positives:** 19
- **False positives:** 10 = 4 true nevi predicted `mel` + 6 true nevi
  predicted `other`
- **True negatives:** 70
- **False negatives:** 21 = 14 actual melanomas predicted `nv` + 7 actual
  melanomas predicted `other`
- **Predicted-other errors:** 13 total = 6 on true nevi and 7 on true
  melanomas; these are included within FP/FN above, not added a second time

| Metric | HAM-preprocessed |
|---|---:|
| Accuracy | 0.7416667 |
| Melanoma precision | 0.6551724 |
| Melanoma recall/sensitivity | 0.4750000 |
| Specificity | 0.8750000 |
| Melanoma F1 | 0.5507246 |

Accuracy is `(TP + TN) / 120`; errors in the `other` column remain incorrect.

## 5. RAW vs HAM metric comparison

| Metric/count | RAW | HAM-preprocessed | Difference |
|---|---:|---:|---:|
| TP | 0 | 19 | +19 |
| FP (including `other` on true `nv`) | 15 | 10 | −5 |
| TN | 65 | 70 | +5 |
| FN (including `other` on true `mel`) | 40 | 21 | −19 |
| Predicted-other errors | 18 | 13 | −5 |
| Accuracy | 0.5416667 | 0.7416667 | +0.2000000 |
| Melanoma precision | 0 | 0.6551724 | +0.6551724 |
| Melanoma recall/sensitivity | 0 | 0.4750000 | +0.4750000 |
| Specificity | 0.8125000 | 0.8750000 | +0.0625000 |
| Melanoma F1 | 0 | 0.5507246 | +0.5507246 |
| Predicted `mel` count | 0 | 23 | +23 |
| Predicted `vasc` count | 18 | 1 | −17 |
| Hard prediction changes | — | 48 / 120 | — |

RAW has 18 total `other` errors (15 true `nv`, 3 true `mel`); HAM has 13
(6 true `nv`, 7 true `mel`). These are shown separately while also being
included in binary FP/FN, respectively.

The seven-class transition matrix's nonzero entries are:

| RAW → HAM | Count |
|---|---:|
| `nv` → `nv` | 72 |
| `nv` → `bkl` | 8 |
| `nv` → `mel` | 21 |
| `nv` → `vasc` | 1 |
| `vasc` → `nv` | 12 |
| `vasc` → `bkl` | 4 |
| `vasc` → `mel` | 2 |

These entries total 120; 72 remain unchanged and 48 change predicted class.

## 6. Melanoma probability analysis

ROC-AUC values were independently recomputed from the per-image melanoma
probabilities using tie-aware ranking and agree with `summary_metrics.json`.

| Measure | RAW | HAM-preprocessed | Difference |
|---|---:|---:|---:|
| Melanoma ROC-AUC (`mel` vs `nv`) | 0.660625 | 0.632500 | −0.028125 |
| Mean `p(mel)`, true melanoma (n=40) | 0.185051 | 0.348943 | +0.163892 |
| Median `p(mel)`, true melanoma | 0.184812 | 0.354107 | +0.169295 |
| Mean `p(mel)`, true common nevus (n=80) | 0.140915 | 0.265197 | +0.124283 |
| Median `p(mel)`, true common nevus | 0.133596 | 0.263860 | +0.130264 |

Across all 120 cases, the melanoma probability delta (HAM minus RAW) has mean
**+0.1374856**, median **+0.1433313**, **96 positive**, **24 negative**, and
**0 zero** changes. By true class:

| True class | Mean delta | Median delta | Positive | Negative | Zero |
|---|---:|---:|---:|---:|---:|
| Melanoma | +0.163892 | +0.198927 | 29 | 11 | 0 |
| Common nevus | +0.124283 | +0.125414 | 67 | 13 | 0 |

The average melanoma-probability increase for true melanomas exceeded that
for true nevi by **0.039609**. The increase is therefore not identical across
the two groups, but both groups shift upward substantially. Importantly, the
rank-based AUC decreases by 0.028125 rather than increasing. This indicates
that, despite more melanoma argmax predictions and higher positive-class
probabilities, the ordering of melanoma versus nevus cases did not improve
overall and was lower in this sample.

## 7. Actual melanoma case analysis

Among the 40 true melanoma cases after HAM preprocessing:

| HAM prediction | Count |
|---|---:|
| `mel` | 19 |
| `nv` | 14 |
| `bkl` | 6 |
| `vasc` | 1 |
| `akiec`, `bcc`, `df` | 0 |

The 19 true-positive melanoma IDs are:

`IMD064, IMD080, IMD088, IMD090, IMD219, IMD242, IMD284, IMD348, IMD349,
IMD403, IMD407, IMD408, IMD409, IMD410, IMD406, IMD421, IMD425, IMD426,
IMD435`.

Of the 23 HAM-preprocessed `mel` predictions, **19 are actual melanomas** and
**4 are actual nevi**. True-nevus IDs predicted `mel`:

`IMD022, IMD035, IMD177, IMD378`.

For the 19 true melanoma cases predicted `mel`, `p(mel)` has mean **0.522349**,
median **0.510716**, minimum **0.447180**, and maximum **0.635720**; because
`mel` is top-1 for these rows, this is also their top-1 confidence. For the
four true nevi predicted `mel`, `p(mel)` has mean **0.498933**, median
**0.511941**, minimum **0.440455**, and maximum **0.531394**; their top-1
confidence is the same as `p(mel)`.

Relative to RAW, which had zero true-positive melanoma predictions, the
HAM-preprocessed condition recovers **19 additional actual melanoma cases**
at argmax. It still misses 21 of 40 melanomas (14 assigned to `nv`, 6 to
`bkl`, 1 to `vasc`).

## 8. Original VASC-case transitions

All 18 rows in `vasc_case_analysis.csv` match cases previously predicted as
`vasc` in the verified RAW evaluation. Their HAM-preprocessed transitions are:

| Previous true label | To `nv` | To `bkl` | To `mel` | Total |
|---|---:|---:|---:|---:|
| Common nevus (`nv`) | 12 | 3 | 0 | 15 |
| Melanoma (`mel`) | 0 | 1 | 2 | 3 |
| **Total** | **12** | **4** | **2** | **18** |

Thus the 18 former `vasc` outputs become 12 `nv`, 4 `bkl`, and 2 `mel`.
The two true melanoma cases newly predicted `mel` are `IMD421` and `IMD426`.
The former `vasc` melanoma `IMD420` becomes `bkl`.

## 9. Confirmed findings

- The local artifact IDs and counts are consistent with the verified prior
  PH² results; RAW reproduces all 120 prior hard predictions exactly.
- Applying the HAM-matched pipeline materially changes outputs on the paired
  sample: 48 hard labels change, the `vasc` count drops from 18 to 1, and the
  `mel` count rises from 0 to 23.
- Under the explicit binary metric policy, true melanoma detections increase
  from 0 to 19/40, melanoma recall from 0 to 0.475, and accuracy from 0.5417
  to 0.7417.
- Mean melanoma probability rises for both true melanomas and true nevi; the
  rise is 0.039609 larger on average for true melanomas.
- Melanoma ROC-AUC is lower in the HAM-preprocessed condition (0.6325) than in
  RAW (0.660625). Hard-label improvement therefore does not imply improved
  ranking discrimination.

## 10. Strongly supported interpretations

- The checkpoint is **preprocessing-sensitive** on these PH² cases. The
  paired setup and exact RAW reproduction support attributing the observed
  output differences to changing the preprocessing condition within this
  diagnostic.
- HAM preprocessing substantially improves melanoma **argmax detection and
  thresholded binary metrics for this sample**, recovering 19 melanomas while
  also producing four melanoma predictions on true nevi.
- The melanoma probability increase is partly a broader score shift: 67/80
  true nevi also have increased `p(mel)`, and mean `p(mel)` rises in both
  groups. The larger mean rise among melanomas does not translate into better
  global ranking, as the ROC-AUC falls.
- These results are consistent with preprocessing mismatch contributing to
  the original hard-prediction failure, but do not establish that mismatch as
  the sole cause or prove generalizable benefit.

## 11. Remaining uncertainties

- Only 40 melanomas and 80 common nevi are included; there is no independent
  PH² replication in this paired result.
- The comparison changes a bundle of preprocessing operations (hair removal,
  Gray World, Retinex, and JPEG representation). It does not isolate each
  operation's individual effect.
- Probabilities are model scores, not calibrated clinical probabilities.
- The AUC decrease and higher hard-label recall can coexist because argmax
  decisions and ranking across all positive-negative pairs measure different
  properties.
- The diagnostic establishes sensitivity to this particular HAM-matched
  pipeline on these images, not causal attribution for all external-domain
  errors.

## 12. Recommended next technical step

If further diagnosis is approved, perform a prespecified **component
ablation** on the same frozen 120 cases and checkpoint: compare RAW against
single-step additions and the full preprocessing sequence (hair removal,
Gray World, Retinex, and final JPEG representation), saving per-image
probabilities and using the same RAW reproduction gate. This would determine
which operation or combination drives the observed argmax recovery and AUC
change. Do not retrain, alter the checkpoint, tune thresholds, or modify
existing evaluation artifacts as part of that diagnostic.

## Audit status

- Files inspected: all seven generated diagnostic artifacts plus the existing
  PH² metrics and predictions used for RAW reproduction.
- Artifact verification: PASS for row count, unique ID set, labels, RAW
  reproduction, class counts, prediction transitions, summary metrics,
  confusion matrices, and probability-derived statistics.
- Inference rerun during local review: **NO**.
- Checkpoint or data modified: **NO**.
- Existing PH² evaluation artifacts modified: **NO**.
