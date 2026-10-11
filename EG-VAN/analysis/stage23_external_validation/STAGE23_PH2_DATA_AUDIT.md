# Stage 23 PH2 data and independence audit

This read-only audit was completed **before Stage 23 PH2 inference**. It inspected dataset identity and labels, not Stage 23 external predictions or metrics. The executable preflight is `evaluate_stage23_ph2.py --preflight`.

## Local evidence

| Item | Verified finding |
|---|---|
| Canonical clinical table | `data/external/ph2/metadata/PH2_dataset.txt`; SHA256 `2cc2fdc28eb230d86b56d72a5aa710593d0861cf6564fa64f43a97a89070ccf4`. Its clinical diagnosis codes are 0 common nevus, 1 atypical nevus, 2 melanoma. No patient identifier or patient-to-image grouping field is present. |
| Fixed cohort manifest | `analysis/ph2_external_validation_exp5/ph2_manifest.csv`; SHA256 `3c130b3b44dcc1052ae29656510a95a4e91df4b8f65c121838941efb375d3842`. Exactly 200 unique `IMD` IDs, all matching the clinical table. |
| Frozen mapping | 80 common nevi → `nv`; 40 melanomas → `mel`; all 80 atypical nevi excluded. Exactly 120 included images. This mapping predates Stage 23 external inference and is [frozen in protocol](phase23_ph2_protocol.json). |
| Image availability | All 200 manifest BMPs exist at `data/external/ph2/images/`; all 200 corresponding original package dermoscopic BMPs exist. Preflight checks exact byte hash equality between each flat copy and its original, and verifies that each image decodes with adequate dimensions. It does not use segmentation masks or ROI images. |
| HAM overlap | Exact PH2 `IMD` image IDs do not intersect HAM split image IDs. This does **not** prove patient independence or absence of visually identical cross-dataset images; no cross-dataset patient identifier is available. |
| Stage 23 training boundary | Frozen Stage 23 manifest says `ph2_accessed=false`, and its training/initialization source has no PH2 loader. This supports no **direct PH2 training or selection in the Stage 23 run**. It cannot prove that historical PH2 exposure had no indirect influence on project-level preprocessing or research direction. |

## Task mismatch and historical use

HAM10000 Stage 23 predicts seven mutually exclusive labels: `akiec,bcc,bkl,df,mel,nv,vasc`. PH2's available clinical categories are common nevus, atypical nevus, and melanoma. Atypical nevus has no unique frozen HAM output mapping; including it by relabeling after inspection would alter the endpoint. Therefore the prespecified included cohort is 80 NV + 40 MEL, while the network still produces seven-class probabilities. Predictions into the five remaining HAM classes become **OTHER** and count as incorrect for mapped 2×3 accuracy and class recalls. Binary MEL specificity has a different meaning: any argmax except MEL is negative, including OTHER. Both measures will be reported visibly.

PH2 was previously used in this repository for [preprocessing diagnostics](../../experiments/ph2_preprocessing_diagnostic/), [older-model external evaluation](../ph2_external_validation_exp5/), and [Stage 16 follow-up](../stage16_final_evaluation/). Consequently PH2 is **not an untouched independent external-validation set for the overall project**. The proposed Stage 23 evaluation can be reported only as a frozen **external follow-up**. An independent confirmatory claim requires a genuinely new cohort that was not used in development.

No Stage 23 PH2 performance value exists in this preparation audit. No model inference, training, threshold selection, or source artifact mutation occurred.
