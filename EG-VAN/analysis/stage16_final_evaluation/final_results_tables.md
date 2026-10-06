# Stage 16 final results tables

All values derive from the audited final CSV/JSON artifacts. HAM test and PH² follow-up are distinct cohorts. Stage 15 validation was used for checkpoint selection; HAM test and PH² were evaluation-only.

## Table 1. HAM10000 held-out test, seven classes (n = 1,014)

| Metric | Value |
|---|---:|
| Accuracy | 0.823471 |
| Balanced accuracy | 0.675097 |
| Macro precision | 0.739039 |
| Macro recall | 0.675097 |
| Macro F1 | 0.699582 |
| Weighted F1 | 0.818013 |
| Macro OVR ROC-AUC | 0.958206 |
| Weighted OVR ROC-AUC | 0.946609 |

## Table 2. HAM10000 held-out test, classwise

| Class | Support | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| AKIEC | 40 | 0.571429 | 0.600000 | 0.585366 | 24 | 18 | 16 |
| BCC | 58 | 0.727273 | 0.689655 | 0.707965 | 40 | 15 | 18 |
| BKL | 104 | 0.659091 | 0.557692 | 0.604167 | 58 | 30 | 46 |
| DF | 11 | 1.000000 | 0.636364 | 0.777778 | 7 | 0 | 4 |
| MEL | 107 | 0.622222 | 0.523364 | 0.568528 | 56 | 34 | 51 |
| NV | 676 | 0.893258 | 0.940828 | 0.916427 | 636 | 76 | 40 |
| VASC | 18 | 0.700000 | 0.777778 | 0.736842 | 14 | 6 | 4 |

## Table 3. Stage 15 selected validation versus held-out HAM test

| Metric | Stage 15 validation (n = 986) | HAM test (n = 1,014) |
|---|---:|---:|
| Accuracy | 0.819473 | 0.823471 |
| Macro F1 | 0.661131 | 0.699582 |
| MEL recall | 0.588785 (63/107) | 0.523364 (56/107) |
| MEL F1 | 0.602871 | 0.568528 |
| NV recall | 0.939668 | 0.940828 |

These are descriptive values from different partitions. Validation metrics selected epoch 16; the test metrics were observed afterward. No difference was used to change the model.

## Table 4. PH² external follow-up, mapped NV/MEL truth (n = 120)

| Metric | Value |
|---|---:|
| Sample count | 120 |
| Accuracy | 0.666667 |
| Balanced accuracy | 0.568750 |
| MEL precision | 0.687500 |
| MEL recall | 0.275000 |
| MEL F1 | 0.392857 |
| NV recall | 0.862500 |
| MEL probability ROC-AUC | 0.542813 |

PH² had prior use elsewhere in the project and is **external follow-up evidence**, not untouched external validation. Of 200 PH² cases, the frozen protocol included 80 common nevi and 40 melanomas and excluded 80 atypical nevi. Predictions outside NV/MEL remained OTHER and counted as incorrect.
