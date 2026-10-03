# Stage 12: frozen baseline versus reconstructed EG-VAN

This is analysis of saved predictions only. Both checkpoints, the HAM split, and Stage 10/11 outputs were hash-verified. No training, inference, threshold tuning, calibration fitting, or Grad-CAM was performed. PH² is an external follow-up with prior project exposure, not untouched independent validation.

## HAM internal test (1,014 paired cases)

| Metric | Experiment #5 | EG-VAN | EG-VAN minus baseline |
|---|---:|---:|---:|
| accuracy | 0.833333 | 0.824458 | -0.008876 |
| balanced_accuracy | 0.693940 | 0.681575 | -0.012365 |
| macro_precision | 0.708827 | 0.735027 | +0.026200 |
| macro_recall | 0.693940 | 0.681575 | -0.012365 |
| macro_f1 | 0.699414 | 0.706053 | +0.006639 |
| weighted_f1 | 0.831740 | 0.821034 | -0.010705 |

Paired correctness: {'both_correct': 796, 'egvan_only_correct': 40, 'experiment5_only_correct': 49, 'both_wrong': 129}. Exact McNemar b=40, c=49, p=0.396570. This is a paired accuracy test, not evidence about all class-specific metrics.

Melanoma recall: baseline 0.551402 (59/107), EG-VAN 0.523364 (56/107). NV recall: baseline 0.931953, EG-VAN 0.939349. Per-class precision, recall, F1 and support are in `ham_model_comparison.csv`.

## PH² mapped external follow-up (120 paired cases; 80 NV, 40 MEL)

| Metric | Experiment #5 | EG-VAN | EG-VAN minus baseline |
|---|---:|---:|---:|
| accuracy | 0.625000 | 0.675000 | +0.050000 |
| balanced_accuracy | 0.543750 | 0.575000 | +0.031250 |

MEL recall: 0.300 (12/40) versus 0.275 (11/40). NV recall: 0.7875 (63/80) versus 0.8750 (70/80). The six additional correct EG-VAN predictions overall come from a net seven extra correct NV and one fewer correct MEL case; this is a class-level accounting, not a causal explanation.

Paired correctness: {'both_correct': 68, 'egvan_only_correct': 13, 'experiment5_only_correct': 7, 'both_wrong': 32}; exact McNemar b=13, c=7, p=0.263176. This does not support a statistical superiority claim.

True MEL prediction distribution: baseline {'bkl': 11, 'nv': 15, 'mel': 12, 'df': 1, 'vasc': 1}; EG-VAN {'nv': 23, 'df': 1, 'mel': 11, 'bkl': 5}. True NV distribution: baseline {'nv': 63, 'mel': 8, 'bkl': 9}; EG-VAN {'nv': 70, 'mel': 4, 'bkl': 6}. Predictions into other HAM classes remain errors.

MEL paired cases: HAM {'both_correct': 50, 'both_wrong': 42, 'experiment5_only_correct': 9, 'egvan_only_correct': 6}; PH² {'both_wrong': 24, 'experiment5_only_correct': 5, 'both_correct': 7, 'egvan_only_correct': 4}. The case-level transitions are in `melanoma_failure_comparison.csv`.

## Confidence and uncertainty from saved probabilities

ECE uses the project's 10 fixed equal-width top-class confidence bins; Brier is the mean seven-class sum of squared errors; NLL uses natural log with 1e-12 floor; high-confidence error means confidence ≥0.75. No calibration was fitted.

| Dataset | Model | Mean confidence | Mean entropy | ECE | Brier | NLL | High-confidence errors |
|---|---|---:|---:|---:|---:|---:|---:|
| HAM | EfficientNetV2S Experiment #5 | 0.8099 | 0.5140 | 0.0390 | 0.2408 | 0.4757 | 38 |
| HAM | Reconstructed EG-VAN | 0.7865 | 0.5393 | 0.0390 | 0.2552 | 0.5052 | 25 |
| PH2 | EfficientNetV2S Experiment #5 | 0.6899 | 0.7737 | 0.0741 | 0.5171 | 0.9838 | 11 |
| PH2 | Reconstructed EG-VAN | 0.6448 | 0.8141 | 0.0744 | 0.4987 | 0.9826 | 8 |

Confidence and entropy split by correct/incorrect predictions are in `uncertainty_comparison.csv`. ECE, Brier and NLL are descriptive on these cohorts; PH² has only NV/MEL truth but all seven predicted classes are retained.

## HAM to PH² descriptive changes

| Model | Metric | HAM | PH² | PH² minus HAM |
|---|---|---:|---:|---:|
| EfficientNetV2S Experiment #5 | accuracy | 0.833333 | 0.625000 | -0.208333 |
| EfficientNetV2S Experiment #5 | melanoma_recall | 0.551402 | 0.300000 | -0.251402 |
| EfficientNetV2S Experiment #5 | nevus_recall | 0.931953 | 0.787500 | -0.144453 |
| Reconstructed EG-VAN | accuracy | 0.824458 | 0.675000 | -0.149458 |
| Reconstructed EG-VAN | melanoma_recall | 0.523364 | 0.275000 | -0.248364 |
| Reconstructed EG-VAN | nevus_recall | 0.939349 | 0.875000 | -0.064349 |

The accuracy decrease was numerically smaller for EG-VAN, as was the NV recall decrease. MEL recall remained weak externally and was slightly lower for EG-VAN. HAM is seven-class and PH² has mapped NV/MEL truth only, so these are descriptive cross-dataset changes, not causal domain-shift magnitudes or directly equivalent population risks.

## Improvement diagnosis and claim boundaries

The safest next model-development experiment is a pre-registered HAM train/validation-only melanoma-objective/sampler ablation with a fixed selection rule. This would be exploratory because the HAM test and PH² follow-up have already been viewed. A new untouched external dataset is needed for final confirmation. A new training run is scientifically justifiable only as such a pre-specified exploratory ablation, not as PH²-driven optimization. No new training was started.

`improvement_hypotheses.csv` separates HAM-only proposals, PH²-driven choices to reject, and untouched external confirmation. `claim_evidence_matrix.csv` gives safe paper wording. No claim of statistical or clinical superiority is supported here.
