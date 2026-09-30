# PH2 External Follow-Up Evaluation: Experiment #5

## Scope and frozen model

This is a PH2 external follow-up evaluation of the frozen Controlled Experiment #5 checkpoint. It is not a completely untouched independent external validation. The Experiment #5 checkpoint and HAM split are required to remain unchanged, and PH2 results must not be used to modify the model.

PH² had previously been used for preprocessing diagnostics and an earlier model evaluation before Experiment #5. Therefore this analysis is reported as an external follow-up evaluation rather than a completely untouched independent external validation.

- Checkpoint SHA256: `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`
- Frozen split SHA256: `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`
- Selected epoch: 15
- Class order: `akiec, bcc, bkl, df, mel, nv, vasc`

## Cohort and mapping

- Total PH2 records: 200
- Evaluated direct-overlap cases: 120 (80 common nevus to `nv`; 40 melanoma to `mel`)
- Excluded: 80 atypical nevi; no HAM label is assigned.
- No absent HAM classes are treated as evaluation failures.

## Preprocessing

The approved HAM-matched diagnostic transform was reused: original PH2 BMP decoded as RGB, converted to BGR, processed by the frozen Phase 4B hair-removal, Gray World, and multi-scale Retinex functions, JPEG-encoded/decoded in memory, converted to RGB, resized to 384x384, converted to tensor, and ImageNet-normalized. No augmentation, masks, ROI crops, test-time augmentation, or threshold tuning were used.

## Metrics

- Total samples: 120
- Accuracy: 0.625000
- Balanced accuracy: 0.543750
- `nv` precision / recall / F1: 0.807692 / 0.787500 / 0.797468
- `mel` precision / sensitivity-recall / F1: 0.600000 / 0.300000 / 0.400000
- Macro precision / recall / F1 over `nv` and `mel`: 0.703846 / 0.543750 / 0.598734
- Melanoma AUROC using `p_mel`: 0.6046875

Confusion rows are true `[nv, mel]`; columns are predicted `[nv, mel, other]`. Predictions of `akiec`, `bcc`, `bkl`, `df`, or `vasc` are retained as `other` and count as incorrect. Metrics are restricted to externally represented `nv` and `mel`; no seven-class macro-F1 is reported.

## Limitations

PH2 was previously used for preprocessing diagnostics and an earlier model evaluation before Experiment #5. Interpret this as a follow-up evaluation with potential prior-data exposure, not as an untouched independent test. The cohort is small and limited to two mapped diagnoses. This evaluation does not establish clinical validity.

Experiment #5 is frozen and will not be modified based on these PH2 results. No PH2 training or fine-tuning is performed.
