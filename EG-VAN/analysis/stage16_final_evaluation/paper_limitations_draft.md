# Limitations draft

1. HAM10000 is imbalanced: NV accounts for 676 of 1,014 test images. Aggregate accuracy can obscure lower melanoma recall.
2. The HAM test contains 107 melanoma examples. The observed 56/107 recall has sampling uncertainty; no clinical sensitivity claim is supported.
3. Dermatofibroma test support is only 11, limiting interpretation of its apparently high precision and F1.
4. The Stage 15 epoch-16 selected checkpoint was recovered by deterministic continuation from an archived epoch-13 state after artifact persistence failed.
5. The original epoch-16 weight file is unavailable; byte-for-byte identity between it and the recovered checkpoint cannot be established.
6. Replayed epochs 14–16 exactly matched the available history rows, validation predictions, and metrics. This is exact **observable** replay, not direct proof of missing weight identity.
7. PH² has a different true-label space from seven-class HAM10000. Only NV and MEL truth were mapped; five other predicted HAM classes remained OTHER and counted as errors.
8. Eighty atypical nevi were excluded according to the frozen PH² protocol, so the 120-case result does not characterize performance on that lesion group.
9. PH² was used previously elsewhere in the project, so this is an external follow-up rather than untouched independent validation.
10. External melanoma recall was substantially lower (11/40) than internal HAM test recall (56/107), but the two datasets differ and the observed gap does not isolate a cause.
11. No threshold optimization was performed on HAM test or PH². Reported confusion matrices use the frozen seven-class argmax rule.
12. No further training or candidate selection was performed after final test access. Any future development must be a separately designed study without selection on these closed evaluation results.
