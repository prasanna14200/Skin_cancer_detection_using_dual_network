# Stage 16 final research report

**Final analysis status: COMPLETE for the frozen recovered Stage 15 epoch-16 candidate.** The checkpoint SHA256 is `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`. Saved HAM and PH² evaluation artifacts, manifests, source hashes, prediction IDs, confusion matrices, metrics, and AUC values passed the independent read-only [audit](final_evaluation_audit.md). No model inference or training was repeated in this analysis.

## Final evidence

The selected validation epoch met every frozen checkpoint gate: accuracy 0.819473, macro F1 0.661131, MEL recall 63/107 (0.588785), MEL F1 0.602871, and NV recall 0.939668. The original epoch-16 weight file is missing; recovered epochs 14–16 matched available original observables exactly, but identity to the missing weights cannot be established.

On the held-out HAM10000 test (1,014 images), accuracy was 0.823471, macro F1 0.699582, and macro OVR AUC 0.958206. Melanoma recall was 56/107 (0.523364), with 51 false negatives. Forty-one of these 51 misses (80.39%) were classified NV. The fixed seven-class results are in the [tables](final_results_tables.md), and the [error analysis](final_error_analysis.md) details class transitions.

In the separate mapped PH² external follow-up (120 included images), accuracy was 0.666667 and melanoma recall was 11/40 (0.275000). Eighteen of 40 melanomas were predicted NV and 11 OTHER. PH² was previously used in this project, and its label space differs from HAM; it must not be represented as untouched external validation. The 80 atypical nevi were excluded under the established mapping.

Eight publication figures were produced from saved predictions and metrics only: HAM count and normalized confusion matrices, classwise F1 and recall, seven OVR ROC curves, PH² count and normalized mapped confusion matrices, and a descriptive melanoma-recall comparison across validation, HAM test, and PH² follow-up. All are in `figures/` at 350 DPI.

## Interpretation and closure

The frozen Stage 15 candidate demonstrated strong internal seven-class HAM10000 discrimination but moderate melanoma sensitivity. Melanoma performance decreased substantially under the PH² external follow-up protocol, indicating limited transfer in this evaluated setting. The selective FP32 nonlocal3 q@k policy addressed numerical overflow; no classification benefit is attributed to it. The experiment remains closed, with no test-driven optimization or additional candidate selection. Next work is paper and report integration only.

Draft manuscript text: [Results](paper_results_draft.md), [Discussion](paper_discussion_draft.md), and [Limitations](paper_limitations_draft.md).
