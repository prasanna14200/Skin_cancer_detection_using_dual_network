# Stage 4D readiness decision

## READY FOR EXPLORATORY TRAINING — preparation only

The v2 [manifest](stage4d_split_manifest_v2.json) and [independent audit](STAGE4D_SPLIT_INTEGRITY_AUDIT.md) pass the defined **refined-screen containment gate**: all 40 frozen Stage 4C candidate edges are in development training; graph closure contains the two further edges exposed by a direct move; and the full v2 re-screen has zero cross-boundary pairs meeting both original dHash≤4 and pHash≤8 cutoffs. All 8,015 original train images remain included once, with 6,477 for development training and 1,538 for internal selection. Selection includes 163 melanoma images / 95 lesions. The Stage 23 checkpoint and Stage 4B v1 are unchanged.

**Scope of readiness:** this permits a *separately authorized, exploratory* comparison of two freshly initialized models under the unchanged [Stage 4B single-factor control-versus-Gaussian-blur protocol](../stage4b_development_split/FRESH_CONTROL_VS_TREATMENT_PROTOCOL.md), using exactly the **same v2 manifests** for both arms. It does not authorize training in this task; no training runner or checkpoint was created. Frozen Stage 23 cannot be evaluated on this selection set because its training included these images.

**Limits:** No image pair has been human-adjudicated; patient IDs are absent; 5,189 dHash-only cross-boundary pairs remain a low-specificity screen signal; unflagged near duplicates may remain. This is leakage containment under a fixed image-hash rule, **not independently verified patient-level separation**. Any future treatment result is exploratory and cannot replace Stage 23 without a new untouched patient-linked confirmation cohort. Reviewer adjudication and source linkage remain valuable, especially if a stronger independence claim is needed. [Remaining limitations](STAGE4D_REMAINING_LIMITATIONS.md) must accompany any report.

**Next action:** review this v2 audit and decide separately whether to authorize fresh-control/treatment implementation. Do not train, infer, tune thresholds, use old HAM validation/test or PH2 for selection, or make Stage 23 replacement claims on v2.
