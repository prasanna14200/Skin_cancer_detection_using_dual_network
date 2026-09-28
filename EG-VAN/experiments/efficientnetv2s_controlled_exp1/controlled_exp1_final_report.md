# Controlled Experiment #1 Final Report

**Outcome: FAIL**

## Verified Run

- Model: EfficientNetV2S
- Recorded training epochs: 25
- Selected checkpoint epoch: 5
- Selection rule: minimum validation loss
- Checkpoint SHA256: `478bb1aa9c48897accceb800a103e7ee76c1381dc6514c7355996babbfaa902f`
- Validation samples: 986
- Test samples: 1014
- Training rerun: NO
- Inference rerun: NO
- PH2 inference: NO

## Prespecified Validation Criteria

| Criterion | Controlled value | Required minimum | Result |
|---|---:|---:|---|
| Melanoma F1 | 0.461538461538 | 0.575773195876 | FAIL |
| Validation macro-F1 | 0.557890438366 | 0.631658238136 | FAIL |
| Nevus recall | 0.761689291101 | 0.900000000000 | FAIL |

Baseline epoch 6 validation metrics recovered from the frozen baseline training history: melanoma F1 0.525773195876, macro-F1 0.651658238136, nevus recall 0.956259426848.

Controlled test metrics are included in `comparison_metrics.json`; they were not used for checkpoint selection or success-criterion evaluation. Baseline ROC-AUC values are unavailable from the saved baseline artifacts and are intentionally omitted.

## Finalization Finding

The runner writes comparison_metrics.json and experiment_manifest.json only after its post-training baseline evaluation. Those baseline outputs are absent, so the saved files show that finalization did not reach its report-writing stage (or those outputs were later removed). The exact interruption/removal cause cannot be determined: no run log or error record is present. The runner has no code path that writes controlled_exp1_final_report.md.

The reports were reconstructed solely from hash-verified checkpoints, the frozen split, saved metric JSON, training histories, and prediction CSVs. No image data was loaded and no model inference was run.
