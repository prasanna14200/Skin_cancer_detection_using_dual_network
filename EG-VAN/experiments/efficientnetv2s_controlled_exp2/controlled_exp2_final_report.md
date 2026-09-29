# Controlled Experiment #2 Final Report

**Outcome: FAIL**

## Verified Run

- Model: EfficientNetV2S
- Recorded epochs: 25
- Selected epoch: 8
- Checkpoint selection: minimum frozen HAM10000 validation loss
- Checkpoint SHA256: `9dc38968e91a98693de5e6a8c4ee720a69ef80e275887780a8d714077c6ea93f`
- Training sampler: deterministic `WeightedRandomSampler`
- Melanoma weight: 2.44312387785314; non-melanoma weight: 1.0
- Original baseline focal loss retained: alpha 0.25, gamma 2.0; no class-specific loss weights
- PH2 accessed: NO

## Frozen Validation Criteria

| Criterion | Experiment #2 | Required minimum | Result |
|---|---:|---:|---|
| Melanoma F1 | 0.525821596244 | 0.575773195876 | FAIL |
| Validation macro-F1 | 0.707427727325 | 0.631658238136 | PASS |
| Nevus recall | 0.903469079940 | 0.900000000000 | PASS |

## Checkpoint-Matched Comparison

| Partition / checkpoint | Mel precision | Mel recall | Mel F1 | Nevus recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Baseline validation epoch 6 | 0.586207 | 0.476636 | 0.525773 | 0.956259 | 0.651658 |
| Experiment #2 validation epoch 8 | 0.528302 | 0.523364 | 0.525822 | 0.903469 | 0.707428 |
| Baseline matched test epoch 6 | 0.523256 | 0.420561 | 0.466321 | 0.957101 | 0.620137 |
| Experiment #2 test epoch 8 | 0.460938 | 0.551402 | 0.502128 | 0.890533 | 0.695649 |

Baseline test predictions were used only after verifying their checkpoint SHA256 against the frozen baseline checkpoint and their image IDs/labels against the frozen test partition. Baseline `test_metrics.json` was not used as epoch-6 evidence.

This report was assembled from saved metrics and predictions. It does not rerun training or inference. Test metrics are descriptive and were not used for checkpoint or hyperparameter selection.
