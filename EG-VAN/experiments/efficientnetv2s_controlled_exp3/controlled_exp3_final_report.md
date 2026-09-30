# Controlled Experiment #3 Final Report

**Outcome: FAIL**

- Selected checkpoint epoch: 6 (minimum validation loss)
- Checkpoint SHA256: `8615a9f5afad48c17c21b559aeab01e5e506fa3cbce3f544eed14b47b53377b9`
- Melanoma sampler weight: 1.5630495442733532; non-melanoma weight: 1.0
- PH2 accessed: NO

## Frozen Validation Criteria

| Criterion | Experiment #3 | Required minimum | Result |
|---|---:|---:|---|
| Melanoma F1 | 0.474576271186 | 0.575773195876 | FAIL |
| Validation macro-F1 | 0.704980813801 | 0.631658238136 | PASS |
| Nevus recall | 0.933634992459 | 0.900000000000 | PASS |

## Checkpoint-Matched Validation Comparison

| Experiment | Mel precision | Mel recall | Mel F1 | Nevus recall | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| baseline epoch 6 | 0.586207 | 0.476636 | 0.525773 | 0.956259 | 0.651658 |
| experiment_1 epoch 5 | 0.392157 | 0.560748 | 0.461538 | 0.761689 | 0.557890 |
| experiment_2 epoch 8 | 0.528302 | 0.523364 | 0.525822 | 0.903469 | 0.707428 |
| experiment_3 epoch 6 | 0.600000 | 0.392523 | 0.474576 | 0.933635 | 0.704981 |

Test results are evaluation only. They were not used for checkpoint selection or tuning. Baseline test metrics come from saved uncertainty predictions whose checkpoint hash and frozen test IDs/labels were verified; the final-epoch baseline `test_metrics.json` was not substituted for selected epoch 6.

The comparison and report were reconstructed from saved artifacts only. No inference was rerun during finalization.
