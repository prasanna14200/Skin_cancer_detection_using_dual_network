# Controlled Experiment #4 Final Report

**Outcome: FAIL**

- Selected epoch: 6 by minimum validation loss
- Checkpoint SHA256: `c09e2d6b77891a1fcf8b9e7b8aa2f702a4407c84c5bd8798ae189f1f15357b73`
- Adamax weight decay: 0.0001
- Experiment #3 Adamax weight decay: 0.0
- PH2 accessed: NO

## Frozen Validation Criteria

| Criterion | Experiment #4 | Required minimum | Result |
|---|---:|---:|---|
| Melanoma F1 | 0.504587155963 | 0.575773195876 | FAIL |
| Validation macro-F1 | 0.663900640131 | 0.631658238136 | PASS |
| Nevus recall | 0.903469079940 | 0.900000000000 | PASS |

## Checkpoint-Matched Validation Comparison

| Run | Epoch | Mel precision | Mel recall | Mel F1 | Nevus recall | Macro-F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline epoch 6 | 0.586207 | 0.476636 | 0.525773 | 0.956259 | 0.651658 | 0.807302 |
| experiment_1 epoch 5 | 0.392157 | 0.560748 | 0.461538 | 0.761689 | 0.557890 | 0.710953 |
| experiment_2 epoch 8 | 0.528302 | 0.523364 | 0.525822 | 0.903469 | 0.707428 | 0.814402 |
| experiment_3 epoch 6 | 0.600000 | 0.392523 | 0.474576 | 0.933635 | 0.704981 | 0.820487 |
| experiment_4 epoch 6 | 0.495495 | 0.514019 | 0.504587 | 0.903469 | 0.663901 | 0.795132 |

Test results are EVALUATION ONLY. They do not inform checkpoint selection or tuning. Baseline test evidence is from checkpoint-hash-verified saved uncertainty predictions; final-epoch baseline test metrics are not substituted for selected-epoch evidence.
