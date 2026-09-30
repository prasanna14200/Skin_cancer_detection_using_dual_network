# Final Internal Error Analysis

This analysis uses only the already-saved, checkpoint-matched Experiment #5 validation and test prediction CSVs. No inference was run. Validation was used for selection; test results below are descriptive evaluation only.

Selected checkpoint: `experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt`  
SHA256: `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`  
Class order: `akiec, bcc, bkl, df, mel, nv, vasc`

## Validation (n=986)

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| akiec | 30 | 0.551724 | 0.533333 | 0.542373 |
| bcc | 58 | 0.716981 | 0.655172 | 0.684685 |
| bkl | 104 | 0.719101 | 0.615385 | 0.663212 |
| df | 9 | 0.500000 | 0.555556 | 0.526316 |
| mel | 107 | 0.638298 | 0.560748 | 0.597015 |
| nv | 663 | 0.894813 | 0.936652 | 0.915254 |
| vasc | 15 | 0.705882 | 0.800000 | 0.750000 |

Confusion matrix (rows=true, columns=predicted):

```text
[[16, 2, 7, 0, 2, 3, 0],
 [ 3,38, 3, 0, 0,14, 0],
 [ 4, 6,64, 4, 8,18, 0],
 [ 1, 2, 0, 5, 0, 1, 0],
 [ 2, 2, 5, 1,60,34, 3],
 [ 3, 3,10, 0,24,621,2],
 [ 0, 0, 0, 0, 0, 3,12]]
```

The most frequent validation error is melanoma→nevus (34). There are 47 melanoma false negatives: 34 predicted `nv`, 5 `bkl`, 3 `vasc`, 2 `akiec`, 2 `bcc`, and 1 `df`. There are 34 melanoma false positives: 24 true `nv`, 8 `bkl`, and 2 `akiec`.

## Test (n=1,014; evaluation only)

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| akiec | 40 | 0.600000 | 0.675000 | 0.635294 |
| bcc | 58 | 0.811321 | 0.741379 | 0.774775 |
| bkl | 104 | 0.702128 | 0.634615 | 0.666667 |
| df | 11 | 0.666667 | 0.545455 | 0.600000 |
| mel | 107 | 0.567308 | 0.551402 | 0.559242 |
| nv | 676 | 0.914369 | 0.931953 | 0.923077 |
| vasc | 18 | 0.700000 | 0.777778 | 0.736842 |

Confusion matrix (rows=true, columns=predicted):

```text
[[27, 0, 8, 0, 1, 4, 0],
 [ 4,43, 2, 1, 3, 4, 1],
 [10, 4,66, 1,13,10, 0],
 [ 0, 2, 0, 6, 0, 3, 0],
 [ 3, 1, 6, 0,59,34, 4],
 [ 1, 3,12, 1,28,630,1],
 [ 0, 0, 0, 0, 0, 4,14]]
```

There are 48 melanoma false negatives (34 predicted `nv`, 6 `bkl`, 4 `vasc`, 3 `akiec`, and 1 `bcc`) and 45 melanoma false positives (28 true `nv`, 13 `bkl`, 3 `bcc`, and 1 `akiec`).

## Error Pattern

Across the two saved partitions, the leading pair is melanoma→nevus (68 cases), followed by nevus→melanoma (52). The pattern indicates that melanoma/nevus confusion remains the principal internal error mode. The test partition was not used for checkpoint selection, model selection, or tuning.

No clinical-generalization conclusion follows from this internal analysis.
