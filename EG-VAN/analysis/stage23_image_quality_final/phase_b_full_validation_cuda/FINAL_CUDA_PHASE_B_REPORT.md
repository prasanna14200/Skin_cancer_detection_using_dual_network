# Stage 23 Phase B full-validation CUDA robustness report

The frozen Stage 23 checkpoint was evaluated on **all 986 validation images** in the original split order with batch size 16 and CUDA FP16 autocast. No training or checkpoint selection occurred. HAM test and PH2 were not accessed.

Checkpoint SHA256: `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`. GPU/runtime: `Tesla T4`, PyTorch `2.11.0+cu130`. Selective FP32 nonlocal3 q@k remains installed by the frozen loader.

Complete baseline gate: **PASS**, 986/986 labels agree with saved Stage 23 predictions; maximum seven-class probability delta `0` (required ≤0.005). No degraded condition began until this gate passed.

| Condition | Accuracy | Macro F1 | MEL recall | Flips | Mean entropy Δ (nats) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.8296 | 0.6713 | 0.6542 (70/107) | 0/986 | +0.0000 |
| blur_r1 | 0.7241 | 0.3940 | 0.1495 (16/107) | 212/986 | -0.1414 |
| blur_r2 | 0.6917 | 0.2722 | 0.0187 (2/107) | 273/986 | +0.0201 |
| underexposure_070 | 0.7961 | 0.5929 | 0.3084 (33/107) | 133/986 | -0.0640 |
| overexposure_130 | 0.6846 | 0.4395 | 0.6262 (67/107) | 217/986 | +0.2037 |
| contrast_070 | 0.7870 | 0.5472 | 0.2523 (27/107) | 152/986 | -0.0841 |
| jpeg_q40 | 0.7617 | 0.5097 | 0.3551 (38/107) | 177/986 | +0.1085 |

All changes are paired with the unchanged baseline. `cuda_statistical_results.json` records classwise support and recall, correct→incorrect/incorrect→correct transitions, and 1,000-repetition lesion-cluster percentile intervals (seed 2310) for accuracy, macro F1, melanoma recall, flip rate, and entropy changes.

These fixed synthetic raw-image perturbations are exploratory. They do not measure clinical technical adequacy, establish a quality gate, or justify threshold tuning. PH2 and HAM test outcomes were outside this validation-only study.

The outputs are reproducible from `run_cuda_full_validation.py` and `analyze_cuda_full_validation.py`; `cuda_execution_manifest.json` contains frozen input, source and output SHA256 hashes. Existing CPU chunks in the separate `phase_b_full_validation/` directory were not combined or modified.
