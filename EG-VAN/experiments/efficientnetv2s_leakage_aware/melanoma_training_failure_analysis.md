# Melanoma Training/Checkpoint Failure Analysis

## Executive summary

This read-only analysis finds no evidence of a class-index mapping error or an absent melanoma classifier output. The verified epoch-6 EfficientNetV2S checkpoint does learn to predict `mel` on HAM10000: it detects 51/107 validation melanomas and 45/107 held-out test melanomas when evaluated through existing saved results keyed to the checkpoint SHA-256. Its melanoma-vs-nevus ROC-AUC on the HAM10000 test cases is about 0.865.

The PH² gap is real: the same checkpoint has 0% melanoma argmax recall on the 40 included RAW PH² melanoma cases, while its HAM10000 test melanoma recall is 42.1%. The paired PH² preprocessing diagnostic confirms that changing from RAW to HAM-matched preprocessing substantially shifts scores and labels, recovering 19/40 melanomas as argmax `mel`; however, melanoma ROC-AUC falls from 0.6606 to 0.6325. Therefore preprocessing sensitivity is confirmed, but improved melanoma ranking is not. The paired result does not establish preprocessing as the sole cause of the external failure.

Training history also shows a substantial generalization gap: train loss continues to fall after the epoch-6 minimum validation loss, while validation loss rises. The frozen training data are imbalanced (`nv:mel` ≈ 5.97:1), and the shared focal-loss alpha is not class-specific. These are plausible contributors, not individually proven causes.

## 1. Scope and evidence reviewed

This report was produced from existing files only. The split CSV was counted directly; saved history and metrics were parsed; the uncertainty predictions were analyzed as existing probabilities; and checkpoint metadata and classifier-head parameters were loaded safely on CPU with `weights_only=True`. No model forward pass was run.

Primary evidence:

- `data/splits/split_leakage_aware.csv`
- `experiments/efficientnetv2s_leakage_aware/config.json`
- `experiments/efficientnetv2s_leakage_aware/training_history.json`
- `experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt`
- `experiments/efficientnetv2s_leakage_aware/test_metrics.json`
- `experiments/uncertainty/predictions.csv`
- `experiments/uncertainty/uncertainty_summary.json`
- `src/train.py`, `src/dataset.py`, `src/run_baseline.py`, `src/evaluate.py`, `src/preprocessing.py`
- `experiments/ph2_external_eval/metrics.json` and `predictions.csv`
- `experiments/ph2_preprocessing_diagnostic/summary_metrics.json` and `preprocessing_comparison.csv`

## 2. Frozen split class distribution

Counts and proportions below are recomputed from the frozen leakage-aware split; no split was changed.

| Class | Train (n=8,015) | Train % | Validation (n=986) | Validation % | Test (n=1,014) | Test % |
|---|---:|---:|---:|---:|---:|---:|
| `akiec` | 257 | 3.206% | 30 | 3.043% | 40 | 3.945% |
| `bcc` | 398 | 4.966% | 58 | 5.882% | 58 | 5.720% |
| `bkl` | 891 | 11.117% | 104 | 10.548% | 104 | 10.256% |
| `df` | 95 | 1.185% | 9 | 0.913% | 11 | 1.085% |
| `mel` | 899 | 11.216% | 107 | 10.852% | 107 | 10.552% |
| `nv` | 5,366 | 66.949% | 663 | 67.241% | 676 | 66.667% |
| `vasc` | 109 | 1.360% | 15 | 1.521% | 18 | 1.775% |

The train `nv:mel` ratio is **5.969:1**. Melanoma is not the smallest class (95 `df`, 109 `vasc`, 257 `akiec` training cases), although the dominant nevus class is about six times larger.

## 3. Class mapping and checkpoint integrity

The class order is consistent across `src/dataset.py`, the run config, checkpoint metadata, and evaluator:

| Index | Label |
|---:|---|
| 0 | `akiec` |
| 1 | `bcc` |
| 2 | `bkl` |
| 3 | `df` |
| 4 | `mel` |
| 5 | `nv` |
| 6 | `vasc` |

Checkpoint SHA-256 verified locally: `f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848`. It records epoch 6, EfficientNetV2S, and the above class order. The classifier layer has shape `[7, 1280]`, so all seven output rows, including `mel`, exist.

Classifier output-row parameter summary (descriptive only):

| Class | Weight L2 norm | Bias |
|---|---:|---:|
| `akiec` | 0.82195 | −0.03939 |
| `bcc` | 0.86564 | −0.01790 |
| `bkl` | 0.64789 | +0.01208 |
| `df` | 0.97121 | −0.03680 |
| `mel` | 0.62517 | −0.01210 |
| `nv` | 0.63109 | +0.01980 |
| `vasc` | 1.06499 | −0.02710 |

The melanoma row is nonzero and its norm is similar to the nevus row. These parameter norms and biases alone cannot establish class separability, calibration, or why any particular prediction won. They do not indicate a missing or obviously zeroed melanoma head.

## 4. Training objective and input preprocessing

The saved configuration specifies pretrained torchvision EfficientNetV2S weights, 384×384 inputs, batch size 16, 25 epochs, Adamax at 0.001, and focal loss with `alpha=0.25`, `gamma=2.0`. The source computes a single shared scalar alpha for all samples; it is not class-specific alpha or inverse-frequency weighting. The training `DataLoader` shuffles but does not use a weighted sampler. No per-class loss weights or resampling were found in the audited run path.

Training and evaluation read **processed HAM images** from `data/processed/images/{image_id}.jpg`. The repository preprocessing is hair removal → Gray World balancing → multi-scale Retinex. The model transforms then resize to 384×384, convert to tensor, and use ImageNet normalization. Random flips and rotation are training-only; validation and test are deterministic.

The PH² RAW path instead uses original BMPs converted to RGB, then the same 384×384 resize, tensor conversion, and ImageNet normalization. The paired diagnostic applies the HAM preprocessing operations to PH² originals before the same model transform. Thus the matched diagnostic isolates that preprocessing bundle while keeping the checkpoint and the 120 cases fixed; it does not isolate each operation individually.

## 5. Epoch-by-epoch validation and training evidence

Melanoma precision, recall, and F1 below are derived from each saved validation confusion matrix. `mel predicted` is the sum of the prediction column for `mel`; melanoma support is 107 at every validation epoch.

| Epoch | Train loss | Val loss | Val accuracy | Val macro-F1 | Mel precision | Mel recall | Mel F1 | Mel predicted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.108173 | 0.087095 | 0.7667 | 0.4916 | 0.4714 | 0.3084 | 0.3729 | 70 |
| 2 | 0.076302 | 0.082808 | 0.7840 | 0.5659 | 0.7568 | 0.2617 | 0.3889 | 37 |
| 3 | 0.063298 | 0.087965 | 0.7901 | 0.5919 | 0.6000 | 0.4486 | 0.5134 | 80 |
| 4 | 0.052824 | 0.080784 | 0.8012 | 0.6320 | 0.5600 | 0.3925 | 0.4615 | 75 |
| 5 | 0.044782 | 0.076120 | 0.8043 | 0.6524 | 0.5517 | 0.4486 | 0.4948 | 87 |
| 6 | 0.037086 | **0.076013** | 0.8073 | 0.6517 | 0.5862 | 0.4766 | 0.5258 | 87 |
| 7 | 0.031582 | 0.080979 | 0.7972 | 0.6284 | 0.5960 | 0.5514 | 0.5728 | 99 |
| 8 | 0.028996 | 0.083070 | 0.8073 | 0.6288 | 0.7551 | 0.3458 | 0.4744 | 49 |
| 9 | 0.016130 | 0.079671 | 0.8245 | 0.7042 | 0.6522 | 0.4206 | 0.5114 | 69 |
| 10 | 0.013057 | 0.085866 | 0.8286 | **0.7288** | 0.6719 | 0.4019 | 0.5029 | 64 |
| 11 | 0.008289 | 0.088711 | 0.8398 | 0.7235 | 0.7059 | 0.4486 | 0.5486 | 68 |
| 12 | 0.006504 | 0.090174 | 0.8337 | 0.7283 | 0.6629 | 0.5514 | 0.6020 | 89 |
| 13 | 0.005235 | 0.091458 | 0.8276 | 0.7103 | 0.6322 | 0.5140 | 0.5670 | 87 |
| 14 | 0.004412 | 0.091690 | 0.8276 | 0.7083 | 0.6500 | 0.4860 | 0.5561 | 80 |
| 15 | 0.003716 | 0.092346 | 0.8337 | 0.7022 | 0.6667 | 0.5421 | 0.5979 | 87 |
| 16 | 0.003465 | 0.095327 | 0.8367 | 0.7252 | 0.7051 | 0.5140 | 0.5946 | 78 |
| 17 | 0.003096 | 0.094428 | 0.8357 | 0.7122 | 0.6867 | 0.5327 | 0.6000 | 83 |
| 18 | 0.003278 | 0.095694 | 0.8357 | 0.7128 | 0.7162 | 0.4953 | 0.5856 | 74 |
| 19 | 0.002989 | 0.097923 | **0.8418** | 0.7283 | 0.7105 | 0.5047 | 0.5902 | 76 |
| 20 | 0.002579 | 0.099493 | 0.8357 | 0.7104 | 0.6750 | 0.5047 | 0.5775 | 80 |
| 21 | 0.002556 | 0.098506 | 0.8387 | 0.7146 | 0.6951 | 0.5327 | 0.6032 | 82 |
| 22 | 0.002420 | 0.095719 | 0.8347 | 0.7086 | 0.6860 | 0.5514 | **0.6114** | 86 |
| 23 | 0.002680 | 0.094990 | 0.8337 | 0.7188 | 0.6413 | 0.5514 | 0.5930 | 92 |
| 24 | 0.002366 | 0.098116 | 0.8347 | 0.7138 | 0.6867 | 0.5327 | 0.6000 | 83 |
| 25 | 0.002564 | 0.099048 | 0.8347 | 0.7136 | 0.6705 | 0.5514 | 0.6051 | 88 |

Epoch 6 is the **best-validation-loss** checkpoint. Other history extrema are:

- Best validation macro-F1: epoch 10, 0.7288.
- Best validation accuracy: epoch 19, 0.8418.
- Best melanoma recall: 0.5514 at epochs 7, 12, 22, 23, and 25.
- Best melanoma F1: epoch 22, 0.6114.

The model's train loss falls 97.6% from epoch 1 to epoch 25 (0.10817 → 0.00256), while validation loss bottoms at epoch 6 then rises to 0.09905 by epoch 25, about 30.3% above its minimum. At epoch 25, training accuracy/macro-F1 are 0.9884/0.9866 versus validation 0.8347/0.7136; training melanoma recall is 0.9600 versus validation 0.5514. This is strongly consistent with overfitting after the early validation-loss minimum, though the later validation accuracy and melanoma metrics fluctuate rather than declining monotonically.

The checkpoint criterion creates an observed tradeoff: epoch 22 has higher validation melanoma F1 (0.6114 vs 0.5258 at epoch 6) and recall (0.5514 vs 0.4766), but higher validation loss (0.09572 vs 0.07601). Epoch 22 weights are not present in the inspected checkpoint artifact; this comparison is of saved metrics only and does not imply epoch 22 would generalize better.

### Epoch-6 validation class metrics

| Class | Support | Predicted count | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| `akiec` | 30 | 38 | 0.4474 | 0.5667 | 0.5000 |
| `bcc` | 58 | 47 | 0.8085 | 0.6552 | 0.7238 |
| `bkl` | 104 | 54 | 0.6481 | 0.3365 | 0.4430 |
| `df` | 9 | 11 | 0.6364 | 0.7778 | 0.7000 |
| `mel` | 107 | 87 | 0.5862 | 0.4766 | 0.5258 |
| `nv` | 663 | 727 | 0.8721 | 0.9563 | 0.9122 |
| `vasc` | 15 | 22 | 0.6364 | 0.9333 | 0.7568 |

The epoch-6 melanoma row has 51 correct `mel`, 46 `nv`, 4 `vasc`, 4 `akiec`, 1 `bcc`, and 1 `bkl` predictions. The checkpoint is not melanoma-blind in-domain, but its validation misses are predominantly melanoma→nevus.

## 6. Best-checkpoint internal test evidence vs final-epoch metrics

`experiments/uncertainty/uncertainty_summary.json` and `predictions.csv` are keyed to the exact verified checkpoint SHA-256. Recalculation from those existing test probabilities gives:

- 1,014 HAM10000 test cases; accuracy 0.7988 and macro-F1 0.6201.
- Melanoma: 45/107 correct; precision 45/86 = 0.5233, recall 45/107 = 0.4206, F1 = 0.4663.
- The melanoma test confusion row (actual `mel`, predicted in class order above) is `[5, 1, 8, 0, 45, 42, 6]`.
- Melanoma probability mean/median: 0.3930/0.3904 for actual melanoma; 0.1077/0.0500 for actual nevus.
- ROC-AUC for `mel` vs `nv` on the 783 such test cases is 0.8653; for `mel` vs all six non-melanoma classes on all 1,014 cases it is 0.8617.

By contrast, `experiments/efficientnetv2s_leakage_aware/test_metrics.json` was written by `src/run_baseline.py` after the final epoch-25 model. It reports 55/107 melanoma recall (0.5140), not the epoch-6 checkpoint's 45/107. Those final-epoch metrics must not be attributed to the frozen epoch-6 checkpoint. The uncertainty artifact is the existing checkpoint-matched test evidence.

## 7. PH² comparison

The verified external RAW PH² result for 80 common nevi and 40 melanomas is 0/40 melanoma argmax recall, with all predictions `nv=102`, `vasc=18`, `mel=0`. The separate paired diagnostic used the same 120 cases and frozen checkpoint:

| Evidence | HAM10000 test, checkpoint-matched | PH² RAW | PH² HAM-preprocessed |
|---|---:|---:|---:|
| Melanoma argmax recall | 0.4206 | 0.0000 | 0.4750 |
| Melanoma-vs-nevus ROC-AUC | 0.8653 | 0.6606 | 0.6325 |
| Mean `p(mel)`, true melanoma | 0.3930 | 0.1851 | 0.3489 |
| Mean `p(mel)`, true nevus | 0.1077 | 0.1409 | 0.2652 |

The held-out HAM test and PH² are independent datasets, and the PH² sample is small; these values are evidence of an external-domain performance gap, not a controlled estimate of a generalization effect. The within-PH² paired result is the controlled part: HAM-matched preprocessing changed 48/120 predictions and raised mean `p(mel)` by 0.13749, with increases for both true melanoma (+0.16389 mean) and true nevus (+0.12428 mean). It increased argmax melanoma recall to 19/40, but lowered AUC by 0.02813. This indicates preprocessing sensitivity and a class-wide score shift, not demonstrated ranking improvement.

## 8. Evidence-weighted interpretation

### CONFIRMED

- The checkpoint class-index mapping, output shape, and evaluator mapping agree; no index permutation or absent `mel` output was found.
- The exact checkpoint detects melanomas on HAM10000 validation and test (47.7% and 42.1% recall, respectively), but often confuses them with `nv`.
- The train set has a roughly 5.97:1 `nv:mel` ratio; no class-specific alpha or weighted sampler is used in the audited path.
- The saved training history shows a rising validation loss after epoch 6 while training loss keeps falling.
- On the paired 120-case PH² sample, HAM preprocessing materially changes scores and hard predictions; it increases argmax melanoma detections from 0 to 19, but AUC falls rather than rises.

### STRONGLY SUPPORTED

- A meaningful external-domain gap exists: checkpoint-matched in-domain HAM10000 test AUC is materially above either PH² AUC, and the RAW PH² melanoma recall is lower than internal recall.
- Overfitting after the early validation-loss minimum is a strong training concern, evidenced by train/validation divergence and worsening validation loss.
- Input preprocessing/domain shift contributes to PH² output behavior: the paired same-image experiment changes many predictions, including former `vasc` outputs, and raises melanoma scores in both true classes.
- The current validation-loss checkpoint selection does not optimize melanoma recall/F1 directly; saved later-epoch history contains better melanoma recall/F1 values than epoch 6, with worse validation loss.

### POSSIBLE, NOT PROVEN

- The absence of class-specific weighting/resampling and the `nv` frequency may bias the learned decision boundary against melanoma. Class imbalance is present, but the in-domain checkpoint has substantial melanoma sensitivity and the available evidence does not isolate the loss/sampling contribution.
- Broader PH²-vs-HAM differences (acquisition, image appearance, lesion composition, and dataset size) may contribute beyond preprocessing; they are not separated by this paired experiment.
- Selecting the epoch by a melanoma-sensitive criterion could improve melanoma metrics, but the history alone does not prove that a corresponding checkpoint would generalize better. Later epoch weights were not evaluated on PH² and should not be selected using PH².

### NOT SUPPORTED

- A simple class-index/class-label mapping bug.
- A claim that HAM preprocessing improves melanoma ranking/discrimination on PH²: the measured AUC decreases.
- A claim that preprocessing alone explains the external failure, or that class imbalance alone caused it.

## 9. Recommended next technical step (proposal only; not executed)

Run one controlled HAM10000-only loss ablation if retraining is approved: retain the same frozen leakage-aware split, image preprocessing, model initialization, optimizer, scheduler, batch size, epochs, and seed; change only the shared-alpha focal objective to a prespecified class-balanced focal weighting. Do not use PH² for training, checkpoint selection, threshold selection, or class-weight tuning.

Before running, define validation success against the epoch-6 baseline: at least a **0.05 absolute increase in melanoma F1** over 0.5258, while validation macro-F1 remains no more than 0.02 below the epoch-6 value of 0.6517 and nevus recall remains at least 0.90. Also save validation probabilities so melanoma ROC-AUC and score calibration can be reported, rather than judging only argmax counts. If these conditions fail, conclude that this single weighting change did not meet the prespecified validation target. A validation pass would justify a separately approved, single evaluation on the frozen HAM test set; it would not establish external PH² benefit.

No retraining or experiment is authorized or performed by this report.

## Integrity status

- Training executed for this analysis: **NO**.
- Model inference executed for this analysis: **NO**.
- Checkpoint modified: **NO**.
- HAM10000 data or frozen split modified: **NO**.
- PH² data or artifacts modified: **NO**.
- Existing training, test, and uncertainty artifacts modified: **NO**.
- New file created: **this report only**.
