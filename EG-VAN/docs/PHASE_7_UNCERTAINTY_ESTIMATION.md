# Phase 7 - Uncertainty Estimation

Date: 2026-09-26
Status: COMPLETE. CUDA inference and artifact audits passed in Google Colab on Tesla T4.

## Objective

Estimate post-hoc predictive uncertainty for the existing leakage-aware EfficientNetV2S checkpoint without retraining, fine-tuning, changing labels, or changing frozen data/splits.

## Existing checkpoint and dataset

- Checkpoint: `experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt`
- Checkpoint epoch: 6 (best validation-loss checkpoint; SHA256 below matches the local file).
- Model: plain torchvision EfficientNetV2S
- Classifier: 7 outputs in class order `akiec,bcc,bkl,df,mel,nv,vasc`
- Split: frozen `data/splits/split_leakage_aware.csv`
- Expected test rows: 1,014 unique images
- Input: processed HAM10000 images, resized to 384x384
- Normalization: ImageNet mean/std `(0.485,0.456,0.406)` and `(0.229,0.224,0.225)`
- Inference policy: `model.eval()`, `torch.inference_mode()`, no augmentation, no optimizer, no parameter updates.

Static checkpoint validation passed locally: checkpoint exists, contains `model_state`, records leakage-aware configuration, and its classifier tensor shape is `(7, 1280)` with bias shape `(7,)`.

## Implemented analysis

Module: `src/run_uncertainty_analysis.py`.

It is prepared to generate:

- `predictions.csv`: image ID, true/predicted labels, correctness, confidence, entropy, and class probabilities.
- `calibration_bins.csv`: ten equal-width confidence bins with count, mean confidence, empirical accuracy, and absolute gap.
- `uncertainty_summary.json`: accuracy, macro-F1, confidence/entropy summaries, ECE, confusion matrix, per-class uncertainty, and selective prediction results.
- `per_class_uncertainty.csv`.
- `selective_prediction.csv` at thresholds 0.50, 0.60, 0.70, 0.80, 0.90, and 0.95.
- `reliability_diagram.png`.
- `confidence_distribution.png`.
- `entropy_distribution.png`.
- `quality_uncertainty.csv` when the Phase 6 quality table is available.

Definitions:

- Confidence: `max_i p_i` from the seven-class softmax vector.
- Predictive entropy: $H(p)=-\sum_i p_i\log(p_i)$ with probabilities clipped to `1e-12` for numerical stability.
- ECE: sum over ten confidence bins of bin coverage multiplied by the absolute difference between mean confidence and empirical accuracy.
- Selective prediction: retain samples whose confidence is at least a selected threshold, then report coverage and retained-sample accuracy. This is not a clinical safety analysis.

## Execution result

Inference ran in CUDA-enabled Google Colab:

- GPU: Tesla T4
- PyTorch: `2.11.0+cu128`
- CUDA: `12.8`
- Python: `3.13.15`
- Test predictions: 1,014 unique images, exactly aligned to the frozen leakage-aware test partition.
- Correct: 810; incorrect: 204.
- Accuracy: `0.7988165680473372`
- Macro-F1: `0.6201368398477792`
- Mean / median confidence: `0.7755111562312237` / `0.8195773065090179`
- Mean / median predictive entropy (natural-log units): `0.5802945969745708` / `0.5426918864250183`
- Mean confidence, correct / incorrect: `0.8284780717190401` / `0.5652013447354821`
- Mean predictive entropy, correct / incorrect: `0.47187217525838887` / `1.0107953890829402`
- ECE, 10 equal-width bins: `0.05242056577398463`
- Checkpoint SHA256: `f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848` (matches the local checkpoint file).

### Checkpoint-state comparability note

The Stage A `test_metrics.json` reports accuracy `0.8412228796844181`, while this inference reports `0.7988165680473372`. Inspection of `src/run_baseline.py` shows it saves the best validation checkpoint during training but evaluates the still-in-memory final epoch model after the loop; it does not reload the best checkpoint before test evaluation. The uncertainty run explicitly loaded `best_checkpoint.pt` from epoch 6. Therefore these two test results are from different checkpoint states and must not be treated as a like-for-like score comparison. Neither result was changed or rerun in this documentation audit.

Per-class counts, accuracy, mean confidence, and mean entropy:

| True class | N | Accuracy | Mean confidence | Mean entropy |
|---|---:|---:|---:|---:|
| AKIEC | 40 | 0.6750 | 0.6787 | 0.9162 |
| BCC | 58 | 0.6034 | 0.6312 | 0.9796 |
| BKL | 104 | 0.3365 | 0.5836 | 0.9830 |
| DF | 11 | 0.5455 | 0.7162 | 0.8381 |
| MEL | 107 | 0.4206 | 0.6218 | 0.8668 |
| NV | 676 | 0.9571 | 0.8460 | 0.4204 |
| VASC | 18 | 0.8333 | 0.8681 | 0.3655 |

Confusion matrix (rows=true, columns=predicted; class order `akiec,bcc,bkl,df,mel,nv,vasc`):

```text
[[27, 3, 5, 0, 0, 5, 0],
 [5, 35, 7, 1, 1, 5, 4],
 [18, 2, 35, 0, 23, 25, 1],
 [0, 2, 0, 6, 0, 3, 0],
 [5, 1, 8, 0, 45, 42, 6],
 [1, 2, 8, 0, 17, 647, 1],
 [0, 1, 0, 1, 0, 1, 15]]
```

Calibration-bin counts and observed gaps:

| Confidence interval | N | Mean confidence | Empirical accuracy | Absolute gap |
|---|---:|---:|---:|---:|
| [0.0, 0.1) | 0 | - | - | - |
| [0.1, 0.2) | 0 | - | - | - |
| [0.2, 0.3) | 1 | 0.2724 | 0.0000 | 0.2724 |
| [0.3, 0.4) | 41 | 0.3649 | 0.1951 | 0.1698 |
| [0.4, 0.5) | 80 | 0.4630 | 0.4000 | 0.0630 |
| [0.5, 0.6) | 98 | 0.5458 | 0.5204 | 0.0254 |
| [0.6, 0.7) | 135 | 0.6506 | 0.7259 | 0.0754 |
| [0.7, 0.8) | 124 | 0.7530 | 0.8871 | 0.1341 |
| [0.8, 0.9) | 147 | 0.8513 | 0.9048 | 0.0535 |
| [0.9, 1.0] | 388 | 0.9646 | 0.9742 | 0.0096 |

Selective prediction results:

| Minimum confidence | Retained | Coverage | Accuracy among retained |
|---:|---:|---:|---:|
| 0.50 | 892 | 0.8797 | 0.8632 |
| 0.60 | 794 | 0.7830 | 0.9055 |
| 0.70 | 659 | 0.6499 | 0.9423 |
| 0.80 | 535 | 0.5276 | 0.9551 |
| 0.90 | 388 | 0.3826 | 0.9742 |
| 0.95 | 291 | 0.2870 | 0.9863 |

The ECE value above was calculated from the corrected ten bins; bin counts sum to the full 1,014 test predictions.

### Calibration artifact correction

Audit found the first saved calibration-bin artifact incorrectly placed every confidence value below 1.0 into the final bin because its lower-bound condition was missing. The bin selection was corrected and calibration bins, ECE, reliability plot, and quality-uncertainty summaries were recomputed from the already-saved `predictions.csv`; model inference was not rerun. Corrected bin counts sum to 1,014, and corrected ECE is `0.05242056577398463`.

The quality-tercile calculation was also corrected to use rank-based equal-sized groups and separate `quality_entropy` from predictive entropy. Its table was recomputed from saved predictions and the Phase 6 quality table only.

Quality and uncertainty comparisons are descriptive; each feature's low/high terciles contain 338 samples:

| Quality proxy | Accuracy, lowest -> highest | Mean confidence, lowest -> highest | Mean predictive entropy, lowest -> highest |
|---|---:|---:|---:|
| Brightness | 0.8402 -> 0.7249 | 0.7957 -> 0.7238 | 0.5379 -> 0.7019 |
| Contrast | 0.7899 -> 0.8639 | 0.7585 -> 0.8116 | 0.6444 -> 0.4696 |
| Sharpness | 0.7988 -> 0.7840 | 0.7535 -> 0.7772 | 0.6256 -> 0.5846 |
| Saturation | 0.7544 -> 0.7811 | 0.7481 -> 0.7717 | 0.6416 -> 0.5994 |
| Dark-pixel ratio | 0.7189 -> 0.8432 | 0.7187 -> 0.8011 | 0.7228 -> 0.5165 |
| Bright-pixel ratio | 0.7515 -> 0.8728 | 0.7302 -> 0.8379 | 0.7060 -> 0.4180 |
| Image entropy | 0.8580 -> 0.7130 | 0.8113 -> 0.7124 | 0.5068 -> 0.7194 |
| Illumination variation | 0.6982 -> 0.8669 | 0.7009 -> 0.8118 | 0.7628 -> 0.4910 |

Patterns differ by feature, so there is no single monotonic image-quality/uncertainty relationship in this descriptive analysis. These associations do not establish causality or clinical significance.

## Artifacts and validation

Under `experiments/uncertainty/`:

- `predictions.csv` - 1,014 unique test predictions and seven probabilities per row.
- `calibration_bins.csv` - corrected ten-bin counts and gaps; total count 1,014.
- `uncertainty_summary.json` - metrics, ECE, per-class summaries, confusion matrix, runtime and checkpoint checksum.
- `selective_prediction.csv`
- `per_class_uncertainty.csv`
- `quality_uncertainty.csv`
- `reliability_diagram.png`
- `confidence_distribution.png`
- `entropy_distribution.png`
- `confidence_correct_vs_incorrect.png`

Audit passed: predictions exactly match the 1,014 IDs and true labels in the frozen test split, all predicted/true labels are among the seven classes, every probability vector has seven values summing to one, and quality joins cover the same test IDs. No training or parameter updates occurred.

The initial local preflight stopped before model execution because the Windows runtime has no CUDA. The inference was then completed in Colab on a Tesla T4. No CPU fallback was used.

## Exact Colab command

The Colab inference command used for the completed run was:

```bash
%cd /content/drive/MyDrive/EG-VAN
!PYTHONPATH=src python -u src/run_uncertainty_analysis.py \
  --project-root /content/drive/MyDrive/EG-VAN \
  --checkpoint /content/drive/MyDrive/EG-VAN/experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt \
  --output-dir /content/drive/MyDrive/EG-VAN/experiments/uncertainty
```

It has already completed; do not rerun inference merely to reproduce the report.

The runner verifies the checkpoint, 1,014 unique test rows, and every referenced processed image. The existing baselines and frozen files remain untouched. For postprocessing already-saved predictions without repeating inference:

```bash
PYTHONPATH=src python src/run_uncertainty_analysis.py \
  --project-root . \
  --checkpoint experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt \
  --output-dir experiments/uncertainty \
  --postprocess-existing
```

## Limitations

- Confidence and entropy differed descriptively between correct and incorrect predictions; this does not make them guaranteed error detectors.
- ECE depends on the stated ten-bin convention.
- Quality tercile patterns varied across features and do not imply causality.
- Selective-prediction accuracy applies only to retained samples and trades off coverage.
- PH² remains blocked and is unrelated to this uncertainty preflight.
- This phase does not evaluate the full EG-VAN architecture.

## Next project step

Phase 7 is complete. The next roadmap decision remains separate; do not retrain the baseline or start full EG-VAN reconstruction automatically.
