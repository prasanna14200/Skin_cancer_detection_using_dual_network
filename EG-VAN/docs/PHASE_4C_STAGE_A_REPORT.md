# EG-VAN Phase 4C Stage A Report

Date: 2026-09-23

## 1. Research question

Does lesion-level separation materially change EfficientNetV2S baseline performance relative to an image-level split? This comparison measures a potential evaluation effect; it does not establish causality and is not a complete EG-VAN architecture comparison.

## 2. Experimental setup

- Dataset: HAM10000, 10,015 images, 7 classes.
- Model: plain torchvision EfficientNetV2S.
- Weights: `EfficientNet_V2_S_Weights.IMAGENET1K_V1`.
- Input: Phase 4B processed images, resized to 384x384 in the model transform.
- Seed: 42.
- Optimizer: Adamax, learning rate 0.001.
- Loss: focal loss, alpha 0.25, gamma 2.
- Scheduler: ReduceLROnPlateau, factor 0.5, patience 1.
- Batch size: 16.
- Maximum epochs: 25.
- Mixed precision: enabled on CUDA.
- Metrics: accuracy, macro-F1, per-class recall, confusion matrix.

Both completed runs used the same documented training-only augmentation approximation and deterministic validation/test transforms.

## 3. Split definitions

- **Naive:** image-level stratified 80/10/10 split; images from one lesion may cross partitions.
- **Leakage-aware:** grouped by `lesion_id` before assignment; every lesion remains in exactly one partition.

Integrity reported for both splits: 10,015 rows and 7,470 unique lesions. Naive crossing lesions: 764. Leakage-aware crossing lesions: 0.

## 4. Dataset integrity

The local verified HAM10000 metadata and processed-image audit found 10,015 image files matching metadata and frozen splits. No local dataset or split modifications were made for this comparison.

## 5. Results

### Naive EfficientNetV2S

Run directory: `experiments/efficientnetv2s_naive`

- Accuracy: `0.8838133068520357`
- Macro-F1: `0.8073501563707641`
- AKIEC recall: `0.7058823529411765`
- BCC recall: `0.9038461538461539`
- BKL recall: `0.7027027027027027`
- DF recall: `0.6666666666666666`
- MEL recall: `0.625`
- NV recall: `0.9657228017883756`
- VASC recall: `1.0`

### Leakage-aware EfficientNetV2S

Run directory: `experiments/efficientnetv2s_leakage_aware`

- Accuracy: `0.8412228796844181`
- Macro-F1: `0.7114084504160505`
- AKIEC recall: `0.725`
- BCC recall: `0.7241379310344828`
- BKL recall: `0.6057692307692307`
- DF recall: `0.45454545454545453`
- MEL recall: `0.514018691588785`
- NV recall: `0.9526627218934911`
- VASC recall: `0.8333333333333334`

Current local artifact audit: the synchronized workspace now contains `experiments/efficientnetv2s_naive/test_metrics.json` and `experiments/efficientnetv2s_leakage_aware/test_metrics.json`, including both confusion-matrix arrays. The master documentation reproduces both matrices from those files.

## 6. Side-by-side comparison

| Metric | Naive | Leakage-aware | Naive - leakage-aware |
|---|---:|---:|---:|
| Accuracy | 0.883813 | 0.841223 | 0.042590 (4.259 pp) |
| Macro-F1 | 0.807350 | 0.711408 | 0.095942 (9.594 pp) |
| AKIEC recall | 0.705882 | 0.725000 | -0.019118 (-1.912 pp) |
| BCC recall | 0.903846 | 0.724138 | 0.179708 (17.971 pp) |
| BKL recall | 0.702703 | 0.605769 | 0.096933 (9.693 pp) |
| DF recall | 0.666667 | 0.454545 | 0.212121 (21.212 pp) |
| MEL recall | 0.625000 | 0.514019 | 0.110981 (11.098 pp) |
| NV recall | 0.965723 | 0.952663 | 0.013060 (1.306 pp) |
| VASC recall | 1.000000 | 0.833333 | 0.166667 (16.667 pp) |

Largest absolute recall changes were DF, BCC, VASC, MEL, and BKL. AKIEC recall increased slightly under the leakage-aware split; NV changed comparatively little.

## 7. Interpretation

Baseline performance decreased when evaluation used lesion-level separation rather than image-level splitting: accuracy decreased by 4.259 percentage points and macro-F1 decreased by 9.594 percentage points. This is a descriptive comparison of two protocols. It does not prove that lesion overlap caused the entire difference, does not establish that the paper's 98.2% was definitely wrong, and does not claim support or rejection of the broader hypothesis.

This is an EfficientNetV2S baseline comparison, not a comparison of the complete EG-VAN architecture.

## 8. Limitations

- The completed Colab run directories are now synchronized into the local workspace, but repeated runs were still not performed.
- Repeat runs were not performed, so statistical stability of the observed delta has not been established.
- Only one baseline architecture was compared.
- No external validation, calibration, ablation, efficiency benchmarking, or EG-VAN attention modules were evaluated.

## 9. Stage A gate and decision

A real descriptive accuracy and macro-F1 delta is present. However, the frozen wording requires a stable delta, and stability has not been independently established by repeated runs. Therefore the result is sufficient to motivate the next planned external-validation preparation, but the stability caveat must remain explicit before treating the gate as fully satisfied.

## 10. Next protocol step

Stage B is external validation in the frozen order: PH2 first, then ISIC2019. PH2 verification and an evaluation-only plan have been prepared separately. No PH2 images were downloaded or evaluated in this step.

## 11. Artifact verification update

The completed Colab run directories are now present in the local workspace. `test_metrics.json`, `config.json`, `split_integrity.json`, `training_history.json`, best checkpoints, and final checkpoints are present for both naive and leakage-aware runs. The synchronized `test_metrics.json` files include the scalar metrics and confusion matrices used in `docs/PROJECT_DOCUMENTATION.md`.

## 12. Deviations

No new deviation was introduced in this comparison. Existing preprocessing, resize, normalization, and training-augmentation choices remain those recorded in `docs/DEVIATIONS.md`.
