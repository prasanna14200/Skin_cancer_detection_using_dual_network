# EG-VAN Phase 4C GPU Smoke-Test Report

Date: 2026-09-20
Status: GPU smoke test passed in Google Colab. No 25-epoch experiment was started.

> Current-status note: this is a historical smoke-test report. The later completed 25-epoch EfficientNetV2S baseline runs are documented in `docs/PHASE_4C_STAGE_A_REPORT.md`, `experiments/efficientnetv2s_naive/`, `experiments/efficientnetv2s_leakage_aware/`, and `docs/PROJECT_DOCUMENTATION.md`.

## Environment

- GPU: Tesla T4
- CUDA available: True
- CUDA runtime: 12.8
- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128

## Split integrity

The existing split files were validated without recreation or modification:

| Split | Rows | Unique lesions | Crossing lesions |
|---|---:|---:|---:|
| Naive | 10,015 | 7,470 | 764 |
| Leakage-aware | 10,015 | 7,470 | 0 |

## Model

- Model: plain torchvision EfficientNetV2S
- ImageNet weights: `EfficientNet_V2_S_Weights.IMAGENET1K_V1`
- Classifier output: 7 classes
- No SCGA, NLB, MFF, fusion, external validation, calibration, or ablation was used.

## Dataset and input pipeline

- Input: Phase 4B processed images only.
- Raw HAM10000 images and H-MNIST CSV files were not used.
- Smoke-test batch shape: `(2, 3, 384, 384)`.
- Labels shape: `(2,)`.
- Labels were valid for all seven classes.
- The processed files were not modified.

## Training mechanics smoke test

- Forward pass: PASS
- Output shape: `(2, 7)`
- Focal loss: PASS
- Smoke-test focal loss value: `0.3575671315193176`
- Backward pass: PASS
- Optimizer step: PASS
- Mixed precision: PASS

The focal-loss value above is a mechanics smoke-test value only. It is not model performance and is not a research result.

## Transform audit

Current `make_transforms()` in `src/train.py` applies:

- Both training and evaluation: resize to `384 x 384`, convert to tensor, ImageNet normalization from `EfficientNet_V2_S_Weights.DEFAULT`.
- Training only: `RandomHorizontalFlip`, `RandomVerticalFlip`, and `RandomRotation(15)`.
- Validation/test: no random augmentation.

Classification:

| Decision | Classification | Basis |
|---|---|---|
| 384 x 384 resize | PAPER-UNDERSPECIFIED / IMPLEMENTATION CHOICE | The paper does not provide an operational model input size; this value was already documented and was used in the successful smoke test. |
| ImageNet normalization from torchvision weights | IMPLEMENTATION CHOICE | Required by the legitimate pretrained torchvision weights; not specified operationally by the paper. |
| Horizontal flip | PAPER-UNDERSPECIFIED / IMPLEMENTATION CHOICE | The paper mentions augmentation but does not enumerate reproducible transformations. |
| Vertical flip | PAPER-UNDERSPECIFIED / IMPLEMENTATION CHOICE | Same ambiguity; added in the current Colab-tested change. |
| Rotation by 15 degrees | PAPER-UNDERSPECIFIED / IMPLEMENTATION CHOICE | Same ambiguity; angle is not specified by the paper. |
| Training-only random augmentation | PAPER-SPECIFIED in scope, IMPLEMENTATION CHOICE in content | The protocol separates training augmentation from deterministic preprocessing; validation/test remain deterministic. |
| No augmentation in validation/test | PAPER-SPECIFIED in scope / FROZEN PROTOCOL | Random training augmentation must not affect validation or test data. |

## Exact change audited

The earlier prepared implementation used only `RandomHorizontalFlip()` for training. The current Colab-tested implementation uses:

```python
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
transforms.RandomRotation(15),
```

This is a documented augmentation approximation, not a claim of reproducing the paper's unspecified twenty transformations exactly.

## Stage A preparation

Target run: **STAGE A - BASELINE 1 - NAIVE SPLIT**.

Frozen configuration:

- Model: EfficientNetV2S with `EfficientNet_V2_S_Weights.IMAGENET1K_V1`
- Dataset: HAM10000 Phase 4B processed images
- Split: `data/splits/split_naive.csv`
- Classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`
- Seed: 42
- Optimizer: Adamax
- Learning rate: 0.001
- Focal loss: alpha 0.25, gamma 2
- Scheduler: ReduceLROnPlateau, factor 0.5, patience 1
- Batch size: 16
- Epochs: 25 unless the approved early-stopping rule applies
- Mixed precision: enabled on CUDA
- Input: 384 x 384
- Metrics: accuracy, macro-F1, per-class recall, confusion matrix

## Preparation gap before launch

Historical note: at the time of this smoke-test report, `src/train.py` defined the frozen configuration, transforms, loss, optimizer, scheduler, and dataset construction helpers, but the executable Stage A runner had not yet been completed. The later runner and completed baseline artifacts now exist; see `docs/PHASE_4C_STAGE_A_REPORT.md` and `docs/PROJECT_DOCUMENTATION.md`.

The GPU smoke-test gate remains valid and complete. This report itself did not start a training run.

## Files used

- `src/dataset.py`
- `src/models/baseline_effnet.py`
- `src/train.py`
- `src/evaluate.py`
- `notebooks/phase4c_gpu_smoke_test.ipynb`
- `data/processed/images/`
- `data/processed/metadata_clean.csv`
- `data/splits/split_naive.csv`
- `data/splits/split_leakage_aware.csv`

## Status

**PHASE 4C GPU SMOKE TEST COMPLETE**

This smoke-test report predates the completed Stage A baseline runs; see `docs/PHASE_4C_STAGE_A_REPORT.md` for the later training results.
