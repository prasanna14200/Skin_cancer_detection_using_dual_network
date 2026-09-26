# EG-VAN Research Project - Complete Project Documentation

## 1. Project Overview

This repository supports an EG-VAN skin lesion classification research project. The implemented work so far focuses on building a careful HAM10000 data foundation, reproducing paper-informed preprocessing where operational details are available, and training a plain EfficientNetV2S baseline under both image-level and lesion-level split protocols.

The current classification task is 7-class HAM10000 lesion diagnosis:

| Label | Count |
|---|---:|
| `akiec` | 327 |
| `bcc` | 514 |
| `bkl` | 1,099 |
| `df` | 115 |
| `mel` | 1,113 |
| `nv` | 6,705 |
| `vasc` | 142 |
| Total | 10,015 |

The local dataset audit verified 10,015 HAM10000 metadata rows, 10,015 dermoscopic JPGs, and 7,470 unique lesion IDs. The project context is the supplied EG-VAN paper, "A Global and Local Attention-Based Dual-Branch Ensemble Network With Advanced Color Balancing for Multi-Class Skin Cancer Recognition." The repository does not currently implement the full EG-VAN architecture. It implements paper-informed preprocessing and an EfficientNetV2S baseline used to evaluate split behavior.

## 2. Research Goal

The original EG-VAN objective is multi-class skin cancer recognition using an attention-based dual-branch ensemble network and advanced color balancing. The current repository objective is narrower: establish a verified data/preprocessing foundation and compare a plain EfficientNetV2S baseline under naive and leakage-aware evaluation.

COMPLETED:

- HAM10000 metadata and image verification.
- Deterministic preprocessing: hair removal, Gray World correction, and Retinex correction.
- Frozen naive and leakage-aware split files.
- EfficientNetV2S baseline training on both split protocols.
- Descriptive leakage-aware comparison.
- PH2 secondary-mirror pre-run audit and external-evaluation script.
- Image-quality proxy extraction and descriptive distributions.
- Post-hoc softmax uncertainty, entropy, ECE, selective prediction, and quality-uncertainty analysis.

BLOCKED:

- PH2 external inference remains blocked; the local package is a secondary mirror and no PH2 inference result exists.

PLANNED / NOT YET IMPLEMENTED:

- Full EG-VAN modules such as SCGA, NLB, MFF, dual-branch fusion, and paper-level architecture comparison.
- Repeated-run stability analysis.

Baseline experiments matter because they separate a working, auditable comparison from claims about the complete EG-VAN architecture. Leakage-aware evaluation matters because HAM10000 contains multiple images for some lesions; image-level random splitting can put related lesion images in train and validation/test partitions. PH2 external validation is intended to test cross-dataset behavior on direct class overlap only. Image-quality and uncertainty analyses are implemented; full EG-VAN modules and repeated-run stability work remain future work.

## 3. Complete Project Architecture

```text
Raw Dataset
    -> Preprocessing
    -> Processed Images
    -> Metadata
    -> Dataset Splitting
    -> Naive Split / Leakage-Aware Split
    -> EfficientNetV2S Baseline
    -> Training
    -> Validation
    -> Test Evaluation
    -> Metrics
    -> Research Analysis
    -> Additional Validation
        -> Image Quality Assessment (descriptive analysis complete; error-link analysis integrated later)
        -> PH2 External Validation (blocked before inference)
        -> Uncertainty Estimation (completed for frozen test predictions)
```

Raw Dataset: `data/raw/HAM10000_metadata.csv` and HAM10000 image directories provide the starting point. Metadata includes `lesion_id`, `image_id`, `dx`, `dx_type`, `age`, `sex`, and `localization`.

Preprocessing: `src/preprocessing.py` applies deterministic per-image operations: blackhat/Telea hair removal, Gray World color correction, and multi-scale Retinex. Cached outputs are written to `data/processed/images/`.

Metadata: `prepare_metadata.py` validates required columns, class counts, missing values, and image correspondence, then writes `data/processed/metadata_clean.csv`.

Splitting: `build_splits.py` creates `split_naive.csv` and `split_leakage_aware.csv`. The naive split stratifies by image. The leakage-aware split groups by `lesion_id`.

Model: `src/models/baseline_effnet.py` builds a torchvision EfficientNetV2S backbone with a replaced 7-class classifier.

Training: `src/run_baseline.py` trains the baseline on a selected split with focal loss, Adamax, ReduceLROnPlateau, CUDA mixed precision, and checkpointing.

Evaluation: `src/evaluate.py` and the training runner compute accuracy, macro-F1, per-class recall, and confusion matrices.

Additional validation: `src/external_eval.py` is prepared for PH2 inference-only evaluation, but no PH2 inference result exists.

## 4. Phase-by-Phase History

### Phase 1 - Project Setup / Research Understanding

The project started from the supplied EG-VAN paper and a frozen experimental protocol. `docs/DEVIATIONS.md` records differences between the paper and the implemented repository. The main research discipline is to avoid claiming full reproduction when only a baseline or preparation step exists.

### Phase 2 - Dataset / Data Understanding

HAM10000 is the verified dataset. It contains 10,015 images and 7,470 unique lesions. The seven class labels are `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, and `vasc`. The raw H-MNIST CSV files are present but unused. No metadata rows were removed in the verified preparation.

### Phase 3 - Preprocessing

The implemented preprocessing pipeline is:

```text
Original image -> hair removal -> Gray World correction -> Retinex correction -> processed image
```

The paper does not fully specify several numerical parameters. The repository fixes those parameters explicitly and records them in `docs/DEVIATIONS.md`: hair kernel size 17, threshold 10, Telea radius 1.0, Retinex sigmas 15/80/250, equal Retinex weights, epsilon 1e-6, and per-channel 1st-99th percentile normalization. Cached processed files preserve the source dimensions of 450x600.

### Phase 4A - Preprocessing Validation

The sample gate processed 10 deterministic images using seed 42. All sample outputs opened successfully, retained valid three-channel dimensions, and produced before/after sample images. Full preprocessing processed all 10,015 images with zero reported failures.

### Phase 4B - Split / Data Foundation

Actual verified numbers:

| Measurement | Naive | Leakage-aware |
|---|---:|---:|
| Train images | 8,010 | 8,015 |
| Validation images | 998 | 986 |
| Test images | 1,007 | 1,014 |
| Total images | 10,015 | 10,015 |
| Unique lesions | 7,470 | 7,470 |
| Cross-partition lesions | 764 | 0 |

The naive split can place images from one lesion in multiple partitions. The leakage-aware split assigns each lesion group to exactly one partition and is the preferred honest evaluation split.

### Phase 4C - EfficientNetV2S Baseline

EfficientNetV2S was selected as the current baseline backbone. The model uses ImageNet-pretrained torchvision weights and replaces only the final classifier with a 7-output linear layer. It is not the complete EG-VAN architecture.

Verified configuration:

- Input size: 384x384 model transform.
- Normalization: EfficientNetV2S ImageNet weights mean/std.
- Training transforms: resize, random horizontal flip, random vertical flip, random rotation 15 degrees, tensor conversion, normalization.
- Validation/test transforms: resize, tensor conversion, normalization.
- Optimizer: Adamax, learning rate 0.001.
- Loss: focal loss, alpha 0.25, gamma 2.0.
- Scheduler: ReduceLROnPlateau, factor 0.5, patience 1.
- Epochs: 25.
- Batch size: 16.
- Mixed precision: enabled on CUDA.
- GPU/Colab setup: completed runs report PyTorch 2.11.0+cu128 on Linux Colab; smoke-test report records Tesla T4 and torchvision 0.26.0+cu128.

Flow:

```text
image -> transform -> tensor -> EfficientNetV2S -> classifier -> logits
      -> focal loss -> backpropagation -> Adamax optimizer
      -> validation loss -> best checkpoint -> test evaluation
```

### Phase A / Stage A - Naive vs Leakage-Aware Baseline

Naive baseline:

- Accuracy: 0.8838133068520357
- Macro F1: 0.8073501563707641
- Per-class recall: `akiec` 0.7058823529411765, `bcc` 0.9038461538461539, `bkl` 0.7027027027027027, `df` 0.6666666666666666, `mel` 0.625, `nv` 0.9657228017883756, `vasc` 1.0.
- Confusion matrix, rows=true and columns=predicted in class order `akiec,bcc,bkl,df,mel,nv,vasc`:

```text
[[24, 3, 4, 2, 0, 1, 0],
 [0, 47, 0, 0, 4, 1, 0],
 [4, 2, 78, 1, 11, 15, 0],
 [0, 1, 0, 8, 0, 3, 0],
 [3, 1, 8, 0, 70, 30, 0],
 [0, 0, 3, 1, 19, 648, 0],
 [0, 0, 0, 0, 0, 0, 15]]
```

Leakage-aware baseline:

- Accuracy: 0.8412228796844181
- Macro F1: 0.7114084504160505
- Per-class recall: `akiec` 0.725, `bcc` 0.7241379310344828, `bkl` 0.6057692307692307, `df` 0.45454545454545453, `mel` 0.514018691588785, `nv` 0.9526627218934911, `vasc` 0.8333333333333334.
- Confusion matrix, rows=true and columns=predicted in class order `akiec,bcc,bkl,df,mel,nv,vasc`:

```text
[[29, 1, 5, 0, 0, 5, 0],
 [3, 42, 2, 0, 4, 4, 3],
 [8, 3, 63, 0, 14, 16, 0],
 [0, 1, 0, 5, 2, 3, 0],
 [1, 0, 12, 0, 55, 36, 3],
 [0, 2, 4, 0, 26, 644, 0],
 [0, 0, 0, 1, 0, 2, 15]]
```

Accuracy difference: 0.04259042716761758, or 4.259 percentage points. Macro-F1 difference: 0.0959417059547136, or 9.594 percentage points. This is descriptive only because repeated runs were not performed. It does not prove a single causal explanation, does not disprove or reproduce the paper's final EG-VAN result, and does not compare the complete EG-VAN architecture.

## 5. Current PH2 External Validation Status

PH2 external validation was proposed to test whether the frozen HAM10000 baseline generalizes to a different dermoscopic dataset on direct label overlap. PH2 contains common nevi, atypical nevi, and melanoma.

Intended mapping:

| PH2 class | HAM10000 class | Decision |
|---|---|---|
| Common nevus | `nv` | Include |
| Melanoma | `mel` | Include |
| Atypical nevus | none | Exclude |

Atypical nevus must not be force-mapped to `bkl` or any other HAM10000 label because the repository has no approved clinical mapping.

Current local PH2 status:

- `data/external/ph2/images/` contains 200 BMP images.
- `data/external/ph2/metadata/ph2_manifest.csv` contains 200 rows.
- Manifest distribution: 80 common nevus, 80 atypical nevus, 40 melanoma.
- Valid overlap for evaluation would be 120 images after excluding atypical nevus.
- `src/external_eval.py` exists and is inference-only.
- Intended checkpoint: `experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt`.
- Inference was not performed.
- Status: BLOCKED before inference because the local Windows context lacks the required CUDA/torchvision runtime.

No PH2 accuracy, F1, AUROC, confusion matrix, prediction file, or sample-level result exists.

## 6. Image Quality Assessment

Phase 6 is complete. `src/image_quality.py` computes deterministic brightness, contrast, sharpness, saturation, dark/bright pixel ratios, entropy, and illumination-variation proxies. `src/run_quality_analysis.py` creates a strict quality table, descriptive split/class statistics, and plots under `experiments/image_quality/`.

All 10,015 frozen leakage-aware rows were analyzed. No images were removed, labels changed, splits changed, preprocessing rerun, or baseline retrained. These are image-quality proxies, not clinical quality labels.

Quality extraction and distribution analysis were completed locally. Phase 7 later added a Colab inference-only test-set prediction table and descriptive quality-versus-confidence/entropy/error terciles; these remain associations only, not causal evidence.

## 7. Uncertainty Estimation

Phase 7 is complete. The existing leakage-aware checkpoint was evaluated post hoc in Colab without retraining. Outputs include per-image softmax probabilities, confidence, predictive entropy, ten-bin ECE, reliability plot, selective prediction tables, per-class summaries, and quality-uncertainty joins. Corrected ECE is recorded in `docs/PHASE_7_UNCERTAINTY_ESTIMATION.md`. This is predictive uncertainty from softmax, not Bayesian or epistemic uncertainty.

The Phase 7 predictions use the validation-selected best checkpoint (epoch 6). The Stage A `test_metrics.json` was produced by evaluating the final in-memory epoch (epoch 25), as the runner does not reload the best checkpoint before testing. Their test scores are not directly comparable as the same model state; both historical results remain unchanged.

## 8. Repository Structure

```text
EG-VAN/
├── checkpoints/
├── data/
│   ├── external/
│   ├── processed/
│   ├── raw/
│   └── splits/
├── docs/
├── experiments/
├── notebooks/
├── src/
│   ├── models/
│   ├── build_splits.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── external_eval.py
│   ├── prepare_metadata.py
│   ├── preprocessing.py
│   ├── run_baseline.py
│   ├── run_quality_analysis.py
│   ├── run_uncertainty_analysis.py
│   └── train.py
├── build_splits.py
└── prepare_metadata.py
```

Important files:

`prepare_metadata.py`: validates HAM10000 metadata and image correspondence, reports counts, writes `data/processed/metadata_clean.csv`.

`build_splits.py`: creates deterministic naive and leakage-aware split CSVs using seed 42.

`src/prepare_metadata.py` and `src/build_splits.py`: wrapper entry points that execute the root scripts.

`src/preprocessing.py`: implements Phase 4B preprocessing, sample validation, full-dataset processing, and logging.

`src/dataset.py`: defines class names, split integrity validation, and `HAM10000Dataset`.

`src/models/baseline_effnet.py`: builds the plain EfficientNetV2S baseline and replaces the classifier.

`src/train.py`: defines constants, seed control, focal loss, transforms, dataset/model assembly, optimizer, scheduler, and configuration serialization.

`src/run_baseline.py`: executable CUDA training runner with checkpointing, history writing, validation, and test metrics.

`src/evaluate.py`: evaluates an existing checkpoint on a split without training.

`src/external_eval.py`: evaluates mapped PH2 samples with a frozen checkpoint; rejects ambiguous mappings and requires CUDA.

`experiments/efficientnetv2s_naive/`: completed naive baseline artifacts.

`experiments/efficientnetv2s_leakage_aware/`: completed leakage-aware baseline artifacts.

`docs/DEVIATIONS.md`: deviation log for paper-underspecified or changed implementation choices.

## 9. Code Execution Flow

Local CPU environment:

```bash
python prepare_metadata.py
python build_splits.py
python src/preprocessing.py --sample-only
```

Full preprocessing can run locally, but it rewrites cached processed outputs and should not be rerun casually:

```bash
python src/preprocessing.py
```

Local CPU baseline training is intentionally blocked by `src/run_baseline.py`.

Colab GPU environment:

```bash
python src/run_baseline.py --project-root . --split-csv split_naive.csv --run-dir experiments/efficientnetv2s_naive
python src/run_baseline.py --project-root . --split-csv split_leakage_aware.csv --run-dir experiments/efficientnetv2s_leakage_aware
```

Evaluate a checkpoint:

```bash
python src/evaluate.py experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt . split_leakage_aware.csv --output experiments/efficientnetv2s_leakage_aware/test_metrics.json
```

PH2 inference command shape, still blocked locally:

```bash
python src/external_eval.py data/external/ph2/metadata/ph2_manifest.csv experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt experiments/ph2_external_eval
```

## 10. Important Configuration Decisions

| Decision | Current value | Why |
|---|---|---|
| Dataset | HAM10000 | Verified local 7-class dataset |
| Number of classes | 7 | HAM10000 labels in `CLASS_NAMES` |
| Input size | 384x384 | EfficientNetV2S baseline transform choice |
| Cached processed size | 450x600 | Phase 4B preserves source dimensions |
| Backbone | EfficientNetV2S | Current baseline |
| Pretrained weights | torchvision EfficientNetV2S ImageNet weights | Transfer learning baseline |
| Normalization | torchvision weights mean/std | Match pretrained backbone |
| Main split strategy | leakage-aware | Prevent lesion IDs crossing partitions |
| Loss | focal loss alpha 0.25, gamma 2.0 | Approved baseline config |
| Optimizer | Adamax, lr 0.001 | Approved baseline config |
| Scheduler | ReduceLROnPlateau factor 0.5, patience 1 | Approved baseline config |
| Device | CUDA for training and PH2 inference | Runner blocks CPU training/inference |

## 11. Deviations From Original Paper

| Paper component | Our implementation | Status | Reason |
|---|---|---|---|
| Hair removal | Blackhat mask, 17x17 ellipse, threshold 10, Telea radius 1.0 | PARTIAL / PAPER-UNDERSPECIFIED | Paper lacks exact parameters |
| Gray World | Per-image BGR pairwise gains with clipping | PARTIAL / PAPER-UNDERSPECIFIED | Channel notation and clipping not operationally exact |
| Retinex | Sigmas 15/80/250, equal weights, epsilon 1e-6, percentile normalization | PARTIAL / PAPER-UNDERSPECIFIED | Paper lacks scales, weights, epsilon, normalization |
| Crop/resize after preprocessing | Cached images not cropped/resized; model transform resizes to 384x384 | DEVIATION | Paper does not specify operational dimensions |
| Augmentation | Horizontal flip, vertical flip, rotation 15 degrees | APPROXIMATION | Paper mentions transformations but does not enumerate them reproducibly |
| EG-VAN architecture | Not implemented | PLANNED | Current work is a baseline |
| SCGA/NLB/MFF/fusion | Not implemented | PLANNED | Not in current source |
| External validation | PH2 script and audit prepared; no inference | BLOCKED | CUDA runtime needed |

## 12. Research Results So Far

| Experiment | Dataset split | Model | Accuracy | Macro F1 | Status |
|---|---|---|---:|---:|---|
| EfficientNetV2S naive baseline | Image-level naive | EfficientNetV2S | 0.8838133068520357 | 0.8073501563707641 | COMPLETED |
| EfficientNetV2S leakage-aware baseline | Lesion-level leakage-aware | EfficientNetV2S | 0.8412228796844181 | 0.7114084504160505 | COMPLETED |

These results mean the baseline scored lower under lesion-level separation than under image-level splitting in the completed run pair. They do not prove that leakage caused the entire difference, do not estimate statistical stability, do not evaluate PH2, and do not evaluate full EG-VAN.

## 13. Data Leakage Analysis

Lesion-level leakage means images belonging to the same lesion ID appear in more than one split partition. If a model sees one image of a lesion during training and a related image during testing, test performance can be easier than a fully independent lesion-level evaluation.

Naive split statistics:

- Train 8,010, validation 998, test 1,007.
- 764 lesions cross at least two partitions.
- 380 lesions cross train/test according to the Phase 4A report.

Leakage-aware split statistics:

- Train 8,015, validation 986, test 1,014.
- 0 cross-partition lesion IDs.

The leakage-aware experiment is important because it evaluates on lesions not represented in training. The observed lower baseline score is descriptive evidence of a protocol difference, not a proof of a specific causal mechanism.

## 14. Frozen / Protected Artifacts

Do not casually modify:

- `data/raw/` HAM10000 metadata and images.
- `data/processed/images/` cached preprocessed outputs.
- `data/processed/metadata_clean.csv`.
- `data/splits/split_naive.csv`.
- `data/splits/split_leakage_aware.csv`.
- `experiments/efficientnetv2s_naive/`.
- `experiments/efficientnetv2s_leakage_aware/`.
- `checkpoints/efficientnetv2s_leakage_aware_best.pt`.
- Reported baseline metrics and confusion matrices.

These artifacts anchor reproducibility and research claims. Changing them would invalidate the documentation unless the change is explicitly approved, rerun, and logged.

## 15. Current Project Status

| Component | Status | Evidence |
|---|---|---|
| Dataset verification | COMPLETED | `docs/PHASE_4A_DATASET_REPORT.md` |
| Preprocessing | COMPLETED | `docs/PHASE_4B_PREPROCESSING_REPORT.md`, `data/processed/preprocessing_log.csv` |
| Split foundation | COMPLETED | `data/splits/*.csv`, split reports |
| GPU smoke test | COMPLETED | `docs/PHASE_4C_GPU_SMOKE_TEST_REPORT.md` |
| Naive baseline | COMPLETED | `experiments/efficientnetv2s_naive/test_metrics.json` |
| Leakage-aware baseline | COMPLETED | `experiments/efficientnetv2s_leakage_aware/test_metrics.json` |
| PH2 preparation | PARTIALLY COMPLETED | `data/external/ph2/metadata/ph2_manifest.csv`, `src/external_eval.py` |
| PH2 inference | BLOCKED | No `experiments/ph2_external_eval/` metrics |
| Image quality assessment | COMPLETED | `experiments/image_quality/`, `docs/PHASE_6_IMAGE_QUALITY_ASSESSMENT.md`; quality/error relationship explored descriptively in Phase 7 |
| Uncertainty estimation | COMPLETED | `experiments/uncertainty/`, `docs/PHASE_7_UNCERTAINTY_ESTIMATION.md` |
| Full EG-VAN | PLANNED | No SCGA/NLB/MFF/full architecture source found |

## 16. What We Can Honestly Tell the Guide

The project has verified HAM10000 data, deterministic preprocessing, frozen naive and lesion-level splits, and completed EfficientNetV2S baseline experiments. The naive baseline reached 88.38% accuracy and 80.74% macro-F1. The leakage-aware baseline reached 84.12% accuracy and 71.14% macro-F1. This is a baseline comparison, not a full EG-VAN reproduction.

PH2 external validation is prepared but not run. Image-quality proxies and post-hoc uncertainty analysis are implemented; neither is a clinical validation claim. The full EG-VAN architecture remains unimplemented.

## 17. Next Steps

Immediate:

1. Resolve PH2 provenance/access and runtime conditions before deciding whether a PH2 external evaluation is appropriate.
2. Consider repeated-run stability or statistical analysis only under a separately approved protocol.
3. Plan full EG-VAN reconstruction only after choosing and documenting the next research gate.

After PH2 access/runtime:

1. Confirm secondary-mirror limitations in any report.
2. Consider official PH2 provenance if access becomes available.

Image quality:

1. Define objective quality metrics.
2. Implement without changing frozen baseline results.

Uncertainty:

1. Choose an uncertainty method.
2. Keep it inference-only unless a new experiment is approved.

Full EG-VAN:

1. Implement paper modules separately from the baseline.
2. Compare only after architecture and training protocol are auditable.

Final research comparison:

1. Repeat key runs if stability is required.
2. Report baseline, full model, external validation, and limitations separately.

## 18. Glossary

HAM10000: A dermoscopic image dataset used for 7-class skin lesion classification.

Lesion: A lesion ID can have multiple images; it is the grouping unit used to prevent leakage.

Image-level split: A split that assigns individual images independently.

Lesion-level split: A split that keeps all images for one lesion in one partition.

Data leakage: Evaluation contamination where related information appears in both training and validation/test data.

EfficientNetV2S: The torchvision CNN backbone used for the current baseline.

Transfer learning: Starting from ImageNet-pretrained weights instead of random weights.

Pretrained weights: Learned parameters from a prior dataset, here ImageNet.

Macro F1: Average F1 score across classes, treating classes equally.

Recall: True positives divided by actual class samples.

Confusion matrix: Table of true class versus predicted class counts.

External validation: Evaluating a frozen model on a different dataset.

PH2: External dermoscopic dataset with common nevus, atypical nevus, and melanoma categories.

Uncertainty estimation: Methods that quantify prediction confidence or ambiguity.

Grad-CAM: A visualization method for class-discriminative image regions; not implemented here.

Focal loss: Loss that down-weights easy examples using gamma and alpha terms.

SCGA: Paper-specific attention component; not implemented here.

NLB: Paper-specific non-local block; not implemented here.

MFF: Paper-specific multi-feature fusion; not implemented here.

## 19. If I Open This Project After 6 Months

This is an EG-VAN-related skin lesion classification project, but the working code currently represents a verified HAM10000 preprocessing and EfficientNetV2S baseline pipeline, not the full EG-VAN architecture.

The HAM10000 dataset is under `data/raw/`; processed images are under `data/processed/images/`. Use `data/splits/split_leakage_aware.csv` for the main honest split. The current working model is `src/models/baseline_effnet.py`, trained through `src/run_baseline.py`.

The last completed experiment is the naive versus leakage-aware EfficientNetV2S baseline comparison. Checkpoints and metrics are in `experiments/efficientnetv2s_naive/` and `experiments/efficientnetv2s_leakage_aware/`. The leakage-aware best checkpoint is also copied to `checkpoints/efficientnetv2s_leakage_aware_best.pt`.

PH2 external validation is blocked before inference. Do not modify processed images, split files, labels, metadata, checkpoints, or result JSONs unless you are deliberately starting a new approved experiment. The next technical step is to run `src/external_eval.py` in a CUDA-enabled Colab runtime and document only the metrics that are actually produced.
