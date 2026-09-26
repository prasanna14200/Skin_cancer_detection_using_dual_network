# EG-VAN Phase 5 PH² External Validation Report

Date: 2026-09-23
Status: BLOCKED before inference. Secondary Kaggle mirror audit completed.

## Dataset source

Official source verified:

- University of Porto, Faculty of Sciences, ADDI PH² Database:
  `https://www.fc.up.pt/addi/ph2%20database.html`

The official page exposes PH² sections for data description, terms of use, download, and support. It also exposes a restricted/login area. Official access was not available for the completed local audit, so the local PH2 files must be treated as a secondary mirror package rather than an official University download.

## Access and terms

The package is explicitly treated as a secondary Kaggle mirror, not as an official University download. The Kaggle URL/name was not included in the synchronized project files, so it remains unverified. The mirror's bundled `Readme.txt`, `PH2_dataset.txt`, and `PH2_dataset.xlsx` were preserved unchanged. No license permission is inferred from the mirror.

## Dataset version/date obtained

The mirror package was present locally on 2026-09-23. Original package files were copied unchanged to `data/external/ph2/original/`. The source package has no recorded Kaggle URL/name in the available files. SHA256 values were recorded for the bundled metadata/documentation files; no archive checksum was available.

## Pre-run availability audit

- PH² original dermoscopic images: 200 BMP files.
- PH² auxiliary masks: 250 BMP files; excluded from evaluation.
- PH² labels/metadata: 200 rows in `PH2_dataset.txt` and corresponding workbook present.
- Clinical diagnosis distribution: common nevus 80, atypical nevus 80, melanoma 40.
- Image-to-label correspondence: 200/200 matched; no unlisted originals.
- Image readability: 200/200 readable by OpenCV.
- Leakage-aware HAM10000 checkpoint: present locally.
- Naive HAM10000 checkpoint: absent from the current local workspace.
- CUDA runtime: unavailable in the current Windows execution context.

No PH² inference was started because the required CUDA/torchvision runtime is unavailable locally.

## Valid mapping policy

| PH² class | HAM10000 class | Decision |
|---|---|---|
| Common nevus | `nv` | Include after official metadata verification |
| Melanoma | `mel` | Include after official metadata verification |
| Atypical nevus | None | Exclude; never map to `bkl` |

Verified mapping counts: 80 common-nevus images map to `nv`; 40 melanoma images map to `mel`; 80 atypical-nevus images are excluded. Evaluated sample count would be 120 after the runtime blocker is removed.

## Model checkpoint

The intended checkpoint is the frozen HAM10000 leakage-aware model:

`experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt`

It is present locally. Static checkpoint audit passed: checkpoint configuration identifies EfficientNetV2S, the seven frozen class names, leakage-aware split, CUDA training, and classifier tensors shaped `(7, 1280)` / `(7,)`. Full model construction could not be executed because local torchvision is unavailable and local CUDA is absent.

## Evaluation policy

When unblocked, evaluation must be inference-only:

- no PH² training;
- no fine-tuning;
- no checkpoint selection using PH²;
- no calibration;
- no HAM10000 data modification;
- resize to 384x384 and ImageNet normalization only;
- report accuracy, macro-F1, per-class recall, and confusion matrix for the valid overlap.

## Artifacts not created

The following have not been generated because PH2 inference has not run in the required CUDA runtime:

- `experiments/ph2_external_eval/metrics.json`
- `experiments/ph2_external_eval/confusion_matrix.csv`
- `experiments/ph2_external_eval/predictions.csv`
- `experiments/ph2_external_eval/evaluation_config.json`

No metrics, predictions, or confusion matrix were created because inference did not run.

## Blocker and required manual action

Phase 8 later audited the secondary mirror and classified provenance as **PARTIALLY VERIFIED**: expected IDs, labels, and package structure are present, but the Kaggle URL/uploader/revision, traceability to the official release, and license/permission evidence are missing. The current evaluator also has pre-inference issues documented in `docs/PHASE_8_PH2_PROVENANCE_AUDIT.md`. PH² inference remains **BLOCKED — DATA PROVENANCE INSUFFICIENT** and must not be run until provenance/access is resolved and the evaluator is corrected.

No PH² result is reported, and no ISIC2019, calibration, ablation, or model-training work was started.
