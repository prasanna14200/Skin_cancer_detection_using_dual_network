# EG-VAN PROJECT STATUS

Last updated: 2026-09-26

## 1. Research Objective

The source paper proposes EG-VAN: a dual-branch skin-lesion classifier combining EfficientNetV2-S and a modified ResNet50 with SCGA attention, a Non-Local Block, Multi-Scale Feature Fusion, color balancing, augmentation, focal loss, and Grad-CAM analysis. This repository is a documented reconstruction/extension, not the authors' original implementation. The full EG-VAN architecture has not yet been implemented.

## 2. Dataset

- Dataset: HAM10000.
- Metadata rows and processed images verified: 10,015.
- Classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.
- Unique lesion IDs: 7,470.
- Phase 4B processed images preserve image IDs and are stored at `data/processed/images/`.
- H-MNIST CSV files are unused.
- Frozen splits remain unchanged: image-level naive split and `lesion_id`-grouped leakage-aware split.

## 3. Completed Phases

| Phase/component | Purpose | Implementation | Evidence/artifacts | Status |
|---|---|---|---|---|
| Dataset verification | Verify HAM10000 metadata/images/classes | Metadata validation and correspondence checks | `docs/PHASE_4A_DATASET_REPORT.md`, `data/processed/metadata_clean.csv` | COMPLETE |
| Preprocessing | Paper-informed hair removal, Gray World, Retinex | Deterministic cached processing | `src/preprocessing.py`, `data/processed/preprocessing_log.csv`, `docs/PHASE_4B_PREPROCESSING_REPORT.md` | COMPLETE |
| Naive split | Image-level stratified 80/10/10 | Frozen CSV | `data/splits/split_naive.csv` | COMPLETE |
| Leakage-aware split | Lesion-level 80/10/10 | Frozen CSV, zero lesion crossings | `data/splits/split_leakage_aware.csv`, split reports | COMPLETE |
| GPU smoke test | Verify EfficientNetV2S GPU mechanics | Colab Tesla T4 smoke test | `docs/PHASE_4C_GPU_SMOKE_TEST_REPORT.md` | COMPLETE |
| EfficientNetV2S naive baseline | Baseline on image-level split | ImageNet pretrained, seven-class head | `experiments/efficientnetv2s_naive/` | COMPLETE |
| EfficientNetV2S leakage-aware baseline | Baseline on lesion-level split | Same frozen baseline configuration | `experiments/efficientnetv2s_leakage_aware/` | COMPLETE |
| Stage A comparison | Describe split protocol difference | Accuracy, macro-F1, recalls, confusion matrices | `docs/PHASE_4C_STAGE_A_REPORT.md` | COMPLETE, descriptive |
| PH² provenance/access audit | Assess official source, secondary mirror, labels, checkpoint, and readiness | Mirror structure/labels internally consistent; source chain/license unverified; evaluator synthetic tests/dry-run pass | `docs/PHASE_8_PH2_PROVENANCE_AUDIT.md` | BLOCKED — DATA PROVENANCE INSUFFICIENT |
| Image Quality Assessment | Measure deterministic quality proxies and distributions | Eight proxies over all frozen rows | `experiments/image_quality/`, `docs/PHASE_6_IMAGE_QUALITY_ASSESSMENT.md` | COMPLETE |
| Uncertainty Estimation | Post-hoc confidence, entropy, calibration, selective prediction | Frozen leakage-aware checkpoint inference in Colab; no training | `experiments/uncertainty/`, `docs/PHASE_7_UNCERTAINTY_ESTIMATION.md` | COMPLETE |
| PH² external validation | Evaluate direct-overlap classes externally | No inference result; provenance/access unresolved; evaluator prepared | `docs/PHASE_8_PH2_PROVENANCE_AUDIT.md` | BLOCKED |
| Full EG-VAN reconstruction | Implement SCGA, NLB, MFF, dual branch and fusion | Not implemented | No full architecture modules in `src/` | NOT STARTED |
| Repeated-run stability | Quantify baseline/analysis run variability | No repeated runs performed | No repeat-run artifacts | NOT STARTED |
| Final integrated analysis | Integrate baseline, quality, uncertainty and external findings | Not produced; PH² remains blocked and architecture incomplete | Phase reports are separate | NOT STARTED |

## 4. Important Actual Baseline Results

| Protocol | Accuracy | Macro-F1 |
|---|---:|---:|
| Naive image-level | 0.8838133068520357 | 0.8073501563707641 |
| Leakage-aware lesion-level | 0.8412228796844181 | 0.7114084504160505 |

The descriptive naive-minus-leakage-aware differences are 4.259 percentage points in accuracy and 9.594 percentage points in macro-F1. This comparison does not establish causality, statistical stability, or a result for the complete EG-VAN architecture.

## 5. Phase 6 Result

10,015 leakage-aware rows were analyzed using brightness, contrast, Laplacian sharpness, saturation, dark/bright pixel ratios, entropy, and illumination variation. No samples were filtered or modified. Phase 7 later joined the saved test predictions to quality values and produced equal-count quality tercile summaries. These are computational proxies, not clinical quality labels; associations do not imply causality.

## 6. Phase 7 Result

The frozen leakage-aware test set produced 1,014 predictions, aligned exactly to the frozen test IDs.

- Accuracy: 0.7988165680473372
- Macro-F1: 0.6201368398477792
- Mean confidence: 0.7755111562312237
- Median confidence: 0.8195773065090179
- Mean predictive entropy: 0.5802945969745708
- Median predictive entropy: 0.5426918864250183
- Ten-bin ECE (corrected): 0.05242056577398463
- Correct / incorrect: 810 / 204
- Mean confidence correct / incorrect: 0.8284780717190401 / 0.5652013447354821
- Mean entropy correct / incorrect: 0.47187217525838887 / 1.0107953890829402

Selective prediction showed increasing accuracy at higher confidence thresholds with decreasing coverage; this is a descriptive trade-off, not a clinical safety claim. Corrected results and artifact audit are in `docs/PHASE_7_UNCERTAINTY_ESTIMATION.md`.

## 7. PH² Status

PH² remains a separate external-validation component and is not complete. The official University of Porto ADDI download workflow was problematic. A locally audited secondary Kaggle mirror exists, but it is not represented as an official University package, and its source URL/license provenance is incomplete. No PH² model inference metrics exist. Do not claim official PH² validation or substitute the mirror while calling it official.

## 8. Current Research Contribution

Verified implemented components beyond a single baseline include lesion-aware split construction and audit, descriptive naive-versus-lesion-level baseline comparison, deterministic image-quality proxy analysis, and post-hoc softmax confidence/entropy/ECE/selective-prediction analysis. PH² evaluation and full EG-VAN modules are not implemented as completed experiments.

The uncertainty method is softmax-based predictive uncertainty. It is not Bayesian uncertainty or a clinically validated uncertainty system.

Checkpoint comparability caveat: Phase 7 used `best_checkpoint.pt` (epoch 6). The Stage A training runner evaluates the final in-memory epoch (epoch 25) for `test_metrics.json` without reloading the best checkpoint. Thus Phase 7 test metrics describe the best checkpoint and are not directly comparable to the Stage A test metrics until a checkpoint-consistent evaluation is approved and run.

## 9. Remaining Work

**BLOCKED**

- PH² external inference: no completed evaluation result; Kaggle URL/uploader/revision/license and traceability to official PH² are unavailable. Evaluator dry-run passes, but provenance and CUDA gates prevent inference.
- A second CUDA run may be required for any approved analysis that cannot be reproduced locally; do not retrain baselines casually.

**PLANNED**

- Full EG-VAN reconstruction: SCGA, Non-Local Block, MFF, dual-branch integration, and paper-specific comparison.
- Repeated-run stability analysis and appropriate statistical analysis.
- Any additional external validation, only under the frozen protocol and after PH² decision.

**OPTIONAL / approval required**

- Ablations and efficiency benchmarking, which belong to later protocol stages.
- Calibration extensions beyond the Phase 7 ECE diagnostic.

## 10. Recommended Immediate Next Step

First obtain traceable PH² source/revision and license/access evidence or the official ADDI package. If that cannot be established, decide whether to omit PH² rather than describe the secondary mirror as official. The evaluator is prepared, but inference remains gated until provenance is VERIFIED and CUDA is available. Do not launch PH² inference automatically; full EG-VAN reconstruction remains a later, separate decision.
