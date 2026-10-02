# Stage 7 — final research gap and evidence audit

## Decision

**REQUIRES_CRITICAL_EXPERIMENT** for a paper claiming to extend or externally validate the *full EG-VAN architecture*. The frozen Experiment #5 is a plain EfficientNetV2S classifier with a replaced seven-class head. The modified ResNet50, SCGA, Non-Local Block, MFF, and dual-branch fusion are absent. Therefore the current PH² findings cannot be attributed to the original EG-VAN architecture. A narrower paper about the paper-informed EfficientNetV2S baseline can be drafted now if all full-architecture claims are removed and the title, abstract and methods name the evaluated model precisely.

## Original paper and project scope

The original paper ([Saeed et al., IEEE Access 2025](https://doi.org/10.1109/ACCESS.2025.3561240)) proposes a dual-path EfficientNetV2S/modified-ResNet50 architecture with attention and fusion, plus color balancing. Its reported nine-class results are not directly comparable to this project's seven-class HAM test or mapped NV/MEL PH² follow-up. The repository explicitly documents incomplete architecture reconstruction in `docs/PROJECT_STATUS.md`; `src/models/baseline_effnet.py` replaces only the torchvision EfficientNetV2S classifier. The current study's real addition is a well-provenanced external-validity and reliability audit **of that baseline**.

## Repository inventory and provenance

Frozen checkpoint `experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt`: `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`. Frozen leakage-aware split `data/splits/split_leakage_aware.csv`: `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. Experiment #5 config, 25-epoch training history, selected epoch 15, validation predictions/metrics, and HAM test predictions/metrics are present. HAM test has 1,014 images with seven-class accuracy 0.8333. PH² manifest and predictions include 120 mapped cases (80 NV, 40 MEL); PH² accuracy is 0.6250, balanced accuracy 0.54375, and melanoma recall 0.30. The original PH² manifest includes 80 excluded atypical nevi. PH² was previously used in diagnostics/earlier evaluation, so this is follow-up external evidence, not a pristine untouched external set.

Image quality exists for HAM in `experiments/image_quality/` and for the NV/MEL cross-domain cohorts in `analysis/domain_shift_exp5/`. Grad-CAM manifests, frozen maps, reviewed 12-case table, and human-review summary exist. Stage 5 uncertainty/calibration and Stage 6 case synthesis exist; Stage 6 source hashes and joined counts were checked. Split/data provenance is documented in Phase 4A and PH² provenance documents. Preprocessing implementation and numerical deviations are recorded in `src/preprocessing.py` and `docs/DEVIATIONS.md`. The loss, augmentation and weighted sampler are in `src/train.py` and Experiment #5 config. There are baseline and controlled EfficientNetV2S experiments, but no full architecture or component ablations. A preprocessing timing log exists, but no controlled Experiment #5 parameter/FLOP/latency benchmark was found. `docs/PROJECT_STATUS.md` predates later PH² and Stage 6 results; its statements about those later stages are stale, while its architecture-incomplete statement is consistent with the inspected model code.

## Paper-to-project crosswalk

The full row-level audit is in `original_paper_project_crosswalk.csv`. The distinction that controls interpretation is that EfficientNetV2S is present as a standalone baseline; the four defining ResNet/fusion components are absent. Hair removal and color balancing are paper-informed approximations with documented choices. Internal metrics, PH² external metrics, descriptive domain comparisons, uncertainty/calibration, and reviewed Grad-CAM are supported for the baseline.

## Evidence chain and claim boundaries

`evidence_chain_audit.csv` marks each link. Frozen checkpoint → HAM evaluation → PH² evaluation → observed lower PH² accuracy is supported by saved artifacts. Domain/image-quality differences and changed uncertainty metrics coexist with degradation, but no causal intervention isolates their effects. Eleven PH² errors occupy the saved HIGH/VERY_HIGH confidence bands. The 12 fixed Grad-CAM cases are deliberately stratified; their reviews cannot characterize all PH² errors or quantify population attention prevalence. `claim_evidence_matrix.csv` gives permitted wording for each central claim.

## Sample sizes and legitimate conclusions

HAM test n=1,014 supports held-out seven-class performance for this split. PH² n=120, with 80 NV and 40 MEL cases, supports descriptive follow-up results on those mapped classes. There are 45 PH² errors and 11 HIGH/VERY_HIGH errors, which support explicit counts and case inspection. Twelve fixed Grad-CAM cases support selected-case, human-reviewed observations only. Cross-domain accuracy, calibration and descriptor differences are descriptive; attention and image-quality explanations remain exploratory and hypothesis-generating. No current artifact supports broad clinical generalization, causation, or claims about the full original architecture.

## Missing experiments

The only critical experiment **for full EG-VAN external-validity claims** is to implement and train the actual dual-branch architecture with the documented frozen HAM split, then evaluate its frozen checkpoint internally and on the mapped PH² cohort under a prespecified protocol. Saved EfficientNetV2S predictions are insufficient; this would require training and new inference, which Stage 7 does not perform. Component ablations, confidence intervals, additional untouched external cohorts and controlled efficiency measurements are useful but optional for a narrowly descriptive baseline paper. Do not claim component gains or computational efficiency without their respective measurements.

## Paper novelty statement

Original EG-VAN contributes the proposed dual-branch attention/fusion architecture and color balancing. This repository contributes a reproducible, paper-informed **single-branch EfficientNetV2S baseline** and a linked external follow-up reliability audit: HAM/PH² performance, image/domain descriptors, uncertainty and calibration, high-confidence failures, and fixed human-reviewed Grad-CAM cases. This does not establish how the original dual-branch model would perform externally.

## Reproducibility and audit limits

All Stage 7 work used saved artifacts and read-only inspection. No model training, inference, prediction generation, Grad-CAM regeneration, or calibration fitting occurred. The source SHA256 inventory and generated-file hashes appear in `stage7_experiment_manifest.json`. The manifest excludes its own hash to avoid recursion. The original paper was consulted via its DOI; repository evidence determines implementation status.
