# Experiment #5 Domain-Shift Analysis Plan

**Status:** Plan only. No new measurements, inference, Grad-CAM, training, or model changes are performed in this document-preparation step.

## Research question

Characterize which observable image-domain, confidence, and error-pattern differences accompany the lower Experiment #5 performance on the PH² follow-up cohort than on the HAM10000 internal test. Evaluate descriptive associations only; do not claim that any one feature caused the performance gap.

The primary image comparison will contrast the frozen HAM10000 model-input domain with PH² after the already approved HAM-matched processing. Raw-image comparisons are secondary diagnostics that describe acquisition/preprocessing differences; they are not substitutes for the model-input comparison.

## Datasets being compared

### HAM10000

- Use the frozen leakage-aware split and saved Experiment #5 predictions, especially `experiments/efficientnetv2s_controlled_exp5/test_predictions.csv` and `validation_predictions.csv`.
- The internal test includes seven HAM classes (1,014 total: 676 `nv`, 107 `mel`, and the other five classes). For direct comparison with PH², also derive a separate overlap-only internal-test cohort restricted to true `nv` and `mel`, preserving all other predicted HAM classes as `other`.
- For image measurements, the saved Phase 6 `experiments/image_quality/image_quality.csv` describes all 10,015 Phase 4B processed HAM JPGs. Raw HAM files remain under `data/raw/HAM10000_images_part_1/` and `data/raw/HAM10000_images_part_2/`; frozen processed files are under `data/processed/images/`.

### PH²

- Use the existing 200-case metadata and frozen `data/external/ph2/metadata/ph2_manifest.csv`; primary class-comparable analysis uses only the direct-overlap cohort: common nevus → `nv` (80), melanoma → `mel` (40), total 120. Exclude 80 atypical nevi from model-performance and class-conditional comparisons.
- Original dermoscopic BMPs are in `data/external/ph2/images/`; preserved package originals and masks are under `data/external/ph2/original/`.
- Use existing saved probabilities from `analysis/ph2_external_validation_exp5/ph2_predictions.csv`. Do not run prediction inference for the planned saved-output comparisons.

## Frozen checkpoint provenance

- Model: Controlled Experiment #5, selected epoch 15 using `validation_threshold_constrained_min_loss`.
- Checkpoint: `experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt`.
- Checkpoint SHA256: `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`.
- Frozen HAM split SHA256: `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`.
- Class order: `akiec, bcc, bkl, df, mel, nv, vasc`.
- Grad-CAM target, if separately approved later: `model.features[-1]`.

**PH² is treated as an external follow-up dataset and is not used for model selection, hyperparameter tuning, threshold optimization, or training.** PH² was previously used for an earlier model evaluation and a preprocessing diagnostic, so findings remain follow-up evidence, not an untouched independent validation result. The Experiment #5 checkpoint is frozen and will not be changed based on these analyses.

## Available evidence

### Preprocessing and source images

- `docs/PHASE_4B_PREPROCESSING_REPORT.md` and `src/preprocessing.py` document deterministic Phase 4B processing: hair removal → Gray World → multi-scale Retinex. Constants include hair kernel 17, threshold 10, Telea radius 1.0, Retinex sigmas 15/80/250, equal scale weights, and per-channel 1st–99th percentile normalization.
- `data/processed/preprocessing_log.csv` covers the 10,015 processed HAM image outputs; the Phase 4B report records 10,015 successes and preserved 450×600 output dimensions.
- `experiments/ph2_preprocessing_diagnostic/diagnostic_config.json` and `run_diagnostic.py` define the paired PH² raw and HAM-matched paths. The matched path reuses `src/preprocessing.py`, then performs in-memory JPEG encode/decode, RGB conversion, resize to 384×384, tensor conversion, and ImageNet normalization. It uses no augmentation, masks, or ROI crops.
- The PH² diagnostic previously found 48/120 predictions changed between raw and HAM-matched inputs for the **different leakage-aware epoch-6 checkpoint**. Reported accuracy changed from 0.5417 to 0.7417, melanoma sensitivity from 0.000 to 0.475, and AUROC from 0.6606 to 0.6325. These results establish sensitivity to preprocessing for that checkpoint and cohort; they do not establish a causal preprocessing effect for Experiment #5.

### Image quality

- `experiments/image_quality/image_quality.csv` has 10,015 processed-HAM rows and the columns brightness, contrast, sharpness, saturation, dark-pixel ratio, bright-pixel ratio, entropy, and illumination variation.
- The corresponding split/class statistics and plots are in `experiments/image_quality/quality_stats_by_*.csv`, `quality_distributions_by_*.png`, `quality_boxplots_by_split.png`, and related quality plots.
- `src/image_quality.py:compute_image_quality` defines deterministic features. Brightness/contrast use grayscale mean/std; sharpness is Laplacian variance; saturation is normalized HSV saturation; dark/bright ratios use fixed grayscale thresholds 30/225; entropy is grayscale histogram Shannon entropy; illumination variation is the coefficient of variation of a 16×16 grid of local grayscale means.
- Phase 6 quality/error associations in `experiments/uncertainty/quality_uncertainty.csv` and `experiments/uncertainty/uncertainty_summary.json` use a different leakage-aware epoch-6 checkpoint (`f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848`). They are historical context, not Experiment #5 confidence or error evidence.

### Experiment #5 predictions and explainability

- `experiments/efficientnetv2s_controlled_exp5/test_predictions.csv`, `test_metrics.json`, `validation_predictions.csv`, and `validation_metrics.json` contain the selected-checkpoint's saved predictions and probabilities for the frozen HAM partitions.
- `analysis/explainability_exp5/case_manifest.csv`, `qualitative_analysis.csv`, `explainability_config.json`, and `explainability_report.md` document the 24 preapproved validation cases, saved/reproduced probabilities, and visual Grad-CAM review.
- The existing HAM Grad-CAM maps were generated from processed HAM images on the frozen checkpoint at `model.features[-1]`. They are a small, selected validation sample; they are not a representative domain-shift sample.

### PH² predictions, metadata, and provenance

- `analysis/ph2_external_validation_exp5/ph2_manifest.csv`, `ph2_predictions.csv`, `ph2_metrics.json`, `confusion_matrix.csv`, `external_validation_config.json`, and `ph2_exp5_forensic_analysis.md` contain the Experiment #5 follow-up cohort, full seven-class probabilities, metrics, and audit.
- `experiments/ph2_preprocessing_diagnostic/preprocessing_comparison.csv`, `summary_metrics.json`, `vasc_case_analysis.csv`, and both raw/HAM-preprocessed confusion matrices contain paired historical checkpoint outputs.
- `data/external/ph2/metadata/PH2_dataset.txt`, `ph2_manifest.csv`, `provenance.json`, and `verification_report.json` document diagnosis mapping, local ID/label integrity, and provenance scope. The mapped PH² cohort is 80 common nevus and 40 melanoma; atypical nevus is excluded. The package audit notes no patient-linkage table, original Kaggle archive, or acquisition date.

## Proposed HAM-versus-PH² comparisons

### 1. Image geometry and representation

- Record native width, height, channel count, aspect ratio, file format, and bit depth for raw HAM originals and PH² dermoscopic BMPs.
- Separately record the dimensions/aspect ratio of Phase 4B processed HAM JPGs and the PH² HAM-matched representation.
- For the model-input comparison, measure both domains after the exact model transform to 384×384. The fixed square resize removes the visible aspect-ratio difference in tensor dimensions but may distort source geometry; retain native aspect ratio as a separate covariate rather than interpreting the resized dimensions as native properties.

### 2. Brightness, contrast, color, sharpness, and illumination

- Reuse `src/image_quality.py:compute_image_quality` on the 384×384 RGB model-input arrays for the primary matched-resolution comparison.
- Compute the same feature set on raw images as a separate acquisition-domain comparison, clearly labeled raw.
- Report RGB channel means/stds and fixed-bin per-channel histograms, plus HSV hue/saturation summaries. Use the same RGB decoding/channel convention across domains.
- Compare brightness, contrast, Laplacian sharpness, saturation, dark/bright-pixel ratios, entropy, and local illumination variation. Interpret Laplacian sharpness as resolution- and resampling-dependent, not a validated clinical blur score.
- Within PH², compute paired raw-versus-HAM-matched feature differences on the same included image IDs. This is a preprocessing-description analysis, not an Experiment #5 prediction counterfactual.

### 3. Distribution summaries and plots

- Show per-feature histograms, ECDFs, and box/violin plots for processed HAM test inputs versus PH² HAM-matched inputs.
- Stratify primary comparisons by true class (`nv` and `mel`); also provide pooled summaries standardized to a common class weighting so differing class proportions do not drive the pooled contrast.
- Show raw-to-matched PH² paired-difference plots. Separate them visually from the primary processed-HAM versus PH²-matched comparison.
- Report means, medians, quartiles, standardized mean differences, and Cliff’s delta with bootstrap confidence intervals. If formal tests are included, pre-specify them and control the false-discovery rate across image features; do not treat small p-values as evidence of causation.

### 4. Saved prediction, confidence, and confusion comparisons

- From saved Exp5 probabilities calculate maximum softmax confidence, predictive entropy, top-1/top-2 margin, and `p_mel` for internal HAM test and PH² follow-up.
- Compare these quantities by domain and true class; separately describe correct versus incorrect examples and the six `nv`/`mel` outcome groups. Use histograms, ECDFs, and boxplots; do not choose thresholds from these plots.
- Recompute an internal-test overlap-only `nv`/`mel` 2×3 matrix (`nv`, `mel`, `other`) from the saved seven-class predictions and compare it with the PH² 2×3 matrix. Also show the published full seven-class internal test matrix alongside the PH² class distribution, explicitly noting the differing label support.
- Compare class-conditional `p_mel` distributions for true melanoma and true nevus, as well as predicted labels. Show AUROC with confidence intervals as a ranking metric; keep argmax sensitivity separate. AUROC does not imply an operational threshold or adequate sensitivity.
- Do not compare raw seven-class macro-F1 with PH² as though all seven classes were represented. If reporting macro-F1, state that HAM is seven-class while the PH² macro metric covers only `nv` and `mel`.

### 5. Statistical and sampling limitations

- The PH² follow-up contains only 40 melanoma and 80 common-nevus cases. Give denominator counts and uncertainty intervals for class-specific metrics; avoid broad population claims.
- HAM may contain multiple images per lesion; resample at the frozen `lesion_id` cluster for HAM bootstrap intervals rather than treating all images as independent. PH² metadata has one case/image identifier but no patient linkage table; use case-level resampling and state this limitation.
- Feature comparisons are observational and may be confounded by class composition, acquisition device/site, resolution, framing, and preprocessing. Class-stratify and standardize, but do not claim this removes unmeasured confounding.
- This is a follow-up analysis after prior PH² exposure, not a fresh independent test. Do not use the results for threshold, preprocessing, or model selection.

## Explainability comparison plan

### Comparability

Comparable, qualitative Grad-CAM is feasible with the exact same frozen Experiment #5 checkpoint and target layer `model.features[-1]`. Existing HAM maps use processed HAM JPGs with the evaluation transform. Proposed PH² maps must use the existing approved HAM-matched PH² preprocessing, then the same 384×384 resize/tensor/ImageNet normalization. Use `model.eval()` and enable gradients specifically for Grad-CAM; do not wrap CAM generation in `torch.inference_mode()` or `torch.no_grad()`.

The planned maps compare spatial emphasis, not absolute activation magnitude: each CAM is normalized independently, the feature layer is spatially coarse, and PH² versus HAM lesions/capture geometry may differ. Use the same overlay rendering and label panels; do not interpret Grad-CAM as causal evidence.

### Deterministic candidate PH² cases

Select two lexicographically lowest `image_id` values within each saved prediction outcome group. This fixed rule uses no probability ranking, visual inspection, or map output for selection. Candidate IDs from current saved PH² predictions:

| Outcome group | Candidate IDs | Current saved outcomes |
|---|---|---|
| Correctly detected melanoma (`mel` → `mel`) | `IMD065`, `IMD168` | 2 cases available |
| Melanoma → nevus | `IMD061`, `IMD063` | 15 total available |
| Melanoma → other | `IMD058`, `IMD085` | Both are currently predicted `bkl`; 13 total available |
| Correctly classified nevus (`nv` → `nv`) | `IMD003`, `IMD009` | 2 cases selected |
| Nevus → melanoma | `IMD035`, `IMD045` | 8 total available |
| Nevus → other | `IMD010`, `IMD020` | Both are currently predicted `bkl`; 9 total available |

These 12 cases are a deterministic review sample, not a representative sample. For each case, create predicted-class and true-class CAMs; when target labels are equal, compute once and record/reuse the map. Compare with the existing HAM validation Grad-CAM only for matching outcome types that are actually represented in its 24-case manifest. PH² `mel`→other and `nv`→other groups do not have guaranteed exact counterparts in that selected HAM map set; report these PH²-only groups separately rather than inventing matched pairs.

If later approved, optional PH² lesion-mask overlap can be reported as an auxiliary localization descriptor only, after verifying mask/image alignment. Masks must never enter the model input, crop, preprocessing, case selection, or prediction path. The existing HAM Grad-CAM set has no paired lesion-mask annotations in this analysis.

## Leakage safeguards

- Do not train, fine-tune, change weights, modify Experiment #5 artifacts, or create Experiment #6.
- Do not rerun PH² classification inference for this analysis plan. Use the saved Exp5 probabilities and the saved internal test probabilities.
- Do not use PH² labels, scores, or CAMs to select checkpoints, tune preprocessing, adjust thresholds, or revise the model.
- Keep the exact Experiment #5 checkpoint and split hashes in every later result/config; recheck them before and after any separately approved computation.
- Keep the frozen 120-case direct-overlap manifest. Do not add atypical nevi or change cohort membership after viewing results.
- Keep raw PH², HAM-matched PH², and processed HAM comparisons in clearly separated conditions; do not treat the old epoch-6 diagnostic predictions as Experiment #5 predictions.
- No masks/ROI crops, segmentation, color normalization beyond the frozen preprocessing, hair-removal changes, augmentation, or test-time augmentation.
- Preserve the external follow-up limitation statement in all outputs.

## Exact execution steps

1. **Read-only preflight:** inventory source tables; verify the Experiment #5 checkpoint and split hashes, PH² manifest count/mapping, prediction IDs, class order, and output-path nonexistence. Do not load the model or run a forward pass.
2. **Saved-output analysis:** independently recompute Exp5 PH² and internal-test confidence/error summaries from existing CSV probabilities; derive the internal-test `nv`/`mel` overlap-only comparison without changing the seven-class predictions.
3. **Image-domain feature extraction:** run only deterministic feature measurements, not model inference. Measure native geometry and the frozen quality proxies for raw HAM/PH²; measure matched 384×384 model-input proxies for processed HAM and HAM-matched PH². Perform paired raw-to-matched PH² summaries by image ID. Record exact source, transform, sizes, and feature code version.
4. **Statistics and plots:** generate predeclared histograms, ECDFs, and boxplots; stratify/standardize by `nv` and `mel`; report effect sizes and cluster-aware/case-level uncertainty intervals with the limitations above. No thresholds or model settings are chosen from results.
5. **Explainability, only after separate approval:** use the 12 fixed PH² IDs above, the frozen Experiment #5 checkpoint, HAM-matched transform, and `model.features[-1]`; save predicted/true maps and overlays in an isolated domain-shift analysis subdirectory. Generate no maps during the current plan step.
6. **Synthesis:** describe which image and prediction differences co-occur, list plausible domain-shift contributors, and distinguish observed associations from causal explanations. Maintain the PH² follow-up terminology and conclude without changing Experiment #5.
7. **Final audit:** verify no checkpoint/split changes and no modification of prior PH² diagnostic/evaluation artifacts; report exactly which saved outputs or new analysis artifacts were produced.

## Current-step execution boundary

This file is a design plan only. No quality features were recomputed, no statistical tests or plots were generated, no inference or Grad-CAM was run, no PH² prediction was modified, and no model or split was changed.