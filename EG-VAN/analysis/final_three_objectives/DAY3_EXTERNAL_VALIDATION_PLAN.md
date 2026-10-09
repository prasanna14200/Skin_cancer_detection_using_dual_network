# Day 3 fixed Stage 23 external-domain follow-up plan

**Status: protocol prepared; no PH2 image, manifest row, or performance outcome accessed for this plan.** This is an exploratory **external follow-up**, not untouched external validation: PH2 was used in earlier project diagnostics and Stage 11/16/20 analyses. Stage 23 remains frozen at epoch 14, SHA256 `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`. No external result may change the checkpoint, preprocessing, class order, calibration, or selection decision.

## Available infrastructure and fixed cohort

Read-only path checks show `data/external/ph2/original/`, `images/`, and `metadata/`, the existing `analysis/ph2_external_validation_exp5/ph2_manifest.csv`, and `experiments/stage16_final_evaluation/evaluate_ph2.py` are present. The existing [Stage 16 pre-evaluation protocol](../stage16_final_evaluation/pre_evaluation_audit.md) documents 200 PH2 cases: 80 common nevi mapped to `nv`, 40 melanoma mapped to `mel`, and 80 atypical nevi excluded. The 120 included cases form the fixed mapped cohort. These counts are inherited protocol descriptions; they must be reverified from the manifest and images before Day 3 inference. Predictions into the other five HAM classes stay `other` and count as errors. Do not change this mapping based on Stage 23 output.

## Provenance and overlap checks before inference

1. Hash the Stage 23 checkpoint, frozen registry/manifest, PH2 cohort manifest, preprocessing code, and each included source image. Reject missing, unreadable, duplicate, or label-inconsistent records; do not silently drop cases.
2. Compare exact source-image hashes with available HAM raw image hashes across **all** HAM partitions without reading HAM test *outcomes*. Compare normalized file names/IDs and available lesion/patient identifiers. Because HAM and PH2 identifiers differ and patient-level linkage may be unavailable, an exact-hash negative result cannot prove patient independence. Document unresolved overlap risk.
3. Record PH2 acquisition/source and mapping provenance from existing repository documentation. Do not use observed model errors to revise inclusion, mapping, or preprocessing.

## Frozen inference and outcomes

Use `models/frozen_stage23/load_frozen.py` with strict hash check and eval mode. Follow the already documented PH2 original-BMP path: `src/preprocessing.py:preprocess_image`, in-memory JPEG encode/decode as in the existing PH2 evaluator, then frozen RGB→384×384→tensor→ImageNet normalization. Preserve Stage 23 selective-FP32 `resnet.nonlocal3` q@k numerical policy under CUDA AMP. Reuse the existing PH2 evaluator where possible, adapting only the checkpoint loader/provenance. Run only the fixed included cohort once; save seven probabilities, argmax, mapped label, exclusion audit, confusion matrix and source hashes.

Predeclared mapped outcomes: sample count, 2×3 `nv`/`mel`/`other` confusion matrix, accuracy, balanced accuracy, NV recall, MEL precision/recall/F1, and MEL one-vs-rest ROC-AUC from the unmodified MEL score if both classes are present. Other predictions count as wrong. Report Wilson 95% intervals for binomial accuracy/recall and fixed-seed 2,000-resample stratified case bootstrap percentile intervals for non-binomial metrics where mathematically defined; mark undefined intervals explicitly. Do not interpret repeated images as independent patients if provenance shows grouping.

## Domain shift and uncertainty

Describe PH2 versus Stage 23 HAM-validation distributions separately: image-quality proxies measured on comparable raw/processed views, class composition, max-softmax score and entropy, and continuous risk–coverage curves. Do **not** import the Stage 15/20 entropy cutoff or choose a Stage 23 threshold from PH2. Stage 23 validation-derived uncertainty associations are exploratory; PH2 can only assess their descriptive transfer. Contrast outcomes cautiously because PH2 uses a two-class mapped cohort with an `other` prediction column and excludes atypical nevi.

## Decision boundary

Freeze this analysis before opening PH2 images or performance outcomes on Day 3. No model training, parameter update, threshold fitting, label remapping, or checkpoint switching is permitted after inspecting external results. Report adverse or null transfer results as observed. A genuinely untouched external confirmation cohort is still needed for confirmatory generalization claims.
