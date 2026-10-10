# Phase B: Synthetic image-degradation robustness study

## Frozen protocol status

PHASE_B_STATUS: COMPLETE_BOUNDED_SAMPLE

SAMPLE_SIZE_AND_LESIONS: 21 validation images from 21 distinct lesions, created by selecting 3 lesions per true class with NumPy default_rng(2309), then the first sorted validation image per lesion. This is a class-balanced, lesion-aware bounded sample, not a representative 986-image validation estimate.

BASELINE_REPRODUCED: YES. Baseline predictions reproduced the saved Stage 23 validation outputs for all 21 selected images, with maximum absolute probability delta 0.001466155052185 and all 21 argmax labels matching the saved prediction table.

DEGRADATIONS_COMPLETED: YES. The protocol ran all seven fixed conditions in the required order on raw RGB images before preprocessing: baseline, blur_r1, blur_r2, underexposure_070, overexposure_130, contrast_070, jpeg_q40.

ACCURACY_CHANGES: Baseline accuracy = 15/21 = 0.7143. Conditions: blur_r1 7/21 = 0.3333 (delta -0.3810), blur_r2 5/21 = 0.2381 (delta -0.4762), underexposure_070 14/21 = 0.6667 (delta -0.0476), overexposure_130 14/21 = 0.6667 (delta -0.0476), contrast_070 14/21 = 0.6667 (delta -0.0476), jpeg_q40 12/21 = 0.5714 (delta -0.1429).

MELANOMA_RECALL_CHANGES: Baseline melanoma recall = 3/3 = 1.0. Conditions: blur_r1 1/3 = 0.3333 (delta -0.6667), blur_r2 0/3 = 0.0 (delta -1.0), underexposure_070 2/3 = 0.6667 (delta -0.3333), overexposure_130 3/3 = 1.0 (delta 0.0), contrast_070 2/3 = 0.6667 (delta -0.3333), jpeg_q40 2/3 = 0.6667 (delta -0.3333).

ENTROPY_CHANGES: Mean entropy deltas versus baseline: blur_r1 -0.0215 nats, blur_r2 +0.2400, underexposure_070 -0.0049, overexposure_130 +0.1291, contrast_070 -0.0552, jpeg_q40 +0.2676. These are exploratory paired changes on the 21-image bounded sample.

PREDICTION_FLIPS: Prediction flip rates: baseline 0.0, blur_r1 0.4762, blur_r2 0.5714, underexposure_070 0.1429, overexposure_130 0.2381, contrast_070 0.1429, jpeg_q40 0.2857. Correct-to-incorrect transitions were 8, 10, 1, 2, 1, and 4 respectively; incorrect-to-correct transitions were 0, 0, 0, 1, 0, and 1.

CONFIDENCE_INTERVALS: Lesion-cluster bootstrap 95% intervals are stored in `phase_b_statistical_results.json` under `paired_lesion_bootstrap_95` and were computed with fixed seed 2310 and 1000 resamples, grouped at lesion level and respecting the bounded sample design.

TEST_RESULTS: The original image-quality suite passed (`pytest tests/test_stage23_image_quality_final.py -q`, 5 tests). The dedicated Phase B integrity suite passed (`python -m pytest tests/test_stage23_quality_phase_b.py -q`, 5 tests); it verifies frozen protocol timing, hashes, complete paired predictions, baseline agreement, recomputed summaries, and figure resolution.

LIMITATIONS: This is a synthetic raw-image degradation study only. The sample is class-balanced and small (21 images/21 lesions), intentionally not representative of the 986 validation set. There is no clinical image-quality threshold, no claim of adequacy, and the study does not establish clinical robustness. Multiple severities and classes are exploratory; no multiplicity adjustment was applied.

IMAGE_QUALITY_OBJECTIVE_STATUS: The objective was met in the bounded sense: the frozen Stage 23 validation classifier was evaluated under fixed raw-image degradations with reproducible inference and paired analysis. It does not establish a general clinical image-quality conclusion.

NEXT_ACTION: If a broader CPU-capable run is desired, the full 986-image validation degradation study should be scheduled under a documented, resource-aware protocol; otherwise, retain this bounded study as the transparent Phase B evidence base and avoid overgeneralizing from the restricted sample.

## Files generated

The following artifacts were saved under `analysis/stage23_image_quality_final/phase_b/`:

- `PHASE_B_PROTOCOL.md`
- `phase_b_protocol.json`
- `phase_b_manifest.json`
- `phase_b_predictions.csv`
- `phase_b_summary.csv`
- `phase_b_melanoma_analysis.csv`
- `phase_b_statistical_results.json`
- `phase_b_robustness_curves.png`
- `phase_b_prediction_transitions.png`
- `phase_b_melanoma_robustness.png`
- `FINAL_PHASE_B_REPORT.md`

## Frozen checkpoint and protocol references

- Checkpoint SHA256: `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`
- Protocol seed: `selection_seed = 2309`
- Bootstrap seed: `2310`
- Reconstruction rule: raw RGB degradation before frozen preprocessing, with no post-preprocessing degradation or tuning.
- Data access constraint: no HAM test outcomes and no PH² images/outcomes were accessed.

## Summary of reproducibility

This run used only the frozen Stage 23 validation artifacts and unchanged preprocessing path. The benchmarked degradation study is fully scripted with:

- `analysis/stage23_image_quality_final/phase_b/run_phase_b.py`
- `analysis/stage23_image_quality_final/phase_b/analyze_phase_b.py`
- `analysis/stage23_image_quality_final/phase_b/freeze_protocol.py`

The complete output manifest includes SHA256 digests for the input sources and generated outputs in `phase_b_manifest.json`.
