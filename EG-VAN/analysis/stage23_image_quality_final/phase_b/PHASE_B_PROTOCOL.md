# Phase B protocol — frozen before inference

**Status:** predeclared, Stage 23 validation only. The Stage 23 epoch-14 classifier is frozen at SHA256 `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`. This study does not train, select, or tune it. HAM test and PH2 data are excluded.

## Inclusion and fixed sample

The eligible population is the 986 unique Stage 23 validation images in `data/splits/split_leakage_aware.csv`, with valid raw and processed JPEGs and saved Stage 23 prediction/probability records. Because the existing seven-image/49-forward CPU pilot took about 13 minutes, a full 986×7 run would be prohibitive on the current CPU. **Before inference**, select exactly three distinct lesion IDs per true class using NumPy `default_rng(2309).choice` without replacement from lexicographically sorted eligible lesion IDs. For each selected lesion use its lexicographically first validation image ID. Sort the 21 selected IDs by class then ID, and persist them in `phase_b_protocol.json`. The sample is class-balanced by design and **not representative** of validation class prevalence; accuracy is descriptive for these 21 images, not an estimate of overall validation accuracy. No correctness, entropy, or confidence enters selection. Missing input or baseline mismatch causes a stop; no case replacement is permitted after outcomes are seen.

## Fixed image interventions

Primary experiment: decode the **raw** RGB JPEG, apply intervention, then run the unchanged paper preprocessing (`src/preprocessing.py:preprocess_image`: hair removal → Gray World → Retinex), JPEG encode/decode with OpenCV's default encoding as in the validated preprocessing output, and the frozen RGB `Resize(384,384) → ToTensor → ImageNet Normalize` evaluation transform. No augmentation. No post-preprocessing intervention is included. Conditions for every case, in fixed order:

1. `baseline`: identity.
2. `blur_r1`: PIL Gaussian blur, radius 1 source pixel.
3. `blur_r2`: PIL Gaussian blur, radius 2 source pixels.
4. `underexposure_070`: RGB ×0.70, round and clip to uint8.
5. `overexposure_130`: RGB ×1.30, round and clip to uint8.
6. `contrast_070`: `(RGB − global RGB mean) ×0.70 + global RGB mean`, round/clip.
7. `jpeg_q40`: in-memory RGB JPEG quality 40, subsampling 0, then decode.

These values were fixed before inference, resemble plausible image changes, and do not define a clinical adequacy boundary. The JPEG step after paper preprocessing is part of the **frozen pipeline replay**, not the tested JPEG degradation; only `jpeg_q40` adds a *pre*-preprocessing low-quality re-encode. No severity will be changed after inspecting results.

## Baseline gate

Verify checkpoint/registry/split/prediction hashes, class order, raw/processed file availability, and 21 distinct lesion IDs. First run the unchanged raw-image baseline for all 21. Replay preprocessing and compare the generated processed JPEG pixels with the corresponding saved processed JPEG. Require exact decoded pixel identity; if source pipeline versions prevent this, stop and report rather than proceed. Require every baseline argmax to match the saved Stage 23 validation argmax and maximum absolute probability discrepancy ≤0.005 per case, allowing documented CPU FP32 versus original CUDA numerical differences. If any fail, stop before degraded inference and retain only diagnostic baseline output. Do not replace cases.

## Endpoints and uncertainty

Primary endpoint: paired accuracy change for each degradation relative to the same-case baseline. Secondary endpoints: melanoma recall (3 sampled MEL lesions only), classwise correct counts, prediction flip rate, correct→incorrect and incorrect→correct counts, MEL false-negative and MEL→NV counts, and paired mean/median entropy change. Save all seven-class probabilities, confidence and entropy per image/condition. Lesion-cluster bootstrap (seed 2310, 1000 draws) estimates percentile 95% intervals for paired accuracy, flip-rate and mean entropy changes; group sampling resamples whole lesions. For melanoma, report raw counts and a grouped interval but emphasize the n=3 denominator and extreme imprecision. No hypothesis-testing or multiplicity-adjusted claims; all severity and class comparisons are exploratory.

Any missing file, invalid probability, non-finite tensor, preprocessing mismatch, baseline mismatch, or failed condition aborts the run. No exclusion or replacement based on prediction outcomes. A partial run must be labelled incomplete. Figures and reports are generated only from a complete, baseline-validated prediction table. No accept/reject quality rule, clinical claim, or human-quality label is inferred from synthetic changes.
