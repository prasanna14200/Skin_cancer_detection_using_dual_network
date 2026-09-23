# EG-VAN Phase 4B Preprocessing Report

Date: 2026-09-17

## 1. Paper sections consulted

- Supplied EG-VAN paper, Section III-A, **Data Preprocessing**, equations (1)-(14), pages 72855-72857 in the local PDF.
- Section III-B, **Data Augmentation**, was reviewed to keep augmentation out of Phase 4B.
- Section IV-G, **Impact of Preprocessing Steps**, was reviewed as context only; no accuracy judgment was made in Phase 4B.

## 2. Preprocessing pipeline

The paper's stated sequence is:

```text
Original image
    -> Hair removal
    -> Gray World color correction
    -> Retinex color correction
    -> Processed image
```

The implementation is deterministic and per-image. It does not calculate dataset-wide statistics and does not apply augmentation.

## 3. Hair removal implementation

- **Operation:** grayscale morphology, blackhat response, binary threshold mask, Telea inpainting.
- **Paper specification:** equations (1)-(7) describe convolution, morphological operations, a black top-hat/blackhat response, thresholding, and Telea inpainting.
- **Implementation:** OpenCV grayscale `MORPH_BLACKHAT` with an elliptical 17x17 structuring element, threshold 10, and Telea inpainting radius 1.0.
- **Paper-specified parameters:** none of the kernel shape/size, threshold, or Telea radius is provided.
- **Deviation:** `[PAPER-UNDERSPECIFIED]` values were fixed explicitly and are recorded in `docs/DEVIATIONS.md`.

## 4. Gray World implementation

- **Operation:** calculate per-image channel means, derive pairwise channel gains from equation (9), apply gains, clip to the 8-bit range.
- **Paper specification:** equation (8) defines average RGB channel values. Equation (9) defines the channel correction gains. The text places Gray World correction after hair removal and before Retinex.
- **Implementation:** per-image BGR means, pairwise gains corresponding to the displayed equation, epsilon `1e-6` for zero-denominator protection, and clipping to `[0, 255]`.
- **Paper-specified parameters:** no dataset-wide statistics or additional parameters are specified.
- **Deviation:** `[PAPER-UNDERSPECIFIED]` the printed channel notation is typographically ambiguous, and clipping behavior is not stated. The implementation uses OpenCV BGR order and deterministic clipping.

## 5. Retinex implementation

- **Operation:** Gaussian filtering at multiple scales, logarithmic filtered-minus-input subtraction, weighted multi-scale summation, per-channel normalization.
- **Paper specification:** equations (10)-(12) define Gaussian filtering, logarithmic subtraction, and weighted multi-scale Retinex. Equation (13) combines Retinex channels with Gray World gains.
- **Implementation:** Gaussian sigmas `(15, 80, 250)`, equal weights `(1/3, 1/3, 1/3)`, epsilon `1e-6` before logarithms, and independent 1st-99th percentile normalization per channel to 8-bit output.
- **Paper-specified parameters:** the paper does not provide Gaussian scales, weights, logarithm epsilon, or output normalization details.
- **Deviation:** `[PAPER-UNDERSPECIFIED]` all numerical Retinex parameters and normalization were fixed explicitly and are recorded in `docs/DEVIATIONS.md`.

## 6. Sample validation

- **Sample size:** 10 images selected deterministically with `SEED = 42`.
- **Successful:** 10.
- **Failed:** 0.
- **Checks:** input existence and decoding, output existence and reopening, valid three-channel dimensions, finite pixel values, and non-empty output.
- **Output dimensions:** each sampled output preserved its source dimensions, 450x600x3.
- **Visual artifacts:** before/after side-by-side JPEGs were written under `data/processed/preprocessing_samples/`.

## 7. Full dataset processing

- **Total:** 10,015.
- **Successfully processed:** 10,015.
- **Failed:** 0.
- **Processing duration:** 3,682.79 seconds.
- **Processing log:** `data/processed/preprocessing_log.csv` contains one successful row for every image.
- **Output count:** 10,015 JPG files.
- **Output identity:** processed filenames exactly match source image IDs.
- **Output audit:** all outputs reopened successfully, contained finite pixels, and had dimensions 450x600x3.

## 8. Data integrity

- Original files in `data/raw/images/`, `data/raw/HAM10000_images_part_1/`, and `data/raw/HAM10000_images_part_2/` were used as inputs only.
- No original image was overwritten or deleted.
- Processed images were written to `data/processed/images/`.
- No training augmentation, model code, evaluation, or dataset-wide normalization was performed.

## 9. Deviations

- Hair-removal kernel shape/size, threshold, and Telea radius were not stated by the paper; explicit deterministic values were chosen.
- Gray World channel ordering/clipping behavior was not operationally unambiguous; OpenCV BGR order and 8-bit clipping were chosen.
- Retinex scales, weights, log epsilon, and normalization were not stated; explicit per-image values were chosen.
- The paper mentions crop/resize after color processing but does not specify crop size or model dimensions. Crop/resize was omitted to avoid inventing values and to preserve source dimensions.
- Augmentation was intentionally excluded because it belongs to training/data loading, not deterministic cached preprocessing.
- A bounded four-worker process pool was used for independent per-image processing; it uses no shared or dataset-wide statistics.

## Status

**PHASE 4B COMPLETE**

Phase 4C model implementation has not started. No accuracy or preprocessing-effect claim is made in this phase.
