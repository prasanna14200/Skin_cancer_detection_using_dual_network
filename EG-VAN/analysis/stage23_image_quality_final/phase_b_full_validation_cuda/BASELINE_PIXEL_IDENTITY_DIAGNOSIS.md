# Stage 23 CUDA Phase B baseline pixel-identity failure

## Finding

The Colab exception for `ISIC_0025339` occurred before model inference in validation batch 0. The former `processed_image()` read the raw JPEG with OpenCV, ran `preprocess_image()`, encoded and decoded a **new** JPEG with OpenCV, and compared its decoded **BGR uint8 pixels** to the historical processed JPEG decoded with OpenCV. The assertion did not compare JPEG bytes, normalized model tensors, or predicted probabilities. Only after this extra replay check did it load the historical JPEG with PIL for the model.

That replay requirement was the implementation error. The original Stage 23 validation dataset loaded `data/processed/images/<image_id>.jpg` directly with PIL as RGB and applied resize, `ToTensor`, and normalization. It did not regenerate preprocessing or re-encode the JPEG during validation. Pixel equality between a new OpenCV preprocessing/JPEG round trip in Colab and a previously generated JPEG is a separate reproducibility claim, not a prerequisite for using the frozen Stage 23 baseline input. The Colab exception establishes that those two decoded BGR arrays differed there; it does not reveal the first differing preprocessing operation, JPEG codec behavior, or the pixel delta. The specific lower-level Colab divergence remains unmeasured.

## Local evidence

- `ISIC_0025339` is validation row 0, lesion `HAM_0006755`, true class `bkl`.
- The local `data/raw/images/ISIC_0025339.jpg` and original `data/raw/HAM10000_images_part_1/ISIC_0025339.jpg` have identical SHA256 `72d813753fbe97c0d23ef72d7231f126fa71e08338e2376e0c34e95bae1c7325` and identical decoded pixels.
- The historical processed JPEG SHA256 is `1b3fc11627ffdcb212a2b5c65701e743cebe48bde91916c4ca163be679c39e49`, as already recorded in `image_quality_metrics.csv`. That table's SHA256, `d11570bbb11451eea37073b849b6b8db7d4c7549d7efcc71652628a20d2d8455`, matches `quality_analysis_manifest.json`.
- On this Windows installation (OpenCV 5.0.0, Pillow 11.3.0), replaying `preprocess_image()` and OpenCV JPEG encoding for this image produced **identical JPEG bytes** and zero differing decoded pixels. This local result rules out an intrinsic mismatch for these local files; it does not reproduce or localize the Colab difference.
- The checkpoint hash read during preflight is `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`. The frozen split/reference/source hashes also passed.

## Correction and preservation

The CUDA runner now loads the exact historical processed JPEG through PIL for baseline, as Stage 23 did. Before use, it checks each processed JPEG SHA256 against the previously audited 986-image hash table and requires the actual RGB image returned for inference to be pixel-identical to that saved JPEG decoded through the original PIL path. The pixel-identity check remains strict; a one-pixel mutation fails a regression test. Baseline does not call the raw-image degradation or preprocessing functions and does not re-encode. The six frozen degradation definitions and their raw-image preprocessing path are unchanged. The 0.005 seven-probability tolerance and exact argmax gate remain unchanged.

The read-only `--check` preflight now verifies every processed validation JPEG hash, including inputs for any already committed batches, and validates existing CUDA batch records without overwriting them. A committed batch that passes this preflight and its existing probability/metadata validation needs no invalidation. No CUDA batch records are present in this local copy; any records on Colab must be assessed with `--check`. Since the reported failure was on image 0 of batch 0, that failed attempt could not have committed batch 0.

## Local verification

- Focused CUDA-runner tests: 11 passed. The local TorchVision installation emitted a non-blocking image-extension warning and a Windows DLL diagnostic during import; the test process exited successfully.
- `python -m py_compile` passed for runner and test source.
- `python .../run_cuda_full_validation.py --check`: `PREFLIGHT_PASS`; 986 processed-image identities, checkpoint, split, reference, source and seven frozen conditions validated; local CUDA batch inventory was zero for every condition.
- No CUDA inference, degradation inference, training, checkpoint write, reference-prediction edit, or batch-output write was performed.

## Safe Colab continuation

After syncing the updated runner and running `--check` on Colab, use `--run --baseline-only` if there are no committed CUDA batches, or `--run --resume --baseline-only` if valid CUDA batches already exist. Complete-baseline probability and argmax equivalence must still pass before any degradation run. If Colab's processed JPEG differs from the audited hash, stop and inspect file transfer/provenance rather than bypassing the check.
