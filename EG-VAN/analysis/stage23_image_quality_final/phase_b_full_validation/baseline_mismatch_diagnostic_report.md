# Stage 23 full-validation baseline mismatch diagnosis

**Disposition:** baseline gate remains **FAILED**. This is a read-only diagnosis of saved research evidence; no training, HAM test/PH2 access, checkpoint edit, split edit, or reference-prediction edit occurred. The [machine-readable repeat trace](baseline_mismatch_diagnostic.json) has SHA256 `dc4ae2d1bb458ddf26e235f7115436bb4b0c903f33fc19d3bbb94ba9f5b590fe`.

## Saved progress

The runner's own `--verify-existing-chunks` check passes. There are **50 baseline chunks containing 473 valid, unique, complete baseline records**. Every saved probability vector is finite, nonnegative, sums to one, yields its recorded class/confidence/entropy, and agrees with its saved Stage 23 reference label within the frozen probability tolerance. The largest saved baseline probability delta is 0.0040428042411804. Six other condition chunks contain one valid record each for `ISIC_0024321`; the separate condition CSVs and combined CSV are scoped one-case artifacts, **not** a completed full validation run. In total, 479 chunk records are valid. No full condition is complete. The state file's `last_progress.valid_records=473` agrees with the durable baseline chunks, while `baseline_verified_images=3` reflects a previous bounded scope and must not be interpreted as 986-image verification. None of these files was deleted, rewritten, or regenerated.

## Disputed case

`ISIC_0029026` belongs to lesion `HAM_0002905` and has true class **NV**. Raw JPEG SHA256 is `12db355827e7056441e62a1d9b9fa98274ff32cdb51091f6e1f777006617a5e2`; processed JPEG SHA256 is `f3ab1591eb8e82f6c9430aa5d8aeefdab9f58e90a9f5d77f2f12873f3ddd9469`. Both files decode, and the regenerated paper-preprocessing result is **pixel-identical** to the saved processed JPEG. The resulting 384×384 normalized model input tensor is also **exactly equal** between the raw-replay and direct saved-processed paths.

Class order: `akiec, bcc, bkl, df, mel, nv, vasc`. Frozen checkpoint SHA256 before and after repeats: `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`.

| Class | Saved Stage 23 reference | Current CPU FP32 replay |
|---|---:|---:|
| AKIEC | 0.0332494415 | 0.0332671143 |
| BCC | 0.0551546738 | 0.0547126867 |
| BKL | 0.4337192476 | **0.4365369678** |
| DF | 0.0005138192 | 0.0005110913 |
| MEL | 0.0407554768 | 0.0405478738 |
| NV | **0.4345671535** | 0.4324012399 |
| VASC | 0.0020401482 | 0.0020231197 |

Reference prediction: **NV**, ahead of BKL by **0.0008479059**. CPU prediction: **BKL**, ahead of NV by **0.0041357279**. Maximum absolute probability difference is **0.0028177202** (BKL). This is a **near-tie argmax flip with small probability differences**, not a large probability disagreement. It falls below the frozen 0.005 probability tolerance but violates the separate exact-argmax requirement.

Three repeated CPU raw-replay inferences and three direct saved-processed-image inferences produced exactly the same seven probabilities and BKL argmax. Two neighboring validation controls also reproduced exactly between raw and direct CPU paths: `ISIC_0029017` remained MEL (maximum reference delta 0.0005544), and `ISIC_0029036` remained BKL (delta 0.0000693). These repeats rule out nondeterminism in the tested CPU path and a raw-versus-processed implementation error for these cases.

## Pipeline comparison and localization limit

The original Stage 23 validation loader reads the saved processed JPEG with PIL RGB, applies the registered bilinear resize to 384×384, `ToTensor`, and ImageNet normalization, then evaluates `model.eval()` in batches of **16 under CUDA FP16 autocast**. It computes softmax from FP32-cast logits and serializes seven probabilities in the frozen class order. The full Phase B runner reads raw JPEG with OpenCV BGR, converts to RGB, runs the same hair-removal → Gray World → Retinex preprocessing, JPEG encodes/decodes with OpenCV, then passes a PIL RGB image to the same frozen transform and model loader. Its inference is **single-image CPU FP32**. The processed pixels and transformed tensor agree exactly for the disputed case; model state and class order are hash-verified. Thus the first demonstrated meaningful divergence is **numerical execution**: device/precision and batch shape. The CPU path's stable BKL/NV flip is consistent with the very small reference margin.

The available local environment has no CUDA GPU. It cannot independently replay the original CUDA FP16 batch-16 kernels. Consequently, the **specific arithmetic source** of the difference (CUDA FP16 rounding, batch-shape-dependent kernels, or their combination) remains unresolved. The saved reference is independently checked against its source hash and validation metadata, but this diagnosis does not assume it is inherently more correct than the CPU prediction.

## Safe next step

The baseline gate must remain failed; **do not resume the full degradation run or loosen its threshold**. The new [CUDA batch probe](probe_original_cuda_batch.py) is bounded to the original validation batch containing `ISIC_0029026` (split-order index 130, batch start 128, offset 2). It compares CUDA FP16 and FP32 under singleton and original batch-16 shapes, using the unchanged checkpoint and saved processed validation images. Run it on a T4 from the repository root with:

```bash
python analysis/stage23_image_quality_final/phase_b_full_validation/probe_original_cuda_batch.py
```

It writes a **new** `cuda_baseline_mismatch_probe.json` and touches no chunk or reference artifact. Local syntax checks passed; GPU execution is pending. If matching Stage 23 CUDA execution reproduces the reference, a switch to that execution policy would require separately documented provenance and must not mix newly generated predictions with the existing CPU chunks. If the CUDA probe does not reproduce the reference, investigate runtime/version and exact validation serialization before any protocol change. A probabilistic equivalence policy allowing near-tie label flips would be a **separate prospective protocol amendment**, never a silent reinterpretation of this failed gate.
