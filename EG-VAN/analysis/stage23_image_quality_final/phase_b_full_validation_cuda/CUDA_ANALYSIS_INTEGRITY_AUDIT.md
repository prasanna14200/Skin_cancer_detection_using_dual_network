# CUDA Phase B final-table integrity audit

## Disposition

**PASS: representation-only mismatch.** Read-only validation of the existing final table against all 434 committed batch directories passed. The 6,902 image-condition pairs are unique, ordered as 986 images for each of seven frozen conditions, and have identical categorical fields and identical seven-class probability JSON values. No inference record or batch file was modified.

The final table SHA256 is `ff7c7a454bd896c54628bd4aeb61949789b82d1a69ae3473878690ca11d0d221`.

## Exact first difference

The original analyzer used `DataFrame.equals()`, which requires exact equality of parsed floating-point scalars. The runner created each committed `records.csv` with 17-digit float serialization, then read it through pandas and serialized the combined final CSV again. That second parse/write changed some derived decimal spellings at machine precision.

First differing field: condition `baseline`, row 0, image `ISIC_0025339`, column `confidence`:

| Source | CSV value |
|---|---:|
| Committed `batches/baseline/batch_0000/records.csv` | `0.49667981266975403` |
| Final `cuda_full_validation_predictions.csv` | `0.49667981266975397` |

The absolute difference is `5.551115123125783e-17`. The corresponding seven-class probability JSON is identical in both files, including its first probability `0.49667981266975403`. The maximum absolute committed-versus-final difference in any derived scalar over all conditions was `2.220446049250313e-16`.

| Condition | Batch files | Rows | Changed confidence spellings | Changed entropy spellings | Probability/metadata differences |
|---|---:|---:|---:|---:|---:|
| baseline | 62 | 986 | 494 | 574 | 0 |
| blur_r1 | 62 | 986 | 501 | 675 | 0 |
| blur_r2 | 62 | 986 | 478 | 562 | 0 |
| underexposure_070 | 62 | 986 | 500 | 603 | 0 |
| overexposure_130 | 62 | 986 | 472 | 438 | 0 |
| contrast_070 | 62 | 986 | 465 | 635 | 0 |
| jpeg_q40 | 62 | 986 | 493 | 494 | 0 |

No duplicate, missing, reordered, class-label, image-ID, or probability differences were observed. For the six degraded conditions, the baseline-delta column is missing in both files as designed; baseline values agree.

## Corrected comparison

`analyze_cuda_full_validation.py` now validates all committed batch hashes and metadata as before, and compares every final-table record in its original order. The image/lesion IDs, class labels, condition, batch location, correctness, and reference label must match exactly. All seven JSON probabilities must match exactly as float64 values, with no tolerance. Only the redundant `confidence`, `entropy_nats`, and baseline probability-delta scalar columns permit a maximum absolute CSV round-trip discrepancy of `4 × float64 epsilon` (`8.881784197001252e-16`), with zero relative tolerance. Missing values must agree. The observed maximum was one quarter of that bound. A changed probability, prediction, row, or column schema fails immediately with its first differing value.

The new `--check-only` mode performs this entire validation without writing analysis outputs. It returned `INTEGRITY_PASS`, 986 images per condition, 7 conditions, 6,902 records. Syntax checks passed. Focused CUDA runner and analyzer tests passed: 19 tests, including the observed first-row rounding example, corruption of each of the seven probability components, prediction/metadata corruption, missing or reordered rows, and changed column order. The local TorchVision import emitted an image-extension warning, but tests exited successfully.

## Existing artifact handling

The completed Colab inference files can be analyzed without rerunning inference. Sync the corrected `analyze_cuda_full_validation.py` to Google Drive; the new regression test and this audit can be synced for reproducibility. Preserve `run_cuda_full_validation.py`, `batches/`, `baseline_gate.json`, and `cuda_full_validation_predictions.csv` exactly as they are. Run the corrected analyzer with `--check-only` first, then its normal analysis command. No final robustness conclusion is made in this audit; those outputs should be reviewed after analysis completes.
