# Full-validation Phase B extension status

## Scope

This extension was frozen separately from the bounded 21-image pilot and intentionally does not overwrite or mix the earlier Phase B results. The same frozen Stage 23 checkpoint, validation cohort, and degradation settings are preserved for the full 986-image extension.

## Actual execution status

FULL_VALIDATION_STATUS: PARTIALLY VERIFIED / FULL BASELINE PENDING

IMAGES_AND_LESIONS: The fixed validation cohort is 986 images and 743 lesions. The existing proof-of-work chunks were validated against this cohort and the frozen reference: 3 baseline records and 1 record for each of the six degradation conditions (9 valid records total, no duplicates). A subsequent baseline-only resume test extended baseline coverage to 3 images; it did not run any degradation inference.

BASELINE_REPRODUCED: The baseline gate passed on 3/3 images. The maximum absolute probability delta was 0.000197358429432, below the frozen 0.005 tolerance; preprocessed pixel identity also passed for all three. Full 986-image verification remains pending.

DEGRADATION_CONDITIONS: The protocol is frozen with the same seven conditions as the original bounded Phase B study: baseline, blur_r1, blur_r2, underexposure_070, overexposure_130, contrast_070, and jpeg_q40. All are applied to raw RGB before the frozen preprocessing pipeline.

ACCURACY_AND_RECALL: Not available for the full cohort. No complete full-cohort summary is claimed.

MELANOMA_FINDINGS: Not available for the full cohort pending full completion of the eight-step execution plan.

ENTROPY_AND_CONFIDENT_ERRORS: Not available for the full cohort pending completion of the full prediction table.

PAIRED_CONFIDENCE_INTERVALS: Not available because the complete prediction set and paired baseline comparison were not produced at full scale.

CHECKPOINT_INTEGRITY: Confirmed. The frozen Stage 23 checkpoint SHA256 matches the required value exactly: `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`.

SOURCE_PROVENANCE: Verified. The frozen registry and file hashes match the historical Stage13 provenance. No historical experiment files were silently altered.

TEST_RESULTS: Three CPU-only chunk/resume tests passed. Existing chunks were independently checked: 9/9 current rows valid, no duplicate image-condition keys, and baseline probabilities satisfy the frozen reference tolerance. The multi-image `--limit 3 --baseline-only --resume` test reused the original first baseline chunk, appended two new uniquely numbered chunks, verified all three baseline images, and stopped without running degradation inference.

REMAINING_LIMITATIONS: The host has 7.68 GiB physical RAM with approximately 1.19 GiB free after the bounded run; observed runner peak RSS was 606 MB. The two newly processed baseline images took 22.3 seconds total, or approximately 11.2 seconds per image excluding model startup. Extrapolating the 986-image inference and processed-pixel gate indicates more than three hours for the baseline alone. That duration is not feasible within this agent session. No six-condition full-cohort degradation inference was started.

IMAGE_QUALITY_RESEARCH_CONCLUSION: The bounded Phase B pilot remains a completed and separate artifact. The full 986-image extension is prepared and partially validated, but no full-cohort research conclusion should be drawn until the complete execution and statistical analysis are finished on an appropriate host.

## Refactored execution notes

The runner at `run_phase_b_full_validation.py` now:

- One frozen model instance is loaded once per process.
- CPU inference is used with a single-threaded torch configuration and a small in-memory footprint.
- Images are processed individually; CSV chunks are flushed, fsynced, and atomically installed without replacing existing chunks.
- `--resume` validates every row before skipping it; malformed, duplicate, or incomplete chunk data fails explicitly.
- Progress includes elapsed time, ETA, processed records, and peak RSS.
- `--baseline-only` prevents all degradation inference. Default full execution cannot proceed to degradations until the entire selected baseline gate passes.
- `--verify-existing-chunks` checks the existing chunks and exits without loading the model or modifying state.

## Verified bounded execution

A bounded end-to-end validation succeeded with `--limit 1 --chunk-size 1 --resume`, and a multi-image baseline resume succeeded with `--limit 3 --chunk-size 1 --progress-every 1 --baseline-only --resume`:

- Baseline verified: true for 3/3 selected images, not the full cohort
- Maximum absolute probability delta: 0.00019735842943191528
- Valid records currently on disk: 9 (3 baseline, 1 per degradation condition)
- Full-cohort final results available: false
- Peak RSS: ~605.97 MB

## Manual full-baseline command

From PowerShell, run:

```powershell
Set-Location 'D:\Cancerdetection\EG-VAN'
python analysis\stage23_image_quality_final\phase_b_full_validation\run_phase_b_full_validation.py --baseline-only --resume --chunk-size 10 --progress-every 10
```

This resumes the existing baseline chunks, verifies all 986 unchanged images against the frozen reference and preprocessed pixel identity, and does not start degradation conditions. Continue to use `--baseline-only` until the command reports `BASELINE GATE PASSED: 986/986 images`.

This confirms the execution path is viable for a resumable low-memory run, but not sufficient to claim the full 986-image extension is complete.

## Files present

- `phase_b_full_validation_protocol.json`
- `run_phase_b_full_validation.py`
- `execution_state.json`
- `phase_b_full_validation_predictions.csv`
- `FINAL_PHASE_B_FULL_VALIDATION_REPORT.md`

The full-cohort baseline, six degradation conditions, 6,902-record prediction table, and statistical analyses remain pending. The original bounded pilot remains separate and unchanged.

The pre-existing `phase_b_full_validation_predictions.csv` is only the earlier one-image proof-of-work assembly (7 rows). It is not authoritative after the baseline-only resume extension and must not be interpreted as the current chunk inventory or a full-cohort result. The validated atomic chunk files are the current resumable source.
