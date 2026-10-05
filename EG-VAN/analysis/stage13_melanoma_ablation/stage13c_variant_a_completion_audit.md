# Stage 13C Variant A completion audit — 2026-10-04

**Disposition: 25/25 epochs completed; validation selection failed because no epoch met all preregistered eligibility gates.** This is an audit of the completed selective-q@k-FP32 A run, including continuation from the epoch-8 checkpoint. No training, HAM test inference, PH2 inference, or B/C run was performed during this audit. Existing research artifacts were read only.

## Artifact integrity and resumed history

The completed A folder contains `config.json`, `numerical_protocol.json`, `training_history.csv`, `numerical_events.json`, `last_checkpoint.pt`, and `experiment_manifest.json`. `last_checkpoint.pt` is **699,068,701 bytes**, SHA256 **`4b911c64b1df8da0dbd6d0355961ec73cc78b9c9ec21187842ec7f5049013168`**, matching the manifest. It records Variant A and epoch **25**. Its embedded history and the CSV each have **25** rows, numbered **1–25** exactly once, and all CSV fields match the embedded history. All 25 train and validation losses are finite. The first eight CSV rows plus header are byte-for-byte identical to the previously audited epoch-8 CSV: 1,716 bytes, SHA256 `7d54c95b781cab384f7be9e0ab8dc90daf642c6514331657cb16c2fe5589f348`. Thus the original eight history rows were preserved while epochs 9–25 were appended; this does not establish bit-for-bit equality of model trajectories to an uninterrupted run.

All floating tensors in the saved model state are finite, including parameters and BatchNorm running buffers; all floating optimizer-state tensors are finite. The saved Adamax LR is **9.765625e-7**, equal to the scheduler's `_last_lr`. The ReduceLROnPlateau state has `last_epoch=25`, factor 0.5, patience 1, finite `best=0.07517957464653931`, and a valid final LR. That scheduler `best` is the **unconditional** minimum validation loss, attained at epoch 7; it is not an eligible selected checkpoint. The GradScaler state is valid: scale **131072**, growth factor 2, backoff factor 0.5, growth interval 2000, and growth tracker 293. `numerical_events.json` records two recoverable skipped steps, at epoch 21 batch 98 and epoch 25 batch 208; both backed off 262144 → 131072 and agree with the corresponding history `amp_skips=1` rows. The final checkpoint tensors are finite. Per-batch state for all earlier batches is not stored, so their finiteness is supported by the guarded runner and completion records rather than independently remeasured.

The checkpoint, `config.json`, and registered `variant_config("A")` match exactly. Class order is `akiec, bcc, bkl, df, mel, nv, vasc`. Both checkpoint and run protocol file identify `Stage13C selective nonlocal3 qk FP32 under normal CUDA AMP`; the protocol file pins the audited bounded gate SHA256 `721eeaf6c64740d7dd8d7cdb06890a2ffa1c1be49c336f4d07033e4d172cdb00`. The manifest's frozen split hash `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` and Stage 9 control checkpoint hash `60f34ea4cfa6cbf3dce69e8d8d8813af30f33b2f666a4f06340de0` independently match the local files. The manifest declares no HAM test or PH2 access. No test or PH2 image/loader was used in this audit.

## Eligibility and checkpoint selection

Registered Stage 9 eligibility requires **MEL F1 ≥ 0.5757731958762886**, **macro F1 ≥ 0.6316582381362074**, and **NV recall ≥ 0.9** on the same validation epoch. Recomputing these gates from all 25 history rows yields **zero eligible epochs**, exactly matching every saved `eligible=False`. The highest MEL F1 is **0.5741626794258373** at epoch 15, below its gate by **0.0016105164504513**. Fifteen epochs satisfy the macro-F1 and NV-recall gates together, but none also satisfy the MEL-F1 gate. The minimum validation loss without eligibility filtering is **0.07517957464653931** at epoch 7; it cannot be selected under the preregistered rule. The checkpoint correctly records `best_epoch=None` and `best_validation_loss=Inf`; the final manifest serializes these as `selected_epoch=null` and `best_validation_loss=null` with status `FAIL_NO_ELIGIBLE_CHECKPOINT`. `best_checkpoint.pt` is absent, consistent with the rule. `validation_metrics.json` and `validation_predictions.csv` are also absent because the runner produces them only for a selected eligible checkpoint. No selected-epoch validation metrics exist.

## Final epoch-25 validation and Stage 9 reference

The final row is a validation-only epoch result, **not a selected checkpoint**. Its train loss is **0.015245914120861537** and validation focal loss is **0.08311225181638167**. Final validation metrics: accuracy **0.8144016227180527**, macro F1 **0.6665853585613355**, MEL recall **0.514018691588785**, MEL F1 **0.5472636815920398**, and NV recall **0.9336349924585219**.

| Validation metric | A epoch 25 | Frozen Stage 9 reference | A minus Stage 9 |
| --- | ---: | ---: | ---: |
| Accuracy | 0.8144016227180527 | 0.8225152129817445 | -0.0081135902636917 |
| Macro F1 | 0.6665853585613355 | 0.6709124104985493 | -0.0043270519372138 |
| MEL recall | 0.5140186915887850 | 0.5794392523364486 | -0.0654205607476636 |
| NV recall | 0.9336349924585219 | 0.9472096530920060 | -0.0135746606334841 |

The supplied Stage 9 reference has no MEL F1 or validation loss, so those two measures are not compared. The table compares the **unselected final A epoch** to Stage 9's frozen validation reference; it is not a comparison of selected checkpoints.

## Resume reproducibility limit

The run continued from saved epoch 8 and completed epochs 9–25. The epoch-8 checkpoint saved torch CPU/CUDA and sampler generator states, but did **not** store Python or NumPy RNG states. Consequently the resumed run cannot be claimed bit-for-bit identical to a hypothetical uninterrupted run. The first eight history rows are exactly preserved, and the observed 25-epoch artifact is internally consistent, but exact uninterrupted-trajectory equivalence is unverified.

**Next action:** stop after this audit. Variant A has no eligible checkpoint under the registered gates. Do not begin B/C or use HAM test/PH2 based on this report.
