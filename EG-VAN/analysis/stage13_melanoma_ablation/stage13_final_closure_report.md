# Stage 13 / 13C final closure report

**Final disposition: `NO_CANDIDATE_SELECTED`.** This report closes the preregistered HAM-validation candidate-selection objective. It documents completed Variant A and the reason no B/C candidate can be selected under the frozen rule. No training, checkpoint edit, HAM test or PH2 inference, finalizer run, or Stage 14 work was performed to create this report. `selection_rule.json` remains unchanged.

## A. Original scientific hypothesis

Stage 13 asked whether melanoma-focused changes could improve EG-VAN's HAM validation melanoma recall beyond both the frozen Stage 9 model and a freshly trained A control, while retaining minimum overall/class performance. The Stage 9 validation reference recalled **62 of 107 MEL cases** (recall `0.5794392523364486`). The preregistered minimum gain was one additional correctly recalled validation MEL case beyond both controls. This was an exploratory, validation-only candidate decision; it did not authorize tuning on HAM test or PH2 outcomes. See [Stage 13 protocol](stage13_protocol.md) and [frozen selection rule](selection_rule.json).

## B. Original preregistered A/B/C design

| Variant | True-MEL focal-loss multiplier | Train MEL sampler weight | Scientific role |
| --- | ---: | ---: | --- |
| A | 1.0 | 1.5630495442733532 | Fresh Stage 9 objective/sampler control |
| B | 1.5630495442733532 | 1.5630495442733532 | MEL loss multiplier only |
| C | 1.0 | 2.4431238778531372 | MEL sampler-weight change only |

The frozen design used the same seven-class EG-VAN architecture, HAM train/validation split, seed 42 and initialization policy, 384-pixel images, physical/effective batch 16, 25 epochs, focal alpha 0.25/gamma 2, Adamax LR 0.001/weight decay 0.0001, ReduceLROnPlateau, and unweighted validation focal loss. Within each run, a checkpoint required MEL F1 ≥ `0.5757731958762886`, macro F1 ≥ `0.6316582381362074`, and NV recall ≥ `0.9`; among eligible epochs, minimum validation loss won, with the earlier epoch on ties. Only B/C could be final new candidates. The scientific configurations and frozen selection thresholds were not changed during Stage 13C.

## C. Numerical failures encountered

Early bounded CUDA AMP traces showed first-batch non-finite gradients at the default initial GradScaler scale **65536** for A and B; lower first-batch scales alone did not establish multistep stability. A guarded scale-16384 gate stopped on A batch 2. A preliminary ordinary-AMP A/B/C gate at initial scale 8192 completed A/B but stopped on C batch 7: C batch 2 had a recoverable GradScaler skip/backoff, while batch 7 produced non-finite forward logits and a non-finite BatchNorm running-mean buffer. A memory-safe same-state probe showed C batch-7 FP32 forward was finite and ordinary FP16 AMP forward was not. These are earlier attempts and bounded diagnostics, **not completed Stage 13C full B/C experiments**. See [numerical-stability audit](stage13c_numerical_stability_audit.md).

## D. Root-cause localization

The [arithmetic trace](stage13c_nonlocal3_arithmetic_trace.json) localized the first non-finite output in `resnet.nonlocal3` to **`torch.bmm(q, k)`**. Both input projections `q` and `k` were finite FP16. Their FP16 affinity contained **3 Infs**; the same pre-batch-7 state/input in FP32 produced finite affinity with maximum **69518.90625**, beyond FP16's largest finite value **65504**. Softmax NaNs and downstream model-buffer contamination followed the affinity overflow. The supported numerical root cause for this C batch-7 forward failure is FP16 overflow in the query-key affinity matmul. This does not establish the exact originating operation for the separate early backward-gradient overflows.

## E. Selective-FP32 numerical correction

The diagnostic-only policy computes only `resnet.nonlocal3`'s query-key `torch.bmm(q.float(), k.float())` with CUDA autocast disabled; surrounding model operations remain under CUDA AMP. The [same-state C batch-7 probe](stage13c_selective_qk_fp32_trace.json) recorded finite FP32 affinity, attention, logits, focal loss, scaled/unscaled gradients, model parameters/buffers, and optimizer state, with no batch-7 optimizer update. The [bounded 16-batch A/B/C gate](stage13c_abc_selective_qk_fp32_gate.json) then completed all variants with this same policy. A and B executed 16/16 steps; C safely skipped batches 2–4 with dynamic scaler backoff before completing the remaining window. This gave **bounded numerical clearance**, not a guarantee of 25-epoch stability and not a classification-performance improvement claim. The guarded full-runner applied that same execution policy in A's later full training.

## F. Why this is a numerical policy, not an A/B/C scientific-factor change

The intervention changes the arithmetic precision of the demonstrated overflowing q@k matmul only. It does not change the model's layer graph or attention formula, focal loss, true-MEL multiplier, sampler weights, Adamax/scheduler, split, seed, or selection thresholds. The bounded gate applied the identical in-memory policy to A/B/C; their initial model hashes matched, A/B received the same initial sampled batches, and C used only its preregistered sampler difference. Production architecture source was not modified for this correction. A's successful numerical execution does not imply a gain in melanoma classification.

## G. Completed Stage 13C Variant A result

The [A completion audit](stage13c_variant_a_completion_audit.md) independently verified **25/25** continuous history epochs, finite train/validation losses, finite saved model parameters and buffers, finite optimizer state, and valid scheduler/GradScaler state. The first eight CSV rows match the audited pre-resume history byte for byte. A's last checkpoint is epoch 25, SHA256 `4b911c64b1df8da0dbd6d0355961ec73cc78b9c9ec21187842ec7f5049013168`; final GradScaler scale is 131072. The final, **unselected** epoch-25 validation row has focal loss `0.08311225181638167`, accuracy `0.8144016227180527`, macro F1 `0.6665853585613355`, MEL recall `0.514018691588785`, MEL F1 `0.5472636815920398`, and NV recall `0.9336349924585219`. These epoch-25 metrics are not selected-checkpoint metrics.

## H. Why A failed checkpoint eligibility

Recomputing the frozen three-gate test on every A history row yields **no eligible epoch**. The highest A MEL F1 was **`0.5741626794258373` at epoch 15**, below the required **`0.5757731958762886`**. The unconditional minimum validation loss, `0.07517957464653931` at epoch 7, belonged to an ineligible epoch. The checkpoint therefore correctly records `best_epoch=None` and no eligible best validation loss; `best_checkpoint.pt`, selected validation predictions, and selected validation metrics are absent. The manifest status is `FAIL_NO_ELIGIBLE_CHECKPOINT`. This is a within-run checkpoint-selection failure, separate from the fact that A completed training with finite final state.

## I. Why B/C final selection became impossible under the frozen rule

The [candidate-rule audit](stage13c_candidate_rule_after_a_audit.md) establishes that the final B/C MEL-recall comparison requires **selected A validation recall**: `max(Stage9 MEL recall, A MEL recall) + 1/107`. The JSON does not separately define “A unavailable,” but its no-eligible clause says such a variant has no selected checkpoint, and the finalizer requires A's selected checkpoint/validation metrics. A therefore has no legally usable MEL recall for this comparison. Substituting A's epoch-25 recall, another unselected epoch, or the Stage 9-only lower bound `63/107` would be a post-hoc rule. B/C could theoretically complete scientifically informative runs, but **cannot be selected as final candidates under this preregistered objective** without changing the rule. Stage 13C full B and C training were **not performed**; they are not classified as failed full experiments.

## J. Final disposition and finalizer limitation

The frozen rule explicitly says to report **`NO_CANDIDATE_SELECTED` if A is unavailable** and not relax the rule. That condition now applies, so the final candidate is **none**. The existing `train_ablation.py:finalize` does not gracefully write this terminal result: it raises when A lacks a `COMPLETE_VALIDATION_ONLY` manifest and eligible best checkpoint. Running it would only error, not add valid evidence. It was **not run or modified** to manufacture a result. This closure report records the preregistered disposition directly from the audited rule and A artifacts. `selection_rule.json` was neither modified nor reinterpreted.

## K. Epoch-8 resume reproducibility limitation

A was interrupted after epoch 8 and resumed from the validated `last_checkpoint.pt`, continuing at epoch 9 through epoch 25. Model, optimizer, scheduler, GradScaler, history, sampler generator, and saved torch CPU/CUDA RNG states were restored. The epoch-8 checkpoint did **not** store Python or NumPy RNG states. Therefore bit-for-bit equivalence with a hypothetical uninterrupted A trajectory cannot be claimed, even though the first eight history rows were preserved and the completed run is internally consistent. See [resume audit](stage13c_variant_a_resume_audit.md).

## L. Data accessed and boundary maintained

Stage 13/13C training and numerical diagnostics used HAM train, and A checkpoint selection used HAM validation. Frozen split metadata and Stage 9 validation reference/checkpoint hashes were used for integrity and comparison. No HAM test images, predictions, or performance outcomes, and no PH2 images, predictions, or performance outcomes were accessed for Stage 13/13C performance selection. This report performed documentation-only review. No checkpoint, scientific config, numerical policy, or frozen rule was modified; no Stage 14 work began.

## M. Possible future work as a new experiment

A future objective could be separately preregistered **before new B/C results are observed**, with an explicit comparator for the case where a control has no eligible checkpoint, a defined selection/stop rule, and any numerical execution policy fixed in advance. Alternatively, B/C could be studied descriptively without claiming a candidate under the closed Stage 13 rule. Either path is a **new experiment**, not a continuation or rescue of the current candidate-selection objective. Because earlier stages viewed HAM test/PH2 outcomes, final confirmation would require a genuinely untouched external cohort. No such work is authorized or started here.

## Evidence ledger

| Artifact | Path | SHA256 |
| --- | --- | --- |
| Frozen split | `data/splits/split_leakage_aware.csv` | `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` |
| Stage 9 control best checkpoint | `experiments/egvan_reconstruction_controlled_exp1/best_checkpoint.pt` | `60f34ea4cfa6cbf3dce69e8d8d8813af30f33b2f666a4f06340de0` |
| Frozen final selection rule | `analysis/stage13_melanoma_ablation/selection_rule.json` | `a6524cfd87b7a0ceaad4990af3749b5822be95a08d2dc7b9e0b464471230b659` |
| Non-local arithmetic trace | `analysis/stage13_melanoma_ablation/stage13c_nonlocal3_arithmetic_trace.json` | `ae5ef8c540e926813ff91144f16b4f6c5a00059a1e0cae70f886fa822b459876` |
| Selective C batch-7 trace | `analysis/stage13_melanoma_ablation/stage13c_selective_qk_fp32_trace.json` | `723b8cbabb9ed9e84964fc91f002e3039b24ea4fd3242f27786f2bcce99f5ff5` |
| Selective A/B/C bounded gate | `analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.json` | `721eeaf6c64740d7dd8d7cdb06890a2ffa1c1be49c336f4d07033e4d172cdb00` |
| Completed A last checkpoint | `experiments/egvan_melanoma_ablation_exp/stage13c_selective_qk_fp32_runs/A/last_checkpoint.pt` | `4b911c64b1df8da0dbd6d0355961ec73cc78b9c9ec21187842ec7f5049013168` |
| Preserved A epoch-1–8 CSV prefix | first 1,716 bytes of `stage13c_selective_qk_fp32_runs/A/training_history.csv` | `7d54c95b781cab384f7be9e0ab8dc90daf642c6514331657cb16c2fe5589f348` |

The earlier interrupted A epoch-8 checkpoint had SHA256 `981243c5ac5149dcd86275153f8a614298f1d581807dcf22ebed0f45008ff3d8` when audited. It was subsequently superseded in the same run by the epoch-25 `last_checkpoint.pt`; the first-eight-history prefix above is the preserved local continuity check.
