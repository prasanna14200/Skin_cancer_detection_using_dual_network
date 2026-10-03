# Stage 13B: Variant B numerical root-cause investigation

## 1. Recursive inventory — direct observation

The **downloaded Colab experiment copy** at `colab_artifacts/egvan_melanoma_ablation_exp/` now contains exactly three files: `A/last_checkpoint.pt`, `B/last_checkpoint.pt`, and `C/last_checkpoint.pt`. Its `B_failed_original_20261003/` directory is absent. The **repository experiment directory** at `experiments/egvan_melanoma_ablation_exp/` has no `A/`, `B/`, `C/`, or `B_failed_original_20261003/` output directories; it contains only the three preregistered config files, training runner, test, and newly prepared diagnostic/audit scripts. No files were copied between these directories.

| Directory | `config.json` | `experiment_manifest.json` | `training_history.csv` | `last_checkpoint.pt` | `best_checkpoint.pt` | `validation_predictions.csv` | validation metrics file | Other generated files |
|---|---|---|---|---|---|---|---|---|
| Downloaded `A/` | No | No | No | **Yes** | No | No | No | None |
| Downloaded `B/` | No | No | No | **Yes** | No | No | No | None |
| Downloaded `C/` | No | No | No | **Yes** | No | No | No | None |
| Downloaded `B_failed_original_20261003/` | Directory absent | Directory absent | Directory absent | Directory absent | Directory absent | Directory absent | Directory absent | None |
| Repository `A/`, `B/`, `C/`, `B_failed_original_20261003/` | All four directories absent | — | — | — | — | — | — | Top-level `config_A/B/C.json`, `train_ablation.py`, `test_ablation.py`, `diagnose_first_epoch.py`, `audit_ac_artifacts.py` |

There is a **separate earlier download** at `colab_artifacts/stage13_failed_B/` with `config.json`, `experiment_manifest.json`, `training_history.csv`, and `last_checkpoint.pt` only. It is the original failed/resumed run identified below. The new downloaded experiment copy is therefore incomplete relative to the folder described in the request. Loose Drive checkpoints were neither searched for nor used.

## 2. Run provenance and hashes

| Artifact | SHA256 | Content-based provenance |
|---|---|---|
| Downloaded `A/last_checkpoint.pt` | `1fb2daf381772d49455ac59345243a7c9c16c9c2eef507c844870aa8f0a70863` | Embedded `variant=A`, preregistered A config, epoch 25, selected epoch 15, loss 0.08285253810857822. |
| Downloaded `C/last_checkpoint.pt` | `a19ac7aaa663c3f1a5a91b4f2c11183cb5a60ccde1b290ae608c1bb668b044bc` | Embedded `variant=C`, preregistered C config, epoch 25, no eligible/selected epoch. |
| Downloaded `B/last_checkpoint.pt` | `9a7f460058b89244e8f795b3f01a0b90bc410e8e5696f4e1cb12d4ea6f6ffd41` | Embedded `variant=B`, epoch **18**, 18 NaN loss rows, no eligible epoch, scaler zero. Its first 18 history rows match the epoch-25 original checkpoint, and all 1,110 floating model tensors plus all 2,298 optimizer tensors match that checkpoint bit for bit. It is the original failed run's pre-resume checkpoint lineage, **not the fresh rerun or bounded diagnostic**. |
| Earlier separate `stage13_failed_B/last_checkpoint.pt` | `2c95e409c15dd4aec411820dbced9a89748a24a437de963d0e6d2ed85ea330f2` | Embedded `variant=B`, epoch 25, 25 saved NaN loss rows; matching manifest says `FAIL_NO_ELIGIBLE_CHECKPOINT`. This is the same original run after resume. |
| Fresh B rerun | **No local artifact to hash** | User supplies an epoch-1 NaN Colab log and says the run began from initialization without `--resume`. No fresh-B folder, manifest, history, or checkpoint is present locally to verify its provenance independently. |

The original B run's `config.json` exactly matches the registered B config; its manifest records 25 epochs and the checkpoint hash above. The epoch-18 copy establishes the pre-resume numerical state independently: 58,295,425 model NaNs, 116,174,818 optimizer NaNs, and GradScaler scale zero were already present. No local filename alone was used to classify either B file. The fresh B result must remain separate from the original; its exact checkpoint state is unknown until its directory is supplied.

## 3. A and C checkpoint audit — directly observed, incomplete overall

The available A/C **last checkpoints** were loaded on CPU without model inference. Their embedded histories each contain epochs 1–25 in order. All embedded train losses, validation losses, recorded metrics, and learning rates are finite. Their embedded configs exactly match `config_A.json` and `config_C.json`; class order is `akiec, bcc, bkl, df, mel, nv, vasc`. Each has 1,277 finite model-state tensors and 2,298 finite optimizer-state tensors. Both GradScaler scales are positive and finite at 131,072. Both schedulers have finite best recorded loss (`A: 0.08000872658686542`, `C: 0.0695801383001507`).

For A, the saved eligibility flags agree with the preregistered gates: epochs **15, 16, 23** are eligible. Epoch **15** has minimum validation loss among them, **0.08285253810857822**, matching the checkpoint's `best_epoch`/`best_validation_loss`. For C, no epoch passes all gates, and `best_epoch=None` is consistent. C's lower overall minimum validation loss does not override the eligibility gates.

**A_STATUS: INCOMPLETE. C_STATUS: INCOMPLETE.** Their `training_history.csv`, run manifests, A's selected `best_checkpoint.pt`, validation predictions, and validation metrics files are absent from this copy. Thus the CSV histories, A's selected checkpoint tensor state, validation prediction IDs against the frozen HAM validation partition, and selected validation metrics cannot be independently checked. C's absent best checkpoint is expected from its embedded zero-eligible history, but the missing independent history/manifest still prevents a full artifact audit. No HAM test or PH2 performance file was opened. The Stage 13 training source constructs only HAM train and validation loaders; historical data-use claims for A/C cannot be independently proven without manifests/logs.

The read-only `experiments/egvan_melanoma_ablation_exp/audit_ac_artifacts.py` is prepared to audit the complete A/C folders when supplied. It checks file hashes, 25-epoch CSV histories, finite model/optimizer/scaler states, selected checkpoint rule, and saved validation IDs/metrics; it does not run inference.

## 4. Both B failures — separate evidence

**Original failed/resumed B:** The separate archived CSV and checkpoint contain NaN train and validation loss from **epoch 1** through epoch 25, no eligible epoch, and no best checkpoint. The epoch-25 model has **58,295,425 NaN floating elements**; Adamax state has **116,174,818 NaN floating elements**; GradScaler scale is **0.0**. The checkpoint cannot support valid continuation. Its embedded history shows failure before the reported interruption. See `stage13b_variant_b_failure_audit.md` for details. **ORIGINAL_B_STATUS: INVALID.**

**Fresh B:** The user-provided log from a new run without `--resume` says epoch 1 already has NaN train/validation loss, zero MEL recall, no eligibility, and no best epoch. That independently reported restart rules out resume as the initial cause. The fresh run's first saved non-finite epoch is **reported as 1**; its checkpoint epoch, tensor finiteness, optimizer state, GradScaler state, and full eligibility history cannot be checked because its files are absent. **FRESH_B_STATUS: INCOMPLETE** pending that separate archive. The downloaded `B/last_checkpoint.pt` is the original epoch-18 lineage, so it must not be substituted for the fresh checkpoint.

The fresh B execution cannot be physically archived here because no fresh B file is present. `colab_variant_b_diagnostic_instructions.md` gives a guarded Colab cell to move the actual fresh `B/` directory to `B_failed_fresh_epoch1_20261003/` and verify its checkpoint SHA256 after the move. The first failed/resumed B download remains untouched.

## 5. Batch-level diagnostic and root-cause boundary

The checked-in A/B configurations differ only in `variant`, `description`, `loss.name`, and `loss.mel_multiplier`. The only active training-path difference is the true-MEL loss multiplier: A **1.0**, B **1.5630495442733532**. Both use the same MEL sampler weight **1.5630495442733532**. A local CPU check of eight 384-pixel HAM **train** batches found finite normalized inputs (range -2.1179039478302 to 2.640000104904175) and valid labels. That used `num_workers=0`; it does not establish the exact T4 AMP batch behavior.

The dedicated `analysis/stage13_melanoma_ablation/diagnose_variant_b_colab.py` delegates to `experiments/egvan_melanoma_ablation_exp/diagnose_first_epoch.py`. It recreates seed-42 B then A initialization and sampler, verifies config isolation, compares initial model and batch hashes, and runs **at most one HAM train epoch per variant**, stopping at the first non-finite quantity. It checks, in order: input, labels, logits, softmax/log-softmax, cross entropy, `p_true`, original focal term, multiplier application, per-example loss, batch loss, AMP-scaled loss, gradients, parameters after step, optimizer state, and scaler scale. A failure record includes variant, epoch/batch, tensor, finite min/max, NaN/Inf counts, class distribution, MEL presence, and AMP scale. No checkpoint or research result is saved. Syntax and CPU helper/config checks passed. **The user reports that the T4 diagnostic has run, but its trace JSON is not present in the repository, downloaded experiment folder, or expected local analysis path; it has not been audited here.** Exact guarded Colab cells are in `colab_variant_b_diagnostic_instructions.md`.

**Direct observation:** NaNs exist from epoch 1 in the original run, and a separately initiated fresh B run is reported to fail in epoch 1. A/C last-checkpoint histories and tensors are finite. The user now reports a bounded T4 A/B trace in which both variants start with identical model/batch/labels, remain finite through scaled loss, then first show non-finite values in `resnet.conv1.weight.grad` on batch 1 **before optimizer step**, at GradScaler scale 65,536. The reported A counts are 216 NaNs/685 Infs; B counts are 295 NaNs/840 Infs. The raw trace JSON is still absent locally, so these precise trace values are user-provided rather than independently read.

**Inference:** Resume and the B-only multiplier are not sufficient initiating causes, because fresh B and unmodified A also show the first-batch failure. The first *recorded* non-finite operation is in scaled backward gradients, not the optimizer step or a recorded forward/loss tensor. The pattern is consistent with CUDA FP16 AMP scaling/backward overflow; lower-scale and full-FP32 controls are required to distinguish that mechanism from broader FP16 backward instability. The original final B checkpoint's corrupted model could explain its validation NaN.

**Unresolved:** independent verification of the missing raw T4 trace; whether gradients are finite with a lower AMP scale or full FP32; the precise internal backward operation that overflows; fresh-B checkpoint condition; A/C selected validation files and full manifests. The training runner does lack a fail-fast non-finite guard, which let the original failed run continue, but that does not explain the initial NaN. `stage13b_amp_diagnostic_report.md` and `diagnose_amp_scaling_colab.py` specify the bounded controls. No preregistered setting or training artifact was changed. Any stabilization change would require a newly named experimental configuration, not a silent B replacement.

## Boundaries and next action

Only this report/diagnostics JSON and diagnostic-only scripts/instructions were written. Existing A, B, C, frozen split, Stage 9 checkpoint, configurations, and selection rule were not modified. No HAM test or PH2 performance data was accessed, and no inference or full training was started locally. Stage 14 was not started.

Supply the **already generated** bounded T4 trace JSON, plus the missing A/C and fresh-B experiment-folder files. Archive fresh B separately if that has not yet been done. Do not run another 25-epoch B experiment. The Stage 13 selection rule remains unchanged.

**DISPOSITION: B_ROOT_CAUSE_UNRESOLVED**
