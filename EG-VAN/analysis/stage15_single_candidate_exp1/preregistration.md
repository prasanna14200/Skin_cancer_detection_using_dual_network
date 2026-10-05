# Stage 15: one midpoint true-MEL objective candidate

**Status: preregistered and prepared; full training not started.** This is a new experiment. Stage 14 remains closed and untouched.

## Single scientific factor

The reference for the *factor change* is the completed Stage 14 training configuration. Change only the true-MEL per-example focal-loss multiplier from `1.5630495442733532` to the exact decimal arithmetic midpoint between it and 1.0:

`(1.0 + 1.5630495442733532) / 2 = 1.2815247721366766`.

The JSON registers that decimal literal; Python stores its nearest binary floating-point value for execution. The multiplier still applies after the ordinary focal term only to samples whose **true** class is MEL. The original sampler MEL weight stays at `1.5630495442733532`. This is one candidate, not a sweep or a fit to Stage 14's validation loss. The hypothesis is exploratory: a smaller MEL objective perturbation may keep a one-case recall gain while reducing the multiclass F1 cost. It may fail; no monotonic trend is assumed.

All other scientific training factors are identical to Stage 14: seven-class Stage 8B EG-VAN architecture, frozen HAM lesion-isolated split, 384×384 pipeline and augmentation, seed 42 and initialization, physical/effective batch 16, 8,015 replacement sampler draws, 25 epochs maximum, Adamax LR 0.001/weight decay 0.0001, ReduceLROnPlateau, and focal alpha 0.25/gamma 2. The common validation focal loss remains **unweighted by MEL multiplier**. There is no gradient clipping, early stopping, control rerun, architecture edit, or sampler change.

## Numerical execution and data boundary

Keep the established `resnet.nonlocal3` q@k affinity `torch.bmm(q.float(), k.float())` in FP32 with CUDA autocast disabled only for that operation, and normal CUDA AMP elsewhere. Use dynamic GradScaler with initial scale 8192. Normal overflows may skip/back off; corrupted model/optimizer state or persistent non-finite loss causes fail-fast stop. The prior bounded selective-FP32 A/B/C gate supports this execution policy at the historical endpoint settings, but it did **not** test the exact midpoint for 25 epochs. A new failure would require investigation; no additional diagnostic chain is preregistered now.

Only frozen HAM train may update weights. Only frozen HAM validation may determine checkpoint eligibility and selection. HAM test and PH2 images, predictions, and metrics are excluded from development and selection. Earlier Stage 12 test/external outcomes are not optimization targets. The archived Stage 9 validation result is a descriptive reference, not a contemporaneous causal control.

## Frozen checkpoint rule

Each epoch must pass **all** of these unchanged Stage 14 gates:

| Gate | Minimum |
| --- | ---: |
| Correctly recalled validation MEL | 63 of 107 |
| MEL recall | 0.5887850467289719 |
| MEL F1 | 0.5757731958762886 |
| Macro F1 | 0.6509124104985493 |
| Accuracy | 0.8025152129817445 |
| NV recall | 0.917209653092006 |

Among eligible epochs, select the **lowest common unweighted validation focal loss**, breaking exact ties by earlier epoch. If none is eligible, report `NO_CANDIDATE_SELECTED`. Never select an ineligible closest or minimum-loss epoch instead.

## Prospective artifact and resume contract

After **each epoch**, save `last_checkpoint.pt` with model, Adamax, scheduler, GradScaler, full history, sampler generator, Python/NumPy/torch CPU/all CUDA RNG states, and the latest compact validation payload. Save `best_checkpoint.pt` only when the current eligible epoch strictly improves eligible validation loss. Save `training_history.csv` and numerical-overflow events.

Also save `validation_epochs/epoch_NNN_predictions.csv` and `validation_epochs/epoch_NNN_metrics.json` for **every** epoch. Predictions include image ID, true class, predicted class, correctness, and seven probabilities. The metric file includes support, TP, FP, FN, precision, recall, F1 for every class; the 7×7 confusion matrix; accuracy, balanced accuracy, macro F1, MEL recall/F1, NV recall, common validation focal loss, eligibility, and failed gates. At completion, copy the selected epoch's validated evidence to `validation_predictions.csv` and `validation_metrics.json` only if a checkpoint was selected. The manifest hashes all generated artifacts. Per-epoch prediction/metric files are compact; no 25-checkpoint archive is planned.

The explicit `--resume --train` path requires an existing last checkpoint and never starts fresh silently. It validates provenance, config, frozen rule, numerical policy, finite state, history, and per-epoch artifact hashes; restores every saved RNG and training state; and resumes at saved epoch + 1. The checkpoint embeds the latest validation payload so an interruption after its atomic save can repair a missing latest epoch artifact without another inference pass. It also permits recovery of a newly eligible best checkpoint if the last checkpoint was saved immediately before it. Earlier missing or altered validation evidence causes a hard stop. The run directory is separate from Stage 14.

## Evidence hashes and expected compute

| Artifact | SHA256 |
| --- | --- |
| Stage 15 `config.json` | `3ad8c205e06cde8203a869be0a6cc9643ed0ca5b6beb553b57d53d4dce9d25f2` |
| Stage 15 `selection_rule.json` | `daf07a1bac3538b5a35ed59121bd5561aba4797a6889e852fd0ca07ad88be735` |
| Stage 15 `numerical_protocol.json` | `f59010b7e2509c0bf1633b7126ff1d8d629afab39b515f57e95012fb78eb0196` |
| Closed Stage 14 manifest | `ac166de17da0f9c41cf9237412ae0355b20b50c495f0fa4c54e94ff465bd1759` |
| Frozen HAM split | `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` |
| Selective-FP32 bounded gate | `721eeaf6c64740d7dd8d7cdb06890a2ffa1c1be49c336f4d07033e4d172cdb00` |

One T4 run of at most 25 epochs requires about 501 HAM train batches and 62 validation batches per epoch. Predictions are collected during the existing validation pass, not an extra pass. A reliable wall-clock estimate is unavailable; full training was not started locally.
