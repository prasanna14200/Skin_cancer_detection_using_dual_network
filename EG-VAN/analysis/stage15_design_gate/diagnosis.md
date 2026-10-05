# Stage 15 design gate: Stage 14 validation evidence

**Historical decision for the original class-directed design question: `INSUFFICIENT_EVIDENCE`.** At the time of this audit, no Stage 15 experiment, factor value, runner, or preregistration was created. A later artifact-recovery search confirmed that the missing epoch-specific files do not exist. The subsequent, narrower one-midpoint-factor Stage 15 preregistration is documented separately in [`../stage15_single_candidate_exp1/preregistration.md`](../stage15_single_candidate_exp1/preregistration.md); it does not claim class-specific attribution. Stage 14 remains closed as `NO_CANDIDATE_SELECTED`.

## Evidence and scope

This diagnosis uses only saved Stage 14 validation history and the frozen Stage 9 validation predictions. Stage 14's run folder contains `training_history.csv`, final epoch-25 `last_checkpoint.pt`, configuration/protocol/rule snapshots, numerical events, and a manifest. It contains **no epoch-11, epoch-16, or epoch-22 checkpoint**, no per-epoch validation predictions, no per-epoch confusion matrix, and no selected checkpoint/predictions because no epoch was eligible. The epoch-25 model cannot reconstruct earlier epoch predictions. No inference was run.

The Stage 14 history SHA256 is `33d419d79087954c0b5fdeec89b71dcfd1565b79d0f6742506180005cac2c367`. The Stage 9 validation-prediction SHA256 is `76c0587dd642cac903128b7242f62dc71eed11bb6ae698369f6787625237c9ef`. The frozen HAM split SHA256 is `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. The saved Stage 14 checkpoint is epoch 25 and hashes to `3faa3d1390182e05268ad20e19f4f78f312497b0665b2888d9f7bce9676ed60c`. The [Stage 14 completion audit](../stage14_single_mel_objective_exp1/completion_audit.md) established the run's numerical validity and lack of eligible epochs.

## Stage 9 class-wise validation reference

These counts and metrics are independently reconstructed from the 986 saved Stage 9 validation predictions. They describe the **reference**, not Stage 14's epoch-11/16/22 predictions.

| Class | Support | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| akiec | 30 | 0.560000 | 0.466667 | 0.509091 | 14 | 11 | 16 |
| bcc | 58 | 0.790698 | 0.586207 | 0.673267 | 34 | 9 | 24 |
| bkl | 104 | 0.678571 | 0.548077 | 0.606383 | 57 | 27 | 47 |
| df | 9 | 0.625000 | 0.555556 | 0.588235 | 5 | 3 | 4 |
| mel | 107 | 0.601942 | 0.579439 | 0.590476 | 62 | 41 | 45 |
| nv | 663 | 0.883263 | 0.947210 | 0.914119 | 628 | 83 | 35 |
| vasc | 15 | 0.916667 | 0.733333 | 0.814815 | 11 | 1 | 4 |

Stage 9 confusion matrix, rows = true class and columns = predicted class, both in `akiec, bcc, bkl, df, mel, nv, vasc` order:

| True class | akiec | bcc | bkl | df | mel | nv | vasc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| akiec | 14 | 2 | 7 | 0 | 4 | 3 | 0 |
| bcc | 2 | 34 | 4 | 0 | 0 | 18 | 0 |
| bkl | 5 | 2 | 57 | 2 | 14 | 24 | 0 |
| df | 1 | 0 | 1 | 5 | 1 | 1 | 0 |
| mel | 3 | 4 | 4 | 1 | 62 | 33 | 0 |
| nv | 0 | 1 | 11 | 0 | 22 | 628 | 1 |
| vasc | 0 | 0 | 0 | 0 | 0 | 4 | 11 |

Stage 9 macro F1 is `0.6709124104985493`, accuracy `0.8225152129817445`, MEL recall 62/107, and NV recall `0.947209653092006`.

## What Stage 14's saved history actually identifies

The support for each Stage 14 validation class is fixed by the frozen partition: `akiec=30, bcc=58, bkl=104, df=9, mel=107, nv=663, vasc=15`. For epochs 11, 16, and 22, the history saves MEL precision/recall/F1 and NV recall. It also saves aggregate accuracy, balanced accuracy, and macro F1. It does **not** save the other class-wise precision, recall, F1, FP, or confusion entries. The table marks unknown quantities explicitly.

| Epoch | Class | Support | Precision | Recall | F1 | TP | FP | FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | mel | 107 | 0.535088 | 0.570093 | 0.552036 | 61 | 53 | 46 |
| 11 | nv | 663 | unknown | 0.885370 | unknown | 587 | unknown | 76 |
| 16 | mel | 107 | 0.611650 | 0.588785 | 0.600000 | 63 | 40 | 44 |
| 16 | nv | 663 | unknown | 0.942685 | unknown | 625 | unknown | 38 |
| 22 | mel | 107 | 0.594340 | 0.588785 | 0.591549 | 63 | 43 | 44 |
| 22 | nv | 663 | unknown | 0.929110 | unknown | 616 | unknown | 47 |

For **akiec, bcc, bkl, df, and vasc at each of these three Stage 14 epochs**, only the support values above are known. Their precision, recall, F1, TP, FP, and FN are **not recoverable** from the saved artifacts. No Stage 14 epoch-11/16/22 confusion matrix exists. The MEL false positives above are computable from saved MEL precision and TP, but the true classes of those false positives cannot be determined. The NV true positives and false negatives are computable from its saved recall and support; its false positives and precision are unknown.

The observed epoch-specific aggregate metrics are:

| Epoch | Reason inspected | Accuracy | Macro F1 | MEL recall | MEL F1 | NV recall | Validation loss |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | minimum validation loss | 0.790061 | 0.624722 | 61/107 | 0.552036 | 0.885370 | 0.07159741316557414 |
| 16 | maximum MEL F1 | 0.817444 | 0.643062 | 63/107 | 0.600000 | 0.942685 | 0.07758982457658824 |
| 22 | closest to all gates | 0.815416 | 0.646466 | 63/107 | 0.591549 | 0.929110 | 0.08110617038380151 |

At epoch 22, MEL F1 is **0.0010731052984575 above** Stage 9 MEL F1, while macro F1 is **0.0244466790842984 below** Stage 9. Algebraically, the average F1 of the six non-MEL classes is therefore **0.0286999764814244 below** their Stage 9 average. This proves a **combined non-MEL F1 decline** but does not identify which of the six classes accounts for most of it. The corresponding average non-MEL F1 declines are `0.0340794926278437` at epoch 16 and `0.0474819139798151` at epoch 11. These aggregate identities do not reconstruct class-wise F1 or confusion matrices.

One identifiable partial pattern is that NV recall fell from Stage 9's `0.947209653092006` to `0.9291101055806938` at epoch 22 (628 to 616 correctly recalled NV cases), while remaining above its registered floor. Without NV precision and the other classes' metrics, this does not establish NV as the largest cause of the macro-F1 gap. The Stage 9 confusion matrix cannot be assigned to Stage 14.

## Design disposition

The saved evidence confirms that epoch 22 missed only the frozen macro-F1 gate by `0.0044466790842984`; it does not show **which non-MEL class or error flow** caused that gap. A particular class-directed adjustment, or a numerical value for a reduced MEL multiplier, would therefore be post-hoc guesswork rather than an intervention justified by the requested class-level diagnosis. No Stage 15 scientific change is selected, and no training runner or preregistration is prepared.

If authentic archived epoch-11/16/22 validation predictions or corresponding epoch checkpoints become available with verifiable provenance, they could support a new read-only diagnosis. The current epoch-25 checkpoint cannot supply those earlier predictions. No HAM test or PH2 image, prediction, or metric was accessed; no training or inference was performed.
