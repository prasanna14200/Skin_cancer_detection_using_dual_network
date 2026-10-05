# Stage 14 single MEL-objective experiment: completion audit

**Disposition: `NO_CANDIDATE_SELECTED`.** This is a validation-only result. The downloaded run completed 25 epochs and remained numerically finite, but no epoch met every frozen eligibility gate. No checkpoint is selected. This audit made no changes to training artifacts or thresholds and did not run inference.

## Scope and provenance

- Experiment: `experiments/stage14_single_mel_objective_exp1/run/`.
- Frozen configuration, rule, and numerical protocol match their run snapshots and the completed checkpoint exactly. Their SHA256 values are respectively `b9a83c4f281f747fa6dea5a8c5f64732a880452386864dcbf8f38446fb5a0de9`, `41b0b30641458799c9b1c8697a1494bdbfbe4f41f657df17b0f205d36161cb32`, and `5329ad4a11af4ad6865f4c21ea9ca9c46f28bf4ea601cceff9a230a793d7e6ea`.
- The frozen HAM split hashes to `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. The Stage 9 reference checkpoint hashes to `60f34ea4cfa6cbf3dce69e8d8bcc82292d8d8813af30f33b2f666a4f06340de0`. The prior bounded selective-FP32 gate hashes to `721eeaf6c64740d7dd8d7cdb06890a2ffa1c1be49c336f4d07033e4d172cdb00`.
- `last_checkpoint.pt` SHA256: `3faa3d1390182e05268ad20e19f4f78f312497b0665b2888d9f7bce9676ed60c`. The run manifest SHA256 is `ac166de17da0f9c41cf9237412ae0355b20b50c495f0fa4c54e94ff465bd1759`. Every artifact hash inside the manifest matches its downloaded file. The saved runner hash `ec52468e26b36241a794955b4e2ad2d4c3438959cbb69da20d4ce2e767ba8845` matches the current runner.
- The checkpoint records epoch 25 and contains 25 history rows; the CSV contains the same 25 rows, numbered exactly 1–25 with no duplicate or gap. Epochs 1–4 and 5–25 form a continuous recorded history. The reported resume line (`saved_epoch=4 next_epoch=5 scale=16384`) agrees with this sequence and the runner's explicit resume behavior. The downloaded folder does not include the epoch-4 checkpoint or execution log, so an independent bit-for-bit reconstruction of the interruption boundary is unavailable. The final checkpoint does include the complete history and all specified RNG states.
- The checkpoint has `best_epoch=None` and `best_validation_loss=+Inf`, the runner's sentinel for no eligible epoch. The manifest reports `NO_CANDIDATE_SELECTED`. `best_checkpoint.pt`, selected `validation_metrics.json`, and selected `validation_predictions.csv` are absent, as required for this outcome; their absence is not a download defect.

## Configuration and numerical state

The frozen setup has seven classes in order `akiec, bcc, bkl, df, mel, nv, vasc`; 384-pixel images; seed 42; batch 16; 25 epochs; Adamax with LR 0.001 and weight decay 0.0001; ReduceLROnPlateau; focal alpha 0.25 and gamma 2; and a true-MEL loss multiplier of **1.5630495442733532**. The original MEL sampler weight is retained. The numerical protocol is the same selective FP32 `resnet.nonlocal3` `torch.bmm(q.float(), k.float())` affinity under otherwise normal CUDA AMP, with dynamic GradScaler initially 8192. The source installs this policy for the training model; the checkpoint and run snapshots identify the policy, though a checkpoint alone cannot prove the arithmetic dtype of every executed operation.

All 25 train losses, validation losses, and recorded validation metrics are finite. All 1,277 model-state tensors are free of NaN/Inf, including 501 BatchNorm running-statistic/counter entries. All 766 populated optimizer slots are free of NaN/Inf. The final scheduler state is internally consistent: `last_epoch=25`, mode `min`, factor 0.5, patience 1, best validation loss `0.07159741316557414` (epoch 11), and last LR `3.90625e-06`, matching Adamax. The GradScaler state is valid with scale **65536**, growth factor 2, backoff factor 0.5, growth interval 2000, and nonnegative growth tracker 1676.

The checkpoint contains Python, NumPy, torch CPU, one CUDA, and sampler-generator RNG states. The Python, NumPy, torch, and sampler state structures can be restored by their corresponding local APIs; the CUDA state is a nonempty byte tensor for the recorded single T4. These are final-epoch states. The independent epoch-4 RNG values were not downloaded, so exact replay of the resume boundary cannot be established from the final folder alone.

The history records **12,523 optimizer updates** and two recoverable GradScaler skips: epoch 21 batch 352 backed scale off `262144 -> 131072`, and epoch 22 batch 328 backed it off `131072 -> 65536`. The checkpoint and `numerical_events.json` agree. Neither event produced persistent model or optimizer corruption; training subsequently completed through epoch 25. The final epoch has train loss `0.015026059741629561` and validation focal loss `0.08270047829272963`.

## Independent eligibility recomputation

Frozen gates: MEL recall at least **63/107** (`0.5887850467289719`), MEL F1 at least `0.5757731958762886`, macro F1 at least `0.6509124104985493`, accuracy at least `0.8025152129817445`, and NV recall at least `0.917209653092006`. Each history row's saved `eligible=False` agrees with independent recomputation. The following abbreviations list every failed gate; an epoch with just one failed gate is near qualifying, but remains ineligible.

| Epoch | MEL correct / 107 | Failed gates |
| ---: | ---: | --- |
| 1 | 73 | MEL F1, macro F1, accuracy, NV recall |
| 2 | 37 | MEL recall, MEL F1, macro F1, accuracy |
| 3 | 64 | MEL F1, macro F1, accuracy, NV recall |
| 4 | 52 | MEL recall, MEL F1, macro F1, accuracy |
| 5 | 60 | MEL recall, MEL F1, macro F1, accuracy, NV recall |
| 6 | 45 | MEL recall, MEL F1, macro F1, accuracy |
| 7 | 44 | MEL recall, MEL F1, macro F1, accuracy |
| 8 | 63 | MEL F1, macro F1, accuracy, NV recall |
| 9 | 61 | MEL recall, MEL F1, macro F1, accuracy |
| 10 | 58 | MEL recall, MEL F1, macro F1, accuracy |
| 11 | 61 | MEL recall, MEL F1, macro F1, accuracy, NV recall |
| 12 | 57 | MEL recall, MEL F1, macro F1, accuracy |
| 13 | 61 | MEL recall |
| 14 | 67 | MEL F1, macro F1, accuracy |
| 15 | 64 | macro F1, accuracy |
| 16 | 63 | macro F1 |
| 17 | 59 | MEL recall, MEL F1, macro F1 |
| 18 | 65 | macro F1 |
| 19 | 60 | MEL recall, macro F1 |
| 20 | 61 | MEL recall |
| 21 | 57 | MEL recall, macro F1 |
| 22 | 63 | macro F1 |
| 23 | 59 | MEL recall, macro F1 |
| 24 | 56 | MEL recall |
| 25 | 60 | MEL recall |

No epoch is eligible. The minimum observed validation loss (`0.07159741316557414`, epoch 11) cannot substitute for an eligible checkpoint. Its MEL recall was only 61/107 and four other gates also failed. There is no selected checkpoint to hash or selected-prediction file to validate.

## Descriptive extrema and nearest miss

- Highest MEL recall: **73/107 = 0.6822429906542056**, epoch 1. At that epoch MEL F1 `0.454828660436137`, macro F1 `0.3868221462450477`, accuracy `0.6845841784989858`, and NV recall `0.8009049773755657` all missed their floors.
- Highest MEL F1: **0.6**, epoch 16, with exactly 63/107 MEL recalled. Its macro F1 `0.6430619610352275` missed the `0.6509124104985493` floor.
- Lowest validation loss: **0.07159741316557414**, epoch 11, ineligible.
- Highest macro F1: **0.6599617188348142**, epoch 13; MEL recall was only 61/107.

The preregistration does not define a “closest” epoch. For descriptive reporting only, choose among epochs failing the fewest gates, then use the smallest *relative shortfall* from the sole failed gate. This gives **epoch 22**: MEL recall **63/107**, MEL F1 `0.5915492957746479`, accuracy `0.8154158215010142`, NV recall `0.9291101055806938`, macro F1 `0.6464657314142509`, and validation loss `0.08110617038380151`. Its **only** failed gate is macro F1, short by `0.0044466790842984`. This descriptive ranking does not select a checkpoint or alter the frozen rule.

## Frozen Stage 9 comparison and scientific interpretation

Stage 9 validation accuracy was `0.8225152129817445`, macro F1 `0.6709124104985493`, MEL recall **62/107**, and NV recall `0.947209653092006`. Epoch 22 recalled one additional MEL case (63/107), with accuracy lower by about `0.00710` and NV recall lower by about `0.01810`; both remained above their registered floors. Its macro F1 was lower by about `0.02445`, exceeding the allowed 0.02 decline. Other epochs achieved higher MEL recall, but failed one or more preservation gates. In particular, the maximum-recall epoch had large seven-class performance losses.

Thus the run shows **validation-only melanoma-sensitivity signals at some epochs**, but **does not demonstrate the preregistered joint objective** of improved MEL recall with acceptable seven-class performance. No candidate is selected. A causal claim that the multiplier itself helped is also unsupported: the reference is an archived Stage 9 run, while this run uses a different numerical execution policy and training trajectory and has no contemporaneous control.

The runner and imported data builder construct HAM train and validation loaders only. The manifest records no HAM test or PH2 use; this audit read no HAM test or PH2 image, prediction, or performance file. No new experiment was designed or started.
