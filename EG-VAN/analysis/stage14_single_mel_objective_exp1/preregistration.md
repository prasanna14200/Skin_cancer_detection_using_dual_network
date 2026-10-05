# Stage 14: one MEL-objective candidate — preregistered before training

**Status:** prepared, not trained. Stage 13 is closed as `NO_CANDIDATE_SELECTED`; this is a new experiment with its own immutable config and validation-only selection rule. No A/B/C sweep or repeated control run is planned.

## Hypothesis and evidence audit

Increasing the training loss contribution of true melanoma examples by a modest, prespecified factor may increase HAM validation MEL recall and MEL F1 while preserving the seven-class accuracy, macro F1, and NV recall floors. The factor is **1.5630495442733532**, the train-count-derived factor already registered as Stage 13 B's single objective perturbation. Its **classification effect is unknown** because Stage 13C full B training was never performed. The prior selective-FP32 16-batch B gate establishes numerical feasibility for the exact B training objective/sampler on that bounded T4 window, not efficacy. Stage 13C A trained 25 finite epochs but had no eligible checkpoint; its highest MEL F1 was 0.5741626794258373, just below the old 0.5757731958762886 gate. This motivates a targeted MEL objective change without adding sampler or architecture changes. There is no train/validation-only evidence requiring an architecture, optimizer, LR, or seed change.

## Single scientific factor and common infrastructure

The reference scientific training configuration is the original A/Stage 9 focal objective and sampler. This candidate changes **only the true-MEL per-example focal-loss multiplier**, from 1.0 to 1.5630495442733532. The underlying focal term remains `0.25 * (1-p_true)^2 * cross_entropy`; the multiplier applies after that term only to true-MEL samples. The training sampler remains the original train-count-derived MEL weight 1.5630495442733532. The candidate's training fields exactly match the audited Stage 13 B config, whose effect was not evaluated in a completed Stage 13C run. The new experiment identity and selection rule are new documentation, not extra training factors.

The same seven-class Stage 8B architecture, 384×384 transforms, frozen HAM split, seed 42 and initialization policy, 25 epochs maximum, physical/effective batch 16, 8,015 replacement sampler draws, Adamax LR 0.001/weight decay 0.0001, ReduceLROnPlateau, and common **unweighted** HAM validation focal loss are fixed. No gradient clipping or early stopping is introduced. The established numerical execution policy computes only `resnet.nonlocal3` q@k affinity in FP32 under otherwise normal CUDA AMP. Its initial dynamic GradScaler scale is 8192, supported by the prior bounded A/B/C selective policy gate. It is infrastructure, not a classification-performance claim.

## Frozen reference, eligibility, and outcome

The only comparator is the archived Stage 9 epoch-15 **HAM validation** reference: accuracy 0.8225152129817445, macro F1 0.6709124104985493, MEL recall 62/107 = 0.5794392523364486, NV recall 0.947209653092006. Because the new policy and training trajectory differ from Stage 9 and no simultaneous control is run, qualifying a candidate is **not causal proof** that the loss multiplier caused an improvement. This one-candidate design favors speed and an interpretable single scientific intervention; a later untouched cohort would be needed for confirmation.

Before training, the candidate checkpoint eligibility is frozen to require **all**:

| Measure | Minimum | Rationale |
| --- | ---: | --- |
| Correctly recalled validation MEL | 63 of 107 | One more than Stage 9's 62 |
| MEL recall | 0.5887850467289719 | Exactly 63/107 |
| MEL F1 | 0.5757731958762886 | Prior Stage 9/13 floor; prevents recall-only precision collapse |
| Macro F1 | 0.6509124104985493 | Stage 9 minus 0.02, previously registered preservation tolerance |
| Overall accuracy | 0.8025152129817445 | Stage 9 minus 0.02 |
| NV recall | 0.917209653092006 | Stage 9 minus 0.03 |

Among eligible epochs, select **minimum common unweighted HAM validation focal loss**, breaking exact ties by earlier epoch. Never use HAM test, PH2, or earlier Stage 12 test/external outcomes. If no epoch qualifies, report `NO_CANDIDATE_SELECTED`; do not select the lowest-loss ineligible epoch or relax any floor. A selected result is an exploratory validation-only candidate. The rule is frozen in `experiments/stage14_single_mel_objective_exp1/selection_rule.json` before training.

## Numerical safety, checkpointing, and resume

The runner reuses the audited Stage 13C batch guard and selective q@k-FP32 policy. It checks inputs, labels, logits, per-example/reduced/scaled loss, model parameters and buffers, optimizer state, GradScaler scale, and validity of optimizer step/skip. A normal GradScaler overflow may skip and back off while state remains finite; events are logged. State corruption, unexplained skips, or an epoch with no optimizer update aborts. Source/config/rule/numerical hashes, frozen split, Stage 9 checkpoint, and the prior bounded B gate are checked before a run.

After **every completed epoch**, the runner atomically saves `last_checkpoint.pt`, writes the full history CSV and numerical-event log, and updates `best_checkpoint.pt` only for a strictly lower eligible validation loss. The checkpoint includes model, Adamax, scheduler, GradScaler, complete history, best selection, sampler generator, **Python RNG, NumPy RNG, torch CPU RNG, and all CUDA RNG states**. Explicit `--resume --train` requires an existing checkpoint, validates state and artifact history, restores all saved states, and begins at `saved_epoch + 1`; it never silently starts fresh. The snapshot from `last_checkpoint.pt` is authoritative if a CSV/event log write was interrupted after an epoch checkpoint. No research artifact is written by `--check`.

Successful completion writes configuration/numerical/rule snapshots, history, checkpoints, selected validation predictions and metrics if eligible, and a hashed manifest. No test or PH2 loader exists in the execution path. The runner refuses a fresh run in an existing output directory.

## Scope and future interpretation

Only HAM train and validation images may be used for this experiment. Frozen split metadata and Stage 9 validation artifacts may be read for preflight and comparison. HAM test/PH2 images, predictions, and metrics are excluded from development and selection. The final output is either one validation-only candidate or `NO_CANDIDATE_SELECTED`; it does not revise Stage 13's terminal decision. Any later test/external confirmation must use an appropriately untouched cohort, because earlier stages viewed HAM test/PH2 outcomes.

**Compute plan:** one T4 candidate run of at most 25 epochs (about 501 HAM train batches and 62 validation batches per epoch). No repeated control or hyperparameter sweep. Saved artifacts do not contain a reliable wall-clock benchmark, so hours are not estimated. The prior selective-FP32 bounded B gate already covers the exact objective/sampler/numerical policy, so another T4 diagnostic is unnecessary before this run unless the read-only preflight or a new numerical failure reveals a problem.

## Immutable evidence and preregistration hashes

| File | SHA256 |
| --- | --- |
| `config.json` | `b9a83c4f281f747fa6dea5a8c5f64732a880452386864dcbf8f38446fb5a0de9` |
| `selection_rule.json` | `41b0b30641458799c9b1c8697a1494bdbfbe4f41f657df17b0f205d36161cb32` |
| `numerical_protocol.json` | `5329ad4a11af4ad6865f4c21ea9ca9c46f28bf4ea601cceff9a230a793d7e6ea` |
| Frozen HAM split | `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` |
| Frozen Stage 9 best checkpoint | `60f34ea4cfa6cbf3dce69e8d8d8813af30f33b2f666a4f06340de0` |
| Prior selective-FP32 A/B/C bounded gate | `721eeaf6c64740d7dd8d7cdb06890a2ffa1c1be49c336f4d07033e4d172cdb00` |
| Imported Stage 13C guarded training helper | `d5cdfbade60e834e56acf9d67fbb3f83d90f0c6f96473dff9b9d3dd4c399ef1a` |

The Stage 13 selection rule remains unchanged and closed; this new rule was registered before any Stage 14 training.
