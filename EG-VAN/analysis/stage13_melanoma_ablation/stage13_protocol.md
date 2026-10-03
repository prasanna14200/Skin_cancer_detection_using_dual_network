# Stage 13: pre-registered HAM-only melanoma ablation

## Scope and data boundary

This stage prepares three EG-VAN training configurations. No full training has occurred. Training, validation, checkpoint selection and final candidate choice may use only the frozen HAM train and validation partitions. The frozen split SHA256 is `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`; the Stage 9 reference best checkpoint SHA256 is `60f34ea4cfa6cbf3dce69e8d8bcc82292d8d8813af30f33b2f666a4f06340de0`. Test and PH² predictions, metrics, case outcomes and maps are not inputs to the runner or the selection rule. Split metadata is read only to validate lesion isolation and filter train/validation. No test or PH² image loader exists in the Stage 13 training path.

The Stage 9 train partition has 8,015 images; its class counts are AKIEC 257, BCC 398, BKL 891, DF 95, MEL 899, NV 5,366 and VASC 109. Validation has 986 images, including 107 MEL. The existing Stage 9 epoch-15 validation reference is accuracy 0.8225152129817445, macro F1 0.6709124104985493, MEL recall 0.5794392523364486 (62/107), and NV recall 0.947209653092006. These are validation-only anchors, verified against the saved validation predictions.

## Exact Stage 9 loss and sampler audit

Stage 9 computes per-example focal terms `0.25 * (1 - p_true)^2 * cross_entropy(logits, target)`, then takes the batch mean. Alpha is a **single global scalar**, not a class-specific vector. Gamma is 2.0. Stage 9 samples 8,015 train rows with replacement using `WeightedRandomSampler` and a CPU generator seeded 42. Train rows labelled MEL have weight `(5366/899)^(1/4) = 1.5630495442733532`; every other row has weight 1.0. Validation is unweighted and unsampled. Stage 9 checkpoint eligibility requires MEL F1 at least 0.5757731958762886, macro F1 at least 0.6316582381362074 and NV recall at least 0.9; among eligible epochs the lowest validation focal loss wins, with earlier epoch on ties.

## Frozen A/B/C design

| Variant | Training focal true-MEL multiplier | Training MEL sampler weight | Other settings |
|---|---:|---:|---|
| A: fresh control | 1.0 | 1.5630495442733532 | Exact Stage 9 objective/sampler |
| B: objective only | `(5366/899)^(1/4) = 1.5630495442733532` | 1.5630495442733532 | Only true-MEL loss terms change |
| C: sampler only | 1.0 | `(5366/899)^(1/2) = 2.4431238778531372` | Only train MEL sampling weight changes |

A is a fresh run with the same seed and initialization policy, not a warm start from the Stage 9 checkpoint. The archived Stage 9 validation result remains a second reference. B changes one per-example loss multiplier after the original focal term; it retains alpha 0.25 and gamma 2. C retains the exact original loss. All three use the same training augmentation and deterministic validation transform, 384×384 input, seven-class Stage 8B architecture, ImageNet EfficientNetV2S branch initialization, random ResNet50 and new module initialization, Adamax LR 0.001/weight decay 0.0001, ReduceLROnPlateau on **common unweighted validation focal loss**, 25 epochs, CUDA AMP, physical/effective batch 16, sampler replacement and 8,015 draws, Python/NumPy/PyTorch/CUDA seed 42, and cuDNN deterministic/benchmark settings inherited from `src/train.py:set_seed`. No clipping or early stopping is added. These are two single-factor perturbations, not a sweep.

The chosen B/C numerical weights are fixed functions of **train** NV/MEL counts. They do not use any test or external result. B increases the MEL contribution in a batch that is already MEL-oversampled; C increases MEL draw frequency with no loss change. Neither is guaranteed to improve recall.

## Checkpoint and final candidate selection

Within each A/B/C run, retain the Stage 9 eligibility gates and select minimum common **unweighted** validation focal loss among eligible epochs. This keeps scheduler and checkpoint comparison on the same validation objective even for B. If a run has no eligible epoch, it has no best checkpoint. All 25 epoch histories and per-class selected validation predictions/metrics are retained; no test inference is part of this workflow.

After all three runs complete, only B and C can be new candidates. A is the direct control. A candidate must gain **at least one correctly recalled validation MEL case** beyond both the frozen Stage 9 reference and A: MEL recall ≥ `max(0.5794392523364486, A MEL recall) + 1/107`. It must also have accuracy ≥ `0.8225152129817445 − 0.02`, macro F1 ≥ `0.6709124104985493 − 0.02`, NV recall ≥ `0.947209653092006 − 0.03`, and MEL F1 ≥ `0.5757731958762886`. These absolute tolerances allow a small overall validation decrease but reject a clear class/overall collapse. They are design tolerances, not empirically optimized cutoffs.

Among qualifying B/C candidates, select higher MEL recall, then higher MEL F1, macro F1, accuracy, lower common validation loss, and finally B before C. If A is unavailable or neither B nor C qualifies, report `NO_CANDIDATE_SELECTED`; do not relax the rule. The runner verifies all selected validation rows against the frozen validation IDs and recomputes per-class metrics before finalizing. Any selected checkpoint is frozen and hashed; Stage 13 stops before HAM test or PH² evaluation.

## Reproducibility and interpretation

Each future variant directory will contain a configuration snapshot, 25-epoch history, last checkpoint, best eligible checkpoint where available, selected validation predictions/metrics, and manifest with runtime, GPU, git state and hashes. Resume is restricted to the same variant/configuration. The local five-test suite checks focal equivalence, single-factor configurations, selection boundaries, data-loader names, and a synthetic seven-logit forward/backward/optimizer/scheduler step. Local CPU verification is a pipeline check, not a research run.

Any selected variant is exploratory. The HAM test and PH² follow-up have already been viewed in earlier stages, so repeated tuning against them would invalidate clean evaluation. A new untouched external cohort is required for final confirmation. No Stage 14 work begins automatically.
