# Ranked EG-VAN+ experiment plan

## Decision boundary

The current 82.3471% HAM result is frozen and test performance is not an experiment-selection signal. The 95% full-cohort target is not demonstrated, and the available evidence cannot estimate whether it is attainable under the lesion-isolated protocol. The experiment sequence below is a set of validation-stage hypotheses, not a forecast or promise. Each candidate changes one factor, uses the existing lesion-isolated train/validation partitions, and leaves HAM test and PH2 unavailable for selection.

| Rank | Controlled experiment | Expected benefit | Evidence and justification | GPU cost | Difficulty | Main risk |
|---|---|---|---|---|---|---|
| 1 | Change only ResNet50 initialization from random to ImageNet pretrained; preserve architecture, sampler, loss, augmentation, optimizer, schedule, seed, and selection gates. | Moderate potential; magnitude unknown. | The current Stage15 ResNet50 is random-initialized while the HAM training set is limited; better initialization is a plausible generalization/convergence hypothesis. The paper's ResNet pretraining status is NOT SPECIFIED IN PAPER, so this is an ablation, not a paper-faithful correction. | High: one full training run. | Low to moderate. | Paper initialization wording is internally ambiguous; transfer weights may not improve the task and alter training dynamics. |
| 2 | Change only the sampler from the Stage15 MEL-weighted sampler to uniform sampling; keep focal alpha/gamma and the MEL loss multiplier fixed. | Uncertain for overall accuracy; may reduce majority-class-to-MEL false positives, but could lower MEL recall. | Current test errors include NV-to-MEL (18) while 41 MEL cases go to NV; the existing upweighting therefore has not resolved the trade-off. This is hypothesis generation from frozen summaries, not causal evidence. | High: one full training run. | Low. | Could worsen melanoma sensitivity or macro F1. Compare the full preregistered validation metric set and preserve MEL floors. |
| 3 | Change only the training augmentation policy to one predeclared, mild image-space policy; validation remains deterministic and unaugmented. | Low to moderate potential. | Training loss continues down while validation loss rises after epoch 7, which supports testing regularization. Paper augmentation is unreproducible (20 transformations in one passage, 13 in another), so do not label a chosen policy paper exact. | High: one full training run. | Moderate. | Color/geometry transforms can distort diagnostic cues; validation selection can overfit if policies are repeatedly revised. |

## Experiments not currently justified

- **Longer training alone:** validation loss is lowest at epoch 7 (0.08127) and reaches 0.08860 at epoch 24 while train loss falls to 0.01691. More epochs alone are not supported as a route to higher held-out accuracy.
- **A claim of paper-faithful preprocessing:** crop geometry, image size, normalization, morphology constants, and augmentation operators are not sufficiently specified. The current pipeline already implements hair removal, Gray World, and Retinex approximately. A preprocessing ablation may be considered only after its single operational change is specified before training.
- **Architecture replacement:** the major named paper modules are present. Exact feature taps, MFF alignment/carry, and some attention parameter choices are ambiguous, but no missing block or measured architecture defect explains the gap.
- **Higher resolution:** there is no current error analysis showing that resolution is the limiting factor; it adds GPU cost and confounds architecture/training comparisons.
- **More aggressive class weighting:** MEL weighting is already present. Any weighting study must preserve fixed validation safety floors and disclose accuracy/sensitivity trade-offs; it is not a credible shortcut to 95% overall accuracy.

## Execution and evaluation rules

1. Before any next training, freeze a one-factor config, source hashes, seed, primary metric, MEL/macro-F1 guardrails, checkpoint-selection rule, and stopping rule. Do not change the test split or Stage16/20 artifacts.
2. Use only train data for gradient updates and the existing validation data for the registered candidate assessment. Validation has already been used repeatedly in project work; treat the comparison as exploratory, not confirmatory.
3. Do not evaluate HAM test or PH2 during this experiment or use either to rank candidates. If the next candidate is advanced, establish a genuinely untouched confirmation cohort before making a new generalization claim. The already exposed HAM test cannot serve as an untouched lockbox for the new candidate.
4. Report full-cohort accuracy together with macro F1, balanced accuracy, classwise recall/F1, and especially MEL sensitivity. Selective coverage is a separate operational analysis and never replaces full-cohort accuracy.
5. Stop if validation gates fail; do not respond with test-driven threshold changes, sample removal, post-hoc label mapping, or repeated unregistered variants.

## Highest-value next experiment

Rank 1, ImageNet initialization of the ResNet50 branch, is the most informative single-factor model experiment because it tests a major current training choice without changing the reconstructed topology. It does **not** have evidence supporting an expected 12.65-point accuracy increase or a 95% result. The next immediate action is to register this train/validation-only comparison; no training is part of Stage22.
