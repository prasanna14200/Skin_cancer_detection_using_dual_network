# Final Stage 23 model: evaluation-boundary audit

This audit records **existing project exposure** before any further evaluation of the newly frozen Stage 23 classifier. It did not load an evaluation dataset, run inference, recalculate test metrics, or use test/external outcomes to change checkpoint selection.

## 1. Validation-selected evidence

The final classifier is the original Stage 23 ImageNet-V2 ResNet50 EG-VAN checkpoint at epoch 14, selected from the frozen 986-image HAM validation partition. Stage 23 eligible epochs were 14, 15, and 19; epoch 14 had the lowest common validation focal loss. Stage 24 had no eligible epoch. Stage 25 selected an eligible epoch 12 within its own run but was `MIXED_VALIDATION_RESULT` against Stage 23: small aggregate validation gains came with lower melanoma recall/F1. The melanoma-sensitive final choice to retain Stage 23 was made from these saved validation comparisons. This is **validation-selected evidence**, not proof of superiority on held-out or external data.

## 2. Previously inspected HAM test outcomes

Stage 16 evaluated the **earlier Stage 15 recovered checkpoint** on the HAM10000 held-out test partition and saved final test predictions/metrics in `analysis/stage16_final_evaluation/`. Stage 20 used those saved test probabilities for reliability and selective-review analysis. The Stage 20 audit explicitly records that Stage 16 outcomes had already been viewed before its extension, so Stage 20 is exploratory rather than an untouched confirmation. Stage 22's performance-gap audit reviewed the Stage 16 confusion matrix and classwise failures when planning subsequent development; its ranked plan explicitly says the already exposed HAM test cannot serve as a fresh lockbox for a later candidate.

Stage 23–25 training/eligibility paths load HAM **train and validation** only. Their manifests report no HAM test access in those runs. We found no verified Stage 23 HAM test inference in this freeze. Nevertheless, because prior HAM test results were inspected and influenced the broader research direction, this test partition must **not** be described as untouched for Stage 23. Re-running it would be an exploratory follow-up, not a new independent confirmatory test. None of its prior outcomes was used to alter the frozen Stage 23 checkpoint in this task.

## 3. PH2 and external follow-up

PH2 was evaluated earlier in the repository (`analysis/egvan_ph2_external_followup_exp1/`, Stage 11) and again under the Stage 16 external follow-up protocol for the older Stage 15 checkpoint. Stage 20 also analyzed saved PH2 probabilities. The project records explicitly state PH2 had prior use; it is **external follow-up evidence**, not untouched external validation. Stage 23–25 training/selection did not access PH2 in their recorded paths, and this freeze made no new PH2 inference. Prior PH2 evidence cannot be repurposed as untouched confirmation for Stage 23.

## 4. Is any untouched evaluation cohort verified?

**None is verified in the current repository evidence.** Stage 22 discussed independent external cohorts as future possibilities, but a proposed dataset is not a frozen, audited, non-overlapping evaluation set. Before any confirmatory generalization claim, a separate cohort would need a prespecified acquisition/overlap check, label mapping, exclusion policy, endpoint, and locked analysis. This audit does not create or authorize such an evaluation.

## Boundary conclusion

Freeze Stage 23 on validation evidence only. Treat Stage 16/20 HAM test findings and PH2 findings as **previously exposed project evidence**. Do not use them to choose a different checkpoint, tune preprocessing or thresholds, or claim prospective confirmation. The freeze does not reset the evaluation boundary.

Evidence inspected: `analysis/stage16_final_evaluation/final_evaluation_audit.md`, `analysis/stage20_reliability_completion/final_stage20_audit.md`, `analysis/stage22_performance_gap/stage22_audit.md`, `analysis/stage22_performance_gap/ranked_experiment_plan.md`, the Stage 23/24/25 validation manifests/rules, and Stage 11 PH2 follow-up report. No test or PH2 prediction file was read for this boundary audit.
