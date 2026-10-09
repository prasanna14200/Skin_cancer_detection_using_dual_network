# Stage 22 performance-gap audit

## Disposition

**Stage22 audit: COMPLETE; research objective: NOT COMPLETE.** This is a read-only synthesis of the local EG-VAN IEEE Access paper, existing source, registered Stage15 configuration, and frozen Stage16/20 reports. The specified recovery checkpoint is present at `experiments/stage15_single_candidate_exp1/stage15_single_candidate_exp1/recovery_epoch16/best_checkpoint.pt`; its SHA256 matches the supplied `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`. The checkpoint was not loaded. No training, fine-tuning, HAM test inference, PH2 inference, model/threshold change, manuscript edit, submission, or artifact deletion was performed.

## Findings

### Paper result and comparability

The paper's 98.20% result is for an **aggregated nine-class dataset**: seven HAM10000 diagnoses plus malignant and benign labels from an additional ISIC 2017 dataset. The paper separately reports 97.80% for a seven-class experiment. Neither result is directly comparable with the frozen 82.3471% on the project's 1,014-image lesion-isolated seven-class HAM test set: source data, split construction, augmentation, class distribution, and evaluation protocol are not established as equivalent.

Table 2 totals 13,312 originals and 1,332 held-out test images. Reported training and validation totals are 37,883 and 6,686, respectively, both described as containing original and augmented images. Those counts imply approximately 10% test and 85/15 train/validation ratios, but the paper gives no executable split procedure, lesion/patient isolation, or partition IDs. Table 2's augmentation totals produce 46,534 train/validation images, 1,965 more than the stated 44,569. The paper also refers to 20 transformations in one section and 13 augmentation techniques in another. These are reporting/reconstruction uncertainties, not proof of leakage. The seven-class 97.80% should be acknowledged but cannot validate a same-split reproduction.

### Most-supported current limitation

The strongest measured limitation is classwise generalization, especially melanoma confusion with nevi, alongside a train/validation gap. At selected validation epoch 16, accuracy was 81.95%, macro F1 0.6611, MEL recall 63/107 (58.88%), and NV recall 93.97%. Validation loss reached 0.08127 at epoch 7, then was 0.08387 at epoch 16; train loss decreased to 0.01691 by epoch 24 while validation loss was 0.08860. The saved training curve does not support "more epochs alone" as the leading fix.

On the frozen HAM test, accuracy was 82.3471%, macro F1 0.699582, macro ROC-AUC 0.958206, MEL recall 56/107 (52.34%), and MEL F1 0.568528. Class recalls were AKIEC 60.00%, BCC 68.97%, BKL 55.77%, DF 63.64%, MEL 52.34%, NV 94.08%, and VASC 77.78%. Forty-one of 51 melanoma misses were MEL-to-NV. Current evidence supports a minority-class/generalization problem; it does not identify a single causal preprocessing or architecture defect.

### Most-likely explanations for the paper-to-project difference

| Candidate | Evidence level | Evidence boundary |
|---|---|---|
| Different dataset and nine-versus-seven class task for the 98.20% claim | HIGH | Explicit paper datasets and result labels; direction/magnitude of task difficulty is not known. |
| Non-equivalent split and evaluation protocol | HIGH | Paper includes augmented images in train and validation and does not specify lesion isolation; project split is lesion-isolated. No leakage occurrence is proven. |
| Training differences | HIGH that differences exist; MEDIUM that they explain accuracy | Random ResNet initialization, MEL weighting/sampling, physical batch 16, and selection rules differ; isolated effects are unmeasured. |
| Preprocessing differences | MEDIUM for implementation differences; LOW for causal impact | Current method approximates paper hair/color steps; crop is absent and paper constants are missing. No controlled impact estimate. |
| Architecture reconstruction choices | MEDIUM uncertainty; LOW causal impact evidence | Named modules exist; exact feature taps, MFF alignment/carry, and some internal attention policies remain assumptions. |
| Insufficient convergence | LOW | Validation loss worsens after epoch 7 while training loss continues falling; extra epochs alone are weakly supported. |
| FP32 non-local q@k numerical policy | HIGH that policy differs; UNSUPPORTED as accuracy cause | Policy addresses observed FP16 overflow; no accuracy gain is attributed to it. |
| PH2 domain shift as explanation of paper/HAM gap | LOW | PH2 is a two-class mapped follow-up with prior project use; it is not a direct comparator for the paper's result. |

### Quality, uncertainty, and external validation

- **Image quality: PARTIAL.** Stage20 technical proxies are descriptive and computed on processed images. Recommended completion is Option A: prespecified technical-quality labels, blinded independent rating, and a held-out lesion-disjoint quality validation. No gate or cutoff was created here.
- **Uncertainty: COMPLETE WITH LIMITATION.** Preserve the validation-selected predictive-entropy threshold `0.7675495327940953`. HAM retained coverage was 79.39%, retained accuracy 90.93%, and errors captured 59.22%; the latter accuracy is conditional, not full-cohort classifier accuracy. Stage20 is exploratory because Stage16 test outcomes had already been viewed. Confirmatory reliability claims need a new untouched cohort, not threshold retuning.
- **External validation: FOLLOW-UP ONLY.** PH2 findings remain 66.67% accuracy, MEL recall 27.5%, and MEL F1 0.392857 on the included 120 cases; prior project use means this is not untouched validation. BCN20000 (DOI 10.1038/s41597-024-03387-w; 18,946 Barcelona dermoscopic images, eight diagnostic categories plus a test-set OOD class) is the recommended candidate for a future lockbox. It has not been downloaded or evaluated. First verify exact label mapping, local/prior/pretraining use, overlap, and OOD denominators.

## Recommended next experiment

Register one train/validation-only ablation changing only ResNet50 initialization from random to ImageNet pretrained. Preserve the current split, architecture, seed, sampler, focal loss, augmentation, optimizer, schedule, selection rule, and MEL/macro-F1 guardrails. This tests a plausible training gap but is not known to recover a particular fraction of accuracy and does not reproduce an unspecified paper setting. Do not use HAM test or PH2 to select it. If no meaningful validation improvement is shown under the frozen criteria, do not escalate to a broad sweep.

Detailed reports: [paper reconstruction](paper_pipeline_reconstruction.md), [preprocessing crosswalk](paper_preprocessing_vs_ours.md), [architecture matrix](architecture_gap_matrix.md), [training crosswalk](training_protocol_gap.md), [accuracy analysis](accuracy_gap_analysis.md), [quality plan](quality_completion_plan.md), [external validation plan](external_validation_plan.md), [ranked experiments](ranked_experiment_plan.md), and [target pipeline](final_egvan_plus_target_pipeline.md).

## Stage22 checklist

| Check | Result |
|---|---|
| Paper reconstructed with explicit unknowns | COMPLETE WITH LIMITATIONS |
| Preprocessing and architecture compared | COMPLETE; no implementation changes |
| Training protocol and accuracy gap audited | COMPLETE; causal attribution remains unproven |
| Frozen model, classwise results, Stage20 reliability reviewed | COMPLETE from saved reports/metrics |
| Quality objective | PARTIAL; no quality gate created |
| Uncertainty objective | COMPLETE WITH LIMITATION; rule preserved |
| Untouched external validation | NOT COMPLETE; candidate only |
| Full-cohort 95-98% target | NOT ACHIEVED; feasibility under this protocol remains undetermined |
| Training performed | NO |
| HAM test inference performed | NO |
| PH2 inference performed | NO |

**Next action:** preregister the single ResNet50 initialization ablation and run only after Stage22 closure; keep HAM test and PH2 sealed. Before any future external test, establish an independently sourced, non-overlapping cohort and freeze its label/OOD mapping.
