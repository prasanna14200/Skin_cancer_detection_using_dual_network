# PH² External Validation Plan

**STATUS: PLANNED — NOT RUN**  
**PH2_ACCESSED: FALSE**

## Locked Model and Preprocessing

- Source experiment: Controlled Experiment #5.
- Checkpoint: `experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt`.
- SHA256: `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`.
- Selected epoch: 15 under `validation_threshold_constrained_min_loss`.
- Architecture/classes: EfficientNetV2S; `akiec,bcc,bkl,df,mel,nv,vasc`.
- Frozen PH² input pipeline for primary external evaluation: apply the same processed-HAM steps used during training (hair removal → Gray World → multi-scale Retinex), then resize to 384×384, tensor conversion, and ImageNet mean/std normalization. Evaluation only; no random augmentation.
- Do not compare alternative PH² preprocessing conditions during this run. The prior paired preprocessing diagnostic is historical evidence only and will not be used to tune this checkpoint.

## Label Mapping and Cohort

Use the already established overlap policy without label invention:

- Common nevus → `nv`.
- Melanoma → `mel`.
- Atypical nevus → excluded (`null` mapping).

Expected protocol cohort: 120 included cases (80 common nevi, 40 melanomas); 80 atypical nevi excluded. Validate source labels, unique IDs, mapping, and included/excluded counts before inference. This plan does not inspect PH² files now.

## One-Time Evaluation

After explicit approval, freeze the manifest and run the selected checkpoint once. Keep argmax as the prediction rule and retain any other seven-class outputs as `other` for binary-overlap scoring rather than forcing them into `nv` or `mel`.

Report:

- Seven-class prediction distribution and per-class counts.
- Binary confusion matrix: true rows `[nv, mel]`, prediction columns `[nv, mel, other]`.
- Melanoma sensitivity/recall, precision, F1, specificity, and accuracy under the established `other`-counts-as-error policy.
- Melanoma probability ROC-AUC using stored probabilities, with cohort and class definitions stated.
- Per-case predictions/probabilities, checkpoint hash, preprocessing provenance, mapping, and excluded-case counts.

No fine-tuning on PH². No checkpoint reselection, hyperparameter selection, threshold tuning, preprocessing tuning, or repeated development evaluation using PH². Test or PH² outcomes must not be fed back into internal model selection. Report limitations from the small external cohort and distribution shift; do not claim clinical validation.
