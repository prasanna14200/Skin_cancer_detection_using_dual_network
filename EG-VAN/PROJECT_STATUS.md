# EG-VAN+ Project Status

## Completed

- HAM10000 data preparation and frozen lesion-aware split.
- Leakage-aware EfficientNetV2-S baseline.
- Original PH² external evaluation and paired preprocessing diagnostic (historical; no new PH² access in current stage).
- Controlled Experiments #1–#5, including Experiment #5 recovery and final internal model selection.
- Experiment #5 selected internal checkpoint frozen by its pre-registered validation-constrained minimum-loss rule.
- Saved-prediction internal error analysis.
- Explainability and PH² external-validation protocols prepared.

## Current

- HAM10000 internal model: Experiment #5 checkpoint, epoch 15; SHA256 `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`.
- All three frozen validation criteria passed at the selected checkpoint.
- Model is internally selected only; the selected Experiment #5 checkpoint has not yet undergone its locked PH² external validation.
- No clinical-generalization claim is made.

## Next

- Execute the planned internal Grad-CAM case review after approval.
- Perform the single locked PH² evaluation after explicit approval; do not tune from it.
- Continue EG-VAN+/dual-branch architecture research as a separate stage after the internal/external evaluation review.
- Complete final model comparison and research documentation.
- Deployment: not started.
