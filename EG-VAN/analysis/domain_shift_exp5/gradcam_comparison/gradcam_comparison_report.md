# Domain Shift Stage 2: HAM–PH² Grad-CAM Comparison

## Status

The fixed 12-case PH² subset passed the saved-prediction reproduction gate on the Colab Tesla T4 before Grad-CAM generation. This is exploratory visual review only; no causal interpretation has been made. The per-case review table intentionally remains `UNCERTAIN` until a human inspects the generated images.

PH² is treated as an external follow-up dataset and is not used for model selection, hyperparameter tuning, threshold optimization, or training.

Grad-CAM is an exploratory visualization of model sensitivity and does not establish causal reasoning or clinical validity.

## Frozen model and method

- Experiment #5 checkpoint SHA256: `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a` (epoch 15).
- Frozen split SHA256: `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`.
- Target layer: `model.features[-1]`.
- Cases: exactly the 12 fixed IDs; no other PH² images were passed through the model.
- Reproduction uses the original evaluator batch shape (16); 4 batch slots repeat already-approved inputs and their duplicate outputs are discarded.
- PH² input: existing HAM-matched preprocessing helper, then the same 384×384 ImageNet-normalized transform. No masks, ROI crops, augmentation, or threshold changes.
- Grad-CAM: existing `src/explainability/gradcam.py` utility and HAM overlay renderer. Correct cases reuse one target map for true and predicted labels.

## Reproduction gate

- Required absolute per-class probability tolerance: 0.0005.
- Cases checked: 12.
- Top-1 matches: 12/12.
- Maximum absolute probability difference: 0.
- Mismatches: 0.

## Case maps and review

Each `ph2_cases/<image_id>/` directory contains the processed 384×384 input, predicted-class heatmap/overlay, and true-class heatmap/overlay. Correct cases use byte-identical copies for true/predicted targets and are marked in `ph2_gradcam_manifest.csv`. `gradcam_review.csv` is a blank qualitative review worksheet; the runner does not label attention regions automatically.

Compare these selected maps with existing HAM maps only descriptively and only where case/outcome groups are comparable. No lesion masks are used to calculate localization accuracy. The visualization does not establish causal reasoning or clinical validity.
