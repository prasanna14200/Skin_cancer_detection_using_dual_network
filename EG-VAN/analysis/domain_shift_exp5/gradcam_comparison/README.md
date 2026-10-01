# Stage 2 Grad-CAM Comparison

- Run type: exploratory, selected-case PH² follow-up comparison.
- Approved PH² cases completed: 12.
- Reproduction tolerance: 0.0005; gate passed before maps.
- Target layer: `model.features[-1]`.
- Training/fine-tuning: NO.
- Full PH² inference rerun: NO.
- Threshold tuning: NO.
- PH² is treated as an external follow-up dataset and is not used for model selection, hyperparameter tuning, threshold optimization, or training.
- Grad-CAM is an exploratory visualization of model sensitivity and does not establish causal reasoning or clinical validity.

Complete `gradcam_review.csv` only after visually reviewing the saved case images. Do not treat qualitative entries as segmentation metrics.
