# Internal Explainability Plan

**Status: PLANNED — NOT RUN**  
**Model:** frozen HAM10000 internal model, Experiment #5 epoch 15  
**Checkpoint SHA256:** `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`

## Existing Files to Reuse

- `experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt` — immutable selected checkpoint.
- `experiments/efficientnetv2s_controlled_exp5/config.json` — model, class order, input transforms, and provenance.
- `experiments/efficientnetv2s_controlled_exp5/validation_predictions.csv` and `test_predictions.csv` — select correct-melanoma, melanoma-FN, melanoma-FP, and non-melanoma case IDs without rerunning evaluation to find them.
- `src/models/baseline_effnet.py` — EfficientNetV2-S construction and seven-class classifier.
- `src/train.py` — frozen 384×384 evaluation transform and ImageNet normalization.
- `src/dataset.py` — frozen HAM10000 split and class-index mapping.
- `src/preprocessing.py` — established processed-HAM pipeline (hair removal, Gray World, multi-scale Retinex); do not reprocess already processed images.

No Grad-CAM/Grad-CAM++ or attention-visualization implementation was found in the searched `src/` and `experiments/` Python sources.

## Planned Case Groups

Use the saved validation predictions to define examples before looking at heatmaps:

- Correctly classified melanoma (`true_label=mel`, `predicted_label=mel`).
- Melanoma false negatives, grouped by predicted label; prioritize `nv` and `bkl` confusion.
- Melanoma false positives, grouped by true label; prioritize `nv` and `bkl`.
- Representative correctly classified `nv`, `bkl`, `bcc`, `akiec`, `df`, and `vasc` cases.

Use a deterministic, documented image-ID ordering and a fixed cap per group. Preserve the full case manifest so the sample is reproducible. Do not select examples based on visually compelling maps.

## Execution and Interpretation

1. Strict-load the hash-verified checkpoint into the repository EfficientNetV2-S model; do not train or modify weights.
2. Generate Grad-CAM for the predicted class and the true class on the prespecified cases, using a documented final convolutional feature block as the target layer and the frozen evaluation transform.
3. Save original image, overlay, class labels, predicted probabilities, target layer, and checkpoint hash together in a case manifest.
4. Qualitatively assess whether highlighted regions overlap lesion tissue, background, borders, hair/artifacts, or color/illumination regions. Use blinded/consistent rubric and retain negative or ambiguous examples.
5. Treat maps as qualitative diagnostics, not proof of causal model reasoning or clinical validity. No test-based model selection, threshold tuning, or changes to the frozen model are allowed.

## Planned New Files

- `src/explainability/gradcam.py` — hook-based CAM implementation and tests.
- `experiments/explainability/run_internal_gradcam.py` — deterministic case selection, frozen-checkpoint loading, heatmap generation.
- `analysis/explainability_plan.md` — this protocol.
- `experiments/explainability/` — generated case manifest, overlays, and execution provenance after approval.

No explainability execution or model inference has been performed for this plan.
