# Experiment #5 Internal Explainability Analysis

## 1. Objective

Qualitatively review the existing Step 3 originals and predicted-class/true-class Grad-CAM overlays for the 24 preapproved Experiment #5 validation cases. No inference or map generation was performed during this analysis. The observations describe visible activation patterns only; they do not establish causal model reasoning.

## 2. Frozen Model and Provenance

- Model: Controlled Experiment #5, selected epoch 15.
- Checkpoint SHA256: `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`.
- Frozen split SHA256: `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`.
- Target layer: `model.features[-1]`.
- Step 3 runtime: Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, Tesla T4.
- Step 3 reports 24 cases completed, 14 correct and 10 incorrect; 24 predicted-class maps and 24 true-class map outputs, with 14 identical-target maps reused.

## 3. Grad-CAM Method

The generated maps use the final spatial feature block, hook-captured activations and gradients, spatially averaged gradients as channel weights, weighted channel summation, ReLU, and per-map min-max normalization. The normalized map is resized to 384 x 384 and overlaid on the processed image. Correctly classified cases use the same target for true and predicted class; the true-class visualization is a reused copy, not an independent second computation.

## 4. Cases Analyzed

All 24 original images and their predicted/true overlays were visually inspected using labeled review sheets made from the existing Step 3 PNGs. No underlying image or heatmap was altered. One case-level observation is recorded for each image in `qualitative_analysis.csv`.

The preapproved categories were preserved: 2 correct melanoma; 2 melanoma-to-nevus false negatives; 2 nevus-to-melanoma false positives; 2 BKL-to-melanoma false positives; 2 each of correct nevus, AKIEC, BCC, BKL, DF, and VASC; 2 BKL-to-nevus errors; and 2 BCC-to-nevus errors.

Review contact sheets are in `figures/`, including `melanoma_correct.png`, `melanoma_false_negative.png`, `nevus_false_positive.png`, `bkl_false_positive.png`, `bkl_to_nevus_errors.png`, `bcc_to_nevus_errors.png`, and two sheets for other correctly classified cases. Larger `detail_*.png` sheets provide a closer view of the same existing images.

## 5. Correct Melanoma Cases

In both correct melanoma examples, activation is concentrated over the visible lesion interior and/or inner border. One is more broadly distributed across the central lesion region than the other. Neither reviewed overlay appears dominated by image corners or clearly separate background regions. With two curated examples and no lesion masks, this supports only the narrow observation that these two maps appear lesion-associated, not a general tendency for correct melanoma predictions.

## 6. Melanoma False Negatives

For both melanoma-to-nevus errors, the predicted-nevus map and true-melanoma map remain in or near the visible lesion. Their strongest subregions shift between targets, with partial rather than exact overlap. The maps do not show a consistent shift from lesion to background in these two examples. This suggests class-target-dependent emphasis within the same broad image region, but does not explain the classification errors.

## 7. Melanoma False Positives

### Nevus to melanoma

Both predicted-melanoma maps emphasize visible lesion regions. The corresponding true-nevus maps also overlap those regions, while changing the relative hotspot location or extent. No obvious ruler, marker, color chart, or image-corner focus is apparent in these overlays.

### BKL to melanoma

Both predicted-melanoma maps emphasize the central visible lesion, with substantial overlap with the true-BKL maps. The predicted maps differ in relative intensity and spatial spread, but are not conspicuously dominated by non-lesion background. These four false positives therefore do not provide clear visual evidence that the melanoma target consistently attends to an obvious external artifact.

## 8. Other Correct and Incorrect Classes

### Correct nevus, AKIEC, BCC, BKL, DF, and VASC

Most maps show compact or moderately broad activation over a central visible lesion/feature. The two examples within a class are not identical: some are compact, while others include more of the lesion margin or nearby tissue. In particular, at least one AKIEC/DF example appears broader or less sharply localized than its paired case. The small per-class counts preclude a class-level localization claim.

### BKL to nevus errors

Both predicted-nevus maps emphasize the visible lesion region. Their true-BKL maps overlap substantially or partially, but spread the activation differently; one prediction appears more concentrated in a subregion than the broader true-class map.

### BCC to nevus errors

Both predicted-nevus and true-BCC maps remain centered on the visible lesion, with shifts in the strongest subregion. Their overlap is partial to substantial rather than clearly disjoint. These maps do not establish why the model preferred nevus.

## 9. Cross-Case Patterns

Across this selected set, the predominant qualitative pattern is lesion-associated activation with varying focus on interior, margin, or both. In the misclassified cases, predicted-versus-true maps usually overlap within the broad lesion region while differing in hotspot placement or extent. The observed differences are generally within-lesion shifts, not unambiguous switches to unrelated image regions.

Correct melanoma examples look lesion-associated, but so do the melanoma false negatives and false positives. These maps therefore do not reveal a simple visual distinction between correct and incorrect melanoma-related predictions in this small sample.

## 10. Artifact / Background Attention

No repeated strong focus on image corners, ruler/marker, or color chart was apparent in the reviewed contact sheets. Some maps extend beyond the apparent lesion into nearby skin, and several are broad enough that lesion spill and surrounding-skin activation cannot be cleanly separated. Hair was not a consistent conspicuous hotspot; fine hair or skin texture cannot be ruled out as contributing locally. Without masks and higher-resolution map-generation layers, absence of an obvious artifact focus is not proof that artifacts were unused.

## 11. Implications for EG-VAN+

The lesion-associated and within-lesion shifts observed here motivate testing whether an explicit attention branch can improve localization consistency or classification robustness. That is a design hypothesis only. It should be evaluated with controlled ablations, frozen splits, appropriate localization annotations where available, and independent evaluation; these Grad-CAM examples do not show that an attention branch will improve performance.

## 12. Limitations

- The 24 images were deliberately selected representative cases, not a random or prevalence-representative sample.
- Groups contain only two cases each; visual patterns cannot support statistical or class-wide claims.
- There are no lesion masks or annotated artifact locations, so focus and overlap judgments are qualitative and observer-dependent.
- `model.features[-1]` produces spatially coarse activations that are resized for display; apparent borders are not precise lesion boundaries.
- Per-map normalization and overlay rendering can change apparent intensity and make direct magnitude comparisons misleading.
- Grad-CAM indicates class-score-associated activation patterns, but does not prove causal reasoning, clinical interpretability, or that a highlighted region was necessary for the prediction.
- No clinical assessment, external validation, PH² access, inference, or model modification was part of this analysis.

## 13. Conclusion

The reviewed maps are predominantly centered on or near visible lesions, including in both correct and incorrect melanoma-related cases. Class-target changes in errors commonly alter the within-lesion emphasis rather than producing clearly disjoint maps. A few maps are broad or uncertain, and no repeated obvious background/artifact focus was identified. The findings are descriptive, limited to these 24 cases, and suitable only to motivate further controlled investigation of EG-VAN+ attention mechanisms.