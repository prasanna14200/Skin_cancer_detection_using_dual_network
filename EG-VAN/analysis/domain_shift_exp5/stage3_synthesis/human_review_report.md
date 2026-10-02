# Domain Shift Stage 3 Human Qualitative Grad-CAM Review

## Scope and provenance

This review uses only existing Stage 2 PH2 panels and existing HAM/PH2 comparison panels. No model inference, preprocessing, Grad-CAM generation, or access to additional PH2 source images occurred.

- Frozen Experiment #5 checkpoint SHA256: `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`.
- Frozen HAM split SHA256: `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`.
- Selected epoch: 15.
- Target layer: `model.features[-1]`.
- Reviewed PH2 cases: the fixed 12-case set from `ph2_gradcam_manifest.csv`.

PH2 is treated as an external follow-up dataset and is not used for model selection, hyperparameter tuning, threshold optimization, or training. PH2 had previously been used for preprocessing diagnostics and an earlier model evaluation before Experiment #5. Therefore this analysis is an external follow-up review, not an untouched independent external validation.

PH² is treated as an external follow-up dataset and is not used for model selection, hyperparameter tuning, threshold optimization, or training.

Grad-CAM is an exploratory visualization of model sensitivity and does not establish causal reasoning or clinical validity.

## Review method

The existing six PH2 category panels and four available HAM/PH2 comparison panels were visually inspected. The updated `stage3_case_comparison_reviewed.csv` carries one qualitative record for each PH2 case. It preserves the source row's IDs, classes, probabilities, confidence, paths, categories, and provenance; only the five qualitative fields were populated.

Attention labels describe visible heatmap overlap with the apparent lesion-like region, margins, or surrounding image. No lesion masks were used; neither a localization score nor localization accuracy was computed. Lesion boundaries can be ambiguous. For correctly classified cases, true-class maps were reused from the predicted target, so no target-class difference is assessable.

## Correct melanoma

The two selected cases do not show one uniform pattern. IMD065 has conspicuous hotspots near lower-left and right-side margins, while much of the center is less active. IMD168 has a stronger compact focus over a central-right portion of the visible lesion-like region. These are observations from two selected maps, not evidence of a general correct-melanoma localization pattern.

## Correct nevus

IMD003 shows two compact central foci over the visible lesion-like feature. IMD009 shows a compact central focus, with weaker activation near the lower image edge. Both are described as mainly feature-associated in the case table, while edge spill and lack of masks limit boundary judgments.

## Melanoma → nevus

IMD061 has prominent lower-edge/peripheral activation in both target maps; the true and predicted maps are broadly similar in that sweep, and the apparent lesion boundary is difficult to separate from the surrounding area. IMD063 is more fragmented: the predicted-nevus map emphasizes left-central and upper-right regions, while the true-melanoma map distributes hotspots differently along the right/lower periphery. The two examples therefore differ in map organization, but do not establish why either prediction occurred.

## Melanoma → other

For IMD058, predicted-BKL and true-melanoma maps emphasize different patches around the visible central feature. For IMD085, the predicted-BKL map has a central hotspot, whereas the true-melanoma response is weaker centrally and more peripheral, including at the lower edge. In IMD085, whether the lower response is lesion border or background remains uncertain without a lesion mask.

## Nevus → melanoma

IMD035 shows overlapping central activation for both targets, with the predicted-melanoma map broader than the true-nevus map. IMD045 also has overlapping central activation, but the predicted map is broader/horizontal while the true-class map has a more crescent-like shape. This is a visual difference in target-conditioned sensitivity, not evidence of causal reasoning.

## Nevus → other

IMD010 shows separated emphasis: the predicted-BKL map favors upper-left/central areas, while the true-nevus map shifts toward the right-center and nearby tissue. IMD020 shows similar upper-central emphasis for both classes, with the predicted-BKL map spreading somewhat farther downward than the true-nevus map.

## HAM versus PH2 panels

The existing correct-melanoma panels show central/lesion-associated hotspots in the selected HAM examples, while the two PH2 examples differ from each other: one is peripheral and one is more central. In the selected melanoma-to-nevus comparison, PH2 includes strong lower/peripheral activation, whereas the HAM examples show hotspots nearer visible lesion regions, though still with spatial variation. The selected correct-nevus examples in both domains show activation around visible central features. The nevus-to-melanoma panels show overlapping central activation in both domains, with case-specific differences in extent and hotspot shape.

These panels are not paired images and the HAM examples come from a small, preselected validation set. They support only a descriptive comparison of these selected maps. No HAM Grad-CAM counterpart was selected for PH2 melanoma→other or nevus→other; those categories remain PH2-only in the existing panels.

## Summary counts

- Cases reviewed: 12 (4 correct, 8 incorrect).
- Mainly lesion-associated activation: 5 cases.
- Noticeable (moderate/substantial) border or background activation: 11 cases.
- Misclassified cases with visibly different predicted- and true-target maps: 5 cases; the other three were judged similar or only modestly different under the rubric.
- Remaining explicit uncertainty: 1 case (IMD085, lesion versus lower-edge/background relation). Other entries remain qualitative and are limited by the absence of lesion masks.

## Interpretation and limitations

Across these examples, some correct cases show compact lesion-associated activation, while several errors show fragmented, peripheral, or target-dependent patterns. HAM and PH2 panels can look different in selected examples, but the samples are small, selected, and not image-paired. Appearance of a hotspot does not show that it caused a class prediction.

This review does not establish causal explanation, localization accuracy, clinical validity, or that domain shift caused any error. Grad-CAM is an exploratory visualization of model sensitivity, not a proof of model reasoning. No conclusions should be generalized to the full HAM10000 or PH2 cohorts from these 12 PH2 examples.
