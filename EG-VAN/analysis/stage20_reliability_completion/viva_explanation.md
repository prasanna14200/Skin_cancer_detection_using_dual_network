# EG-VAN+ viva explanation

## Problem

EG-VAN combines EfficientNetV2S and an attention-enhanced ResNet50 to classify dermoscopic lesions. A classification score alone does not tell us whether an image is technically usable, whether a particular prediction is uncertain, or whether performance transfers to another dataset. Those are the three reliability questions behind EG-VAN+.

## Proposed EG-VAN+

We proposed technical image-quality screening, uncertainty-aware review, and external follow-up evaluation around the classifier. The classifier itself uses two branches, local and non-local attention, multi-scale fusion and a seven-class output. Its checkpoint remains frozen. The original paper's approximately 98.20% accuracy belongs to the paper's different reported task; our seven-class held-out HAM accuracy is 82.35%, so the figures should not be directly compared.

## What we measured

The repository measures eight technical quality proxies, including brightness, contrast and sharpness. In the final model's validation set, low-sharpness images had a 22.19% error rate versus 14.33% in the middle sharpness tercile. This is an association only; we have no expert quality labels or validated threshold for rejecting images. The quality gate is unfinished.

The final model saved seven probabilities for each validation, HAM and included PH² image. Using validation only, we compared confidence and entropy as error-detection scores and selected entropy. We set a threshold to retain about 80% of validation images. On HAM, the unchanged threshold retained 805 of 1,014 images and flagged 106 of 179 model errors. It flagged 23 of 51 missed melanomas, so it did **not** catch every dangerous-looking error. We also examined calibration with ECE, Brier score and reliability diagrams; these describe probability behavior but do not make softmax outputs clinically reliable.

On the full HAM test, the unchanged model achieved 82.35% accuracy, 0.6996 macro F1 and 0.9582 macro one-versus-rest AUC. It recognized 56 of 107 melanomas. In the PH² mapped follow-up, it recognized 11 of 40 melanomas; 80 atypical nevi were excluded. PH² had earlier project use, so this is external follow-up rather than untouched external validation. The melanoma drop indicates limited transfer in this evaluated setting, not a proven cause or clinical failure.

An earlier baseline has Grad-CAM examples. We have prepared eight fixed final-model examples for a bounded GPU explanation pass, but those maps have **not** yet been generated or reviewed. The final EG-VAN+ pipeline is therefore partly demonstrated: classification, calibration analysis and exploratory review are evidenced; a validated pre-image quality gate and final-model explanations remain pending. We have not retrained, altered predictions, or tuned a rule on HAM test or PH². Since Stage 16 results were already known when Stage 20 began, the new reliability analysis is explicitly exploratory.
