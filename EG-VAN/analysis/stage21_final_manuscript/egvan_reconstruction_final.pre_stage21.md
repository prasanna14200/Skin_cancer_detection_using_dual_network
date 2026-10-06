# Independent EG-VAN Reconstruction for Seven-Class Skin Lesion Recognition: HAM10000 Test and PH² Follow-up

**Manuscript status:** research draft integrating the closed Stage 15–16 experiment. This is an independently reconstructed EG-VAN implementation; it is not a claim of identity with the original authors' code.

## Abstract

**Background:** Aggregate accuracy can obscure melanoma misses in imbalanced skin-lesion datasets. **Methods:** We evaluated an independent, seven-class EG-VAN reconstruction with one preregistered true-melanoma focal-loss candidate. A lesion-isolated HAM10000 split supplied training, validation-based checkpoint selection, and a 1,014-image test partition; a frozen 120-case NV/MEL mapping supported a separate PH² external follow-up. FP16 overflow in one non-local q@k operation was addressed by executing that affinity in FP32 under CUDA automatic mixed precision elsewhere. The selected epoch-16 checkpoint was recovered by exact observable replay, although the missing original weight file precludes byte-level identity verification. **Results:** On HAM10000 test, accuracy was 0.8235, macro F1 0.6996, and macro one-versus-rest ROC-AUC 0.9582. Melanoma recall was 56/107 (0.5234); 41 of 51 melanoma false negatives were predicted as nevus. In the PH² follow-up, accuracy was 0.6667, melanoma recall 11/40 (0.2750), and melanoma-probability ROC-AUC 0.5428. **Conclusion:** The frozen model showed good internal aggregate discrimination but moderate HAM melanoma sensitivity and limited transfer under the evaluated PH² protocol. PH² had prior project use and is external follow-up evidence, not untouched validation.

## 1. Introduction

Skin lesion classification requires class-specific evaluation alongside aggregate discrimination. HAM10000 contains seven common pigmented-lesion categories with substantial imbalance [2]. A seven-class model can therefore show high accuracy while missing a meaningful fraction of melanomas. We reconstructed the EG-VAN architecture described by Saeed and colleagues [1] and evaluated one preregistered melanoma-objective candidate. This manuscript reports the validation-selected model, its HAM10000 test performance, and a separately defined PH² external follow-up. It does not present the reconstruction as the original authors' released implementation or claim clinical readiness.

### 1.1 Related work and study scope

The published EG-VAN combines EfficientNetV2 and a modified ResNet branch with local and non-local attention and multiscale fusion [1]. EfficientNetV2 [4], residual networks [5], non-local operations [6], and focal loss [7] provide the established components from which this implementation was assembled. HAM10000 [2] supplied the seven-class development and internal evaluation data; PH² [3] supplied a separate dermoscopy cohort for the previously defined overlapping-label follow-up. This study reconstructs architectural ideas from [1] under repository-documented assumptions and evaluates a **seven-class HAM10000 task**. The published EG-VAN article reports a nine-class task, so its headline performance numbers are not a matched comparator for the results below. No direct performance ranking or replication claim is made.

## 2. Methods

### 2.1 Data and model

The frozen, lesion-isolated HAM10000 split [2] contained 8,015 training, 986 validation, and 1,014 test images. Image IDs were unique; each lesion ID belonged to only one partition. The split file is `data/splits/split_leakage_aware.csv` (SHA256 `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`). This test partition was withheld from **Stage 15** selection, although an earlier project stage evaluated a different model on the same HAM test partition; it should not be called never-before-accessed project data. The seven diagnosis labels, in model order, were actinic keratoses/intraepithelial carcinoma (AKIEC), basal cell carcinoma (BCC), benign keratosis-like lesions (BKL), dermatofibroma (DF), melanoma (MEL), melanocytic nevi (NV), and vascular lesions (VASC).

The reconstruction paired EfficientNetV2-S [4] with a modified ResNet50 branch [5]. Spatial-context group attention followed the first two retained ResNet stages, and non-local blocks [6] followed the next two. Features from four EfficientNet tap stages and four corresponding ResNet stages entered serial multiscale fusion units; each unit received the preceding fusion output after the first unit. Aligned fusion outputs and both terminal branch features were concatenated before projection, global average pooling, and seven-class classification. The published paper [1] does not fully specify the exact tap subset, carry operator, spatial alignment, or all attention parameters. These were explicit local reconstruction choices documented in the Stage 8B paper-to-code audit, not claims of author-code identity.

HAM images entered the model at 384×384 pixels after the repository's frozen hair-removal, Gray World, and multiscale Retinex preprocessing and ImageNet normalization. The published EG-VAN paper motivates the color-balancing concepts [1]; the exact implementation is defined by this repository's frozen preprocessing code and protocol. Random flips and rotation were restricted to training; validation and test used deterministic resize, tensor conversion, and normalization. Stage 15 training used seed 42, 25 epochs, a physical and effective batch size of 16, weighted replacement sampling with 8,015 draws per epoch, Adamax (learning rate 0.001, weight decay 0.0001), ReduceLROnPlateau, and focal loss [7] (α = 0.25, γ = 2). The registered candidate multiplied focal-loss terms for **true** melanoma examples by 1.2815247721366766 while retaining the original melanoma sampler weight 1.5630495442733532. This was one prespecified candidate, not a parameter sweep. Only HAM training examples updated weights; only HAM validation determined checkpoint eligibility and selection.

### 2.2 Numerical execution and selection

A bounded numerical investigation localized FP16 overflow to the q@k affinity `torch.bmm(q, k)` in `resnet.nonlocal3`. The corresponding FP32 affinity exceeded the FP16 representable range. Only that multiplication was executed in FP32; the surrounding model used CUDA automatic mixed precision. This execution policy addressed numerical stability and is not evidence of improved classification accuracy.

Before test evaluation, an epoch was eligible only if validation melanoma recall was at least 63/107 (0.5887850467289719), melanoma F1 at least 0.5757731958762886, macro F1 at least 0.6509124104985493, accuracy at least 0.8025152129817445, and nevus recall at least 0.917209653092006. These are the exact frozen thresholds in `selection_rule.json`. The lowest common unweighted validation focal loss among eligible epochs determined the checkpoint, with earlier epoch breaking an exact tie. Epoch 16 passed all gates and was frozen before any final HAM test or PH² result was used. The original epoch-16 weight file failed to persist; a bounded recovery from an archived epoch-13 checkpoint exactly reproduced available epoch-14–16 training-history and validation artifacts. The recovered selected checkpoint was independently audited and used for all final evaluations. Its SHA256 is `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`.

### 2.3 Final evaluation

HAM10000 final evaluation used the frozen 1,014-image test partition, deterministic evaluation transforms, seven-class argmax, `model.eval()`, inference mode, and the registered selective-FP32 policy. No threshold was fitted on test predictions. Accuracy and classwise metrics were computed from the fixed argmax labels; one-versus-rest (OVR) ROC-AUC was computed from the saved seven-class probabilities. We report accuracy, balanced accuracy, macro and weighted F1, classwise precision/recall/F1, confusion matrix, and macro/weighted OVR ROC-AUC. Ranking discrimination measured by AUC does not imply the same sensitivity at the fixed argmax decision rule.

The separately defined PH² database [3] contributed 80 common nevi mapped to NV and 40 melanomas mapped to MEL; 80 atypical nevi were excluded under the existing project protocol. Original PH² BMP images were converted to RGB, passed through the frozen hair-removal/Gray World/Retinex preprocessing, encoded and decoded as JPEG in memory to match the HAM representation, then resized to 384×384 and ImageNet-normalized. Predictions into any of the five other HAM classes remained OTHER and counted as incorrect; the mapped confusion matrix consequently has two true rows and three predicted columns. PH² had prior use in the project and is an **external follow-up evaluation**, not untouched external validation. Neither final dataset was used to tune thresholds, preprocessing, checkpoint choice, or the model.

### 2.4 Reproducibility boundary

The final checkpoint, frozen split, Stage 15 preregistration, Stage 16 evaluator source, prediction CSVs, metrics JSONs, manifests, and figure inputs are retained at the paths listed in the repository's final experiment freeze. Both evaluation manifests record SHA256 values for the checkpoint, sources, and outputs. The recovered epoch-16 checkpoint exactly reproduced available epoch-14–16 training-history and validation observables, but the original epoch-16 weight file is unavailable for a byte-level comparison. All final results therefore refer to the audited recovered checkpoint.

## 3. Results

### 3.1 Validation selection

Selected epoch 16 had validation focal loss 0.083874, accuracy 0.819473, macro F1 0.661131, MEL recall 63/107 (0.588785), MEL F1 0.602871, and NV recall 0.939668. These values met all frozen eligibility gates. They are **validation** results used for selection; the subsequent figures and test metrics were not selection inputs.

**Table 1. Selected HAM10000 validation versus held-out HAM10000 test.** The partitions serve different roles; differences are descriptive.

| Metric | Validation (n = 986) | Held-out test (n = 1,014) |
|---|---:|---:|
| Accuracy | 0.819473 | 0.823471 |
| Macro F1 | 0.661131 | 0.699582 |
| MEL recall | 0.588785 (63/107) | 0.523364 (56/107) |
| MEL F1 | 0.602871 | 0.568528 |
| NV recall | 0.939668 | 0.940828 |

### 3.2 Held-out HAM10000 test

On 1,014 test images, accuracy was 0.823471, balanced accuracy 0.675097, macro F1 0.699582, and weighted F1 0.818013 (Table 2). Macro OVR ROC-AUC was 0.958206. The seven-class confusion matrix is shown in Figure 1 and classwise outcomes in Table 3.

**Table 2. Overall seven-class HAM10000 held-out test performance.**

| Metric | Value |
|---|---:|
| Accuracy | 0.823471 |
| Balanced accuracy | 0.675097 |
| Macro precision | 0.739039 |
| Macro recall | 0.675097 |
| Macro F1 | 0.699582 |
| Weighted F1 | 0.818013 |
| Macro OVR ROC-AUC | 0.958206 |
| Weighted OVR ROC-AUC | 0.946609 |

**Table 3. Classwise HAM10000 held-out test performance.**

| Class | Support | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| AKIEC | 40 | 0.571429 | 0.600000 | 0.585366 | 24 | 18 | 16 |
| BCC | 58 | 0.727273 | 0.689655 | 0.707965 | 40 | 15 | 18 |
| BKL | 104 | 0.659091 | 0.557692 | 0.604167 | 58 | 30 | 46 |
| DF | 11 | 1.000000 | 0.636364 | 0.777778 | 7 | 0 | 4 |
| MEL | 107 | 0.622222 | 0.523364 | 0.568528 | 56 | 34 | 51 |
| NV | 676 | 0.893258 | 0.940828 | 0.916427 | 636 | 76 | 40 |
| VASC | 18 | 0.700000 | 0.777778 | 0.736842 | 14 | 6 | 4 |

![HAM10000 seven-class confusion matrix](../analysis/stage16_final_evaluation/figures/ham_confusion_matrix.png)

**Figure 1.** HAM10000 held-out test confusion matrix (n = 1,014). Rows are true classes and columns are predicted classes; counts use the frozen seven-class argmax.

![HAM10000 one-versus-rest ROC curves](../analysis/stage16_final_evaluation/figures/ham_roc_curves.png)

**Figure 2.** HAM10000 held-out test one-versus-rest ROC curves from saved class probabilities. Macro OVR AUC was 0.958206; curves describe ranking and do not replace the fixed-argmax classwise metrics.

### 3.3 Melanoma and classwise errors

The model identified 56 of 107 melanomas (recall 0.523364) and missed 51. Of those 51 false negatives, 41 (80.39%) were predicted NV, equal to 38.32% of all test melanomas. Five melanomas were predicted BKL; the remaining five were distributed across AKIEC, BCC, and VASC. The reverse NV→MEL transition occurred 18 times among 676 nevi. NV recall was 0.940828 and F1 was 0.916427, but melanoma F1 was 0.568528. DF precision was 1.000000 on only 11 examples and should not be overinterpreted.

### 3.4 PH² external follow-up

For the frozen 120-case mapped PH² cohort, accuracy was 0.666667 and balanced accuracy 0.568750. Melanoma precision was 0.687500, recall 11/40 (0.275000), F1 0.392857, and melanoma-probability ROC-AUC 0.542813. Nevus recall was 0.862500. Among 40 true melanomas, 18 were classified NV, 11 MEL, and 11 OTHER (all BKL in the seven-class output). Figure 3 displays the mapped confusion matrix. These PH² metrics belong to a different cohort and label space from the seven-class HAM test.

**Table 4. PH² external follow-up under the frozen NV/MEL/OTHER scoring policy.**

| Metric | Value |
|---|---:|
| Included samples | 120 |
| Accuracy | 0.666667 |
| Balanced accuracy | 0.568750 |
| MEL precision | 0.687500 |
| MEL recall | 0.275000 |
| MEL F1 | 0.392857 |
| NV recall | 0.862500 |
| MEL probability ROC-AUC | 0.542813 |

![PH2 external follow-up confusion matrix](../analysis/stage16_final_evaluation/figures/ph2_confusion_matrix.png)

**Figure 3.** PH² external follow-up confusion matrix for 80 mapped NV and 40 mapped MEL cases. OTHER retains predictions from the five HAM classes outside NV/MEL and counts as incorrect; 80 atypical nevi were excluded.

![Melanoma recall across distinct datasets](../analysis/stage16_final_evaluation/figures/internal_external_melanoma_comparison.png)

**Figure 4.** Descriptive melanoma recall across Stage 15 validation (63/107), held-out HAM10000 test (56/107), and PH² external follow-up (11/40). These datasets and label spaces are not statistically interchangeable.

## 4. Discussion

The frozen reconstruction achieved 82.35% HAM test accuracy, 0.6996 macro F1, and 0.9582 macro OVR AUC. The high ranking-based AUC should be read alongside the fixed-argmax melanoma sensitivity of only 52.34%; these metrics answer different questions, and no post-test threshold was explored. Class imbalance matters because NV contributed 676 of 1,014 test examples. The frequent MEL→NV transition was the dominant melanoma failure pattern; its cause cannot be isolated from the saved predictions alone. This degree of missed melanoma is a substantial barrier to any clinical deployment claim.

Under the mapped PH² external follow-up, melanoma sensitivity was lower at 27.5% and melanoma-probability AUC was 0.5428. These observations support limited transfer in the evaluated external setting, while not establishing a specific causal source of the difference. PH² differs in image source and true-label space; atypical nevi were excluded, and predictions outside NV/MEL were explicitly retained as OTHER. Its 40 included melanomas and prior project use further limit claims of independent external confirmation. The selective FP32 q@k computation resolved a demonstrated numerical overflow but does not establish any classification improvement or explain the melanoma generalization pattern.

## 5. Limitations

HAM10000 class imbalance makes aggregate accuracy less informative for the smaller classes. Development and checkpoint selection used only HAM10000 training and validation; the same dataset family supplied internal testing. Although this test partition was withheld from Stage 15 selection, an earlier repository stage had evaluated another model on it, so it was not untouched at the project level. The final candidate missed 51 of 107 test melanomas, and dermatofibroma support was only 11, limiting subgroup interpretation. No prospective or clinical validation was performed.

The Stage 15 selected checkpoint was recovered from epoch 13 after the original epoch-16 weights failed to persist. Epochs 14–16 reproduced the available history and validation files exactly, but byte identity with the missing original weights cannot be proven; conclusions apply to the audited recovered checkpoint. The local numerical precision policy addressed an observed FP16 overflow and does not validate architecture superiority or melanoma generalization.

PH² has a different label space and acquisition domain, and the frozen overlap protocol excluded 80 atypical nevi. PH² had prior project use and cannot be described as untouched validation. Its 40 melanoma cases and substantially lower melanoma performance are compatible with domain shift but do not establish its mechanism or support broad generalization claims. No threshold adjustment, preprocessing change, checkpoint reselection, or training followed final HAM test or PH² access. This study does not establish clinical readiness or clinical failure.

## 6. Conclusion

The frozen recovered Stage 15 candidate showed good internal seven-class HAM10000 discrimination, with 0.8235 accuracy, 0.6996 macro F1, and 0.9582 macro OVR AUC. Melanoma sensitivity was more limited at 56/107 (52.34%), and 41 of 51 missed melanomas were called nevus. In the separate PH² external follow-up, melanoma sensitivity was 11/40 (27.5%). The findings underscore the need to report class-specific and external-domain outcomes alongside aggregate internal metrics. The closed evaluation did not trigger further model development.

## Supplementary figures

The row-normalized HAM and PH² confusion matrices and HAM classwise F1/recall plots are presented with numbered captions in the [supplementary figure sheet](egvan_supplementary_figures.md). They derive from the same saved prediction and metric artifacts as Figures 1–4.

## References

1. Saeed A, Shehzad K, Ahmed S, Azar AT. *EG-VAN: A Global and Local Attention-Based Dual-Branch Ensemble Network With Advanced Color Balancing for Multi-Class Skin Cancer Recognition*. IEEE Access. 2025. DOI: [10.1109/ACCESS.2025.3561240](https://doi.org/10.1109/ACCESS.2025.3561240). Architecture source; this manuscript reports an independent reconstruction.
2. Tschandl P, Rosendahl C, Kittler H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Scientific Data. 2018;5:180161. DOI: [10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161).
3. Mendonça T, Ferreira PM, Marques JS, Marçal ARS, Rozeira J. *PH²—A dermoscopic image database for research and benchmarking*. 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). 2013:5437–5440. DOI: [10.1109/EMBC.2013.6610779](https://doi.org/10.1109/EMBC.2013.6610779).
4. Tan M, Le Q. *EfficientNetV2: Smaller Models and Faster Training*. Proceedings of the 38th International Conference on Machine Learning. 2021;139:10096–10106. [Publisher record](https://proceedings.mlr.press/v139/tan21a.html).
5. He K, Zhang X, Ren S, Sun J. *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016. [CVF open access](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
6. Wang X, Girshick R, Gupta A, He K. *Non-Local Neural Networks*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018:7794–7803. [CVF open access](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html).
7. Lin TY, Goyal P, Girshick R, He K, Dollár P. *Focal Loss for Dense Object Detection*. Proceedings of the IEEE International Conference on Computer Vision. 2017:2980–2988. [CVF open access](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html).
