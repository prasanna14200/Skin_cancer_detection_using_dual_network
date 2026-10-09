# EG-VAN paper pipeline reconstruction

Source: Saeed et al., *IEEE Access* 13 (2025), 72852–72874, DOI [10.1109/ACCESS.2025.3561240](https://doi.org/10.1109/ACCESS.2025.3561240); local PDF `D:/Cancerdetection/EG-VAN_A_Global_and_Local_Attention-Based_Dual-Branch_Ensemble_Network_With_Advanced_Color_Balancing_for_Multi-Class_Skin_Cancer_Recognition.pdf`, SHA256 `bac1b99167ad820232ecc6b47a3d562d2d9ca55ce19de36fba97b919e8aacee3`. Table 2 was visually inspected. See also the [full-text author publication](https://www.researchgate.net/publication/390831641_EG-VAN_A_Global_and_Local_Attention_based_Dual-Branch_Ensemble_Network_with_Advanced_color_balancing_for_Multi-Class_Skin_Cancer_Recognition).

## Datasets and counts

The paper combines **seven HAM10000 diagnoses** with two additional **ISIC 2017 malignant/benign** labels to make nine output classes (Table 2). Its 98.20% overall accuracy refers to this **aggregated nine-class** task. The paper separately reports **97.80% on its seven-class task** (Table 4 and text), but its seven-class evaluation does not establish equivalence to our frozen lesion-isolated split.

| Class | Source | Original | Test | Original train | Augmented images, Table 2 |
|---|---|---:|---:|---:|---:|
| AKIEC | HAM10000 | 327 | 33 | 294 | 2,206 |
| BCC | HAM10000 | 514 | 51 | 463 | 2,537 |
| BKL | HAM10000 | 1,099 | 110 | 989 | 7,011 |
| DF | HAM10000 | 115 | 12 | 103 | 1,397 |
| MEL | HAM10000 | 1,113 | 111 | 1,002 | 6,998 |
| VASC | HAM10000 | 142 | 14 | 128 | 1,372 |
| NV | HAM10000 | 6,705 | 671 | 6,034 | 0 |
| Malignant | ISIC 2017 | 1,497 | 150 | 1,347 | 6,653 |
| Benign | ISIC 2017 | 1,800 | 180 | 1,620 | 6,380 |
| **Sum** | Combined | **13,312** | **1,332** | **11,980** | **34,554** |

The reported counts imply a 10% original-image test partition (1,332/13,312) and an approximately 85/15 train/validation division of the stated 44,569 post-augmentation images. These are arithmetic implications of the reported counts, not a fully specified split algorithm; exact randomization, stratification, and partition IDs are **NOT SPECIFIED IN PAPER**.

**Internal count discrepancy:** Table 2 implies 11,980 original non-test images + 34,554 augmented = **46,534** potential train/validation images. The methods state **37,883 train + 6,686 validation = 44,569**, a difference of **1,965**. Exclusions, deduplication, and exact partition membership are **NOT SPECIFIED IN PAPER**. The paper says it first held out 1,332 original test images, augmented training data, then divided that into train and validation; it explicitly says both train and validation contained originals and augmented images. Whether transformed siblings of a single original crossed the train/validation boundary is **NOT SPECIFIED IN PAPER**. The exact 20 transformations are **NOT SPECIFIED IN PAPER**. Lesion- or patient-isolated partitioning, cross-dataset duplicate checks, and an executable split manifest are **NOT SPECIFIED IN PAPER**. A held-out original test set is described, but its source-image/lesion overlap with augmented training is **NOT SPECIFIED IN PAPER**. These gaps create leakage *risk*, not proof of leakage.

**Augmentation-count discrepancy:** The augmentation equation refers to 20 transformations, while the later preprocessing discussion says thirteen augmentation techniques were employed on the nine-class dataset. The exact operators and actual executed count are **NOT SPECIFIED IN PAPER**. Table 2 also reports zero augmented images for NV. These statements are preserved as reported and are not reconciled by inference.

## Processing and model

Paper order: hair removal (grayscale morphology, printed top-hat expression, threshold, Telea inpainting) → Gray World and multi-scale Retinex color balancing → crop → resize to model dimensions → augmentation → pretrained EfficientNetV2S plus modified ResNet50 (early SCGA, later embedded-Gaussian Non-Local Blocks) → paired multiscale MFFs → concatenation → GAP → dense softmax. The top-hat prose and equation are internally inconsistent for dark-hair removal; numerical constants, crop geometry, image dimensions, augmentation operators, exact MFF taps and alignment are **NOT SPECIFIED IN PAPER**. See the two crosswalks for implementation details.

## Training and reporting

Paper: focal loss α=0.25, γ=2; Adamax, initial LR=0.001; 25 epochs; batch 32; LR halving after one epoch without validation-loss improvement, and “stop patience” 3. It says validation saves the best result but does not give a reproducible exact checkpoint tie rule. Its reported 98.20% nine-class and 97.80% seven-class numbers must remain paper-reported results; neither is a same-protocol baseline for our 82.3471% held-out seven-class result.
