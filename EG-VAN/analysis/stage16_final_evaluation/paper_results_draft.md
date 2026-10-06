# 5. Results

## 5.1 Model selection

The Stage 15 candidate used the prespecified true-melanoma focal-loss multiplier and was selected solely from the frozen HAM10000 validation partition. Epoch 16 satisfied every registered eligibility gate and had a common unweighted validation focal loss of 0.083874. Its validation accuracy was 0.819473, macro F1 0.661131, melanoma recall 63/107 (0.588785), melanoma F1 0.602871, and nevus recall 0.939668. The selected checkpoint was frozen before final evaluation. Because the original epoch-16 checkpoint had not persisted, it was recovered by resuming the archived epoch-13 state: epochs 14–16 reproduced the original observable history and validation files exactly. The absent original weight file precludes direct byte-for-byte weight comparison.

## 5.2 Internal HAM10000 test performance

On the held-out seven-class test partition (n = 1,014), the frozen candidate achieved accuracy 0.823471, balanced accuracy 0.675097, macro precision 0.739039, macro recall 0.675097, macro F1 0.699582, and weighted F1 0.818013. Macro and weighted one-versus-rest ROC-AUC were 0.958206 and 0.946609, respectively. These discrimination metrics coexist with lower melanoma recall: 56 of 107 melanomas were classified correctly (recall 0.523364), while 51 were missed. No test-driven threshold or checkpoint adjustment was made.

## 5.3 Classwise performance

Nevus was the largest class (676 images) and had recall 0.940828 and F1 0.916427. Vascular lesions (18 images) had recall 0.777778 and F1 0.736842; basal cell carcinoma (58) had recall 0.689655 and F1 0.707965. Dermatofibroma showed precision 1.000000 and F1 0.777778, but only 11 test examples support these estimates. Actinic keratosis/intraepithelial carcinoma (40) had recall 0.600000 and F1 0.585366; benign keratosis (104) had recall 0.557692 and F1 0.604167. Melanoma precision was 0.622222, recall 0.523364, and F1 0.568528. The classwise values and denominators appear in [Table 2](final_results_tables.md).

## 5.4 Melanoma error analysis

The main melanoma error was classification as nevus: 41 of 51 melanoma false negatives (80.39%), representing 41 of 107 test melanomas (38.32%). Five melanomas were assigned BKL, two AKIEC, two VASC, and one BCC. The reverse NV→MEL transition occurred for 18 of 676 nevi. These are descriptive confusion patterns from fixed argmax predictions; no alternative threshold was tested.

## 5.5 PH² external follow-up

The separate PH² evaluation used the previously frozen overlap mapping: common nevus→NV (80 cases), melanoma→MEL (40), and exclusion of 80 atypical nevi. Predicted HAM classes outside NV/MEL remained OTHER and were scored as incorrect. Among the 120 included cases, accuracy was 0.666667, balanced accuracy 0.568750, melanoma precision 0.687500, melanoma recall 0.275000, melanoma F1 0.392857, nevus recall 0.862500, and melanoma-probability ROC-AUC 0.542813. Of 40 melanomas, 11 were classified as MEL, 18 as NV, and 11 as OTHER. PH² had already been used elsewhere in this project; this result is an **external follow-up**, not untouched external validation.

## 5.6 Generalization across evaluated datasets

Melanoma recall was 0.588785 (63/107) at Stage 15 validation, 0.523364 (56/107) on the held-out HAM test, and 0.275000 (11/40) in the mapped PH² follow-up. The three cohorts differ in sampling and, for PH², true-label space and preprocessing pathway. Their values are therefore descriptive rather than statistically interchangeable. Under the evaluated external follow-up protocol, melanoma sensitivity was substantially lower, indicating limited cross-dataset transfer; this does not by itself establish clinical performance or the cause of the difference.
