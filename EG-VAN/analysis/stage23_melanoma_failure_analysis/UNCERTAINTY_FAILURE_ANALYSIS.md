# Melanoma errors and uncertainty

**Inputs:** frozen Stage 23 [validation uncertainty CSV](../stage23_uncertainty_final/validation_uncertainty.csv), selected [validation predictions](../../experiments/stage23_resnet50_imagenet_init_exp1/run/validation_predictions.csv), and [PH2 saved predictions](../stage23_external_validation/run/predictions.csv). IDs, truth labels, argmax, confidence and entropy were verified from the seven-class vectors before subgroup analysis. No score or cutoff was refitted.

| True-MEL subgroup | HAM validation | PH2 mapped follow-up |
|---|---:|---:|
| Correct / false negatives | 70 / 37 of 107 | 13 / 27 of 40 |
| Mean top-label confidence, correct vs miss | 0.7103 vs 0.6708 | 0.5580 vs 0.4964 |
| Median top-label confidence, correct vs miss | 0.7197 vs 0.6892 | 0.5504 vs 0.4909 |
| Mean entropy in nats, correct vs miss | 0.6814 vs 0.7828 | 1.0514 vs 1.1482 |
| Median entropy in nats, correct vs miss | 0.6811 vs 0.7158 | 1.0616 vs 1.1525 |
| Within-true-MEL error AUROC, larger entropy = more likely miss | **0.5927** | **0.5755** |
| Within-true-MEL error AUROC, lower confidence = more likely miss | 0.5718 | 0.6296 |
| False negatives with confidence ≥0.8 / ≥0.9 | 9 / 2 | 1 / 0 |

Higher entropy is associated with error on average, but the **ranking among melanoma cases is weak** in both cohorts. The existing [Stage 23 uncertainty report](../stage23_uncertainty_final/uncertainty_report.md) found all-class error AUROC 0.8349 on HAM validation; that result must not be transferred to melanoma-miss detection. At its exploratory 80% coverage point, 15/37 HAM melanoma misses were reviewed and **22/37 remained retained**. This is retained-subset analysis on the same validation cohort used for model selection, not independent proof of a safe review threshold.

Some HAM melanoma errors are confidently wrong: ISIC_0026296 was predicted NV with maximum softmax 0.9897 and entropy 0.0629 nats; ISIC_0032862 was predicted VASC with confidence 0.9024. The highest-confidence PH2 melanoma miss is IMD284, predicted DF at confidence 0.8093. These are case-level score observations, not clinical probabilities and not proof of lesion appearance. The existing Stage 15/20 entropy cutoff is inapplicable to Stage 23, and no threshold is chosen here.

For correctly classified MEL cases, `p_mel` is the top score; for missed MEL cases it is necessarily below some other class score. Accordingly the lower median `p_mel` among misses (HAM 0.2179; PH2 0.1677) is a definition-linked pattern, not evidence that a new `p_mel` threshold would improve clinical detection. The PH2 subgroup has only 13 correct and 27 missed melanomas and no patient linkage, so point AUROCs are unstable. Cross-cohort uncertainty levels are also affected by different image sources and label compositions. Phase B further showed entropy can fall under harmful synthetic degradation; predictive entropy is not a technical image-quality detector.
