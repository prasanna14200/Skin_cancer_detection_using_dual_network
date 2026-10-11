# Stage 23 frozen classifier: PH2 external follow-up

## Question and design

The frozen validation-selected Stage 23 epoch-14 EG-VAN classifier was evaluated once on the preregistered PH2 mapped cohort. The [protocol](../STAGE23_EXTERNAL_VALIDATION_PROTOCOL.md) fixed the checkpoint, preprocessing, CUDA T4 batch-16 FP16 execution with selective FP32 `resnet.nonlocal3` q@k, class order, cohort and endpoints before Stage 23 PH2 predictions. The checkpoint SHA256 is `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`. PH2 was **previously used elsewhere in the project**, so this is an **external follow-up**, not independent untouched validation. No model, threshold, preprocessing or label mapping was selected from these outcomes.

PH2 has common nevus, atypical nevus and melanoma diagnoses, unlike HAM10000's seven-class task. The frozen mapping includes 80 common nevi as NV and 40 melanomas as MEL; 80 atypical nevi are excluded. The seven-class model still predicts all seven HAM classes. Predictions outside NV/MEL remain OTHER and count as incorrect for mapped accuracy and class recall. All 120 included case IDs and 200 source image hashes passed the [reproducibility audit](PH2_FOLLOWUP_REPRODUCIBILITY_AUDIT.md).

## Verified outcomes

The [saved prediction file](../run/predictions.csv) yields this true-row/predicted-column matrix:

| True | NV | MEL | OTHER | Total |
|---|---:|---:|---:|---:|
| NV | 68 | 5 | 7 | 80 |
| MEL | 13 | 13 | 14 | 40 |

**Primary melanoma recall is 13/40 = 32.5%** (Wilson 95% CI 20.08–47.98%). Mapped accuracy is 81/120 = 67.5% (58.69–75.22%), and mapped balanced accuracy is **58.75%** (descriptive stratified image-ID bootstrap interval 50.63–66.88%). Exact-label NV recall is 68/80 = 85.0% (75.59–91.21%). Binary melanoma specificity among the 80 true NV cases is **75/80 = 93.75%** (86.19–97.30%): an OTHER prediction is binary negative, although it remains wrong in mapped accuracy. MEL precision is 13/18 = 72.22%, MEL F1 is 0.4483 (bootstrap interval 0.2800–0.6001), and the unmodified MEL probability ROC AUC is 0.5859 (bootstrap interval 0.4653–0.7053). All figures and registered intervals were independently recomputed from the manifest-protected output; full precision and interval conventions are in the [results table](PH2_FOLLOWUP_RESULTS_TABLE.md).

## Melanoma errors and uncertainty

Of 40 melanomas, **27 were not classified MEL**: 13 were classified NV and 14 OTHER. The OTHER predictions comprise 9 BKL, 4 DF and 1 AKIEC. Thus 13/27 (48.15%) of melanoma misses became NV, while 14/27 (51.85%) became another HAM class. Five of 80 true nevi were classified MEL; seven more were classified OTHER (six BKL, one DF). The melanoma false negatives have mean top-label softmax score approximately 0.496 and mean entropy 1.148 nats; the 13 correct melanoma classifications have mean score 0.558 and mean entropy 1.051 nats. One melanoma false negative has confidence at least 0.8, and none at least 0.9. These softmax outputs are **not calibrated clinical probabilities**. Overall entropy error-detection AUROC is 0.7610 on this small mapped cohort; it does not validate a referral threshold.

## Relation to Stage 23 internal validation

Stage 23's selected HAM validation involved **986 images**, seven classes, 107 melanomas and 663 nevi. Its accuracy was 82.96%, seven-class macro F1 0.6713, melanoma recall **70/107 = 65.42%**, and NV recall **625/663 = 94.27%**. On the restricted PH2 follow-up, melanoma recall is **13/40 = 32.5%**, a descriptive difference of **−32.92 percentage points**; NV exact-label recall is 85.0%. HAM's seven-class balanced accuracy/macro F1 and PH2's mapped two-class balanced accuracy are **different endpoints** and should not be ranked as though they measured the same task. PH2 has a different source, restricted label mapping, different class proportions and smaller melanoma denominator. The lower PH2 melanoma recall is evidence of limited transfer under this specific follow-up protocol; no acquisition or biological mechanism is isolated and it does not establish clinical failure.

## Limits and conclusion

PH2 had prior project use, patient identifiers are unavailable, the 80 atypical nevi were excluded, and interval estimates resample image IDs rather than verified patient clusters. Forty melanomas give a wide Wilson interval. The external cohort is not an independent confirmatory set. The result does not justify retraining, checkpoint replacement, threshold adjustment, or a claim of clinical readiness. A new untouched, appropriately labelled and patient-linked cohort would be required for independent confirmation.

**Conclusion:** The frozen Stage 23 classifier identified 13 of 40 melanomas in the mapped PH2 follow-up. This is materially lower than its 70/107 melanoma recall on the distinct HAM validation cohort, while binary melanoma specificity among PH2 common nevi was 75/80. These findings should be reported as qualified external follow-up evidence with the mapping and prior-use limitations explicit.
