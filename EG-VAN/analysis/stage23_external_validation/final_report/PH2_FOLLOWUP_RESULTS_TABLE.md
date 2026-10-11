# Stage 23 PH2 external follow-up: verified results table

**Cohort and scope.** Frozen Stage 23 epoch-14 classifier; 120 included PH2 cases, comprising 80 common nevi mapped to NV and 40 melanomas mapped to MEL. All 80 atypical nevi were excluded by the frozen pre-inference mapping. PH2 was previously used in the project: this is **external follow-up**, not untouched independent external validation. Values below were recomputed from the [saved prediction CSV](../run/predictions.csv) and checked against the [saved metric JSON](../run/metrics.json); [audit code](../audit_saved_followup.py) and [machine-readable audit](recomputed_audit.json) reproduce them. No threshold was fitted.

## Fixed mapped confusion matrix

True rows; predicted columns. `OTHER` contains any of the five HAM outputs outside NV/MEL and counts as incorrect in mapped accuracy and exact-label recall.

| True class | Predicted NV | Predicted MEL | Predicted OTHER | Total |
|---|---:|---:|---:|---:|
| NV | 68 | 5 | 7 | 80 |
| MEL | 13 | 13 | 14 | 40 |

## Point estimates and intervals

| Measure | Verified estimate | 95% interval | Definition / interval method |
|---|---:|---:|---|
| MEL recall / sensitivity | **13/40 = 0.3250** | 0.2008–0.4798 | MEL argmax among true MEL; Wilson. **Primary endpoint.** |
| Binary MEL specificity among true NV | **75/80 = 0.9375** | 0.8619–0.9730 | Any argmax other than MEL is negative, including OTHER; Wilson. |
| NV exact-label recall | 68/80 = 0.8500 | 0.7559–0.9121 | Only NV argmax counts correct; Wilson. |
| Mapped balanced accuracy | **(0.8500 + 0.3250)/2 = 0.5875** | 0.5063–0.6688 | Mean of *exact-label* NV and MEL recall. Descriptive 2,000-draw, fixed-seed 42 class-stratified image-ID bootstrap calculated in the audit. This is **not** the mean of binary MEL sensitivity and binary specificity. |
| Mapped accuracy | 81/120 = 0.6750 | 0.5869–0.7522 | Exact NV or MEL argmax; OTHER wrong; Wilson. |
| MEL precision | 13/18 = 0.7222 | — | Five NV cases predicted MEL. |
| MEL F1 | 0.4483 | 0.2800–0.6001 | Fixed-seed stratified image-ID bootstrap interval, independently reproduced. |
| MEL `p_mel` ROC AUC | 0.5859 | 0.4653–0.7053 | Continuous **unmodified seven-class** `p_mel`; bootstrap interval independently reproduced. |
| MEL `p_mel` average precision | 0.5250 | — | Positive prevalence 40/120 = 0.3333; no stored interval. |
| Mean predictive entropy | 0.9305 nats | 0.8772–0.9807 | Seven-class softmax entropy; bootstrap interval independently reproduced. |
| Entropy error-detection ROC AUC | 0.7610 | — | Mapped misclassification is positive; descriptive, no stored interval. |

All binomial Wilson bounds and stored bootstrap bounds were independently recalculated from the saved predictions and matched; the mapped balanced-accuracy interval was calculated additionally from the same fixed bootstrap protocol. These resamples use **image IDs**, because no patient identifiers exist in the local PH2 metadata. They must not be described as patient-cluster intervals or clinical confidence guarantees. Full precision and interval provenance are in [METRICS_TABLE.csv](METRICS_TABLE.csv).

**Terminology safeguard:** “Nevus specificity” here means specificity of the binary *melanoma decision* among true NV: 75/80. Seven NV images predicted OTHER are binary negatives yet still errors under the mapped 2×3 endpoint. NV exact-label recall is therefore 68/80. The distinction is essential when reading the 0.5875 mapped balanced accuracy.
