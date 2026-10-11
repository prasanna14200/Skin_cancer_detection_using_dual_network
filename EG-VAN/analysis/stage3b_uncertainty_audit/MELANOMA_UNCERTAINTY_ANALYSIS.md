# True-melanoma uncertainty and confidently wrong predictions

The [saved prediction integrity audit](STAGE3B_UNCERTAINTY_AUDIT.md) found **70 correct MEL and 37 false negatives among 107 HAM validation melanomas**, and **13 correct MEL and 27 false negatives among 40 PH2 follow-up melanomas**. HAM has 26 MEL→NV misses. PH2 has 13 MEL→NV and 14 MEL→OTHER misses under the frozen two-class mapping. These are different evaluation populations and label spaces.

| Within true MEL cases | HAM correct | HAM miss | PH2 correct | PH2 miss |
|---|---:|---:|---:|---:|
| Count | 70 | 37 | 13 | 27 |
| Median top-label confidence | 0.7197 | 0.6892 | 0.5504 | 0.4909 |
| Median predictive entropy, nats | 0.6811 | 0.7158 | 1.0616 | 1.1525 |

Entropy's error-detection AUROC **within true MEL** is only 0.5927 on HAM (lesion-bootstrap 95% interval 0.4761–0.7071) and 0.5755 on PH2 (image-ID bootstrap 0.3708–0.7649). `1−confidence` yields 0.5718 and 0.6296; `−margin` yields 0.5591 and 0.6154. None establishes dependable ranking of melanoma misses. Of 37 HAM misses, 15 have entropy **at or below the median of correct HAM melanomas**. These comparisons are descriptive; no score threshold is proposed.

Using confidence ≥0.8 and ≥0.9 as **descriptive bins**, not referral thresholds: **24/168** all-class HAM errors meet ≥0.8 and **8/168** meet ≥0.9; among HAM melanoma misses the counts are **9/37** and **2/37**. For PH2 mapped errors the counts are **1/39** and **0/39**; among PH2 melanoma misses **1/27** and **0/27**. HAM `ISIC_0026296` was predicted NV at approximately 0.9897 confidence; PH2 `IMD284` was predicted DF at approximately 0.8093. Their scores are not clinical probabilities. Existing case-level [HAM](../stage23_melanoma_failure_analysis/HAM_MELANOMA_ERRORS.csv) and [PH2](../stage23_melanoma_failure_analysis/PH2_MELANOMA_ERRORS.csv) tables provide individual saved scores without rerunning inference.

The exploratory Stage 23 validation 80%-coverage entropy scenario referred 15/37 HAM melanoma false negatives, leaving 22/37 retained. A reviewer might therefore see a more accurate retained group while still missing many melanomas. No comparable Stage 23 review threshold was frozen for PH2. The PH2 prior-use caveat and absent patient grouping preclude a prospective safety claim. The melanoma subgroup results answer the present question: **uncertainty does not meaningfully or reliably distinguish melanoma misses in the available evidence**.
