# Stage 20 final evidence-marked EG-VAN+ pipeline

```text
INPUT DERMOSCOPIC IMAGE
  ↓
TECHNICAL IMAGE-QUALITY RISK ASSESSMENT [PARTIAL: descriptive proxies
measured on processed HAM images; not yet a raw-image pre-model gate]
QUALITY ACCEPT / REVIEW FLAG [NOT VALIDATED: no quality labels or cutoff]
  ↓
FROZEN PREPROCESSING [VERIFIED: hair removal, Gray World, Retinex]
  ↓
FROZEN EG-VAN [VERIFIED]
  ├─ EfficientNetV2S branch
  └─ modified ResNet50 + SCGA + Non-Local branch
      (nonlocal3 q@k FP32 under surrounding CUDA AMP)
  ↓
MULTI-SCALE FEATURE FUSION [VERIFIED]
  ↓
SEVEN-CLASS PROBABILITY VECTOR [VERIFIED: saved validation/HAM/PH²]
  ↓
UNCERTAINTY / CALIBRATION ANALYSIS [VERIFIED: Stage 20 frozen outputs]
  ├─ entropy <= 0.7675495327940953 → retain prediction [EXPLORATORY REVIEW RULE]
  └─ entropy >  0.7675495327940953 → flag for review [EXPLORATORY REVIEW RULE]
  ↓
QUALITATIVE GRAD-CAM EXPLANATION [VERIFIED FOR EIGHT FIXED HAM CASES;
not a validated lesion localizer or an integrated per-image service]
  ↓
OUTPUT: class, probability vector, entropy, review status;
         technical-quality profile, not a validated accept/reject judgment

EVALUATION: HAM held-out n=1,014 [VERIFIED];
            PH² mapped external follow-up n=120 [VERIFIED WITH PRIOR-USE LIMITATION]
```

The quality proxies are currently computed **after** preprocessing in the saved HAM table, so the diagram must not be presented as a finished raw-image pre-classifier gate. Stage 20 uncertainty analysis uses immutable saved probabilities. The entropy threshold was selected from Stage 15 validation data for approximately 80% retained coverage and was **not** selected using HAM test or PH². Stage 16 outcomes had previously been viewed, so Stage 20 review performance remains exploratory. On HAM, 805/1,014 (79.39%) were retained and 209/1,014 (20.61%) reviewed; retained accuracy was 90.93%, with 106/179 errors and 23/51 melanoma false negatives sent to review. These are selective-review results, **not** a new full-cohort classifier accuracy.

The eight final-checkpoint Grad-CAM maps passed artifact checks and were visually reviewed. Seven show broad high response around, rather than clearly within, the conspicuous central lesion-like area at the chosen `resnet.nonlocal3` layer; the BCC example is more focal. The maps are qualitative, have no lesion-mask ground truth and cannot establish clinical localization correctness or whole-model causal behavior. The review flag identifies uncertainty, not clinical acceptability.

The unchanged full-cohort results remain: HAM accuracy 82.3471%, macro F1 0.699582, MEL recall 56/107 (52.34%), MEL F1 0.568528; PH² included-cohort accuracy 66.67%, MEL recall 11/40 (27.5%), MEL F1 0.392857. PH² is external follow-up, not untouched external validation, because of prior project use.

Sources: [Stage 20 reliability report](reliability_report.md), [quality protocol](quality_protocol.md), [Grad-CAM audit](gradcam_audit.md), and `analysis/stage16_final_evaluation/final_evaluation_audit.md`.
