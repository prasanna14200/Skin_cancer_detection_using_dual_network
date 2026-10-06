# Stage 20 evidence-marked EG-VAN+ pipeline

```text
INPUT DERMOSCOPIC IMAGE
  ↓
TECHNICAL IMAGE-QUALITY PROXIES [PARTIAL: measured on processed HAM images]
QUALITY ACCEPT / REVIEW FLAG [NOT VALIDATED: no labels or cutoff]
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
FINAL-MODEL GRAD-CAM [PREPARED ONLY: GPU pass and review pending]
  ↓
OUTPUT: class, probability vector, entropy, review status;
         technical-quality profile, not a validated accept/reject judgment

EVALUATION: HAM held-out n=1,014 [VERIFIED];
            PH² mapped external follow-up n=120 [VERIFIED WITH PRIOR-USE LIMITATION]
```

The quality proxies are currently computed **after** preprocessing in the saved HAM table, so the diagram must not be presented as a finished raw-image pre-classifier gate. Stage 20 uncertainty analysis uses immutable saved probabilities. The 80% validation-coverage rule was frozen before Stage 20 HAM/PH² review calculations, but Stage 16 outcomes had previously been viewed; its evaluation is exploratory, not an untouched prospective test. The review flag identifies uncertainty, not clinical acceptability. Grad-CAM maps for the final checkpoint have not yet been generated.

Sources: [Stage 20 reliability report](reliability_report.md), [quality protocol](quality_protocol.md), [Grad-CAM instructions](gradcam_colab_instructions.md), and `analysis/stage16_final_evaluation/final_evaluation_audit.md`.
