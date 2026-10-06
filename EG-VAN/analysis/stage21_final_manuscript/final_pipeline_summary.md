# Evidence-marked evaluator pipeline

```text
Dermoscopic image
  ↓
Technical image-quality risk assessment [PARTIAL]
  Existing proxies were analyzed offline on processed HAM images;
  no validated raw-image accept/reject cutoff exists.
  ↓ intended order only
Frozen preprocessing [VERIFIED]
  ↓
Frozen reconstructed EG-VAN [VERIFIED]
  EfficientNetV2S + modified ResNet50 / SCGA / Non-Local
  → multiscale fusion; only nonlocal3 q@k affinity FP32 under AMP
  ↓
Seven-class probability vector and unchanged argmax [VERIFIED]
  ├─ Full-cohort prediction remains the Stage 16 result
  └─ Predictive entropy / calibration analysis [EVALUATED, EXPLORATORY]
       ├─ H(p) ≤ 0.7675495327940953 → retain prediction
       └─ H(p) > 0.7675495327940953 → flag for review
       (threshold chosen on validation for ~80% retained coverage)
  ↓
Qualitative final-model Grad-CAM [EIGHT FIXED CASES; ONE LAYER]
  ↓
Final report: class, probabilities, entropy, review status and limitations

Evaluation: HAM held-out seven-class test [COMPLETE]
            PH² external follow-up [COMPLETE WITH PRIOR-USE LIMITATION]
```

This is an **evidence-marked research prototype**, not a currently integrated clinical workflow. The pre-classifier quality position depicts the original intention; the measured proxies were computed after preprocessing and there is no validated binary gate. The entropy review rule does not alter the frozen classifier. On HAM it retained 805/1,014 cases with 90.93% accuracy **among retained cases**, while the unchanged full-cohort accuracy was 82.3471%. Eight Grad-CAM maps provide qualitative one-layer visualization without lesion-mask localization evidence. See [Fig8 pipeline](egvan_plus_pipeline.png), [final Stage 20 audit](../stage20_reliability_completion/final_stage20_audit.md) and [manuscript audit](final_manuscript_audit.md).
