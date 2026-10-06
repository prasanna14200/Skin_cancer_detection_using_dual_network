# Original intent and verified final pipeline

## Pipeline A — intended EG-VAN+

```text
Input dermoscopic image
  ↓
Pre-model image-quality assessment → acceptable / flag for review
  ↓ acceptable
Preprocessing and color balancing
  ↓
EG-VAN dual branch
  ├─ EfficientNetV2S
  └─ modified ResNet50 → SCGA local attention → Non-Local global attention
  ↓
Multi-Scale Feature Fusion
  ↓
Seven-class prediction
  ↓
Uncertainty / confidence and calibration assessment
  ↓
Explainability, if evaluated
  ↓
Prediction or review flag
  ↓
External-domain evaluation
```

The paper architecture was reconstructed with documented assumptions; the proposed reliability functions above are project extensions, not claims that every block was delivered in the final model.

## Pipeline B — verified frozen final system

```text
HAM10000 processed dermoscopic image [VERIFIED]
  ↓
Preprocessing: hair removal → Gray World → Retinex, then fixed evaluation
resize/tensor/normalization [VERIFIED]
  ↓
Pre-classifier quality measurement on processed HAM images [PARTIAL]
Accept/flag quality gate [MISSING]
  ↓
EG-VAN: EfficientNetV2S + modified ResNet50 with SCGA and Non-Local blocks;
multi-scale fusion [VERIFIED]
  └─ nonlocal3 q@k in FP32, CUDA AMP around the rest [VERIFIED]
  ↓
Frozen seven-class argmax and probabilities [VERIFIED]
  ↓
Final-model confidence/calibration analysis [MISSING]
Final-model uncertainty-based review flag [MISSING]
Final-model Grad-CAM explanation [MISSING]
  ↓
HAM held-out evaluation: n=1,014 [VERIFIED]
PH² mapped external follow-up: n=120 [VERIFIED, WITH LIMITATION]
```

The existing quality proxy table was calculated from **processed** HAM images, not as a raw-image pre-inference screen. The earlier baseline uncertainty/calibration and Grad-CAM studies are evaluated components of the repository, but they are **not** components of the frozen final EG-VAN inference pipeline. PH² was previously used elsewhere in the project and is an external follow-up rather than untouched validation.

The final checkpoint is the recovered Stage 15 epoch-16 file at `experiments/stage15_single_candidate_exp1/stage15_single_candidate_exp1/recovery_epoch16/best_checkpoint.pt` (SHA256 `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`). The split SHA256 is `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. See `analysis/final_experiment_freeze.md` and `analysis/stage16_final_evaluation/final_evaluation_audit.md` for provenance. No post-test selection is part of this pipeline.
