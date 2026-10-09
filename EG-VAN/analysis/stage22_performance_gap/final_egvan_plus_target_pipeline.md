# EG-VAN+ target pipeline and evidence status

```text
RAW DERMOSCOPIC IMAGE
  |
  v
TECHNICAL IMAGE-QUALITY ASSESSMENT
  Current: PARTIAL; processed-image proxies are descriptive only.
  Target: independently labelled raw-image accept / review / reject;
  no gate or threshold is currently validated.
  |
  v
FROZEN PREPROCESSING
  Hair removal -> Gray World -> multi-scale Retinex; approximate paper match.
  Crop and paper-exact parameters are absent.
  |
  v
EG-VAN, SEVEN-CLASS HAM OUTPUT
  EfficientNetV2S + modified ResNet50 (SCGA + Non-Local Blocks)
  -> four-scale MFF -> fused classifier -> probabilities
  Current: reconstructed model and frozen probabilities audited.
  |
  v
CALIBRATION / UNCERTAINTY / REVIEW
  Current: validation-selected predictive-entropy rule is preserved.
  Low entropy: show prediction; high entropy: flag for human review.
  Review status is not a quality judgment or replacement accuracy.
  |
  v
EXPLANATION
  Current: bounded Grad-CAM set is qualitative, not a validated lesion
  localizer or a gate for model predictions.
  |
  v
EXTERNAL GENERALIZATION
  Current: PH2 is an external follow-up, not untouched validation.
  Future: separately lock a verified, non-overlapping public cohort;
  publish mapping, OOD handling, coverage and full-cohort metrics.
```

## Status boundaries

| Component | Current evidence | Target completion condition |
|---|---|---|
| Image quality | Eight proxy features on processed HAM images; no expert/reference labels or validated pre-model gate. | Define the technical-quality rubric before annotation; obtain blinded independent labels; freeze cutoffs on development data; evaluate on a separate lesion-disjoint quality set and report false rejection/review burden with uncertainty. Do not use disease-model correctness as the quality label. |
| Preprocessing | Existing deterministic hair removal, Gray World, and Retinex implementation; crop absent; several paper parameters unspecified. | Freeze operational choices and disclose deviations before any future training. A change is a separately registered experiment, not a paper-exact reproduction claim. |
| EG-VAN | Seven-class reconstruction with EfficientNetV2S and modified ResNet50/SCGA/Non-Local/MFF; Stage16 evaluation is frozen. | Keep architecture and selection protocol versioned. A performance claim requires a prespecified untouched confirmation cohort for any new candidate. |
| Uncertainty/calibration | Stage20 saved-output metrics and validation-selected entropy threshold `0.7675495327940953`; exploratory because Stage16 outcomes had already been viewed. | Preserve the existing result. For confirmatory claims about coverage/error risk, validate the frozen rule on a new untouched cohort; do not retune the threshold on HAM test or PH2. |
| Human review | On HAM, 805/1,014 retained (79.39%), retained accuracy 90.93%, and 106/179 errors flagged; 23/51 MEL misses flagged. | Report alongside full-cohort classification. This conditional accuracy is not overall classifier accuracy or a safety guarantee. Define reviewer workflow only after prospective evaluation. |
| Explainability | Eight saved final-model Grad-CAM maps visually audited; no lesion-mask ground truth. | Keep qualitative and bounded unless independently evaluated against appropriate localization labels. |
| External validation | PH2: 120 included cases (80 common nevi, 40 melanoma), prior project use, hence follow-up. BCN20000 is a candidate, not yet screened/downloaded/evaluated. | Verify official class labels, prior project/pretraining use, image/lesion overlap, inclusion rules and OOD policy before obtaining/evaluating it. Freeze mapping and model before one evaluation pass. |
| Full-cohort performance | HAM Stage16 accuracy 82.3471%; macro F1 0.699582. | Report target status against an untouched, protocol-matched cohort. Do not substitute selective coverage, class remapping, or test-informed tuning for full-cohort accuracy. |

The scientifically correct system does not reject images based on model uncertainty alone. Technical image quality and predictive uncertainty are distinct: poor technical quality routes to acquisition review; predictive uncertainty routes an otherwise usable image to diagnostic review. OOD inputs require a separately declared response and must not be silently forced into one of the seven HAM classes.

This is a target architecture, not a claim that all stages are implemented or clinically validated. EG-VAN+ remains a qualified research prototype, and the 95-98% full-cohort target is not achieved.
