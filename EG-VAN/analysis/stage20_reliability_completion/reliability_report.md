# Stage 20 frozen-probability reliability analysis

**Status:** saved-probability analysis and bounded final-model Grad-CAM audit complete; a validated quality gate remains pending. No training, temperature fitting, result replacement, or Stage 16 threshold change was performed. This report's CPU probability analysis did not rerun inference; a separate bounded Colab final-model forward/backward pass generated the eight Grad-CAM maps. The Stage 15 recovered epoch-16 checkpoint SHA256 was independently rechecked as `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`. All Stage 16 output hashes still match their manifests.

The [preregistration](preregistration.md) was created before Stage 20 validation calculations. The validation-only rule was then written to [uncertainty_protocol.json](uncertainty_protocol.json) (SHA256 `5423ef77097d1215f60cd9f9d08246b0aab0a94aab000bcf3e16919bd93a1197`) **before** computing HAM or PH² review outcomes. The analysis script uses existing saved probabilities only. These are exploratory analyses added after Stage 16 outcomes were already known to the project; they are not newly untouched confirmatory tests.

## Sources and rule

The selected validation file has 986 unique images and seven class probabilities; the frozen HAM file has 1,014 and PH² has 120 mapped included images. Inputs passed finite, range, sum-to-one, saved-argmax and correctness checks. The selected validation prediction file SHA256 is `d0510c792d36c9cd3132d9495d70be204fa228c8eb4f2a016bcc4dc2a725e148`.

Validation error-detection AUROC was 0.821859 for `1-MSP` and 0.827386 for predictive entropy, so the prespecified rule chose **entropy**. The 80% validation-coverage threshold is **0.7675495327940953 nats**; accept if entropy `<=` this value, review otherwise. Validation retained 789/986 (80.02%) with 88.47% retained accuracy and flagged 87/178 errors. Only 5/44 validation melanoma false negatives were reviewed. The rule therefore does **not** guarantee melanoma-error detection.

| Frozen dataset | N | ECE, 10 bins | Seven-class Brier | NLL | Entropy error AUROC | Coverage | Retained accuracy | Errors reviewed | MEL false negatives reviewed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage 15 selected validation | 986 | 0.019574 | 0.270595 | 0.545024 | 0.827386 | 789/986 = 80.02% | 88.47% | 87/178 = 48.88% | 5/44 |
| Stage 16 HAM held-out | 1,014 | 0.029678 | 0.256018 | 0.507647 | 0.858883 | 805/1,014 = 79.39% | 90.93% | 106/179 = 59.22% | 23/51 |
| Stage 16 PH² external follow-up | 120 | 0.060178 | 0.491627 | 0.926240 | 0.766875 | 71/120 = 59.17% | 83.10% | 28/40 = 70.00% | 20/29 |

On HAM the rule flagged 209 images (20.61%) including 40/107 melanoma cases. On PH² it flagged 49 (40.83%) including 28/40 melanomas. Retained error rates were 9.07% HAM and 16.90% PH². These are **selective-subset** results: retained accuracy is not a replacement for full-cohort accuracy or a claimed model improvement. The high PH² review rate shows the fixed rule behaves differently under domain shift.

ECE bins are ten fixed equal-width MSP bins. Brier is the unnormalized seven-class sum of squared probability errors averaged over images; NLL uses the saved probability for the mapped true class. PH² calibration conditions on 80 common nevi and 40 melanomas; 80 atypical nevi were excluded and predictions into other HAM classes remain errors. It is not evidence of broad external calibration. No softmax output is described as intrinsically calibrated and no temperature scaling was fitted. Calibration diagrams, confidence distributions, and descriptive risk–coverage plots are in this folder.

## Technical image quality

All 986 validation predictions joined uniquely to the existing **processed-image** quality table. The eight available proxies are described in [quality_protocol.md](quality_protocol.md). Prespecified equal-count validation terciles showed:

| Proxy | Low-tercile error | Middle | High | Interpretation |
|---|---:|---:|---:|---|
| Brightness | 15.20% | 18.29% | 20.67% | Higher brightness was not uniformly protective. |
| Contrast | 19.76% | 19.82% | 14.59% | Higher contrast had fewer errors descriptively. |
| Sharpness | 22.19% | 14.33% | 17.63% | Low sharpness had more errors, but relationship was not monotonic. |

Validation MEL false-negative counts by low/middle/high sharpness were 21/14/9. These counts lack control for class mix or lesion difficulty; they do not prove blur caused errors. Existing proxies have **no quality ground-truth labels** and no justified accept/reject cutoff. Consequently no binary quality gate was implemented or evaluated. The quality objective is a descriptive technical-risk analysis, not a complete pre-inference screen.

## Provenance and limits

Source hashes and full precision results are in [validation_reliability.json](validation_reliability.json), [reliability_analysis.json](reliability_analysis.json), [quality_analysis.json](quality_analysis.json), and the frozen rule JSON. The final checkpoint's original in-memory epoch-16 weights are unavailable, despite exact observable recovery replay; the recovered checkpoint is the frozen final model. PH² had prior project use. Existing Stage 16 predictions and metrics are unmodified. No claim of clinical safety, causal image-quality effects, or improved underlying classifier accuracy follows from this analysis.

## Bounded final-model Grad-CAM

The [Grad-CAM audit](gradcam_audit.md) verifies exactly eight fixed HAM cases, eight non-degenerate `384×384` maps and their overlays, source/output hashes, labels, predicted-class targets, and replayed probabilities. Every map was visually reviewed. Seven maps show broad high response around a conspicuous central region at `resnet.nonlocal3`; the BCC map is more focal. These are **qualitative attention visualizations of one layer**, not clinical lesion-localization evidence or a proof of the full model's causal reasoning. The bounded Colab pass loaded the frozen checkpoint for map generation; it was not training or a new HAM classification evaluation. The image-quality gate remains unvalidated.
