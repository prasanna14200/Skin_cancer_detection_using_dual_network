# Final Stage 20 EG-VAN+ reliability completion audit

**Final disposition: PARTIAL ORIGINAL OBJECTIVE; qualified EG-VAN+ research prototype.** The final-model uncertainty, calibration and selective-review analyses are complete from immutable saved probabilities; a bounded final-model qualitative Grad-CAM set was generated and audited. The intended **pre-model quality accept/reject gate remains unvalidated**. No scientific result, model weight, checkpoint, Stage 16 output or entropy threshold was changed in this audit. The PLOS submission copy was not edited.

## Frozen model and source integrity

The recovered selected Stage 15 epoch-16 checkpoint remains SHA256 `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`. The frozen HAM split SHA256 is `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. Stage 16 HAM and PH² prediction/metric file hashes still match their manifests. The original epoch-16 weight file was lost; exact observable recovery replay is documented, but byte identity to the unavailable original weights cannot be established.

The Stage 20 [protocol](preregistration.md) specified MSP/entropy comparison using validation error-detection AUROC and an 80% validation retained-coverage target before Stage 20 reliability results were computed. The selected [review rule](uncertainty_protocol.json), SHA256 `5423ef77097d1215f60cd9f9d08246b0aab0a94aab000bcf3e16919bd93a1197`, uses predictive entropy: retain when `H(p) <= 0.7675495327940953` and flag otherwise. It was selected **only from the 986 Stage 15 epoch-16 validation predictions**, not from HAM test or PH². Stage 16 outcomes had been viewed before Stage 20, so this extension is exploratory, not a new untouched confirmatory test.

The Grad-CAM [selection](gradcam_selection.json) fixed eight cases before visualization; SHA256 `a98b3aee53e439bd2e4b9c1d67bddfb254e28166820b33c8a6296f466fb3598e`. The downloaded `final_model_gradcam_output_v2/gradcam_manifest.json` links that selection and the frozen checkpoint, with eight maps and eight overlays matching their saved hashes. The first failed rendering directory was not used as completed evidence. The separate [Grad-CAM audit](gradcam_audit.md) records every case, confidence, entropy, image integrity and visual observation. The output manifest does not include runtime/GPU or executing-script SHA256, so the reported Tesla T4 provenance cannot be independently proven from the JSON alone. The code path contains no optimizer/parameter update, and the checkpoint file hash is unchanged; it does not record an after-run in-memory parameter hash.

## Reliability results — selective subset, not new classifier accuracy

| Dataset | ECE | Seven-class Brier | Review outcome |
|---|---:|---:|---|
| Stage 15 validation, n=986 | 0.019574 | 0.270595 | 789 retained (80.02%), 197 reviewed; 87/178 errors reviewed. |
| HAM held-out, n=1,014 | 0.029678 | 0.256018 | **805 retained (79.39%), 209 reviewed (20.61%), retained accuracy 90.93%, 106/179 original errors reviewed (59.22%), 23/51 MEL false negatives reviewed.** |
| PH² mapped follow-up, n=120 | 0.060178 | 0.491627 | 71 retained (59.17%), 49 reviewed; 28/40 errors and 20/29 MEL false negatives reviewed. |

All calibration/review figures and exact values are in [reliability_report.md](reliability_report.md) and saved JSON. Retained accuracy is conditional on review; it **does not replace** full-cohort accuracy or demonstrate a more accurate underlying classifier. A substantial fraction of melanoma misses remained unflagged. No calibration transform or test-derived cutoff was used.

## Unchanged classification and external follow-up results

- Final HAM test: **82.3471% full-cohort accuracy**, macro F1 **0.699582**, MEL recall **56/107 = 52.34%**, MEL F1 **0.568528**. These remain the frozen Stage 16 results.
- PH² included cohort: **66.67% accuracy**, MEL recall **11/40 = 27.5%**, MEL F1 **0.392857**; 80 common nevi and 40 melanomas were included, and 80 atypical nevi excluded by the fixed mapping. PH² had prior project use: this is **external follow-up**, not untouched external validation. The lower melanoma sensitivity is a transfer limitation under this protocol, not a demonstrated causal mechanism or clinical claim.

## Quality and explainability conclusions

The repository contains eight processed-image technical quality proxies. All 986 validation prediction IDs joined to quality measurements. Low-sharpness validation tercile error was 22.19%, versus 14.33% in the middle tercile, but relationships were not consistently monotonic. There are no expert quality labels or prospectively validated accept/reject cutoffs; the technical quality-risk assessment is descriptive and **the pre-model binary gate is not complete**. Do not invent a cutoff to upgrade status.

Eight final-checkpoint `resnet.nonlocal3` predicted-class Grad-CAM maps are intact, non-degenerate and visually reviewed. Seven show broad higher response outside the conspicuous central lesion-like area, and the BCC map is more focal on it. This is **qualitative attention visualization at one layer**, not clinical localization correctness or proof of whole-model causal reasoning. Final-model explainability as a bounded research visualization is **complete with limitation**. No post-hoc case replacement was found.

## Original-objective decision

| Objective | Final status |
|---|---|
| EG-VAN reconstruction | COMPLETE WITH LIMITATION |
| Image-quality assessment and pre-model flag | PARTIAL |
| Uncertainty estimation | COMPLETE WITH LIMITATION |
| Calibration evaluation | COMPLETE WITH LIMITATION |
| Validation-derived selective review | COMPLETE WITH LIMITATION |
| Final-model explainability | COMPLETE WITH LIMITATION |
| HAM held-out evaluation | COMPLETE WITH LIMITATION |
| PH² external follow-up | COMPLETE WITH LIMITATION |
| Numerical stability | COMPLETE |

“**EG-VAN+**” is scientifically defensible only as a **qualified reliability-extension prototype** that includes evaluated probability reliability, exploratory selective review and limited qualitative explanation. A claim that the full intended EG-VAN+ pipeline is implemented, clinically validated, or has a validated raw-image quality gate is **not** supported. Overall status remains **PARTIAL**. The [final evaluator pipeline](final_egvan_plus_pipeline.md) marks each component's evidence boundary.

## Publication disposition and next action

The Stage 20 evidence is ready for a **careful manuscript-update decision**, not automatic insertion or submission. The [manuscript update plan](manuscript_update_plan.md) identifies scope, tables, figures, explainability caveats and limitations. Keep submission paused until the author confirms the paper's honest scope and the revised draft is reviewed. Any later quality-gate development requires a separately documented prospective protocol, quality reference labels or justified criteria, and validation-derived cutoffs; do not tune on HAM test or PH². No training or new 25-epoch experiment is warranted by this audit.
