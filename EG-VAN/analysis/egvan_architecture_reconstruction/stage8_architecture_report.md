# Stage 8 architecture reconstruction

## Status

**NOT_READY_FOR_TRAINING**. The reconstruction passes synthetic code verification, but paper-faithful training readiness requires visual verification of Figure 3 and Table 1 against the original PDF. Their images were inaccessible: no PDF was found in the accessible repository and the online PDF endpoint returned HTTP 403. The tap choices and exact fusion wiring therefore remain provisional. Supply the PDF path/attachment and review those figures before freezing the architecture.

## Paper authority and topology

Primary source: [EG-VAN, IEEE Access 2025](https://doi.org/10.1109/ACCESS.2025.3561240). Accessible full text, captions and equations informed the specification; actual Figure 3/Table 1 visuals were not verified. The architecture specification in `docs/egvan_paper_architecture_spec.md` separates explicit statements from inferences and unresolved details. EfficientNetV2S and modified ResNet50 produce four paired scales. The modified ResNet applies SCGA to early stages and embedded-Gaussian Non-Local Blocks to later stages. Each pair enters SA/GMA, concatenation, separable stride-2 convolution, batch normalization and SCGA. Aligned fused outputs enter GAP and a configurable linear classifier.

## Verification

Synthetic forward passed for batch 1 at repository-compatible 384×384 and batch 2 at 128×128; output was finite `[B,7]`. A synthetic batch-1 backward pass propagated finite gradients through all trainable parameter tensors. Component tests cover dimensions and finite gradients. No data images were opened.

## Parameter audit

| Module | Total | Trainable |
|---|---:|---:|
| EfficientNetV2S baseline | 20,186,455 | 20,186,455 |
| ResNet50 baseline | 25,557,032 | 25,557,032 |
| modified ResNet50 | 34,066,114 | 34,066,114 |
| full EG-VAN reconstruction | 55,545,201 | 55,545,201 |
| EGVAN.efficient | 20,177,488 | 20,177,488 |
| EGVAN.resnet | 34,066,114 | 34,066,114 |
| EGVAN.fusions | 1,235,160 | 1,235,160 |
| EGVAN.aggregate | 65,536 | 65,536 |
| EGVAN.pool | 0 | 0 |
| EGVAN.classifier | 903 | 903 |

These are local instantiated counts. The modified branch and full reconstruction have additional parameters; the paper's efficiency/parameter claims are not adopted. MACs/FLOPs were skipped because a consistently scoped profiler was unavailable without adding dependencies.

## Shape trace

See `feature_shape_trace.csv` for exact 384×384 branch and fusion shapes. Pairwise taps happened to align; final four fused scales required explicit resizing to the coarsest map before concatenation.

## Paper ambiguities and preprocessing

See `docs/egvan_implementation_decisions.md` and `implementation_assumptions.csv`. Major uncertainties are exact tap/attention insertion points, GMA group count, key/value reduction, fusion widths, final alignment, input crop/size and augmentation. The paper's printed bright top-hat equation conflicts with the existing blackhat implementation used for dark hair; the preprocessing pipeline and processed data were left untouched. `src/train.py` uses 384×384, ImageNet normalization and three augmentation operations, none of which is established as paper-exact.

## Research scope

The seven-class forward check is architectural verification, not reproduction of the paper's nine-class reported performance. No training, dataset inference, PH² evaluation, Grad-CAM generation, calibration fitting or optimizer update occurred. Stage 9 must prespecify pretrained initialization, frozen split, input policy and architecture assumptions before training.

## Next action

Obtain and visually inspect the original PDF Figure 3 and Table 1, revise the documented reconstruction if needed, then rerun Stage 8 verification. Only after that: Stage 9 — Controlled EG-VAN Training Protocol.
