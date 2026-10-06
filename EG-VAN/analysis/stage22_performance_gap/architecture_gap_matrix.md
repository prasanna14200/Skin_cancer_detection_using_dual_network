# Architecture gap matrix

References: [paper](https://doi.org/10.1109/ACCESS.2025.3561240), Figures 3–5/Table 1; `docs/egvan_paper_architecture_spec.md`; `src/models/egvan/`. Labels: **EXACT** documented operator/channel agreement; **CLOSE** named mechanism implemented with assumptions; **DIFFERENT** explicit difference; **MISSING** absent; **PAPER AMBIGUOUS** insufficient specification. Exact does not mean author-weight identity.

| Component | Paper | Repository implementation and 384² dimensions | Classification | Consequence |
|---|---|---|---|---|
| EfficientNetV2S branch | Pretrained; stage outputs in Table 1 | Torchvision EfficientNetV2S, ImageNet weights; taps indices 2/3/5/7: 48×96², 64×48², 160×24², 1280×12² | CLOSE | Tap subset not fixed by paper |
| Modified ResNet50 | SCGA at early stages, NLB at later stages | Random ResNet50 initialization in Stage15; post-layer1/2 SCGA and post-layer3/4 NLB; outputs 256×96², 512×48², 1024×24², 2048×12² | CLOSE | Stage channel/placement agrees with Table 1; internal insertion and pretraining unclear |
| SCGA spatial path | 1×1 64→16(dilation 2)→8→mask, global mean, feature projection | Implemented with fixed all-ones mask expansion and trainable input projection | CLOSE | Weight policy unreported; 1×1 dilation does not expand receptive field |
| SCGA horizontal/vertical GMA | Group-wise mean/max and axis attention | Eight equal groups, directional mean/max, shared per-group 1×1 weights | CLOSE | Group count and sharing not fixed in paper |
| Non-Local Blocks | Embedded Gaussian; projected q/k/v, softmax, weighted values, output projection, residual; optional spatial reduction | Two blocks after layer3/4, q/k/v C/2, key/value average pooling stride 2, residual | CLOSE | C/2 width, pool kind/factor not fixed |
| MFF inputs/count | Paired branch scales with serial MFF arrows | Four paired scales, previous fused map carried to next stage | CLOSE | Exact tap/serial operator not enumerated |
| MFF attention/downsample | EfficientNet spatial attention, ResNet GMA; concat→separable stride-2 convolution→BN→SCGA | Same order; depthwise 3×3 + pointwise 1×1 to 128 channels | CLOSE | Kernel/width unspecified in paper |
| Terminal fusion | Concatenate MFF stream and terminal branch features→GAP→classifier | Bilinear align four fused maps and both terminal maps; 1×1 reduction→GAP→linear logits | CLOSE | Alignment/reduction assumptions; external softmax is mathematically equivalent for probabilities |
| Numerical policy | CUDA precision **NOT SPECIFIED IN PAPER** | FP32 q@k only in `resnet.nonlocal3`, AMP elsewhere | DIFFERENT / PAPER AMBIGUOUS | Needed to avoid observed FP16 overflow; performance effect not measured |

No clear missing named architecture block was found. The largest architectural uncertainty is **exact wiring/parameterization**, not evidence that a particular absent module explains the accuracy gap. Altering these assumptions would create a new model and require one-factor validation experiments.
