# EG-VAN paper architecture specification for Stage 8

Authority: [Saeed et al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3561240](https://doi.org/10.1109/ACCESS.2025.3561240). This is a documented reconstruction, not a claim of author-code identity. **Stage 8 historical note:** its online text review could not visually inspect Figure 3 or Table 1. **Stage 8B update:** the supplied local PDF was found at the workspace parent and Figures 3–5 plus Table 1 were visually inspected. Tensor dimensions below are locally verified torchvision dimensions; the paper's Table 1 confirms stage channel values but not the chosen subset of MFF taps.

## Input and training pipeline

| Item | Status | Paper statement and Stage 8 interpretation |
|---|---|---|
| Hair removal | PAPER EXPLICIT operation; PAPER UNSPECIFIED constants | The paper describes grayscale morphology, a top-hat/threshold mask and Telea inpainting (equations 1–7). Kernel shape/size, threshold and radius are not fixed. Its printed top-hat equation describes bright structures, while dark-hair removal suggests black-hat; this conflict is unresolved. |
| Gray World | PAPER EXPLICIT concept; PAPER UNSPECIFIED numerical policy | Per-image channel means and gains are described (equations 8–9). Clipping and channel-order details are not fully fixed. |
| Retinex | PAPER EXPLICIT concept; PAPER UNSPECIFIED scales | Gaussian/log decomposition and combination with Gray World are described (equations 10–13). Scales, weights, epsilon and output normalization are not fixed. |
| Crop | PAPER EXPLICIT concept; PAPER UNSPECIFIED size/location | Equation 14 gives a square crop predicate, but not the chosen crop size or lesion localization method. No crop is added in Stage 8. |
| Resize / normalization | PAPER INFERRED necessity; PAPER UNSPECIFIED values | The paper says image size is set to the model. It does not specify an operational input size or normalization for the reconstructed branches. Stage 8 uses the repository's 384×384 input only for compatibility tests; modules support divisible smaller sizes. |
| Augmentation | PAPER EXPLICIT count; PAPER UNSPECIFIED transforms | Equation 15 refers to 20 transformations but does not provide a reproducible list. Existing repository flips/rotation are an adaptation, not exact reproduction. |
| Optimization | PAPER EXPLICIT in experimental text | Paper reports 25 epochs, batch size 32 and Adamax discussion. Stage 8 does not train or choose a training protocol. |

## Branch 1: EfficientNetV2S

The paper explicitly calls for a pretrained EfficientNetV2S branch and MFF inputs from successive feature blocks. It does not unambiguously enumerate all tensor taps in the textual Figure 3 description. Stage 8 uses torchvision EfficientNetV2S feature indices **2, 3, 5, 7** as four decreasing-resolution taps. At 384×384, these are approximately **48×96², 64×48², 160×24², 1280×12²** (channels × space). These numbers are validated from the selected torchvision implementation and are not asserted to be Table 1's author implementation. The branch excludes its classifier.

## Branch 2: modified ResNet50

The paper's Figure 6 text explicitly says SCGA acts in the **first two blocks** and non-local processing in **later stages**, but it does not fix every bottleneck insertion point or count of non-local blocks. Stage 8 interprets “blocks” as the four ResNet stages `layer1`–`layer4`, applying SCGA after `layer1` and `layer2`, and one Non-Local Block after `layer3` and `layer4`. Torchvision stage outputs at 384×384 are **256×96², 512×48², 1024×24², 2048×12²**. Pretraining status of this branch is not established as precisely as EfficientNetV2S; the constructor exposes it as an option. Synthetic verification uses no downloaded weights.

## SCGA: Figure 4 and equations 16–22

The spatial path applies 1×1 convolutions with 64, then 16 (dilation 2), then 8 filters, ReLU between them, and a final 1×1 sigmoid mask. The paper next globally averages the mask, broadcasts it, multiplies it with the local mask, and expands to the input channels with a fixed all-ones linear 1×1 projection. Figure 4 also shows a parallel 1×1 projection of the input feature path, added in Stage 8B; its weight policy is not specified and is trainable in this reconstruction. The refined mask multiplies that projected input. A 1×1 kernel with dilation 2 has the same receptive field as dilation 1.

GMA splits masked features into equal channel groups (equation 16). For each group, horizontal and vertical means/maxima are taken along width/height respectively (17–18), mean and max are concatenated for each direction (19), separate 1×1 convolutions and sigmoid yield direction scores (20), scores are added and passed through another sigmoid before group-wise multiplication (21), and attended groups are concatenated (22). Group number, exact group convolution channel sharing, and ordering are not fixed; Stage 8 uses 8 groups when divisible and shared per-group convolution weights. SCGA preserves `[B,C,H,W]`.

## Non-Local Block: Figure 5 and equations 23–28

The paper states an embedded-Gaussian non-local operation: learned `theta`, `phi`, and `g` projections; pairwise dot-product affinity and softmax; projected weighted values; and a residual connection. It explicitly allows subsampling key/value positions (equation 28), without fixing its factor. Stage 8 uses 1×1 projections, `C//2` embedding channels, 2×2 spatial reduction for keys/values when possible, and a 1×1 output projection. All choices beyond the named operators are documented assumptions. The block preserves `[B,C,H,W]`.

## MFF: Section III-C-5, Figure 3, equations 29–30

For each paired scale, EfficientNet features pass through Spatial Attention and ResNet features through GMA. Then feature maps are concatenated, passed through a **depthwise 3×3 plus pointwise 1×1 separable convolution with stride 2**, BatchNorm, and SCGA. Figure 3 draws an arrow from each MFF output to the next MFF: Stage 8B therefore threads a resized previous fused map into the next MFF concatenation. The carry-input operator is an assumption because the MFF inset depicts only two branch inputs. Stride 2 and the main operator ordering are explicit; separable kernel and output channels are not. Pairwise and carry alignment use bilinear interpolation only when dimensions differ. Four fused scales are then aligned to the coarsest fused resolution, concatenated, reduced with a 1×1 convolution, globally averaged and classified. Final fusion alignment/reduction are assumptions because the paper says concatenate all outputs before GAP but omits how unequal spatial shapes are reconciled.

## Final model and scope

Input → EfficientNetV2S and modified ResNet50 → paired multi-scale MFF → aggregate fused maps → GAP → linear classifier → **logits**. Softmax is intentionally outside the module for stable PyTorch losses; it can be applied for probabilities. Intermediate branch and fusion features are available via `forward_features` for future analysis. `num_classes` is configurable; seven-class synthetic verification is **architecture reconstruction, not reproduction of the paper's nine-class result**. No training or dataset inference occurs in Stage 8.

## Stage 8B visual PDF findings

- **Figure 3 (PDF page 7):** shows both branch feature streams, four drawn MFF boxes with arrows connecting successive MFF boxes, a final concatenation node, GAP, and classification. The diagram does not number exact EfficientNet tap stages or specify how the previous MFF output enters the next MFF's detailed two-input inset. Stage 8B corrected the missing serial MFF connection with an explicit carry input; the precise carry fusion is assumed.
- **Table 1 (PDF page 8):** modified ResNet stages 2/3 output 256/512 channels and are each followed by SCGA; stages 4/5 output 1024/2048 and are each followed by a Non-Local Block. EfficientNet stages 0–7 list output channels 24, 24, 48, 64, 128, 160, 256, 1280. Our selected taps 2/3/5/7 have channels 48/64/160/1280, respectively. Table 1 confirms their dimensions, **not** that these are the unique intended MFF taps.
- **Figure 4 (PDF page 9):** confirms the spatial mask pathway and a separate 1×1 input-feature projection. Stage 8B added the latter. Horizontal/vertical mean/max, addition, sigmoid and grouped concatenation match the described GMA path, subject to the chosen group count and shared weights.
- **Figure 5 (PDF page 10):** confirms three 1×1 Q/K/V projections, affinity multiplication, softmax, weighted aggregation, output projection and residual addition. Equation 28 supplies key/value subsampling, although the diagram does not prescribe a factor.
- **Figure 6 (PDF page 10):** is an illustrative local/global feature scenario, not a fully enumerated ResNet topology. Table 1 provides the decisive stage-level insertion evidence.
