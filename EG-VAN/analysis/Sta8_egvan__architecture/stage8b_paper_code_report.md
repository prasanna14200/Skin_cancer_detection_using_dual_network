# Stage 8B final paper-to-code audit

## Status

**READY_FOR_STAGE_9** for protocol design: the visually verifiable Figure 3/Table 1 topology is implemented, the component tests and synthetic passes succeed, and remaining operator choices are declared assumptions. This does not establish author-code identity or reproduce the paper's classification results. Stage 9 training must be separately approved and prespecified.

## Original PDF provenance and visual review

Local PDF: `D:\Cancerdetection\EG-VAN_A_Global_and_Local_Attention-Based_Dual-Branch_Ensemble_Network_With_Advanced_Color_Balancing_for_Multi-Class_Skin_Cancer_Recognition.pdf`; SHA256 `bac1b99167ad820232ecc6b47a3d562d2d9ca55ce19de36fba97b919e8aacee3`. Figure 3 and Table 1 were visually inspected on PDF pages 7–8; Figures 4 and 5 on pages 9–10. Figure 6 on page 10. The text and visuals were compared directly with the modules. No data images were used.

## Stage 8B corrections

Stage 8 omitted arrows between MFF boxes and the terminal branch inputs to the final concat. Stage 8B added a previous-MFF carry into the next MFF's concatenation and both terminal branch maps to the final concat. The carry operator and spatial alignment remain assumptions because Figure 3 does not specify their tensor operations. Figure 4's input-side 1×1 convolution was also added.

## Table 1 and stage placement

| ResNet stage | 384×384 output | Paper module | Code module | Audit |
|---|---|---|---|---|
| layer1 (paper stage 2) | 256×96×96 | SCGA | SCGA post-stage | Paper-consistent assumption |
| layer2 (paper stage 3) | 512×48×48 | SCGA | SCGA post-stage | Paper-consistent assumption |
| layer3 (paper stage 4) | 1024×24×24 | NLB | NLB post-stage | Paper-consistent assumption |
| layer4 (paper stage 5) | 2048×12×12 | NLB | NLB post-stage | Paper-consistent assumption |

Table 1 confirms the chosen EfficientNet tap channel values 48, 64, 160 and 1280 at stages 2, 3, 5 and 7. The paper does not label their exact MFF connections, so the tap subset remains an assumption.

## SCGA, NLB and MFF

Figure 4 and equations 16–22 support the 64→16(dilation 2)→8→sigmoid mask path, mask GAP refinement, directional mean/max grouping, 1×1 directional projections, addition, sigmoid and group concatenation. Figure 4 also depicts an input-side 1×1 projection, added in Stage 8B. Its trainability is assumed. Figure 5 and equations 23–28 support theta/phi/g projections, embedded-Gaussian affinity, softmax, weighted values, output projection and residual; key/value subsampling is implemented with an assumed factor of two. MFF follows the paper's SA/GMA, concatenation, separable stride-2 convolution, BN and SCGA order. Alignment policy remains assumed.

## Parameter-count discrepancy

Torchvision ResNet50 with its 1000-class head has 25,557,032 parameters. Its feature trunk has 23,508,032. This reconstruction adds 10,885,762 trainable attention parameters, yielding 34,393,794 in modified ResNet50. The full reconstruction has 58,087,409. The largest additions are Q/K/V/output projections in the two NLBs; SCGA contributes smaller convolutional and new input-projection weights. The paper claims unchanged ResNet50 parameter count but does not give enough implementation/accounting detail to reproduce that claim. No layers were removed to force parity. See `parameter_count_breakdown.csv`.

## Tests and synthetic passes

39 repository tests passed. Batch 1 at 384×384 and batch 2 at 128×128 produced finite `[B,7]` logits. A synthetic 64×64 backward pass produced finite gradients on all 766 trainable parameter tensors, including the serial MFF carry. No optimizer step, training, HAM inference or PH² inference occurred.

## Remaining assumptions and next action

Exact EfficientNet tap subset, carry concatenation, eight GMA groups, shared GMA weights, NLB reduction factor, fusion widths, spatial alignment, trainability of Figure 4's input projection and final multi-scale aggregation are not fully specified by the paper. These choices are documented and do not contradict the visible topology. Proceed to Stage 9 protocol design; do not train until that protocol is approved.
