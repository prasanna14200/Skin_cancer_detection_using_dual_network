# Training-protocol crosswalk

Source: [paper](https://doi.org/10.1109/ACCESS.2025.3561240), Sections III-B and IV-B/C; frozen `experiments/stage15_single_candidate_exp1/config.json`, `numerical_protocol.json`, `selection_rule.json`, `train.py`. “Not specified” means precisely **NOT SPECIFIED IN PAPER**.

| Setting | Paper | Frozen Stage15 implementation | Assessment |
|---|---|---|---|
| Pretrained weights | EfficientNetV2S explicitly pretrained; ResNet status not fully specified | EfficientNetV2S ImageNet default; ResNet50 random | EfficientNet match; ResNet unresolved |
| Frozen/unfrozen layers | NOT SPECIFIED IN PAPER | Both branches trained | Unverifiable |
| Optimizer/LR | Adamax, initial 0.001 | Adamax, 0.001, β=(.9,.999), ε=1e-8, decay .0001 | Named method matches; decay not paper-fixed |
| LR scheduler | Halve after one epoch no validation-loss improvement; stop patience 3 described | ReduceLROnPlateau mode min, factor .5, patience 1 | Close; exact stop behavior differs/unclear |
| Epochs | 25 | 25 intended; persisted original CSV through 24, recovered selected epoch16 | Planned duration matches; original archival failure documented |
| Physical/effective batch | 32 | 16/16 | DIFFERENT; T4 memory-constrained |
| Resolution | NOT SPECIFIED IN PAPER | 384×384 | Unverifiable |
| Focal loss | Focal cross entropy, α=.25, γ=2 | Same base focal; true-MEL term multiplier 1.2815247721366766 | Base close; Stage15 MEL objective is additional factor |
| Augmentation | 20 transformations claimed but identities unavailable; both train and validation augmented | Train flips/rotation, validation deterministic and unaugmented | DIFFERENT; paper validation design has split ambiguity |
| Class balancing | Augmented counts by class (Table 2) | WeightedRandomSampler replacement; MEL weight 1.5630495442733532 | DIFFERENT |
| Early stopping | “Stop patience” 3 named, exact implementation NOT SPECIFIED IN PAPER | Fixed 25 epochs, no early stop | DIFFERENT/ambiguous |
| Model selection | Validation saves best; precise metric/tie logic NOT SPECIFIED IN PAPER | Frozen validation eligibility gates, then minimum unweighted validation focal loss | DIFFERENT and more explicit |
| Seed | NOT SPECIFIED IN PAPER | 42 | Unverifiable |
| Mixed precision | NOT SPECIFIED IN PAPER | CUDA FP16 AMP; `nonlocal3` q@k FP32; dynamic GradScaler | Different/unverifiable, numerically justified |
| Split | 1,332 original test; 37,883 train + 6,686 validation including augmentations | 8,015 train/986 validation/1,014 test, lesion-isolated | DIFFERENT and non-comparable |

The Stage15 registered melanoma objective and sampler were intentionally chosen for its own scientific question. Do not describe Stage15 as a paper-exact training reproduction. Its selected recovered epoch16 is the frozen final model; the original missing epoch16 weights cannot be byte-verified.
