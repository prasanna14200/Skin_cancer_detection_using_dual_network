# Stage15 versus Stage23 change matrix

| Factor | Frozen Stage15 | Preregistered Stage23 | Changed? |
|---|---|---|---|
| Experiment identity | `stage15_single_candidate_exp1` | `stage23_resnet50_imagenet_init_exp1` | Metadata only |
| EfficientNetV2S | `EfficientNet_V2_S_Weights.DEFAULT` | Same | No |
| Modified ResNet50 trunk initialization | Random (`weights=None`) | Explicit torchvision `ResNet50_Weights.IMAGENET1K_V2` | **Yes: sole scientific change** |
| ResNet architecture | Modified torchvision ResNet50; stage-level SCGA/NLB | Same | No |
| SCGA / Non-Local modules | Seeded PyTorch initialization | Same seeded initialization; not overwritten by ImageNet source | No |
| MFF / classifier | Four serial MFFs and terminal fusion; PyTorch defaults | Same | No |
| HAM split / lesion isolation | Frozen split SHA `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` | Same | No |
| Preprocessed images and preprocessing code | Existing cache; unchanged preprocessing source | Same cache/source hashes checked | No |
| Input / transforms | 384x384; frozen train augmentation; deterministic validation transform | Same | No |
| Class order | `akiec,bcc,bkl,df,mel,nv,vasc` | Same | No |
| Focal loss | alpha 0.25; gamma 2; MEL multiplier 1.2815247721366766 | Same | No |
| Sampler | WeightedRandomSampler; MEL 1.5630495442733532; others 1.0; replacement; 8,015 samples | Same | No |
| Optimizer / scheduler | Adamax 0.001; betas 0.9/0.999; eps 1e-8; wd 0.0001; ReduceLROnPlateau factor 0.5 patience 1 | Same | No |
| Batch / workers / epochs / seed | 16 / 2 / max 25 / 42 | Same | No |
| Selection eligibility and tie rule | Stage15 registered gates; min unweighted validation focal loss; earlier epoch tie-break | Same byte-equivalent gates and ordering | No |
| AMP / numerical policy | CUDA FP16 AMP; GradScaler 8192 dynamic; FP32 only for `resnet.nonlocal3` q@k; no clipping | Same | No |
| HAM test / PH2 | Not used by Stage15 candidate selection | Prohibited and absent from runner data path | No access |

The Stage23 config changes the experiment name/description as metadata and the ResNet initialization field. Selection and numerical protocol files carry the new experiment identity but retain the same scientific thresholds and numerical settings. Runtime comparisons normalize these metadata fields and fail if any other Stage15 field differs.
