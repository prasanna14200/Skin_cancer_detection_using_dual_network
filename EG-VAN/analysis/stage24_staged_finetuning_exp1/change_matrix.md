# Stage 24 Change Matrix

| Factor | Frozen Stage23 | Stage24 preregistration | Changed? |
|---|---|---|---|
| Architecture / SCGA / Non-Local / MFF / classifier | Stage8B EGVAN dual branch, four serial MFFs | Same | No |
| EfficientNetV2S initialization | torchvision DEFAULT | Same | No |
| ResNet50 initialization | Official `ResNet50_Weights.IMAGENET1K_V2` | Same official source; independently initialized, no Stage23 trained checkpoint load | No |
| ResNet trainability | conv1, bn1, layer1-4 trainable from epoch 1 | Phase A freeze through layer2; Phase B unfreeze layer2 at epoch 6; conv1/bn1/layer1 frozen throughout | **Yes: sole scientific change** |
| Custom SCGA/Non-Local modules | Trainable from epoch 1 at global Stage23 LR | Trainable throughout at base LR multiplier 1.0 | No |
| Learning-rate groups | One global Adamax group | Four explicit groups with multipliers 1.0 / 0.1 / 0.1 / 0.05 | Part of the sole fine-tuning-policy change |
| Scheduler | ReduceLROnPlateau on val loss, factor .5, patience 1 | Same; group ratios explicitly maintained after each step | No scheduler change; ratio handling implements staged policy |
| BatchNorm behavior | All BN modules enter training mode with `model.train(True)` | Frozen BN modules are restored to eval mode; layer2 BN joins training in Phase B | Part of the sole fine-tuning-policy change |
| Split / lesion isolation | Frozen split SHA `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` | Same | No |
| Input / preprocessing / augmentation | 384x384; Stage23 cache/transforms | Same | No |
| Class order | `akiec,bcc,bkl,df,mel,nv,vasc` | Same | No |
| Sampler | Stage23 weighted sampler, MEL weight 1.5630495442733532 | Same | No |
| Loss / MEL multiplier | Focal alpha .25, gamma 2, MEL multiplier 1.2815247721366766 | Same | No |
| Batch / seed / epochs | 16 / 42 / max 25 | Same | No |
| Numerical policy | CUDA FP16 AMP; selective FP32 nonlocal3 q@k; scaler 8192 dynamic | Same | No |
| Selection gates | Stage23 registered gates and minimum eligible validation loss | Same gates, criterion, and tie-break | No |
| Test / PH2 use | Forbidden for candidate selection | Prohibited and absent from Stage24 data loader construction | No access |

The current preparation includes no `run/` outputs and no training. The fine-tuning/BatchNorm items form one preregistered policy factor; no alternative schedules are registered.
