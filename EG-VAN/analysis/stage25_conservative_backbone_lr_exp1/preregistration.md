# Stage 25 preregistration: conservative ResNet50 trunk learning rate

**Status:** Registered before any Stage 25 training. Stage 23 selected epoch 14 remains the current validation candidate. Stage 24 remains closed with `NO_CANDIDATE_SELECTED`.

## Scientific question and hypothesis

Can Stage 23's fully trainable, officially ImageNet-V2 initialized ResNet50 preserve or improve melanoma discrimination and aggregate validation performance when its pretrained trunk receives a lower learning rate than newly initialized EG-VAN modules? The testable hypothesis is that a conservative trunk LR may protect transferable features while leaving the full trunk adaptable from epoch 1. This is a single fixed comparison, not a learning-rate search.

## Frozen factors and single change

Stage 25 independently initializes the ResNet50 trunk with `torchvision.models.ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True, check_hash=True)`; it never loads Stage 23 trained weights. The EfficientNet initialization, EG-VAN architecture, 384-pixel preprocessing and augmentation, lesion-isolated HAM train/validation split (SHA256 `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`), seed 42, physical/effective batch 16, weighted sampler, focal loss (including true-MEL multiplier `1.2815247721366766`), selective FP32 `resnet.nonlocal3` q@k numerical policy with CUDA AMP, maximum 25 epochs, Adamax betas/epsilon/weight decay, and validation measurements are copied from Stage 23.

The **only scientific change** is Adamax LR allocation:

|Group|Parameters|Initial LR|
|---|---|---:|
|Pretrained ResNet50 trunk|`resnet.conv1`, `resnet.bn1`, `resnet.layer1`–`layer4`|`0.0001`|
|New/non-ResNet EG-VAN modules|Every other trainable parameter, including modified ResNet attention modules|`0.001`|

Every parameter is trainable from epoch 1. There is no freeze/unfreeze phase. BatchNorm uses ordinary Stage 23 train mode. Group membership is disjoint and complete by parameter identity. `ReduceLROnPlateau(mode="min", factor=0.5, patience=1)` uses the same validation-loss signal as Stage 23; after each scheduler step the trunk LR is synchronized to one tenth of the new-module LR to preserve the registered ratio when scheduler epsilon suppresses a small group reduction.

## Frozen selection and interpretation

An epoch is eligible only when **all** unchanged Stage 23/Stage 15 gates hold: MEL true positives ≥63/107; MEL recall ≥0.5887850467289719; MEL F1 ≥0.5757731958762886; macro F1 ≥0.6509124104985493; accuracy ≥0.8025152129817445; NV recall ≥0.917209653092006. Among eligible epochs, select the minimum common unweighted validation focal loss; an earlier epoch wins an exact tie. If none qualifies, `NO_CANDIDATE_SELECTED`, with no best checkpoint. Stage 23 epoch 14 is the fixed validation comparison (70/107 MEL TP, MEL F1 0.633484, macro F1 0.671254, accuracy 0.829615, loss 0.074108). Passing the eligibility gates alone is distinct from exceeding Stage 23's selected validation performance; the comparison sidecar reports both.

If Stage 25 preserves/improves both MEL recall and F1, aggregate metrics, classwise metrics, and validation loss, that supports a validation-only improvement. Mixed changes remain mixed. If it misses any gate, it fails candidate selection regardless of lower loss or higher macro F1. A negative result informs the LR-vs-freezing diagnostic but cannot prove the cause of Stage 24's outcome. One validation split and prior project reuse constrain inference; no Stage25 test or external evaluation is authorized here.

## Data boundary, artifacts, and interruption

Only HAM train and frozen validation are loaded. HAM test, PH2, Stage 16 test outcomes, and Stage 20 test-derived reliability outcomes are excluded from training, stopping, and selection. The runner saves per-epoch validation predictions/metrics, a continuous history, recoverable AMP events, atomic last and eligible-best checkpoints, and a final manifest. A resume requires an existing matching last checkpoint, checks finite model/optimizer states and the saved selection history, restores model, optimizer, scheduler, GradScaler, sampler generator, Python/NumPy/torch/CUDA RNG states, and resumes at saved epoch + 1. It never falls back to fresh initialization on `--resume`.

**Training has not been started by this preparation.** The Colab command is documented in `colab_instructions.md` after CPU checks.
