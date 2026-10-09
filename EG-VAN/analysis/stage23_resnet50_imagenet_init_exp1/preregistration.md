# Stage 23 preregistration

**Status: FROZEN BEFORE TRAINING.** No Stage23 training has been run. This is one train/validation-only initialization experiment; no test or external outcomes may select, tune, stop, or rank the candidate.

## Question and hypothesis

**Question:** Does changing only the Modified ResNet50 branch initialization from random initialization to official torchvision ImageNet-pretrained initialization improve generalization of the reconstructed EG-VAN?

**Hypothesis:** ImageNet-pretrained ResNet50 trunk weights may provide useful low/mid-level representations and improve validation performance. This is a hypothesis, not a claim that accuracy will improve or reach 95%.

## Reference and sole change

Reference is the frozen Stage15 candidate and configuration. Stage15 SHA256: `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`. Its executable model builder passes `pretrained_resnet=False`; `ModifiedResNet50` calls `torchvision.models.resnet50(weights=None)`. Stage23 constructs the same seven-class EGVAN architecture and copies official ResNet50 V2 parameters only into matching backbone keys `conv1`, `bn1`, and `layer1` through `layer4`.

The only scientific change is ResNet50 trunk initialization. Stage23 retains the seeded initialization of SCGA, Non-Local Blocks, fusion modules, and classifier. EfficientNetV2S continues to use `EfficientNet_V2_S_Weights.DEFAULT`. No Stage15 checkpoint weights are loaded.

## Weight provenance and compatibility

Use the explicit public torchvision enum `torchvision.models.ResNet50_Weights.IMAGENET1K_V2`, official URL `https://download.pytorch.org/models/resnet50-11ad3fa6.pth`, and `get_state_dict(progress=True, check_hash=True)`. The local project environment reports PyTorch `2.14.0+cpu` and torchvision `0.29.0+cpu`; the enum and URL were inspected without downloading weights. The Colab check must report its own runtime and fail if the enum/API is unavailable. `--check` does not download or load pretrained weights.

The ImageNet source is generic natural-image pretraining, not dermoscopic pretraining. No project-specific dermoscopy data, HAM test, PH2, or external evaluation set is used for initialization. The pretrained classifier tensors `fc.weight` and `fc.bias` are intentionally not copied. Any missing, unexpected, or shape-incompatible backbone key is a hard error. The expected runtime key/shape report is written to `run/initialization_report.json` before epoch 1.

## Frozen protocol

- Data: existing `data/splits/split_leakage_aware.csv`; SHA256 `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`; lesion isolation required; train 8,015, validation 986; test partition forbidden to the runner.
- Class order: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.
- Preprocessed image cache, `src/preprocessing.py`, dataset transforms, 384x384 resolution, EfficientNetV2S initialization, Modified ResNet50 architecture, SCGA, Non-Local Blocks, four MFF modules, and classifier remain unchanged.
- Training configuration: seed 42; 25 maximum epochs; physical/effective batch 16; accumulation 1; two workers; Adamax LR 0.001, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.0001; ReduceLROnPlateau on validation loss, factor 0.5, patience 1; Stage15 focal loss alpha 0.25, gamma 2, true-MEL multiplier 1.2815247721366766; unchanged Stage15 weighted sampler with MEL weight 1.5630495442733532, other weights 1.0, replacement true, 8,015 samples.
- Numerical execution: normal CUDA FP16 AMP with dynamic GradScaler, initial scale 8192; only `resnet.nonlocal3` q@k matmul runs FP32 outside autocast; no gradient clipping. Preserve Stage15 finite-state, overflow logging, and checkpoint validation behavior.
- Checkpoint selection: preserve the Stage15 registered eligibility gates exactly; among eligible epochs select minimum common unweighted validation focal loss, with earlier epoch breaking exact ties. If none qualify, status is `NO_CANDIDATE_SELECTED` and no best checkpoint is created.
- Stopping: maximum 25 epochs, no metric-driven early stopping. Stop immediately on failed provenance, data, numerical, checkpoint, or validation-integrity checks. Resume only from the next epoch in an existing valid `last_checkpoint.pt`.

## Selection measures

Save every validation epoch's accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, per-class precision/recall/F1, support, TP/FP/FN, MEL TP/FN/recall/F1, NV recall, validation focal loss, and 7x7 confusion matrix. The unchanged eligibility gate is: MEL correct >=63/107, MEL recall >=0.5887850467289719, MEL F1 >=0.5757731958762886, macro F1 >=0.6509124104985493, accuracy >=0.8025152129817445, and NV recall >=0.917209653092006. Do not loosen a gate after observing results.

## Artifact and reproducibility policy

Write atomic `last_checkpoint.pt` after each completed epoch, including model/optimizer/scheduler/GradScaler, history, validation artifact hashes/payload, Python/NumPy/torch/CUDA RNG states, sampler generator state, config/rule/numerical hashes, initialization report/hash, architecture metadata, and class order. Also write the cumulative `training_history.csv`, per-epoch validation prediction CSV and metrics JSON atomically. A best checkpoint is written only when an epoch qualifies and wins the frozen rule. Verify an artifact exists before hashing it. Never manufacture a selected checkpoint during finalization.

Fresh training refuses an existing run directory. Resume requires a valid existing `last_checkpoint.pt`, verifies all snapshots/artifacts, restores every RNG/training state, and starts at saved epoch + 1. CPU tests must pass before Colab training. The manifest records code/config/data hashes, runtime and `ham_test_accessed=false`, `ph2_accessed=false`.

## Prohibited access and interpretation

The runner must not instantiate, read, or infer on a HAM test or PH2 dataset, nor use any external test set. Historical Stage16/20 test outcomes were already viewed; Stage23 selection remains restricted to the frozen training and validation partitions and is exploratory. Do not claim the published 98.20% is protocol-comparable. Do not claim Stage23 has improved or reached 95% before the frozen validation comparison exists.
