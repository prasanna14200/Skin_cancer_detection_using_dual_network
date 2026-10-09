# Stage 23 Training-Policy Audit

## Scope and conclusion

This is the Stage24-requested **Stage23 policy audit only**. It verifies executable Stage23 construction, optimizer creation, and training-step behavior. No Stage24 schedule/configuration or training code was created. No training, resume, model forward/inference, HAM test access, or PH2 access occurred.

**Finding:** Stage23 trains the complete EG-VAN, including all Modified ResNet50 trunk stages, from epoch 1. There is no freeze, unfreeze, or per-layer learning-rate policy in the Stage23 code. The ResNet trunk starts from ImageNet V2 tensors; its trainability is unaffected by those tensor copies.

## Executable path

1. `experiments/stage23_resnet50_imagenet_init_exp1/train.py` calls `build_stage23_components(...)` before entering the epoch loop.
2. `experiments/stage23_resnet50_imagenet_init_exp1/initialization.py` constructs `EGVAN(..., pretrained_resnet=False)`. This first constructs the existing ResNet branch with torchvision `weights=None`, preserving the Stage23 seeded initialization for the custom modules.
3. The helper then obtains `ResNet50_Weights.IMAGENET1K_V2` and copies only shape-compatible torchvision tensors under `conv1.`, `bn1.`, and `layer1.` through `layer4.`. It does not replace the ModifiedResNet50 architecture. The four SCGA/Non-Local custom modules remain newly initialized. The saved initialization report confirms all six trunk prefixes loaded and reports no shape mismatches.
4. The same helper calls `base.optimizer_scheduler(model)` on the complete EGVAN model. `experiments/egvan_melanoma_ablation_exp/train_ablation.py` defines this as one `Adamax(model.parameters(), ...)` parameter group; it does not filter or freeze the ResNet parameters.
5. Stage23 calls `checked_train_batch(...)` for each training batch. The inherited helper calls `model.train(True)`, zeroes the single optimizer, computes the forward/loss, backpropagates through the model, then calls `scaler.step(optimizer)` and `scaler.update()`. There is no layer-specific `requires_grad` change before or during this loop.

## ResNet block policy in Stage 23

| Module | Initialization | Trainable from epoch 1? | Other training behavior |
|---|---|---:|---|
| `resnet.conv1` | ImageNet V2 tensor copied into existing module | Yes | Included in the single Adamax group |
| `resnet.bn1` affine parameters | ImageNet V2 tensor copy | Yes | Module is in training mode; BatchNorm running statistics update |
| `resnet.layer1` | ImageNet V2 tensor copy | Yes | Included in the single Adamax group |
| `resnet.layer2` | ImageNet V2 tensor copy | Yes | Included in the single Adamax group |
| `resnet.layer3` | ImageNet V2 tensor copy | Yes | Included in the single Adamax group |
| `resnet.layer4` | ImageNet V2 tensor copy | Yes | Included in the single Adamax group |
| `resnet.scga1`, `resnet.scga2` | New seeded PyTorch initialization | Yes | Included in the same group |
| `resnet.nonlocal3`, `resnet.nonlocal4` | New seeded PyTorch initialization | Yes | Included in the same group; `nonlocal3` q@k uses the existing FP32 precision island |

No explicit `requires_grad` freeze/unfreeze operations were found in Stage23 or the EG-VAN ResNet/model definitions. The normal PyTorch parameter default is `requires_grad=True`; Stage23 passes all model parameters to the optimizer. `model.train(True)` also means frozen BatchNorm running statistics would need deliberate handling if a later stage intends to freeze `bn1` behavior, not only its affine gradients. No Stage24 BatchNorm policy is selected here.

## Optimizer, scheduler, and numerical policy

Stage23 uses the Stage15 optimizer/scheduler implementation unchanged:

- One Adamax parameter group over `model.parameters()`; initial LR `0.001`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.0001`.
- `ReduceLROnPlateau(mode="min", factor=0.5, patience=1)`; Stage23 advances it once per epoch using validation loss. This is a single global learning rate, not discriminative layer rates.
- Maximum 25 epochs, 384x384 inputs, physical/effective batch 16, seed 42.
- Existing Stage23 initialization: EfficientNetV2S `DEFAULT`; ResNet50 `IMAGENET1K_V2` trunk; custom EG-VAN modules retain seeded initialization.
- Existing selective FP32 `resnet.nonlocal3` q@k policy under CUDA FP16 AMP and dynamic GradScaler, initial scale 8192; no gradient clipping.

The Stage23 numerical protocol, sampler, focal loss/MEL multiplier, split, and eligibility/selection rule are not changed by this audit.

## Provenance

The frozen Stage23 manifest records 25 completed epochs, selected epoch 14, and `CANDIDATE_SELECTED_VALIDATION_ONLY`. Stage23 checkpoint SHA256: `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`.

| Audited input | SHA256 |
|---|---|
| Stage23 config | `67a9d77d8464ef51d94f5861004084b179c3c42159e172c680d85952514d7afa` |
| Stage23 selection rule | `a03213d5db62af968a425f3557773aa14c85adeabdbaeb8e6d46bf4d5481111d` |
| Stage23 numerical protocol | `06222c374635f15ad99642e1bdbeaf4c6c69959dd753d6b332cdb753b87abdfb` |
| Stage23 initialization specification | `145758267bb655d21f70d69c8c384259e4b4074a7596dcf99d2dcb0a2f7a9d76` |
| Stage23 runner | `d009803461a52818014f82a0d791650a86abf3737e98fb0d198eb4224136e967` |
| Stage23 initialization helper | `56a5b767c4c204eac64fbbc8a4c91c1ee7694fdcc3e87ed50bdaf9e0d7b483be` |
| Stage23 run manifest | `48e7fe9d75ac9b4b35750c8d632e3cced5939574e0080b3f12026ea959d44076` |
| Stage23 initialization report | `c86113181d9a84ecb7f3dd38e2e43f1e2926b564a682ed3162a16e7f0b4e5fb9` |
| Stage15 reference config | `3ad8c205e06cde8203a869be0a6cc9643ed0ca5b6beb553b57d53d4dce9d25f2` |
| Stage15 reference training history | `2df7edd11a23658e56dc723ee57638476579cb3620d9c5d10039eefa07941fd0` |

Stage15 reference paths and Stage23 artifacts were read only. No Stage24 experiment directory or training outputs were created. This document does not choose a warm-up length, LR multiplier, or Stage24 training schedule; those decisions belong to a later preregistration step.
