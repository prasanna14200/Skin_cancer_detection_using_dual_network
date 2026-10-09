# Stage 24 Preregistration

**Status: FROZEN BEFORE TRAINING.** This prepares one validation-only experiment. Stage24 training has not started. No schedule sweep is authorized.

## Scientific question and hypothesis

**Question:** Can conservative staged fine-tuning of the ImageNet-pretrained Modified ResNet50 trunk preserve Stage23's melanoma improvement while reducing classwise regressions and improving validation behavior?

**Hypothesis:** Keeping early ResNet representations fixed while adapting later stages, then unfreezing layer2 at a lower learning rate, may improve validation generalization. This is a hypothesis, not an expected result or accuracy guarantee.

## Sole change and references

The sole scientific change from Stage23 is the ResNet50 trunk fine-tuning policy: Phase A freezes conv1, bn1, layer1, and layer2 for epochs 1-5; Phase B unfreezes layer2 at epoch 6 while conv1, bn1, and layer1 remain frozen. ResNet layer3/layer4 and the existing EG-VAN non-trunk modules train throughout with registered discriminative learning rates.

Stage24 is initialized independently from official torchvision `ResNet50_Weights.IMAGENET1K_V2` (`https://download.pytorch.org/models/resnet50-11ad3fa6.pth`) using its hash-checked state-dict API. **Do not load Stage23's trained epoch-14 checkpoint.** Stage23 selected checkpoint SHA256 is recorded only as a frozen comparison/reference artifact: `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`.

## Phase and BatchNorm policy

- Epochs 1-5: `conv1`, `bn1`, `layer1`, and `layer2` parameters have `requires_grad=False`; layer2 remains a reserved optimizer group at its registered LR but has no gradients or Adamax state updates. Frozen BatchNorm modules `bn1`, every BN in layer1, and every BN in layer2 are explicitly held in eval mode after every `model.train(True)` call; their running statistics do not update.
- Epochs 6-25: `conv1`, `bn1`, and `layer1` remain frozen; layer2 becomes trainable without recreating optimizer or scheduler. Layer2 BatchNorm modules switch to training mode, while frozen `bn1`/layer1 BatchNorm modules remain in eval mode.
- Layer3/layer4 and all other Stage23-trainable EG-VAN components remain trainable in both phases.

## Optimizer, learning rates, and scheduler

Construct one Adamax optimizer before epoch 1 with all registered parameter groups. Frozen conv1/bn1/layer1 parameters are excluded. Keep the layer2 group present from the beginning at multiplier 0.05 while its parameters are frozen in Phase A. At epoch 6 change only layer2 `requires_grad`; do not recreate the optimizer, scheduler, or other parameter states. Adamax lazily creates layer2 state on its first update; existing group states remain intact.

| Parameter group | Multiplier of Stage23 base LR | Initial LR |
|---|---:|---:|
| EfficientNet, MFF/fusion/classifier, SCGA and Non-Local modules | 1.0 | 0.001 |
| ResNet layer3 | 0.1 | 0.0001 |
| ResNet layer4 | 0.1 | 0.0001 |
| ResNet layer2 (reserved/frozen epochs 1-5) | 0.05 | 0.00005 |

Use Stage23's `ReduceLROnPlateau(mode="min", factor=0.5, patience=1, eps=1e-8)` once per epoch on common unweighted validation focal loss. After each scheduler step, preserve the registered relative rates by setting every group's LR to the current base-group LR times its fixed multiplier and synchronizing scheduler `_last_lr`. Record every group LR each epoch. No alternate rates or schedules will be tested.

## Frozen factors

Architecture, seven-class order, 384x384 input, preprocessing/cache, augmentation, leakage-aware train/validation split and lesion isolation, EfficientNetV2S initialization, ImageNet V2 ResNet initialization source, SCGA/Non-Local/MFF/classifier, seed 42, sampler, focal alpha/gamma and true-MEL multiplier, batch/workers, base optimizer parameters, scheduler criterion/factor/patience, AMP, GradScaler scale, and selective `resnet.nonlocal3` q@k FP32 policy remain Stage23-identical. Maximum duration is 25 epochs with no early stopping, as in Stage23.

## Data boundary

Only HAM train (8,015) is used for optimization and HAM validation (986) for model selection. Stage24 creates no test or PH2 dataset or loader. HAM test, PH2, external sets, Stage16 HAM test predictions, and Stage20 PH2 results are prohibited for training, selection, tuning, stopping, and decision-making. Their prior historical exposure is acknowledged; this is a validation development experiment, not an untouched confirmatory evaluation.

## Eligibility, selection, and comparison

Preserve the Stage23 gates exactly: MEL support 107; MEL correct >=63; MEL recall >=0.5887850467289719; MEL F1 >=0.5757731958762886; macro F1 >=0.6509124104985493; accuracy >=0.8025152129817445; NV recall >=0.917209653092006. Among eligible epochs choose minimum common unweighted validation focal loss; earlier epoch breaks exact ties. If none qualify, report `NO_CANDIDATE_SELECTED`; do not create a best checkpoint.

Compare the selected candidate with Stage15 epoch 16 and Stage23 selected epoch 14 using validation artifacts only. Report loss, accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, all classwise precision/recall/F1, MEL TP/FN and precision/recall/F1, NV recall, MEL->NV, NV->MEL, and confusion matrices. `CLEAR_VALIDATION_IMPROVEMENT` additionally requires no registered metric regressions relative to Stage23 and MEL recall and MEL F1 no lower than Stage23. A result losing the Stage23 melanoma gain cannot be called clear improvement.

## Artifacts, checkpointing, and resume

Before training, pin hashes for Stage24 config, selection, numerical protocol, fine-tuning policy and all frozen Stage15/Stage23 references. Every epoch writes atomic last checkpoint, history, numerical-event log, validation predictions and metrics (including classwise counts and 7x7 confusion matrix). The checkpoint is authoritative and contains model, optimizer, scheduler, scaler, all RNG states, sampler RNG, phase/trainability, group multipliers/LRs, policy/config hashes, initialization report/provenance, history and validation artifact hashes/payload. Save atomic best checkpoint only when an eligible epoch wins the frozen selection rule; verify it exists before hashing. Never synthesize a selected checkpoint.

Resume verifies checkpoint and sidecars against frozen hashes. A sidecar shorter than the checkpoint may be deterministically reconstructed from checkpoint payload after validating its hashes. A history or validation sidecar ahead of the checkpoint, conflicting content, or unknown future epoch fails closed. Resume from saved epoch 4 starts epoch 5 in Phase A; from epoch 5 starts epoch 6 in Phase B; from epoch 6 starts epoch 7 in Phase B. Preserve optimizer/scheduler state through the phase transition.

## Completion boundary

CPU tests and `train.py --check` must pass before GPU use. `--check` must not download weights, construct test/PH2 loaders, perform inference, create `run/`, or train. After preparation and validation, stop and await explicit user action; do not start Stage24 training or final evaluation automatically.
