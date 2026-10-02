# Stage 9 controlled EG-VAN training protocol

## Evidence and frozen comparison

Experiment #5 checkpoint SHA256 is `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`; frozen split SHA256 is `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. Both were verified before and after local work. Experiment #5 history is `training_history.json`, not the CSV named in the Stage 9 request. It records 25 completed epochs and selection of eligible epoch 15. The new runner writes its own CSV history.

The split has 8,015 train, 986 validation and 1,014 test images, 7,470 unique lesions, no lesion crossing, no repeated image IDs, and no missing processed files. Exact class counts are in `dataset_integrity_check.json`. Class order is `akiec,bcc,bkl,df,mel,nv,vasc`. The split is never regenerated.

## Training policy

Use the Stage 8B 58,087,409 parameter EG-VAN with seven logits. The EfficientNetV2S branch loads `EfficientNet_V2_S_Weights.DEFAULT`; its paper pretraining is explicit and this matches Experiment #5. The modified ResNet50 trunk starts randomly because the paper and Stage 8B decision record do not resolve its pretraining. New attention, fusion and classifier modules use PyTorch defaults. The synthetic CPU verification uses untrained branches and does not stand in for the training initialization. Colab will verify the actual weight loading.

Input path: processed HAM RGB JPEG → resize 384×384 → train only horizontal flip, vertical flip and random rotation 15° → tensor → EfficientNetV2S ImageNet mean/std. Validation applies resize, tensor and the same normalization. The processed JPEG itself comes from the existing project pipeline; Stage 9 does not regenerate it. Loss is the same mean focal loss with alpha 0.25 and gamma 2; there are no class specific loss weights. Train sampling uses replacement, 8,015 draws per epoch, melanoma weight 1.5630495442733532, weight 1 otherwise, and generator seed 42.

Use Adamax with LR 0.001, betas (0.9,0.999), eps 1e-8 and weight decay 0.0001. Use ReduceLROnPlateau on validation loss, min mode, factor 0.5, patience 1. Run all 25 epochs without early stopping. Seed Python, NumPy, PyTorch and CUDA with 42; use cuDNN deterministic true and benchmark false, matching Experiment #5. CUDA uses fp16 autocast and GradScaler. Gradient clipping is off by default; the optional CLI value is a documented protocol deviation and should stay unused in the controlled run. Dataloader workers default to two.

The original physical batch is 16. The runner starts its T4 probe at 16 and halves on OOM or insufficient 15% free memory headroom. It records allocated, reserved, peak allocated and free GPU bytes. No physical batch is recommended until this probe runs. If needed, select a divisor of 16 and accumulation steps that give nominal effective batch 16. Gradient accumulation cannot recreate full batch BatchNorm statistics and has different step behavior for incomplete groups, so equality of effective batch does not prove exact training equivalence.

At each epoch, calculate validation MEL F1, macro F1 and NV recall. An epoch is eligible only if they meet 0.5757731958762886, 0.6316582381362074 and 0.9 respectively. Select minimum validation loss among eligible epochs; ties keep the earlier epoch. If none qualify, record `FAIL_NO_ELIGIBLE_CHECKPOINT` and do not claim a best checkpoint. Save `last_checkpoint.pt` after each epoch and best only on improvement. Both include epoch, model/optimizer/scheduler/scaler states, best validation loss, class order, effective config and architecture ID. Last also contains RNG and sampler state for resume. Full training generates history CSV and, if an eligible checkpoint exists, selected checkpoint validation metrics and predictions CSV. Neither HAM test nor PH² is used for model or checkpoint decisions.

## Current evidence and next stages

Local preflight, synthetic forward/loss/backward/optimizer/scheduler/serialization and three tiny HAM optimization steps pass. The local environment has CPU only, so CUDA AMP and T4 memory behavior remain untested. No full training, HAM test inference or PH² inference occurred. After a measured T4 probe, review the reported batch and launch the full run only with explicit approval. A later stage will load the selected best checkpoint, evaluate the frozen HAM test set with seven class metrics, and then run PH² external follow-up with mapped NV/MEL metrics. Those evaluations are outside Stage 9.
