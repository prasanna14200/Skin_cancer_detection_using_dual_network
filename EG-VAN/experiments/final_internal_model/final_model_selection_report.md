# Final Internal Model Selection

**HAM10000 INTERNAL MODEL**  
**NOT EXTERNALLY VALIDATED YET**

## Selection

Selected source: Controlled Experiment #5, checkpoint epoch 15. The checkpoint is referenced in place at `experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt`; its SHA256 is `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`. No weights were copied or changed.

Checkpoint selection followed the rule registered before training: among epochs passing all three frozen HAM10000 validation criteria, select the epoch with the lowest validation loss. Thirteen epochs were eligible; epoch 15 had the lowest loss among them, 0.07589111880930122. No test or PH2 results influenced selection.

## Selected Validation Metrics

- Melanoma precision: 0.638298
- Melanoma recall: 0.560748
- Melanoma F1: 0.597015
- Macro-F1: 0.668408
- Nevus recall: 0.936652
- Accuracy: 0.827586

All three frozen success criteria passed at the selected checkpoint.

## Test Evaluation

Test metrics are reporting-only and did not affect model selection: melanoma F1 0.559242, macro-F1 0.699414, nevus recall 0.931953, accuracy 0.833333.

This is an internally selected HAM10000 model only. It has **not** been externally validated using this selected checkpoint. No claim of clinical generalization is made.
