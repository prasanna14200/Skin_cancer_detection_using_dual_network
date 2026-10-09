# Stage 23 Final Audit

## Disposition

**Final status: COMPLETE; validation outcome: MIXED_VALIDATION_RESULT.** This is a read-only audit of saved Stage23 artifacts plus the frozen Stage15 validation reference. No training, resume, model forward/inference, HAM test access, or PH2 access was performed during this audit. No model or checkpoint was written or modified.

## Training completeness and selection

The Stage23 manifest reports `CANDIDATE_SELECTED_VALIDATION_ONLY`, 25 completed epochs, selected epoch 14, and the frozen split SHA256 `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. The history contains exactly epochs 1-25. Eligibility was independently recomputed from all 25 saved per-epoch metric JSONs using the unchanged registered Stage15 gates. Eligible epochs are **14, 15, and 19**; minimum validation focal loss among them is epoch 14 at `0.07410776640362206`, so the saved selection is correct.

| Selected Stage23 epoch-14 validation metric | Value |
|---|---:|
| Validation focal loss | 0.07410776640362206 |
| Accuracy | 0.8296146044624746 |
| Balanced accuracy / macro recall | 0.6574259626092767 |
| Macro precision | 0.6938315547325767 |
| Macro F1 | 0.6712537990687111 |
| Weighted F1 | 0.8243435992456105 |
| MEL precision | 0.6140350877192983 |
| MEL recall | 70/107 = 0.6542056074766355 |
| MEL F1 | 0.6334841628959277 |
| MEL TP/FN | 70/37 |
| NV recall | 0.942684766214178 |

## Frozen Stage15 validation reference

Comparison uses Stage15 selected epoch 16 only, with the same 986-case validation partition and seven-class order. Stage15 reference metrics SHA256: `630bb7a41c41cd5b125576ca0d14beef7d5b523dba7919680e526501f52c0571`.

| Metric | Stage15 epoch 16 | Stage23 epoch 14 | Delta (Stage23 - Stage15) | Direction |
|---|---:|---:|---:|---|
| Validation loss | 0.08387391282241738 | 0.07410776640362206 | -0.00976614641879532 | Improved |
| Accuracy | 0.8194726166328601 | 0.8296146044624746 | +0.010141987829614507 | Improved |
| Balanced accuracy | 0.632060409296386 | 0.6574259626092767 | +0.0253655533128907 | Improved |
| Macro precision | 0.7110406944440558 | 0.6938315547325767 | -0.017209139711479082 | Regressed |
| Macro recall | 0.632060409296386 | 0.6574259626092767 | +0.0253655533128907 | Improved |
| Macro F1 | 0.6611314670741572 | 0.6712537990687111 | +0.010122331994553924 | Improved |
| Weighted F1 | 0.8134277196826027 | 0.8243435992456105 | +0.010915879563007769 | Improved |
| MEL recall | 0.5887850467289719 | 0.6542056074766355 | +0.06542056074766356 | Improved |
| MEL F1 | 0.6028708133971291 | 0.6334841628959277 | +0.03061334949879857 | Improved |
| NV recall | 0.9396681749622926 | 0.942684766214178 | +0.003016591251885359 | Improved |

### Classwise precision / recall / F1 deltas

| Class | Precision delta | Recall delta | F1 delta |
|---|---:|---:|---:|
| AKIEC | +0.160714 | +0.066667 | +0.107280 |
| BCC | -0.029497 | +0.017241 | -0.000560 |
| BKL | +0.057401 | -0.019231 | +0.010646 |
| DF | -0.300000 | +0.111111 | -0.045113 |
| MEL | -0.003612 | +0.065421 | +0.030613 |
| NV | +0.009236 | +0.003017 | +0.006297 |
| VASC | -0.014706 | -0.066667 | -0.038306 |

MEL TP/FN changed from 63/44 to 70/37. MEL→NV errors decreased from 35 to 26, while NV→MEL errors increased from 23 to 27.

**Decision: `MIXED_VALIDATION_RESULT`.** Accuracy, macro F1, MEL recall, and MEL F1 improved, but macro precision and multiple classwise metrics regressed. This follows the preregistered classification logic; it is not a test-set result and does not establish external or test generalization.

## Checkpoint, numerical, and replay integrity

- `best_checkpoint.pt`: epoch 14, 14 history rows, `best_epoch=14`, best validation loss `0.07410776640362206`; SHA256 `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`.
- `last_checkpoint.pt`: epoch 25, 25 history rows, `best_epoch=14`; SHA256 `8a46b418c79122bbbc9bf82f485c1e1702e5c08d0022e3b02c800d9d4f76e2f3`.
- Both checkpoints contain 1,277 model-state tensors and 766 optimizer slots. Model, optimizer, GradScaler, and numerical-event values are finite. Scheduler `mode_worse=+inf` is its expected sentinel; its active best loss and learning-rate state are finite.
- Best/last embedded history rows match the corresponding CSV prefixes. The last checkpoint's epoch-25 validation payload and numerical events match their saved sidecars.
- All 62 files declared in `experiment_manifest.json` exist and match their SHA256 entries. Manifest SHA256: `48e7fe9d75ac9b4b35750c8d632e3cced5939574e0080b3f12026ea959d44076`.
- The interrupted epoch-23 backup contains 23 history rows matching the completed history prefix. Its epoch-23 validation metrics and prediction CSV are byte-identical to the completed epoch-23 files.
- The numerical log records one GradScaler-skipped update at epoch 21, batch 374, where a nonfinite gradient was detected and the scale backed off from 262144 to 131072. No nonfinite model or optimizer state persisted.

## Comparison report naming

The comparison was **not omitted by finalization**. The runner writes `stage15_vs_stage23_validation_comparison.json`, and the completed manifest includes its SHA256 `c25443d695b072e54d0581c8c2ad827fb5c3b6bbee4617eb4bea489c90f329b5`. The requested short filename `comparison.json` was absent because the runner never writes that basename. The existing comparison was independently recomputed exactly from the frozen Stage15 and Stage23 validation metrics. An audit-folder `comparison.json` is an exact byte-for-byte copy of that manifest-tracked comparison; the experiment manifest and run directory were left unchanged.

## Freeze boundary

Manifest flags are `ham_test_accessed=false` and `ph2_accessed=false`. The audit read only training/validation artifacts and checkpoint state; it did not run inference. The Stage23 checkpoint SHA is unchanged. Stage15 checkpoint/config/history/reference metric hashes were rechecked and unchanged. No Stage24 was started.
