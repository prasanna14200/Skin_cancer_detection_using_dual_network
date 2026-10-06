# Stage 16 final artifact audit

**PASS.** This is a read-only audit of the saved Stage 16 evaluation artifacts. No inference, training, threshold selection, or model alteration was performed in this analysis.

## Provenance and integrity

The recovered Stage 15 selected checkpoint at `experiments/stage15_single_candidate_exp1/stage15_single_candidate_exp1/recovery_epoch16/best_checkpoint.pt` hashes to `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`, matching both final manifests. The checkpoint's epoch, configuration, numerical policy and finite model state were verified in the [pre-evaluation audit](pre_evaluation_audit.md); it was not loaded for this final analysis. The original epoch-16 weights were not persisted, so their byte identity to the recovered checkpoint is unprovable despite exact observable replay in epochs 14–16.

All five HAM outputs and four PH² outputs are present. Every evaluation artifact SHA256 matches its own final manifest. All source paths listed in both manifests match the saved source hashes, including the checkpoint, split, evaluators, selective q@k policy, and PH² preprocessing source. The HAM split hash is `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`; the PH² cohort manifest hash is `3c130b3b44dcc1052ae29656510a95a4e91df4b8f65c121838941efb375d3842`. Both final manifests record `status=COMPLETE`, epoch 16, Tesla T4, CUDA AMP with FP32 nonlocal3 q@k, and `training_performed=false`. The HAM manifest says PH² was not accessed in that execution; the PH² manifest says HAM test was not used for selection.

## Independent result reconstruction

The 1,014 HAM prediction IDs match the frozen HAM test IDs one for one, with matching true labels and the expected class counts (40 AKIEC, 58 BCC, 104 BKL, 11 DF, 107 MEL, 676 NV, 18 VASC). All seven probabilities per row are finite and in [0,1], sum to one within FP32 rounding tolerance, and reproduce the saved argmax and correctness flags. Their 7×7 confusion matrix matches the CSV and JSON. Independently recomputed classwise TP/FP/FN, precision/recall/F1, accuracy, balanced accuracy, macro precision/recall/F1, weighted F1, and macro/weighted OVR AUC agree with the saved metrics.

The 120 PH² prediction IDs match the frozen included cohort (80 NV, 40 MEL), with mapped truth and seven-class argmax intact. The remaining 80 atypical nevi are excluded by the frozen protocol. Recomputed predictions yield the saved 2×3 confusion matrix `[[69,5,6],[18,11,11]]` (true NV/MEL; predicted NV/MEL/OTHER), saved classwise and aggregate metrics, and MEL probability ROC-AUC. The 17 OTHER predictions are BKL; they were **not** relabeled to NV or MEL and count as incorrect. PH² was previously used in this project; these are external follow-up results, not untouched external validation.

Audit implementation: [final_analysis.py](final_analysis.py), using only saved CSV/JSON files and cryptographic hashes. It produced figures from those saved values without model inference. Figure files are at 350 DPI. No scientific settings or final evaluation artifacts were modified.
