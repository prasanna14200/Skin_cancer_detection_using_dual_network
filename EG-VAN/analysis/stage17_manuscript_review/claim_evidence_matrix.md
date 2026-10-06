# Claim-to-evidence matrix

All numeric claims below are supported by frozen local evidence. Values in the main manuscript are rounded from the saved JSON/CSV artifacts; none are substituted with an older experiment. The Stage 16 auditor independently reconstructed predictions, confusion matrices, classwise metrics and AUC values.

| Claim | Manuscript location | Supporting artifact | Supported? | Correction required? | Notes |
|---|---|---|---|---|---|
| Final model is recovered Stage 15 epoch 16, SHA256 `85fcad4b…4d16d5` | Methods 2.2 | Recovery manifest; `recovery_epoch16/best_checkpoint.pt`; freeze note | YES | NO | Exact observable replay; missing original weight identity remains unprovable. |
| Epoch 16 passed all frozen validation gates and was selected before test | Methods 2.2; Results 3.1 | Stage 15 `selection_rule.json`; epoch-016 validation metrics; recovery manifest | YES | NO | Lowest common unweighted validation loss among eligible epochs. |
| Validation loss 0.083874, accuracy 0.819473, macro F1 0.661131 | Results 3.1; Table 1 | `recovery_epoch16/validation_epochs/epoch_016_metrics.json` | YES | NO | Validation only, not test. |
| Validation MEL recall 63/107 and F1 0.602871; NV recall 0.939668 | Results 3.1; Table 1 | Same epoch-016 metrics/predictions | YES | NO | MEL support 107 on validation. |
| HAM train/val/test counts 8,015/986/1,014 and lesion isolation | Methods 2.1 | Frozen split CSV and Stage 16 preflight | YES | CLARIFY | State hash and earlier project test access caveat. |
| HAM test accuracy 0.823471 and balanced accuracy 0.675097 | Abstract; Results 3.2; Tables 1–2 | `final_ham_test_metrics.json`; predictions; manifest | YES | NO | 1,014 image IDs match frozen test. |
| HAM macro F1 0.699582, weighted F1 0.818013 | Abstract; Results 3.2; Table 2 | Same metrics; classwise and confusion CSV | YES | NO | Independently recomputed. |
| HAM macro OVR AUC 0.958206 and weighted OVR AUC 0.946609 | Abstract; Results 3.2; Figure 2; Table 2 | Probabilities in `final_ham_test_predictions.csv`; metrics JSON | YES | NO | OVR, seven classes; ranking metric. |
| HAM MEL TP 56/107, recall 0.523364, F1 0.568528, FN 51 | Abstract; Results 3.3; Table 3 | HAM predictions and confusion matrix | YES | NO | Do not obscure moderate sensitivity. |
| HAM NV recall 0.940828 | Results 3.3; Table 3 | HAM predictions and classwise CSV | YES | NO | 636/676. |
| HAM MEL→NV 41/51 false negatives (80.39%) | Abstract; Results 3.3; Discussion | HAM predictions; confusion-matrix row MEL | YES | NO | Also 41/107 = 38.32% of all test MEL. |
| PH² 200 total; 120 mapped (80 NV, 40 MEL), 80 atypical excluded | Methods 2.3; Results 3.4 | Frozen PH² manifest and Stage 11 protocol | YES | CLARIFY | Explain external preprocessing and OTHER policy. |
| PH² accuracy 0.666667, balanced accuracy 0.568750 | Results 3.4; Table 4 | `final_ph2_metrics.json`; predictions; manifest | YES | NO | Different cohort/label space. |
| PH² MEL 11/40, recall 0.275, precision 0.6875, F1 0.392857 | Abstract; Results 3.4; Table 4 | PH² predictions and 2×3 confusion CSV/JSON | YES | NO | 18→NV, 11→OTHER. |
| PH² MEL probability ROC-AUC 0.542813 | Abstract; Results 3.4; Table 4 | PH² saved `p_mel`; metrics JSON | YES | NO | Binary truth, score `p_mel`. |
| PH² NV recall 0.8625; OTHER remains incorrect | Methods 2.3; Table 4 | PH² predictions/metrics; frozen Stage 11 protocol | YES | NO | No invented seven-class PH² truth. |
| FP16 overflow localized to nonlocal3 q@k; selective FP32 affinity | Abstract; Methods 2.2; Discussion | Stage 13C arithmetic trace/audit; Stage 15 numerical protocol and Stage 16 manifests | YES | NO | Numerical correction only; accuracy effect unproven. |
| “Limited transfer” under evaluated PH² protocol | Discussion; Conclusion | Lower PH² MEL recall/AUC than HAM test plus protocol differences | QUALIFIED | KEEP QUALIFIER | Descriptive association, not a causal or clinical claim. |
| Original EG-VAN model is reconstructed, not author-code identical | Intro; Methods | Local published PDF; Stage 8B architecture audit | YES | CLARIFY | Published task scope differs; no direct headline metric comparison. |

## Post-edit disposition

The three CLARIFY items were addressed in the manuscript: the frozen split hash and earlier project-level test access are explicit; the PH² mapping, preprocessing and OTHER scoring are stated; and the published nine-class task is separated from this seven-class reconstruction. The exact Stage 15 eligibility thresholds are printed from the frozen selection rule. All supported numeric claims remain tied to the same artifacts. The qualified “limited transfer” statement remains descriptive and does not assert a causal domain-shift mechanism or clinical outcome.
