# Stage 10 frozen HAM internal test evaluation

EG-VAN checkpoint `60f34ea4cfa6cbf3dce69e8d8bcc82292d8d8813af30f33b2f666a4f06340de0` at epoch 15; frozen split `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. This test set was used for evaluation only. No training, threshold tuning, checkpoint reselection, or PH² access occurred.

Test samples: 1014; prediction integrity: PASS.

Accuracy 0.824458; balanced accuracy 0.681575; macro F1 0.706053; weighted F1 0.821034.

Melanoma precision 0.528302, recall 0.523364, F1 0.525822; NV recall 0.939349.

Experiment #5 accuracy 0.833333; EG-VAN minus Experiment #5 -0.008876. Descriptive comparison only; no significance claim.

Paired correctness: {'both_correct': 796, 'egvan_only_correct': 40, 'experiment5_only_correct': 49, 'both_wrong': 129}. McNemar exact test not performed.

Largest off-diagonal transitions (count, true, predicted): [(34, 'mel', 'nv'), (28, 'nv', 'mel'), (18, 'bkl', 'mel'), (14, 'bkl', 'nv'), (12, 'mel', 'bkl'), (9, 'nv', 'bkl'), (9, 'bkl', 'akiec'), (9, 'bcc', 'nv'), (7, 'akiec', 'nv'), (5, 'akiec', 'bkl')]. MEL→NV 34; NV→MEL 28.
