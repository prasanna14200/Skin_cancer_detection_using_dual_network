# Stage 11 frozen EG-VAN PH2 external follow-up

PH² is an external follow-up cohort, not untouched independent external validation: it was previously used in project diagnostics/evaluation. The frozen epoch-15 EG-VAN checkpoint and exact Experiment #5 120-image mapped cohort were used. No training, fine-tuning, threshold tuning, model selection, or Grad-CAM occurred.

EG-VAN PH² accuracy 0.675000; mapped balanced accuracy 0.575000. MEL precision/recall/F1 0.733333/0.275000/0.400000; NV precision/recall/F1 0.752688/0.875000/0.809249.

MEL outcomes mel/nv/other: 11/23/6; NV outcomes nv/mel/other: 70/4/6. Experiment #5 outcomes MEL mel/nv/other: 12/15/13; NV nv/mel/other: 63/8/9.

Paired correctness: {'both_correct': 68, 'egvan_only_correct': 13, 'experiment5_only_correct': 7, 'both_wrong': 32}. McNemar exact test: {'performed': True, 'b': 13, 'c': 7, 'exact_p_value': np.float64(0.26317596435546875)}. A p-value, if present, is not a clinical superiority claim.

Internal-to-external differences are descriptive cross-dataset changes, not causal domain-shift magnitudes. HAM is a seven-class population whereas the mapped PH² cohort contains only NV/MEL truth, so overall accuracies have different class composition.

Internal/external metrics and PH² minus HAM differences: [{'model': 'EfficientNetV2S Experiment #5', 'ham_accuracy': 0.8333333333333334, 'ph2_accuracy': 0.625, 'accuracy_difference': -0.20833333333333337, 'ham_melanoma_recall': 0.5514018691588785, 'ph2_melanoma_recall': 0.3, 'melanoma_recall_difference': -0.25140186915887847, 'ham_nevus_recall': 0.9319526627218935, 'ph2_nevus_recall': 0.7875, 'nevus_recall_difference': -0.1444526627218935}, {'model': 'Reconstructed EG-VAN', 'ham_accuracy': 0.8244575936883629, 'ph2_accuracy': 0.675, 'accuracy_difference': -0.14945759368836287, 'ham_melanoma_recall': 0.5233644859813084, 'ph2_melanoma_recall': 0.275, 'melanoma_recall_difference': -0.24836448598130834, 'ham_nevus_recall': 0.9393491124260355, 'ph2_nevus_recall': 0.875, 'nevus_recall_difference': -0.06434911242603547}].
