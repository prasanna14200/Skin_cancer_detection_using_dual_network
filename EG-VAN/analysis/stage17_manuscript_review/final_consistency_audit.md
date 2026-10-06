# Final post-edit manuscript consistency audit

**Scope:** editorial quality control of `manuscript/egvan_reconstruction_final.md` and `manuscript/egvan_supplementary_figures.md`. No model training, inference, experiment, checkpoint selection, threshold change, or raw evaluation-artifact edit was performed.

## Reviewer findings and resolution

The pre-edit reviewer audit recorded **0 CRITICAL, 6 MAJOR, 6 MINOR, and 5 PASS** grouped areas. The six major issues were limited literature coverage, published-model task comparability, split/test provenance, architecture explanation, limitations, and missing references. All six were addressed in the manuscript. The six minor issues (title, abstract, preprocessing/training description, PH² protocol detail, AUC-versus-argmax interpretation, and reproducibility pointer) were also addressed. The main result narrative remains validation selection → held-out HAM test → classwise and melanoma errors → PH² external follow-up.

## Artifact and number checks

| Check | Post-edit finding |
|---|---|
| Frozen checkpoint | `experiments/stage15_single_candidate_exp1/stage15_single_candidate_exp1/recovery_epoch16/best_checkpoint.pt` SHA256 recomputed as `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`; matches both Stage 16 manifests and manuscript. |
| Raw final outputs | All four HAM and three PH² output SHA256 values recomputed and matched their respective manifests. No output was changed. |
| Validation/model selection | Epoch 16 is labeled as validation-selected. Table 1 and Results use the saved epoch-016 validation metrics (loss 0.083874, accuracy 0.819473, macro F1 0.661131, MEL 63/107 and F1 0.602871, NV recall 0.939668). The Methods prints the exact frozen gate values from `selection_rule.json`, not rounded operational thresholds. |
| HAM final test | All 17 checked headline quantities agree with `final_ham_test_metrics.json` or `final_ph2_metrics.json` at the displayed precision. HAM n=1,014; accuracy 0.823471, macro F1 0.699582, macro OVR AUC 0.958206, MEL 56/107 and F1 0.568528, NV recall 0.940828. |
| HAM classwise table | Every Table 3 support, precision, recall, F1, TP, FP and FN entry matches the final HAM metrics JSON at six decimals. MEL→NV is 41/51 melanoma false negatives (80.39%) and 41/107 total melanomas (38.32%). |
| PH² follow-up | n=120 (80 NV, 40 MEL); accuracy 0.666667, MEL 11/40, F1 0.392857, AUC 0.542813. Confusion rows are `[69,5,6]` and `[18,11,11]` for NV/MEL/OTHER. The 80 atypical nevi are excluded and OTHER counts as incorrect. |
| Split and task scope | Frozen split SHA256 `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`; train/validation/test n=8,015/986/1,014. The published article's nine-class task is not compared directly with this seven-class study. |

The source Stage 16 analysis had already independently reconstructed metrics from saved prediction CSVs and checked confusion matrices and AUC. The present audit additionally rechecked manuscript tokens and classwise rows against those saved JSONs; it did not produce predictions or load the model for inference.

## Figures, tables, references and language

- Main Tables **1–4** and Figures **1–4** occur once each in order; supplementary Figures **S1–S4** occur once each in order. All eight linked PNGs exist, have at least 300 DPI metadata, and use the frozen Stage 16 figure set. Main and supplementary links resolve. The figure review found legible labels and the expected seven-class or NV/MEL/OTHER axes; no stale experimental result was found.
- Table 1 distinguishes validation from held-out test; Tables 2–3 are HAM test only; Table 4 is PH² external follow-up only. Figure captions identify the same partitions and argmax or probability basis as the saved artifacts.
- References **[1]–[7]** are present, have matching in-text citations, and were checked against the local source paper or primary publisher/publication records. No unverified citation was invented.
- PH² is called **external follow-up**, with prior project use disclosed. It is never described as untouched external validation. Earlier project-level access to the HAM test for a different model is disclosed while Stage 15 test-independent selection is preserved.
- The Methods states that only `resnet.nonlocal3` q@k `torch.bmm` runs in FP32 under CUDA AMP elsewhere, and that this is numerical stabilization with no demonstrated accuracy gain.
- The missing original epoch-16 weight file, exact observable replay, and inability to prove byte identity are disclosed. Limitations also cover class imbalance, 51 missed test melanomas, DF support 11, PH² label mapping and exclusion, prior PH² use, possible but unproven domain-shift mechanism, and absent prospective clinical validation.
- Abstract, Results, Discussion, Limitations, and Conclusion preserve moderate HAM melanoma sensitivity and low PH² melanoma sensitivity/AUC. No clinical-readiness, superiority, state-of-the-art or post-test optimization claim is made.

## Checks executed

`python analysis/stage16_final_evaluation/audit_manuscript.py` passed after editing: no stale result tokens, missing final tokens, broken links, numbering gaps, missing sections, or flagged unqualified claims. Separate read-only checks passed for all seven raw-output hashes, the checkpoint hash, all eight figure DPI values, 17 headline metric values, seven classwise table rows, seven citation numbers, and key error counts. No final scientific result or raw evaluation artifact was modified.

**Final disposition:** manuscript science/evidence consistency **PASS**. Remaining submission work is journal-specific formatting and author review of prose, references, figure placement, and disclosure wording; it does not require another model experiment.
