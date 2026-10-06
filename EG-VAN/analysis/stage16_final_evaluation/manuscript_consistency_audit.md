# Final manuscript consistency audit

**PASS for the new editable Markdown manuscript** `manuscript/egvan_reconstruction_final.md` and `manuscript/egvan_supplementary_figures.md`. The repository had no pre-existing editable `.docx` or `.tex` manuscript. The workspace-root EG-VAN PDF is the published Saeed et al. architecture source used in the reconstruction, not an editable manuscript for these new results; it was not modified. The Stage 16 Results, Discussion, and Limitations drafts were the current editable paper material and supplied the new manuscript's evidence and structure.

## Numeric occurrence review

| Occurrence | Classification | Resolution |
|---|---|---|
| Stage 15 selection floors (MEL F1 0.575773, macro F1 0.650912, accuracy 0.802515, NV recall 0.917210) | REFERENCE | Retained only as preregistered eligibility criteria, not final model results. |
| Stage 15 epoch-16 validation accuracy 0.819473, macro F1 0.661131, MEL recall 63/107, MEL F1 0.602871, NV recall 0.939668 | REFERENCE | Labeled validation/model selection throughout; never described as held-out test. |
| Stage 16 HAM accuracy 0.823471, macro F1 0.699582, macro OVR AUC 0.958206, MEL recall 56/107, MEL F1 0.568528 | FINAL | Used in abstract, Tables 1–3, Results, Discussion, and conclusion with appropriate rounding. |
| Stage 16 PH² accuracy 0.666667, MEL recall 11/40, MEL F1 0.392857, MEL AUC 0.542813 | FINAL FOLLOW-UP | Labeled separate PH² external follow-up, not untouched external validation. |
| Old Stage 9, Stage 13, Stage 14, or prior PH² performance values | REPLACE/REMOVE | None found in the new manuscript; no historical performance number was reused as a final result. |

The final numeric statements were checked against `final_results_tables.md`, `final_error_analysis.md`, and the audited Stage 16 result JSON files. MEL→NV is 41/51 HAM melanoma false negatives (80.39%) and 18/40 PH² melanomas (45%). The latter follows the frozen two-truth/three-prediction-category mapping.

## Structural and scientific review

- Main manuscript sections: Abstract, Introduction, Methods, Results, Discussion, Limitations, Conclusion, Supplementary figures, References. The abstract was written after the body and uses only final results.
- Main tables 1–4 occur in order: validation versus test, HAM overall, HAM classwise, PH² follow-up. Figures 1–4 occur in order with dataset-specific captions. Supplementary Figures S1–S4 each have a caption. All eight existing Stage 16 figures are linked without regeneration, and every relative link resolves.
- AKIEC, BCC, BKL, DF, MEL, NV, and VASC are defined in Methods and used consistently with the frozen class order. PH² OTHER remains a predicted category, not an invented true class.
- The checkpoint is consistently called the **recovered, validation-selected epoch-16** checkpoint. The missing original weight file and exact-observable-replay limitation are stated in Methods, Discussion, and Limitations.
- Numerical stabilization is described as FP32 q@k affinity under CUDA AMP elsewhere. The manuscript does not attribute classification gains to this execution policy.
- Validation, held-out HAM test, and PH² external follow-up are separated in text and tables. No test or PH² result is presented as a checkpoint-selection input. No state-of-the-art or clinical-readiness claim was found.
- PH² had prior project use, its atypical nevi were excluded, and its different label space is disclosed. The manuscript does not claim a demonstrated cause for the external performance gap.

Read-only structural checker: `analysis/stage16_final_evaluation/audit_manuscript.py` passed with no stale result tokens, missing final tokens, broken links, section omissions, or numbering gaps. Research artifacts, figures, checkpoints, and evaluation CSV/JSON files were not modified for paper integration.
