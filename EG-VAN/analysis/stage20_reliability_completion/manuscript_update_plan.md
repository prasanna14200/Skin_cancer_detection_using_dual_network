# Manuscript impact plan — submission remains paused

Do **not** edit the existing PLOS submission copy automatically. Stage 20 is an exploratory post-Stage-16 reliability analysis of immutable saved predictions; the paper must state that Stage 16 outcomes were already known before this extension and that no new untouched test claim is made.

| Manuscript part | Proposed change after Stage 20 audit |
|---|---|
| Title/abstract | Keep the present EG-VAN reconstruction scope unless the paper explicitly includes the qualified reliability extension. Do not call the system a complete clinically validated EG-VAN+ pipeline. |
| Methods | Add fixed saved-probability formulas, ten-bin ECE, seven-class Brier, validation-only entropy selection and ≥80% coverage cutoff. State no temperature scaling/inference/retraining. Specify processed-image quality proxies and lack of acceptability labels. |
| Results | Add validation, HAM and PH² calibration/uncertainty table; HAM/PH² selective review counts with full-cohort baseline alongside retained-subset accuracy. Add quality validation-tercile table as descriptive association. |
| Figures | Candidate supplementary figures: three reliability diagrams, confidence distributions and descriptive risk–coverage curves from this folder. Avoid a quality-gate ROC or clinical acceptability figure without labels. |
| Discussion | Explain why the rule reviewed 23/51 HAM MEL false negatives, leaving 28 unreviewed; describe larger PH² review rate and transfer limitations. Do not claim improved classifier accuracy or clinical safety. |
| Limitations | State prior HAM/PH² project exposure, post-Stage-16 exploratory timing, PH² prior use and label exclusions, processed-image quality proxies, absent quality gate, no clinical quality labels, and pending final-model Grad-CAM. |
| Explainability | Add final-model Grad-CAM only after the bounded T4 pass, download audit and qualitative review. Existing Experiment #5 maps cannot be relabeled as final-model explanations. |

The submission remains paused until this Stage 20 analysis and any requested bounded Grad-CAM output are audited. If the paper remains scoped to Stage 16 reconstruction results, Stage 20 findings may be omitted or clearly identified as exploratory supplementary analysis; no frozen result should be replaced.
