# Stage 18 submission-copy audit

**Disposition: PREPARATION PASS; SUBMISSION NOT READY.** The frozen master `manuscript/egvan_reconstruction_final.md` was not edited in Stage 18. Its SHA256 at this audit was `41f6bfb2c663ab2b5a793f3e8ccd9f04d60ce9f98d323deacdbe6b3d0fb65cad`. The separate PLOS source `manuscript/submission/plos_one/manuscript_source.md` had SHA256 `0e27006a360f940ce41f9fe6900ebace43129400cc3ab3526494d0bb767b0413` before author completion.

## Scientific comparison with frozen master and Stage 16 artifacts

| Requirement | Finding |
|---|---|
| HAM results unchanged | Submission source retains n=1,014; accuracy 0.823471; macro F1 0.699582; macro OVR AUC 0.958206; MEL 56/107, F1 0.568528; NV recall 0.940828; 41 of 51 MEL misses called NV. All four master tables, including every classwise row, are present unchanged. |
| PH² results unchanged | n=120 mapped, accuracy 0.666667, MEL 11/40, F1 0.392857, AUC 0.542813 and NV/MEL/OTHER policy preserved. Prior project use remains disclosed; no untouched-validation claim. |
| Checkpoint and split | Selected recovered epoch 16 and SHA256 `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5` remain. Frozen split SHA256 `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` remains. The inability to prove byte identity to the missing original epoch-16 weights remains. |
| Numerical policy and limitations | FP16 `resnet.nonlocal3` q@k overflow and selective FP32 arithmetic under AMP elsewhere remain. No accuracy benefit is attributed to the numerical correction. Earlier project-level HAM test access, PH² prior use, weak melanoma sensitivities, DF support 11 and absence of prospective clinical validation remain. |
| References | The same seven verified works are present; citations are renumbered into first-use order `[1]`–`[7]` only in the submission copy. Final Vancouver punctuation/journal abbreviations still require formatting review. |
| Figure/table layout | Four main figure captions are in numerical order and follow the first citing paragraph; no main image is embedded in the submission text. Tables 1–4 remain editable Markdown tables near first citation. S1–S4 captions are at the end. |

Recomputed hashes of all four HAM and three PH² final output artifacts still match the Stage 16 manifests; the frozen checkpoint SHA256 also matches both manifests. No inference was run. The submission copy was checked for every salient frozen number and all seven classwise table rows; the abstract is 171 words before any author edit. All main TIFF copies are RGB/LZW, 300 dpi, ≤2250 px wide, ≤2625 px high and <10 MB. They are presentation copies of saved Stage 16 PNGs; no figures were recomputed from model outputs. S1–S4 PNG copies are present. A visual check of Fig3 preserved its NV/MEL/OTHER labels and counts `[69,5,6]`, `[18,11,11]`.

## Open blockers

- The source contains explicit author, affiliation, corresponding-contact, acknowledgments and AI-disclosure placeholders. Authors must complete and approve them; no identity or ethics statement was invented.
- A PLOS-accepted editable manuscript upload (DOC/DOCX/RTF, or LaTeX PDF) has **not** been produced because reliable conversion tools are unavailable locally. Conversion instructions are supplied. Double spacing, continuous line numbers, page numbers and final Vancouver styling require inspection after conversion.
- A policy-compliant, author-approved public minimal-data and code plan is unresolved. Local paths are not public links. Dataset rights/privacy, checkpoint release, institutional ethics determination, funding, conflicts, CRediT roles and ORCID need author decisions.
- Exact PLOS ONE individual APC and institutional/assistance eligibility were **NOT VERIFIED** from a reliably labeled official fee table; verify at the live submission route. The cover letter still requires author completion and one-page check.
- Final TIFF text size and journal preview quality, and any submission-system fields not exposed by the checked guidelines, need author verification at upload. No files were uploaded.

**Outcome:** the scientific content has survived venue-specific reformatting unchanged, but the package is not yet ready for submission. Next action is author information and release/ethics decisions, followed by manual conversion and upload preview. No further model-development work is part of this stage.
