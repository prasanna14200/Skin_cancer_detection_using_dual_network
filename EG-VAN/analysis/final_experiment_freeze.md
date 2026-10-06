# EG-VAN final experiment freeze

**Study status: CLOSED through Stage 16.** The final model is the Stage 15 validation-selected, recovered epoch-16 checkpoint. Stage 16 performed the final HAM10000 held-out test and PH² external follow-up. No further model development, threshold tuning, checkpoint selection, or test-driven optimization is part of this study.

| Frozen item | Status / path |
|---|---|
| Final experiment | Stage 15 selected candidate + Stage 16 final evaluation |
| Final checkpoint | `experiments/stage15_single_candidate_exp1/stage15_single_candidate_exp1/recovery_epoch16/best_checkpoint.pt` |
| Final checkpoint SHA256 | `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5` |
| Frozen HAM split SHA256 | `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` |
| HAM final evaluation | COMPLETE; `analysis/stage16_final_evaluation/final_ham_test_manifest.json` |
| PH² external follow-up | COMPLETE; `analysis/stage16_final_evaluation/final_ph2_manifest.json` |
| Further model development | CLOSED for this study |
| Test-driven optimization | NOT PERFORMED |

Authoritative final documents: `analysis/stage16_final_evaluation/final_evaluation_audit.md`, `final_results_tables.md`, `final_error_analysis.md`, `stage16_final_report.md`, and the Results, Discussion, and Limitations drafts in that folder. The manuscript source is `manuscript/egvan_reconstruction_final.md`, with supplementary captions in `manuscript/egvan_supplementary_figures.md`. Figure images are in `analysis/stage16_final_evaluation/figures/`.

The epoch-16 original weight file did not persist. The recovered checkpoint reproduces the available epoch-14–16 history and validation evidence exactly, but byte identity to the missing original weights cannot be established. This limitation remains attached to every result. PH² was used previously in the project and must be called an **external follow-up**, not untouched external validation. The selective FP32 nonlocal3 q@k policy addressed overflow, not demonstrated accuracy gain.

## Suggested Git archival commands

Run from `D:\Cancerdetection` after reviewing `git status`. The repository currently ignores `*.pt`; retain that policy and do not force-add the approximately 699 MB checkpoint.

```powershell
git -c safe.directory=D:/Cancerdetection status --short
git -c safe.directory=D:/Cancerdetection add -- EG-VAN/manuscript EG-VAN/analysis/final_experiment_freeze.md EG-VAN/analysis/stage16_final_evaluation
git -c safe.directory=D:/Cancerdetection diff --cached --stat
git -c safe.directory=D:/Cancerdetection commit -m "Finalize EG-VAN Stage 16 evaluation and manuscript"
git -c safe.directory=D:/Cancerdetection tag -a egvan-final-stage16 -m "Frozen Stage 16 final evaluation"
```

No commit, tag, or push was executed by this documentation task. In particular, do not run `git push` without a separate explicit instruction.
