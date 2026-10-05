# Stage 13C frozen candidate rule after Variant A — 2026-10-04

## Sources and observed A state

Read-only sources: `analysis/stage13_melanoma_ablation/selection_rule.json`, `stage13_protocol.md`, `experiments/egvan_melanoma_ablation_exp/train_ablation.py` (notably `select_candidate` and `finalize`), and the completed Stage 13C A manifest/history audit. The registered rule and source code were not changed. A completed 25/25 finite epochs but has `FAIL_NO_ELIGIBLE_CHECKPOINT`, `best_epoch=None`, no `best_checkpoint.pt`, and no selected `validation_metrics.json` or predictions. It therefore has no selected-checkpoint MEL recall.

## Strict reading

The JSON says `no_eligible: "variant has no selected checkpoint"` and `if_A_unavailable_or_no_B_C_qualifies: "NO_CANDIDATE_SELECTED"`. It does **not** give a standalone definition of “A unavailable.” In context, the protocol requires selected validation predictions/metrics for final comparison; the runner's `finalize` rejects any A manifest whose status is not `COMPLETE_VALIDATION_ONLY` and requires A's best checkpoint and `validation_metrics.json`. Its `select_candidate` then computes `max(Stage9 MEL recall, control["mel_recall"]) + 1/107`. Thus a completed A run with no eligible selected checkpoint is **operationally unavailable for final candidate selection**, despite having a complete training history.

No MEL recall from A can be substituted under the frozen rule. In particular, A's unselected final-epoch recall, highest historical recall, or a Stage 9-only proxy is not a registered replacement for selected A validation recall. The frozen Stage 9 anchor supplies a numerical lower bound of `62/107 + 1/107 = 63/107 = 0.5887850467289719`, but this is **not an executable B/C threshold** without a selected A recall. The `max(Stage9, A)` expression remains undefined for this A run. The other frozen candidate floors are accuracy ≥ 0.8025152129817445, macro F1 ≥ 0.6509124104985493, NV recall ≥ 0.917209653092006, and MEL F1 ≥ 0.5757731958762886; they do not override the A-availability condition.

Accordingly, the preregistered final disposition for this completed A state is **`NO_CANDIDATE_SELECTED`**. B/C may in principle complete training and obtain within-run eligible checkpoints, but they cannot become **final selected candidates under the current registered rule** because the A comparator is unavailable. Training B/C solely to select a candidate is therefore not scientifically justified and would spend GPU time without a possible positive selection outcome. A distinct descriptive ablation objective would require a separately registered decision; it is not implied here.

Three separate outcomes must be kept distinct: (1) **training completion/numerical success**: A completed 25 finite epochs; (2) **within-run checkpoint eligibility**: A has no epoch passing all Stage 9 gates and therefore no selected checkpoint; (3) **final B/C candidate eligibility**: cannot be assessed or satisfied under the frozen rule without selected A validation recall. The absence of an eligible A checkpoint does not change the MEL-F1 gate or permit use of an unselected A epoch.

## Implementation discrepancy and ambiguity

The JSON/protocol explicitly require reporting `NO_CANDIDATE_SELECTED` if A is unavailable. The original `finalize` implementation does not emit that result for this case: it raises `ValueError("Variant A not complete with eligible best checkpoint")` before calling `select_candidate`. This is a reporting-path gap, not authorization to substitute another A recall or relax the rule. No finalizer was run or modified. The word “unavailable” is not explicitly defined in the JSON, but the selected-checkpoint-only data flow makes its application to A's no-eligible-checkpoint state the only interpretation consistent with both protocol and code. Any alternative A recall would require a new post-hoc interpretation and cannot be used for the preregistered decision.

No B/C training, HAM test, or PH2 access occurred in this audit.
