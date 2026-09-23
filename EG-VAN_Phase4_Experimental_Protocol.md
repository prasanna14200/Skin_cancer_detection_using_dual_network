# EG-VAN Phase 4 — Experimental Protocol & Implementation Freeze

## Status

This protocol is frozen as of this document.

Any deviation during implementation must be logged in the "Deviations Log" at the bottom with a date and a reason. It must not be silently changed.

---

## 1. Research Question & Hypothesis (carried from Phase 3)

### Research question

Does EG-VAN's dual local/global attention design (SCGA + NLB) provide a measurable advantage over its individual backbones once evaluated under leakage-aware splitting, and does that advantage persist under external-domain validation?

### Hypothesis

The paper's reported 98.2% accuracy is likely sensitive to lesion-level leakage in its evaluation protocol. The attention modules' true contribution will likely be smaller under a leakage-aware split, but may still prove more robust than plain CNN baselines when tested on external, out-of-distribution data.

This is a testable hypothesis, not a claim we are trying to prove in advance.

---

## 2. Staged Scope (sequential, not simultaneous)

| Stage | Content | Gate to proceed |
|---|---|---|
| Stage A | Reconstruct EG-VAN core on HAM10000. Compare naive image-level split vs. lesion-wise split. | A must produce a real, stable accuracy delta before Stage B starts. |
| Stage B | Add external validation: PH2 first, then ISIC2019. | B's external numbers must be in hand before calibration is added. |
| Stage C | Full ablation (SCGA on/off × NLB on/off, 4 variants) under both split protocols and external data. Efficiency table (params/FLOPs/latency). Calibration (temperature scaling) as the final reliability layer. | — |

### Critical scope rule

If time runs out after Stage A or B, that is a complete, publishable-scope project on its own — not a failure.

This is explicitly stated so scope-creep pressure does not force an unfinished Stage C.

---

## 3. Datasets

### Primary dataset (training + internal evaluation)

- HAM10000
- 7 classes: AKIEC, BCC, BKL, DF, MEL, NV, VASC
- 10,015 images
- Source: Harvard Dataverse (doi:10.7910/DVN/DBW86T)
- License: CC BY-NC 4.0 (non-commercial, attribution required)

### Important scope reduction (frozen)

We are not attempting to reconstruct the paper's merged 9-class dataset.

The identity of the second dataset is unconfirmed by the current Phase 3 finding. This is a deliberate, documented scope reduction.

### External validation datasets (Stage B)

1. PH2
   - ~200 images
   - 3 classes: common nevus, atypical nevus, melanoma
   - Used first as a fast smoke test for the external-validation pipeline

2. ISIC 2019
   - ~25,331 images
   - 8 classes
   - CC BY-NC
   - Used second as a larger, more realistic domain-shift test

### Class mapping caveat (frozen assumption)

Neither PH2 nor ISIC2019 maps 1:1 onto HAM10000's 7 classes.

We will evaluate on the overlapping class subset only (for example, melanoma vs. nevus vs. BCC where present in both), and explicitly report which classes were excluded and why.

We will not force a mapping that misrepresents the data.

---

## 4. Splits (the load-bearing decision of this whole project)

### Split A — Naive split (replicating likely paper protocol)

- random image-level 80/10/10 split
- train/val/test
- stratified by class
- This is what we believe the paper effectively did.
- Important note: HAM10000 images from the same lesion can appear in both train and test.

### Split B — Leakage-aware split

- group by lesion_id before splitting
- all images of the same lesion remain in the same partition
- this is lesion-level, not patient-level
- HAM10000 metadata does not reliably expose patient identity, so we will not claim patient-level control.

### Frozen rule

Both splits use the same class-stratification logic and the same random seed (fixed and logged).

The only variable that changes is the leakage control itself.

---

## 5. Models

| Model | Role |
|---|---|
| EfficientNetV2S alone (ImageNet pretrained, fine-tuned) | Baseline 1 |
| Modified ResNet50 alone (no SCGA, no NLB) | Baseline 2 |
| EG-VAN full reconstruction (EfficientNetV2S + ResNet50 + SCGA + NLB + MFF fusion) | Main model |
| EG-VAN without SCGA | Ablation (Stage C) |
| EG-VAN without NLB | Ablation (Stage C) |
| EG-VAN without SCGA and without NLB (= plain dual-branch, no attention) | Ablation (Stage C) |

### Frozen fallback for memory pressure

If the full Non-Local Block causes OOM at any point, we substitute a lighter global-context block (SE-style squeeze-excitation) for that specific run only.

This must be clearly reported as a memory-constrained run.

We must not silently swap the module and then claim it is the original paper design.

---

## 6. Training Configuration (paper-matched where feasible; deviations logged)

- Framework: PyTorch
- Optimizer: Adamax
- Initial LR: 0.001
- Scheduler: reduce-on-plateau by 0.5, patience 1
- Loss: focal loss, alpha = 0.25, gamma = 2
- Batch size: 16 as the frozen Colab-safe default
- Increase to 32 only if memory allows, and log it per run
- Epochs: 25 unless early stopping triggers earlier
- Mixed precision: enabled by default
- Image size / augmentation: match the paper's described pipeline as closely as feasible
- Hair removal, Gray World + Retinex color balancing, and standard augmentations are included where faithfully reproducible
- If any augmentation cannot be faithfully reproduced because the paper is underspecified, we log it as an approximation

---

## 7. Metrics (frozen per stage)

### Stage A metrics

- Accuracy
- Macro-F1
- Per-class recall, especially MEL and AKIEC
- Confusion matrix
- Report naive split vs. leakage-aware split side by side

### Stage B metrics

- Same metrics as above
- computed separately on PH2 and ISIC2019
- never merged into training

### Stage C metrics

- ECE
- Brier score
- reliability diagram
- parameter count
- FLOPs
- CPU/GPU latency

---

## 8. Success / Failure Criteria (defined before seeing results)

### Reproduction succeeds if

Our naive-split EG-VAN reconstruction lands within a reasonable range of the paper's reported 98.2%.

We are not demanding an exact match because the architecture had to be reconstructed from equations and diagrams, not copied code.

A materially lower number is not a failure if the leakage-aware and architecture story remains coherent. It is reported and explained.

### Hypothesis is supported if

- the naive-vs-leakage-aware gap is non-trivial (for example, several accuracy points), and/or
- the attention-module ablation shows a smaller contribution under leakage-aware conditions than the paper's own ablation table claims.

### Hypothesis is not supported if

Leakage-aware results track closely with naive results.

That itself is a valid finding and should be reported honestly. It would mean EG-VAN's evaluation was not materially leaky, which is useful information regardless of the outcome.

### We are not committed to proving the paper wrong

Either outcome is a valid thesis result.

---

## 9. Implementation Order (strict sequence)

### Stage A

1. Prepare HAM10000 metadata and lesion_id grouping
2. Build naive split baseline pipeline
3. Build leakage-aware split pipeline
4. Train EfficientNetV2S baseline
5. Train ResNet50 baseline
6. Reconstruct EG-VAN full model
7. Run naive-split evaluation
8. Run leakage-aware evaluation
9. Compare results and decide whether Stage B is justified

### Stage B

1. Validate on PH2
2. Validate on ISIC2019
3. Record class-overlap restrictions
4. Compare external-domain performance against internal split results
5. Only then proceed to calibration analysis

### Stage C

1. Run SCGA/NLB ablation matrix
2. Report performance under both split protocols
3. Measure parameters/FLOPs/latency
4. Add temperature scaling and calibration metrics
5. Report final reliability/efficiency story

---

## 10. Experimental Integrity Rules

- No train/test leakage through lesion grouping
- No silent data leakage through preprocessing
- No reusing external test data for tuning
- No adding new metrics retroactively after results are in
- No changing the split logic after the protocol is frozen without logging the deviation
- No replacing the original model with a simpler proxy unless explicitly declared as a fallback for resource constraints
- No claiming novelty without a clear prior-art check that matches the actual contribution

---

## 11. What counts as a valid result

A valid result is one that follows the frozen protocol and reports the experiment honestly.

Examples of valid results:

- EG-VAN outperforms baselines under naive split but not under lesion-aware split
- EG-VAN remains strong under external validation but not much better than simpler backbones
- Calibration removes some overconfidence without changing ranking
- Memory or latency constraints make SCGA/NLB less attractive in practice

Examples of invalid results:

- random split with lesion overlap hidden from the report
- external validation after tuning on the external data
- swapping in a simplified attention module without documenting it
- claiming the model is equivalent to the paper when the architecture was not faithfully reconstructed

---

## 12. If reproduction fails or is unclear

If the EG-VAN reconstruction cannot be made stable or faithful enough, we still proceed with a documented partial result.

This is acceptable because the scope is still valid:

- faithful reconstruction attempt
- leakage-aware benchmark
- internal vs. external evaluation
- reliability and efficiency analysis

The project remains publishable and scientifically meaningful even if the original architecture is not fully reproduced to the exact intended score.

---

## 13. Deviation Log

This section starts empty. Any implementation deviation must be recorded here with:

- Date
- Planned
- Actual
- Reason

Example format:

| Date | Planned | Actual | Reason |
|---|---|---|---|
| YYYY-MM-DD | ... | ... | ... |

---

## 14. Implementation Freeze Statement

This document freezes the experimental design before code implementation begins.

No model code, dataset logic, or training script should be considered final until this protocol has been reviewed and approved.

This is the precise boundary between research design and implementation.

---

## 15. Summary of project status

| Phase | Status | What we have |
|---|---|---|
| Phase 1 | ✅ Complete | EG-VAN paper, architecture, components, dataset, experiments, limitations |
| Phase 2 | ✅ Complete | GitHub/code audit + verification + repository filtering |
| Phase 3 | ✅ Complete | Focused prior-art audit + research question + candidate gap |
| Phase 4 | ✅ Frozen | Experimental protocol and implementation gate |
| Phase 5 | ⏭️ Next | EG-VAN implementation |
| Phase 6 | ⏭️ Next | Real experiments |
| Phase 7 | ⏭️ Next | Results analysis |
| Phase 8 | ⏭️ Next | Report / demonstration |

---

## 16. Working principle for the next step

We are now moving from research design to engineering implementation, but only under a frozen protocol.

This prevents the common failure mode where the model is coded first, then the “experiment” is discovered afterward.

The correct flow is:

1. Freeze the question
2. Freeze the protocol
3. Implement the exact experiment
4. Report what happened honestly
5. Only then discuss implications

---

## Final note

This is the clean and defensible research structure for the project:

- We do not claim the paper is wrong in advance.
- We test whether leakage and external-domain validation materially change the result.
- We do not turn the project into a generic skin-lesion classifier repo.
- We keep the contribution architecture-specific, evaluation-specific, and evidence-driven.

This is the correct setup for the next implementation phase.
