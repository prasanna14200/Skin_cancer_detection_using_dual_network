# Blinded technical-quality review protocol (proposed; not performed)

Review the **raw image** at native resolution without class label, model prediction, confidence, entropy, or feature values. Randomize order; keep repeated images of one lesion in the same review batch but hide lesion identity. Two qualified independent reviewers score every item before discussion. Adjudicate disagreements with a third reviewer or a recorded consensus meeting. Keep original ratings and adjudication separate.

Score each domain 0, 1, 2, or `unable to judge`:

| Domain | 0 | 1 | 2 |
|---|---|---|---|
| Focus / blur | Lesion structures cannot be inspected | Some detail lost; lesion still assessable | Relevant lesion detail visibly sharp |
| Exposure | Relevant detail obscured by clipping/underexposure | Local exposure issue, most lesion visible | No material exposure obstruction |
| Hair / artifact obstruction | Material lesion area obscured | Partial obstruction | No material obstruction |
| Lesion visibility | Lesion not fully visible or boundaries uninterpretable | Visible with limitation | Lesion and surrounding context visible |
| Overall technical adequacy | Repeat acquisition advisable | Borderline; rationale required | Technically adequate for the intended review |

Require a short free-text reason for any 0 or 1, with optional artifact tags (hair, ruler, bubbles, glare, frame, compression, other). Record whether reviewer sees a lesion at all. Adequacy is a human judgment, not a threshold derived from the eight proxies. Disagreement adjudication must be blind to model error status.

Before use, define reviewer qualifications, training examples outside the evaluation set, and an adjudication manual. Report raw agreement and weighted Cohen κ for each ordinal domain (or Krippendorff α if >2 reviewers / missing scores), with lesion-cluster bootstrap intervals. Report adjudicated prevalence and reasons by class, without using model outcomes to revise the rubric.

**Feasibility now:** the repository has the 986 raw validation JPEGs, but no Stage 23 technical-quality annotation file or documented commitment/availability of two independent qualified reviewers. No human-review study has been performed for this Phase A work. Reviewer recruitment and blinded annotation remain pending.
