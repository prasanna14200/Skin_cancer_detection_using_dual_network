# Blinded technical image-quality rubric

Review the **unaltered raw image at native resolution**. Rate what is visible, not diagnosis, malignancy, histology, or suitability for a clinical decision. Do not infer a technical defect solely from dark lesion pigment, low saturation, fine lesion texture, or an unusual lesion appearance. Do not view model predictions, confidence, entropy, proxy scores, class labels or source metadata.

For every domain use `0 = no material technical issue`, `1 = visible issue with most relevant image content still assessable`, `2 = material issue that obscures relevant image content or compromises the intended model input`, or `U = cannot assess`. Record a brief, observable reason for each `1`, `2` or `U`. A nonapplicable domain may be `NA` with a reason; never use `NA` merely because the reviewer is uncertain.

| Domain | What to inspect | Examples of score 2; do not infer diagnosis |
|---|---|---|
| Focus/blur | Sharpness of visible lesion and adjacent skin, including motion blur and focal plane | Substantial lost detail across the region of interest |
| Underexposure | Dark clipping or poor illumination hiding visible detail | Broad shadow/clipping obscuring relevant content; dark pigment alone is not underexposure |
| Overexposure | Bright clipping, glare or saturation hiding visible detail | Broad saturated/glare region obscuring content; pale skin alone is not overexposure |
| Contrast | Whether technical tonal compression makes boundaries or texture visibly indistinct | Broad loss of usable tonal separation; low lesion/skin contrast alone is not necessarily a defect |
| Compression artifacts | Blocking, ringing, banding or other encoding artifacts | Artifacts materially obscure content, beyond normal file-format appearance |
| Occlusion/artifacts | Hair, bubbles, ruler, ink, frame, gel, reflections or other obstructions | Material part of visible lesion/context is obscured; add artifact tags and location |
| Lesion visibility | Is the object of interest in frame and visible enough for the defined *image-input* task? | Cutoff, extreme framing or obstruction prevents technical inspection; do not judge malignancy |
| Overall technical adequacy | Integrate the observable issues above for the intended frozen model input | `0` adequate, `1` borderline with reason, `2` technically inadequate with reason; this is **not clinical diagnostic adequacy** |

For the overall rating, `0` means the image is technically usable without a visible material defect, `1` means usable but technically limited, and `2` means a repeat or replacement image would be advisable **for technical input quality**. `U` means the reviewer cannot determine this from the displayed file. Overall rating need not equal the maximum domain score; explain discordance. Record whether a lesion/target is visible (`yes/no/uncertain`), and any artifact tags (`hair`, `glare`, `ruler`, `bubble`, `ink`, `frame`, `other`, or `none`). Do not mark an image clinically reassuring or concerning.

Two qualified reviewers rate independently. A third reviewer or recorded consensus resolves differences of at least one overall category, any `0` versus `2` domain disagreement, or disagreement about lesion visibility. Preserve the initial scores and adjudication rationale. The coordinator must document reviewer credentials, training examples, display conditions, and any accidental unblinding. The study currently has **no recorded human ratings**.
