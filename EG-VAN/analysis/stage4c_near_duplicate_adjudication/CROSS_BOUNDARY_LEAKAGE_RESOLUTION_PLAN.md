# Cross-boundary leakage resolution plan

## Current state

The 40 refined candidate edges connect **59 distinct lesion IDs in 20 graph components**. The largest component has 11 lesions. Counting **all images belonging to every affected lesion**, not only flagged images, gives 73 development-training images and 46 internal-selection images, or 119 images in all. Of the affected internal-selection images, 5 melanoma images belong to 3 melanoma lesions. No flagged pair has a shared lesion ID, and there is no raw or processed exact-byte duplicate crossing Stage 4B v1. These are hash-screen candidates, not adjudicated duplicates.

## Decision after source review

For every confirmed duplicate, same-lesion/different-capture, likely-same-patient or unresolved high-risk edge, construct a graph on **lesion IDs**. Include every image of each lesion in a connected component. A component touching both split roles must not continue to span development training and internal selection. Preserve reviewer decisions and source evidence; do not relabel lesions or silently move individual images. If source records can establish additional links, add those links before computing components. Unrelated-lookalike decisions may be removed from the risk graph only with recorded reviewer agreement/adjudication.

The conservative default for components still at risk is **quarantine from both model-training and internal-selection manifests**. Keep every quarantined image in a third, explicitly labeled `quarantine` manifest so the original 8,015-image source pool remains accounted for. Deterministically sort lesion IDs and image IDs; never use model outcomes, validation/test outcomes or PH2 results to choose a side. This policy is more cautious than moving all affected components into training, because it avoids using unresolved possible same-patient examples in either arm.

As a **counterfactual upper bound**, quarantining *all 59 presently implicated lesions* would remove 73 images from the training role and 46 from selection, leaving 6,352 development-training and 1,544 internal-selection images, plus 119 quarantined images. Selection melanoma representation would be **163 images / 95 lesions**, down from 168 / 98; all seven classes would remain represented. These are membership counts, **not revised split artifacts or model results**. Actual v2 counts depend on adjudication and any newly linked lesions.

## Version and integrity gate for a future revision

Stage 4B v1 files and hashes remain immutable. If review calls for a changed boundary, write a new `stage4b_*_v2.csv` train, selection and quarantine set plus a v2 manifest and protocol amendment that hashes: v1 inputs, adjudication decisions/evidence, source metadata, algorithm/code, and all v2 outputs. Independently verify every original train image appears in exactly one v2 role; no lesion crosses roles; no confirmed/unresolved risk component crosses roles; raw and processed exact hashes do not cross; original HAM validation/test and PH2 are absent; class/lesion counts and melanoma support remain adequate. Recompute linked components and screen the resulting boundary for remaining candidate pairs. If any high-risk cross-role pair remains, keep training blocked.

Do not revise the fresh-control-versus-treatment hypothesis, outcome rules or historical Stage 23 checkpoint based on these candidate images. Stage 4B v1 cannot evaluate frozen Stage 23 because that checkpoint trained on the entire original HAM training pool. No v2 split is created in Stage 4C preparation.
