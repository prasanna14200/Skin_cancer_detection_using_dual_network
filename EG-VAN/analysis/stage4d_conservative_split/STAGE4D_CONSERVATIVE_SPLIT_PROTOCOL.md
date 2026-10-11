# Stage 4D conservative split protocol, version 2

**Frozen before v2 manifest generation.** Stage 4D creates a new internal development boundary from Stage 4B v1 using lesion-level co-location. This is containment of flagged image similarities, **not** confirmation that pairs are duplicates and **not** verified patient-level separation. Stage 4B v1 and frozen Stage 23 remain immutable. No model outcomes, original HAM validation/test outcomes, PH2 outcomes or model predictions determine membership.

## Inputs and fixed screen

Eligible images are exactly the 8,015 original HAM leakage-aware `train` images already present in Stage 4B v1. Verify the v1 manifest and both partition CSV hashes before any output. The fixed cross-boundary risk edges are the 40 different-lesion pairs in the Stage 4B pHash-refined CSV, verified against the Stage 4C review CSV and evidence manifest. Reviewer decisions are blank; treat every edge as **unresolved for containment**, not as a confirmed identity.

Reproduce the existing two-stage image screen from **raw** JPEGs: 64-bit horizontal grayscale dHash, PIL resize to 9×8 with bilinear interpolation, Hamming distance at most 4; then 63-bit DCT pHash, OpenCV grayscale and INTER_AREA resize to 32×32, top-left 8×8 DCT without DC, median threshold, Hamming distance at most 8. Hash thresholds serve only to define a reproducible containment set. No similarity value proves shared identity.

## Deterministic whole-lesion assignment

1. Form an undirected graph on lesion IDs using the 40 Stage 4C edges. Retain all images of every lesion. The 40 edges form 20 connected components and 59 distinct lesion IDs under the frozen inputs. All v1 development-training lesions stay in development training.
2. Move **every v1 internal-selection lesion in any flagged component**, with all its images, to development training. No image-level move is allowed. This initially identifies 23 selection lesions / 46 images.
3. Moving a selection lesion can create new cross-boundary pairs against lesions that were previously also in selection. Thus, screen **every pair of original v1 selection images from distinct lesions** using the same fixed dHash/pHash rule. Create a graph of all refined within-selection edges. Starting with the selection lesions identified in step 2, take its full graph-connected closure. Move every lesion in that closure, with every image, to development training. Repeat logically until no refined edge connects a moved selection lesion to a retained selection lesion. Sort all IDs lexicographically to make output order deterministic. This closure is defined entirely by source images and fixed hashes, with no model result or label-based exception.
4. Keep all other v1 internal-selection lesions in selection. Keep each original `image_id`, `lesion_id`, diagnosis and raw/processed image hash unchanged. Preserve v1 role as a provenance column and add v2 role. Never draw gradients from v2 internal selection.

Co-location is preferred to quarantine because it retains all eligible development images while containing every presently flagged pair; the selection cohort must still retain all seven classes and melanoma representation. If this policy leaves any **refined** cross-boundary edge after a full v2 re-screen, or causes missing class support, fail the build rather than change the rule. Preserve dHash-only cross-boundary candidates as unresolved screen evidence; they are not independently adjudicated.

## Versioned outputs and audit gate

Create `stage4d_development_train_v2.csv`, `stage4d_internal_selection_v2.csv`, `STAGE4D_CLASS_DISTRIBUTION.csv`, `stage4d_boundary_screen_v2.csv`, and `stage4d_split_manifest_v2.json`. The boundary-screen CSV records all new cross-boundary dHash≤4 pairs with their pHash distances. The manifest records input, code, protocol and output SHA256 hashes; class/image/lesion counts; component and closure counts; and the precise screen status. Write outputs exclusively and never overwrite v1 or another v2 artifact.

An independent read-only audit must verify all original train IDs are assigned once; original HAM validation/test images and lesions are absent; no lesion or raw/processed exact-byte duplicate crosses v2 roles; all 40 frozen edges and all edges arising from co-location have endpoints in one role; no v2 cross-boundary pair passes both hash cutoffs; every recorded hash matches current bytes; and each arm's proposed training/selection manifests are identical across future fresh control and blur-treatment runs. Patient IDs are absent and unflagged duplicates remain possible, so this gate does **not** certify patient-level independence.

The [Stage 4B fresh-control versus one Gaussian-blur treatment protocol](../stage4b_development_split/FRESH_CONTROL_VS_TREATMENT_PROTOCOL.md) remains the scientific comparison. Both fresh models must use the **same v2** development-training and internal-selection manifests. Frozen Stage 23 has already trained on the original HAM train pool and must not be evaluated on v2 selection. Preparing v2 grants no training authorization.
