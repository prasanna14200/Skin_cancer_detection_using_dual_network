# Stage 4B proposed split integrity audit

**Proposed boundary generated; independent read-only audit: PASS_PROPOSED_SPLIT_INTEGRITY.** The audit rehashed all 8,015 raw and processed images and every versioned split artifact, and independently checked assignment, labels, lesion grouping, duplicate separation and deterministic membership. The versioned split is an internal selection resource for **freshly trained** control and treatment models, not an evaluation set for frozen Stage 23. The original leakage-aware split and all frozen research artifacts remain unchanged.

## Membership and class balance

The frozen input is `data/splits/split_leakage_aware.csv` SHA256 `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. Its original `train` pool contains 8,015 images, 5,973 lesions, all seven classes and no conflicting diagnosis within a lesion. The fixed Stage 4B SHA256 lesion-order rule placed **6,425 images / 4,779 lesions** in `development_train` and **1,590 images / 1,194 lesions** in `internal_selection`, retaining every original train image exactly once. The selection set is 19.84% of images and approximately 19.99% of train lesions. No original HAM validation/test image or lesion is in either Stage 4B manifest.

| Class | Development-train images / lesions | Internal-selection images / lesions | Selection images as % of original train class |
|---|---:|---:|---:|
| AKIEC | 204 / 146 | 53 / 36 | 20.62% |
| BCC | 325 / 209 | 73 / 52 | 18.34% |
| BKL | 721 / 465 | 170 / 116 | 19.08% |
| DF | 72 / 46 | 23 / 12 | 24.21% |
| MEL | **731 / 393** | **168 / 98** | **18.69%** |
| NV | 4,282 / 3,458 | 1,084 / 864 | 20.20% |
| VASC | 90 / 62 | 19 / 16 | 17.43% |

The selection fraction was fixed by **lesions per class**, so image-level percentages vary with repeated images. The internal selection denominator for future MEL recall is 168 *images*, grouped into 98 MEL lesions for intervals. These are sample counts, not classification results. DF and VASC remain small and require careful uncertainty reporting.

## Source and artifact integrity

Each CSV row records SHA256 of the exact canonical raw JPEG and the processed JPEG used by the existing Stage 23 loader. The builder read and hashed all 8,015 files in **each** representation; the independent [audit script](audit_stage4b_split.py) recomputes those hashes and the deterministic assignment from source. The [versioned manifest](stage4b_split_manifest_v1.json) records source-code/protocol/config hashes, both role counts, class counts, duplicate-screen outcome and file hashes.

| Versioned output | SHA256 |
|---|---|
| [Development-train CSV](stage4b_development_train_v1.csv) | `8fbbea0afc03bef91f9398b7208292d01bf30dd699433edddf3aabe3ce8d1058` |
| [Internal-selection CSV](stage4b_internal_selection_v1.csv) | `a1b63582e069a45fffc71eeaa08af6c89b9fa829030266e621918bc76c28f8c1` |
| [First-pass dHash candidates](stage4b_near_duplicate_candidates_v1.csv) | `10edd9c8d80fece258691761c135e36b3b3b81088deda3fbc5f1e07a2518907e` |
| [Split manifest](stage4b_split_manifest_v1.json) | `dd2ce3f2d698ff5cbf392ccc2a8889338fe4cb7b1017dcff677214eec392e649` |
| [Second-pass pHash candidates](stage4b_phash_refined_candidates_v1.csv) | `66b9256855efa525c9b493fdad3b2707d125b6a0f324d3a1b1b920e91c8f2b51` |

No lesion ID crosses the new roles. Both known exact-byte raw duplicate pairs stay in their original training lesion and remain in one role. No raw **or processed** exact-byte hash group crosses roles. The original HAM validation/test sets and PH2 were not used in assignment or outcome-based rebalancing.

## Near duplicates and patient linkage

A 64-bit dHash screen on all Stage 4B raw training images flagged **6,192** cross-role, different-lesion pairs at Hamming distance ≤4. This low-resolution hash is intentionally sensitive and produces many lookalikes; these are not 6,192 proven duplicate cases. A separate [second-pass DCT pHash screen](refine_near_duplicate_candidates.py) evaluated the 6,192 fixed candidate pairs: **40** also have 63-bit pHash Hamming distance ≤8 (four ≤4). Its [refinement summary](stage4b_phash_refinement_v1.json) and immutable candidate CSV preserve provenance. The 40 pairs are **unadjudicated**. Similar images may be unrelated patients, and dissimilar hashes do not prove independence. Before training, a reviewer blind to arm/model outcomes should adjudicate these pairs using original full-resolution images and source metadata, keep decisions and hashes, and if a true duplicate crosses roles, **stop and version a new split** rather than silently move/remove cases.

HAM metadata lacks patient IDs, so the split is lesion-disjoint but cannot be certified patient-disjoint. No HAM-to-PH2 patient linkage is available. This limits any eventual interval or generalization claim and remains an explicit risk even after the 40 image candidates are reviewed. No inference, training, threshold selection or original-artifact modification occurred.
