# Lesion, patient and duplicate-image audit

## Direct read-only checks

| Check | Result |
|---|---|
| Leakage-aware split SHA256 | `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` |
| Leakage-aware split images | 10,015 unique IDs; train 8,015; validation 986; test 1,014 |
| Lesion groups | 7,470 distinct; train 5,973; validation 743; test 754; **0 cross-partition lesion IDs** |
| Diagnosis consistency within lesion | **0 lesion IDs with conflicting `dx` labels** in the leakage-aware CSV |
| Raw canonical image coverage | 10,015 `data/raw/images/*.jpg` files; every split ID maps to exactly one basename; no extra basename |
| Exact-byte SHA256 duplicates | Hashed all 10,015 canonical JPG bytes; **2 two-image duplicate groups**, **0 across split partitions** |
| Naive split | Same 10,015 image IDs, 7,470 lesions; **764 lesions cross naive split assignments** |
| HAM patient identifiers | None in `data/processed/metadata_clean.csv`; patient-level isolation cannot be established |
| PH2 identifiers | `IMD` image/case IDs, no verified patient ID; no exact image-ID intersection with HAM, but patient overlap and visual duplicates are not excluded |

The two exact-byte duplicate pairs are `ISIC_0029861` / `ISIC_0024366` (both `HAM_0002300`, NV, train; common SHA256 `b8b37d24e294c5182a1884741c109d8b3bc58edaa993e8c331bd2a15e3e603e6`) and `ISIC_0029938` / `ISIC_0024777` (both `HAM_0001895`, NV, train; common SHA256 `b1780747e0ccd3d31652bcf942a0f4bb22af6e058a5dc8b84202e997b7247b8d`). They do not create an observed cross-partition exact-byte leak in the leakage-aware split. They do affect nominal train-image independence and must remain in one lesion group in any future internal split. The scan was a SHA256 comparison of bytes in the **canonical raw JPG folder**; it was not a perceptual-hash or cross-format duplicate search. It did not change any file.

## Unresolved risks and required future gate

Repeated photographs of one lesion are handled by the `lesion_id` grouping where metadata is correct. Different lesion IDs may still belong to the same patient; site and patient linkage are unavailable. Crops, recompression, near-identical frames or mislabeled lesion IDs may evade exact-byte hashing. PH2 is BMP while HAM is JPEG, so cross-dataset exact-byte matching alone would be inadequate. Public pretraining data overlap and site overlap also cannot be proved absent from the local manifests.

Before a new split or cohort is used, freeze a duplicate/overlap protocol: verify source-image hashes, image IDs, lesion/patient identifiers; run a **predeclared** perceptual-similarity candidate search across proposed partitions and against used HAM/PH2 data; manually adjudicate flagged candidates while blind to model performance; remove or group only according to registered rules; and publish the candidate list, decisions and hashes. A visually similar pair is not automatically an identical patient or pathology case. No such near-duplicate clearance has been completed here, so a new independent cohort is **not yet certified**.

The naive split's 764 crossing lesions are a concrete leakage risk, not a property of the lesion-isolated split. Early naive test metrics must not be compared as though they used the frozen leakage-aware partition. The prior `experiments/efficientnetv2s_leakage_aware/split_integrity.json` also records 0 crossing lesions, consistent with this independent CSV check. No HAM test prediction or PH2 prediction file was opened for this duplicate audit.
