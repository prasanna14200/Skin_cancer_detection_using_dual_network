# EG-VAN Phase 4A Dataset Verification

Date: 2026-09-17

## 1. Dataset

- **VERIFIED source:** HAM10000 metadata and original dermoscopic JPGs supplied under `data/raw/`; the project protocol identifies the source as Harvard Dataverse, DOI `10.7910/DVN/DBW86T`.
- **VERIFIED metadata filename:** `HAM10000_metadata.csv`.
- **VERIFIED image directories:** `HAM10000_images_part_1/` and `HAM10000_images_part_2/`.
- **VERIFIED actual image count:** 10,015 JPG files.
- **VERIFIED metadata row count:** 10,015.
- **VERIFIED unique lesion count:** 7,470.
- **NOT VERIFIED in this run:** license text was not independently re-downloaded. The frozen protocol records the dataset license as CC BY-NC 4.0.
- **VERIFIED additional files:** `hmnist_8_8_L.csv`, `hmnist_8_8_RGB.csv`, `hmnist_28_28_L.csv`, and `hmnist_28_28_RGB.csv` are present and were not used.

## 2. Metadata

- **VERIFIED required columns:** `lesion_id`, `image_id`, `dx`.
- **VERIFIED other columns:** `dx_type`, `age`, `sex`, `localization`.
- **VERIFIED missing values:** `lesion_id`: 0; `image_id`: 0; `dx`: 0; `dx_type`: 0; `age`: 57; `sex`: 0; `localization`: 0.
- **VERIFIED duplicate image IDs:** 0.
- **VERIFIED duplicate metadata rows:** 0.
- **VERIFIED class distribution:**

| Class | Count |
|---|---:|
| AKIEC | 327 |
| BCC | 514 |
| BKL | 1,099 |
| DF | 115 |
| MEL | 1,113 |
| NV | 6,705 |
| VASC | 142 |
| **Total** | **10,015** |

## 3. Image verification

- **VERIFIED part 1 count:** 5,000 JPG files.
- **VERIFIED part 2 count:** 5,015 JPG files.
- **VERIFIED total:** 10,015 JPG files.
- **VERIFIED matching metadata IDs:** 10,015.
- **VERIFIED metadata IDs without an image:** 0.
- **VERIFIED image files without metadata entries:** 0.
- **VERIFIED duplicate image filenames:** 0.
- **VERIFIED readability sample:** 10 deterministic metadata-selected images, seed 42, opened and verified successfully with Pillow.
- **VERIFIED corrupted/unreadable sampled images:** 0.
- **NOT VERIFIED:** exhaustive pixel decoding of every image was not separately performed; filename correspondence covered all 10,015 files.

## 4. Lesion statistics

- **VERIFIED unique lesions:** 7,470.
- **VERIFIED single-image lesions:** 5,514.
- **VERIFIED multi-image lesions:** 1,956.
- **VERIFIED maximum images per lesion:** 6.
- Grouping terminology is **lesion-level**, not patient-level.

## 5. Naive split

Frozen seed: 42. Image-level class-stratified target: 80/10/10.

- **Train:** 8,010 images.
- **Validation:** 998 images.
- **Test:** 1,007 images.
- **Total:** 10,015 images.

| Class | Train | Train % | Validation | Validation % | Test | Test % |
|---|---:|---:|---:|---:|---:|---:|
| AKIEC | 261 | 3.26% | 32 | 3.21% | 34 | 3.38% |
| BCC | 411 | 5.13% | 51 | 5.11% | 52 | 5.16% |
| BKL | 879 | 10.97% | 109 | 10.92% | 111 | 11.02% |
| DF | 92 | 1.15% | 11 | 1.10% | 12 | 1.19% |
| MEL | 890 | 11.11% | 111 | 11.12% | 112 | 11.12% |
| NV | 5,364 | 66.97% | 670 | 67.13% | 671 | 66.63% |
| VASC | 113 | 1.41% | 14 | 1.40% | 15 | 1.49% |

- **VERIFIED unique lesions by partition:** train 6,314; validation 961; test 982.
- **VERIFIED lesions appearing in multiple partitions:** 764.
- **VERIFIED train/test crossing lesions:** 380.
- **VERIFIED train/validation crossing lesions:** 376.
- **VERIFIED validation/test crossing lesions:** 54.

## 6. Leakage-aware split

Frozen seed: 42. Lesion-level class-stratified target: 80/10/10.

- **Train:** 8,015 images.
- **Validation:** 986 images.
- **Test:** 1,014 images.
- **Total:** 10,015 images.

| Class | Train | Train % | Validation | Validation % | Test | Test % |
|---|---:|---:|---:|---:|---:|---:|
| AKIEC | 257 | 3.21% | 30 | 3.04% | 40 | 3.94% |
| BCC | 398 | 4.97% | 58 | 5.88% | 58 | 5.72% |
| BKL | 891 | 11.12% | 104 | 10.55% | 104 | 10.26% |
| DF | 95 | 1.19% | 9 | 0.91% | 11 | 1.08% |
| MEL | 899 | 11.22% | 107 | 10.85% | 107 | 10.55% |
| NV | 5,366 | 66.95% | 663 | 67.24% | 676 | 66.67% |
| VASC | 109 | 1.36% | 15 | 1.52% | 18 | 1.78% |

- **VERIFIED unique lesions by partition:** train 5,973; validation 743; test 754.
- **VERIFIED cross-partition lesion IDs:** 0.

## 7. Comparison

| Measurement | Naive | Leakage-aware |
|---|---:|---:|
| Train images | 8,010 | 8,015 |
| Validation images | 998 | 986 |
| Test images | 1,007 | 1,014 |
| Unique lesions represented | 7,470 | 7,470 |
| Cross-partition lesions | 764 | 0 |
| Train/test lesion overlap | 380 | 0 |

The naive protocol assigns individual images independently, so images from the same lesion can occur in different partitions. The leakage-aware protocol assigns each complete lesion group to exactly one partition. This establishes potential lesion-level overlap; it does not by itself establish inflated accuracy or any model-performance effect.

## 8. Files created/modified

- `data/raw/images/` — consolidated working image directory containing 10,015 JPGs.
- `data/processed/metadata_clean.csv`
- `data/splits/split_naive.csv`
- `data/splits/split_leakage_aware.csv`
- `src/prepare_metadata.py`
- `src/build_splits.py`
- `prepare_metadata.py`
- `build_splits.py`
- `docs/DEVIATIONS.md`
- `docs/PHASE_4A_DATASET_REPORT.md`

## 9. Deviations

- **DEVIATION:** The original ZIP archives were not present in `data/raw/`; their extracted directories were present and validated. The extracted files were consolidated into `data/raw/images/` for the existing validation implementation.
- **DEVIATION:** The existing split writer required a schema projection fix because it declared four CSV fields while receiving full metadata rows. Split seed, grouping logic, assignment ratios, and outputs were unchanged.
- No metadata rows were removed.
- No H-MNIST files were used.

## 10. Status

**PHASE 4A COMPLETE**

Phase 4B preprocessing and all model implementation/training phases have not started.
