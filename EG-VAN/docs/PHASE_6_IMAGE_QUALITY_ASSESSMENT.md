# Phase 6 - Image Quality Assessment

Date: 2026-09-24
Status: COMPLETE (quality/error association analysis completed with Phase 7 predictions)

## 1. Motivation and research question

This phase investigates whether simple image-quality characteristics are associated with model behavior and errors. It is analysis only. Images were not removed, labels were not changed, splits were not changed, and no baseline was retrained.

Question: do deterministic image-quality proxies differ across frozen partitions/classes, and can they later be associated with leakage-aware model errors?

## 2. Quality features

For each Phase 4B processed JPG, the analysis computes:

- `brightness`: mean grayscale intensity, 0-255.
- `contrast`: grayscale standard deviation.
- `sharpness`: variance of the grayscale Laplacian.
- `saturation`: mean HSV saturation normalized to 0-1.
- `dark_pixel_ratio`: fraction of grayscale pixels below 30.
- `bright_pixel_ratio`: fraction of grayscale pixels above 225.
- `entropy`: Shannon entropy of the grayscale histogram in bits.
- `illumination_variation`: coefficient of variation of 16x16 local grayscale means.

The dark/bright thresholds are fixed descriptive image-intensity thresholds, not clinical-quality cutoffs. These features are image-quality proxies, not clinical quality labels.

## 3. Implementation

- Module: `src/image_quality.py`.
- Runner: `src/run_quality_analysis.py`.
- Input: `data/processed/images/`, `metadata_clean.csv`, and frozen `split_leakage_aware.csv`.
- Output table: `experiments/image_quality/image_quality.csv`.
- Calculations are deterministic and read-only with respect to frozen inputs.
- No training augmentation, preprocessing rerun, filtering, or sample removal occurs.

## 4. Dataset and split

- Dataset: HAM10000 processed images.
- Frozen split: lesion-level `split_leakage_aware.csv`.
- Rows analyzed: 10,015.
- Unique image IDs: 10,015.
- Train/validation/test counts: 8,015 / 986 / 1,014.
- Every quality-table image ID maps to the frozen split and metadata.
- No missing processed images or duplicate image IDs were found.

## 5. Quality distributions

Overall descriptive ranges from the 10,015 analyzed images:

| Feature | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| Brightness | 58.0795 | 5.5022 | 146.0611 |
| Contrast | 48.2148 | 25.6373 | 86.7502 |
| Sharpness | 614.1175 | 14.1821 | 13,987.6971 |
| Saturation | 0.3760 | 0.0542 | 0.8888 |
| Dark pixel ratio | 0.3764 | 0.0118 | 0.9804 |
| Bright pixel ratio | 0.0189 | 0.0000 | 0.1075 |
| Entropy | 6.7455 | 2.1578 | 7.8922 |
| Illumination variation | 0.8724 | 0.1761 | 4.5357 |

Split- and split/class-level descriptive tables are saved under `experiments/image_quality/`.

Generated plots:

- `quality_by_split.png`
- `quality_by_class.png`
- `quality_boxplots_by_split.png`
- `quality_distributions_by_split.png`
- `quality_distributions_by_class.png`

## 6. Relationship between quality and errors

Checkpoint inference was completed later under Phase 7 using the frozen leakage-aware checkpoint and test IDs. The resulting `experiments/uncertainty/quality_uncertainty.csv` descriptively groups test samples into equal-count quality terciles and reports accuracy, error rate, mean confidence, and predictive entropy. Examples vary by feature: lowest versus highest brightness tercile accuracy was 0.8402 vs 0.7249; for sharpness it was 0.7988 vs 0.7840. These are associations only and do not establish causality.

## 7. Quality-stratified results

Quality-uncertainty terciles were computed for each of the eight proxies on 1,014 leakage-aware test images, with 338 samples per tercile. Groups are relative ranks within this test set and are not clinical quality categories. See `experiments/uncertainty/quality_uncertainty.csv` and the Phase 7 report.

## 8. Interpretation

The quality table establishes reproducible proxy measurements and their descriptive distributions. It does not establish clinical image quality, causality, or that any quality feature affects model performance.

## 9. Limitations

- The features are computational proxies, not expert or clinical quality assessments.
- Thresholds 30 and 225 are descriptive intensity cutoffs, not validated medical thresholds.
- Checkpoint predictions were generated in CUDA Colab; the local Windows runtime itself still has no CUDA.
- No repeated-run uncertainty or statistical hypothesis testing was added.
- Processed images retain Phase 4B transformations; quality values describe those processed images, not the original raw captures.

## 10. Reproducibility

From the project root:

```bash
$env:PYTHONPATH='src'
python src/run_quality_analysis.py --project-root .
```

To attempt checkpoint-linked analysis in CUDA Colab after syncing the project:

```bash
PYTHONPATH=src python -u src/run_quality_analysis.py \
  --project-root /content/drive/MyDrive/EG-VAN \
  --checkpoint /content/drive/MyDrive/EG-VAN/experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt
```

This phase does not train or update model parameters. Phase 7 uncertainty inference was performed separately and used only to join frozen quality rows to predictions.

## 11. Future experiment possibilities

After explicit approval, verified prediction/error associations could motivate a separate quality-stratified analysis or robustness experiment. No image filtering or retraining should be introduced based on this descriptive phase alone.
