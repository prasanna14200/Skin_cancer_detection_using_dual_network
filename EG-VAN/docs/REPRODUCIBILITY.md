# Reproducibility Notes

## Verified Environment Evidence

| Item | Value | Status | Evidence |
|---|---|---|---|
| Baseline runtime | Google Colab CUDA | VERIFIED | `experiments/*/config.json`, smoke-test report |
| GPU for smoke test | Tesla T4 | VERIFIED | `docs/PHASE_4C_GPU_SMOKE_TEST_REPORT.md` |
| CUDA availability | True in Colab; unavailable locally for PH2 inference | VERIFIED | smoke-test and PH2 reports |
| Python | 3.13.15 in completed baseline configs | VERIFIED | `experiments/*/config.json` |
| PyTorch | 2.11.0+cu128 | VERIFIED | `experiments/*/config.json` |
| torchvision | 0.26.0+cu128 for smoke test | VERIFIED | `docs/PHASE_4C_GPU_SMOKE_TEST_REPORT.md` |
| Local requirements file | None found | VERIFIED ABSENT | `rg --files` audit |

## Dataset

HAM10000 is the verified training/evaluation dataset. The project protocol identifies the source as Harvard Dataverse DOI `10.7910/DVN/DBW86T`. Local verification found 10,015 metadata rows, 10,015 images, 7,470 unique lesions, and seven classes.

The additional H-MNIST CSV files are present under `data/raw/` but are not used by the baseline experiments.

PH2 is present locally from a secondary mirror under `data/external/ph2/`. Its manifest contains 200 rows: 80 common nevus, 80 atypical nevus, and 40 melanoma. PH2 inference has not run.

## Split Files

| Split file | Train | Val | Test | Crossing lesions | Status |
|---|---:|---:|---:|---:|---|
| `data/splits/split_naive.csv` | 8,010 | 998 | 1,007 | 764 | VERIFIED |
| `data/splits/split_leakage_aware.csv` | 8,015 | 986 | 1,014 | 0 | VERIFIED |

## Model Configuration

| Setting | Value | Status |
|---|---|---|
| Model | torchvision EfficientNetV2S baseline | VERIFIED |
| Pretraining | `EfficientNet_V2_S_Weights.DEFAULT` / ImageNet | VERIFIED |
| Classes | `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc` | VERIFIED |
| Input transform | resize 384x384, tensor conversion, ImageNet normalization | VERIFIED |
| Training augmentation | random horizontal flip, vertical flip, rotation 15 degrees | VERIFIED |
| Validation/test augmentation | none | VERIFIED |
| Loss | focal loss, alpha 0.25, gamma 2.0 | VERIFIED |
| Optimizer | Adamax, lr 0.001 | VERIFIED |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 1 | VERIFIED |
| Batch size | 16 | VERIFIED |
| Epochs | 25 | VERIFIED |
| Mixed precision | enabled on CUDA | VERIFIED |
| Early stopping | false | VERIFIED |

## Checkpoints

| Artifact | Status |
|---|---|
| `experiments/efficientnetv2s_naive/best_checkpoint.pt` | VERIFIED PRESENT |
| `experiments/efficientnetv2s_naive/final_checkpoint.pt` | VERIFIED PRESENT |
| `experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt` | VERIFIED PRESENT |
| `experiments/efficientnetv2s_leakage_aware/final_checkpoint.pt` | VERIFIED PRESENT |
| `checkpoints/efficientnetv2s_leakage_aware_best.pt` | VERIFIED PRESENT |

Best validation-loss epochs from synchronized training histories:

| Run | Best epoch | Best val loss |
|---|---:|---:|
| Naive | 12 | 0.042679324317039954 |
| Leakage-aware | 6 | 0.07601256733930256 |

## Commands

Local metadata and split checks:

```bash
python prepare_metadata.py
python build_splits.py
```

Baseline training commands used by the implemented runner:

```bash
python src/run_baseline.py --project-root . --split-csv split_naive.csv --run-dir experiments/efficientnetv2s_naive
python src/run_baseline.py --project-root . --split-csv split_leakage_aware.csv --run-dir experiments/efficientnetv2s_leakage_aware
```

The runner requires CUDA and intentionally fails on CPU.

Evaluation command shape:

```bash
python src/evaluate.py experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt . split_leakage_aware.csv --output experiments/efficientnetv2s_leakage_aware/test_metrics.json
```

PH2 evaluation command shape, not yet run:

```bash
python src/external_eval.py data/external/ph2/metadata/ph2_manifest.csv experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt experiments/ph2_external_eval
```

Phase 7 uncertainty inference was completed in Google Colab using the existing leakage-aware checkpoint, without training or weight updates:

```bash
PYTHONPATH=src python -u src/run_uncertainty_analysis.py \
	--project-root /content/drive/MyDrive/EG-VAN \
	--checkpoint /content/drive/MyDrive/EG-VAN/experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt \
	--output-dir /content/drive/MyDrive/EG-VAN/experiments/uncertainty
```

The verified runtime was Tesla T4, CUDA 12.8, PyTorch 2.11.0+cu128, Python 3.13.15. Outputs are in `experiments/uncertainty/`. The saved predictions were postprocessed once to correct the final calibration-bin boundary and quality-tercile feature collision; model inference was not rerun. Corrected results are documented in `docs/PHASE_7_UNCERTAINTY_ESTIMATION.md`.

## Result Locations

| Result | Location |
|---|---|
| Naive metrics | `experiments/efficientnetv2s_naive/test_metrics.json` |
| Naive config | `experiments/efficientnetv2s_naive/config.json` |
| Naive training history | `experiments/efficientnetv2s_naive/training_history.json` |
| Leakage-aware metrics | `experiments/efficientnetv2s_leakage_aware/test_metrics.json` |
| Leakage-aware config | `experiments/efficientnetv2s_leakage_aware/config.json` |
| Leakage-aware training history | `experiments/efficientnetv2s_leakage_aware/training_history.json` |
| Phase 7 predictions | `experiments/uncertainty/predictions.csv` |
| Phase 7 uncertainty summary | `experiments/uncertainty/uncertainty_summary.json` |
| Phase 7 calibration bins | `experiments/uncertainty/calibration_bins.csv` |

## Not Available / Unverified

- No pinned `requirements.txt`, `pyproject.toml`, or conda environment file was found.
- Official PH2 source package authenticity and license permission are not established by the local secondary mirror.
- PH2 accuracy, macro-F1, AUROC, per-class recall, predictions, and confusion matrix do not exist.
- Phase 7 ECE is a ten-equal-width-bin estimate and depends on the specified binning convention.
- Confidence and entropy are softmax predictive uncertainty measures; they are not Bayesian/epistemic estimates or clinically validated uncertainty.
- Repeated baseline runs were not performed, so stability statistics are unavailable.
