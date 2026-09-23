# EG-VAN Project Quick Reference

PROJECT: EG-VAN skin lesion classification research project.

TASK: 7-class HAM10000 skin lesion classification.

BASELINE: Plain torchvision EfficientNetV2S, ImageNet-pretrained backbone, replaced 7-class classifier.

DATA: HAM10000 metadata and dermoscopic JPGs. Verified local count: 10,015 images, 7,470 unique lesions, 7 classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).

IMAGE SIZE: Processed cached images remain 450x600. Model input transform resizes to 384x384.

SPLIT: Frozen leakage-aware split is the preferred research split.

TRAIN: 8,015

VAL: 986

TEST: 1,014

NAIVE BASELINE: Accuracy = 88.38%, Macro F1 = 80.74%.

LEAKAGE-AWARE BASELINE: Accuracy = 84.12%, Macro F1 = 71.14%.

CURRENT BLOCKER: PH2 external validation has local secondary-mirror PH2 files and a manifest, but inference is blocked until the CUDA/torchvision Colab runtime is used.

NEXT: Run inference-only PH2 evaluation in Colab using `src/external_eval.py`, the verified manifest, and the leakage-aware checkpoint. Do not train on PH2.

Tiny architecture:

```text
HAM10000 raw images + metadata
  -> deterministic preprocessing
  -> processed images + frozen splits
  -> EfficientNetV2S baseline
  -> train/val/test metrics
  -> descriptive leakage analysis
  -> planned/blocked external validation and future research modules
```

Protected artifacts:

- `data/processed/images/`
- `data/splits/split_naive.csv`
- `data/splits/split_leakage_aware.csv`
- `experiments/efficientnetv2s_*`
- `checkpoints/efficientnetv2s_leakage_aware_best.pt`
- Reported baseline metrics.
