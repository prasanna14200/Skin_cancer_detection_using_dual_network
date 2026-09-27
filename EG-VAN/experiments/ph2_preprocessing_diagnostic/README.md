# PH² Preprocessing Sensitivity Diagnostic

## Purpose

This is a paired, frozen-checkpoint diagnostic to determine whether applying
the actual HAM10000 preprocessing pipeline changes predictions on the same PH²
images. It is not a model-improvement or causal-identification experiment.

## Experimental design

- Samples: 120 included PH² images (80 common nevi, 40 melanomas).
- Checkpoint SHA256: `f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848` (epoch 6).
- No training or fine-tuning occurred. No checkpoint, PH² data, split, or
  existing external-evaluation artifact was modified.
- Image IDs, labels, model, class order, resize, tensor conversion,
  normalization, inference mode, and ordering were held constant.
- The changed variable was whether the frozen HAM image preprocessing was
  applied before model input.

## RAW pipeline

Original PH² BMP → RGB → Resize 384×384 → ToTensor → ImageNet normalization.
This is the existing external-evaluation transform.

## HAM-matched pipeline

The same original PH² BMP is converted to BGR for `src/preprocessing.py`, then
processed with blackhat/Telea hair removal (kernel 17, threshold 10, radius
1.0), Gray World, and multi-scale Retinex (sigmas 15/80/250, equal weights,
per-channel 1st–99th percentile normalization). The result is JPEG-encoded and
decoded in memory to match the saved processed-HAM JPG representation, then
converted to RGB and passed through the same resize/tensor/normalization
transform. No augmentation, masks, or ROI crops are used.

## Results snapshot

- RAW predictions: {'akiec': 0, 'bcc': 0, 'bkl': 0, 'df': 0, 'mel': 0, 'nv': 102, 'vasc': 18}
- HAM-preprocessed predictions: {'akiec': 0, 'bcc': 0, 'bkl': 12, 'df': 0, 'mel': 23, 'nv': 84, 'vasc': 1}
- Changed predicted classes: 48 / 120
- Mean change in melanoma probability (HAM − RAW): 0.137486

See `summary_metrics.json`, `preprocessing_comparison.csv`, and
`vasc_case_analysis.csv` for complete results.

## Limitations and interpretation

Any score or prediction change is evidence of preprocessing sensitivity under
this checkpoint and these inputs. It does not, by itself, prove preprocessing
caused the original external-evaluation result. PH² has only 40 included
melanomas and 80 common nevi; there is no independent replication here. The
HAM preprocessing includes image-specific normalization operations and is
applied to an external dataset for this diagnostic only. The RAW condition
must reproduce all 120 previously saved predictions; otherwise no comparison
is interpreted.
