# PH² Zero-Melanoma Forensic Investigation

Read-only review of the verified local PH² outputs, frozen HAM10000 split, source code, training history, and checkpoint metadata. No inference was run and no model, dataset, checkpoint, or existing evaluation artifact was changed. The only file created by this investigation is this report.

## 1. Verified baseline

The existing PH² evaluation reports 200 total cases, 120 included direct-overlap cases (80 common nevi mapped to `nv`, 40 melanomas mapped to `mel`), and 80 atypical nevi excluded. Among the included cases, the 7-class prediction counts are `nv=102`, `vasc=18`, `mel=0`.

| Actual | Predicted `nv` | Predicted `mel` | Predicted `other` |
|---|---:|---:|---:|
| `nv` | 65 | 0 | 15 |
| `mel` | 37 | 0 | 3 |

The reported binary metrics are accuracy 0.5416667, melanoma precision 0, recall/sensitivity 0, specificity 0.8125, and F1 0. The 18 `other` outputs are all `vasc`.

Evidence: `experiments/ph2_external_eval/metrics.json`, `confusion_matrix.csv`, `predictions.csv`.

## 2. Training class distribution

Counts below are from the frozen `data/splits/split_leakage_aware.csv` used by the saved leakage-aware run.

| HAM10000 class | Train | Validation | Test |
|---|---:|---:|---:|
| `akiec` | 257 | 30 | 40 |
| `bcc` | 398 | 58 | 58 |
| `bkl` | 891 | 104 | 104 |
| `df` | 95 | 9 | 11 |
| `mel` | 899 | 107 | 107 |
| `nv` | 5,366 | 663 | 676 |
| `vasc` | 109 | 15 | 18 |
| **Total** | **8,015** | **986** | **1,014** |

Training has about **5.97 nevus images per melanoma** and **49.23 nevus images per `vasc`**. Melanoma is substantially less frequent than `nv`, but it is not one of the smallest classes: there are 899 training melanomas versus 109 `vasc`, 95 `df`, and 257 `akiec`.

This is material class imbalance, but counts alone do not establish that it caused the external result or that the model learned a class bias.

Evidence: `data/splits/split_leakage_aware.csv`, `experiments/efficientnetv2s_leakage_aware/config.json`.

## 3. Class-index mapping

The mapping is explicit in source, not inferred from sorting or `LabelEncoder`:

| Output index | Class |
|---:|---|
| 0 | `akiec` |
| 1 | `bcc` |
| 2 | `bkl` |
| 3 | `df` |
| 4 | `mel` |
| 5 | `nv` |
| 6 | `vasc` |

Trace:

1. `src/dataset.py` declares `CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")`, then builds `CLASS_TO_INDEX` by enumerating that tuple.
2. `HAM10000Dataset.__getitem__` returns `CLASS_TO_INDEX[row["dx"]]`; labels therefore enter the model as indices 0–6 in that order.
3. `src/run_baseline.py` records `list(CLASS_NAMES)` in run config. `src/models/baseline_effnet.py` creates a seven-output classifier. The saved checkpoint metadata records the same order and its classifier has shape `[7, 1280]`.
4. `src/external_eval.py` has the same ordered `HAM_CLASSES` tuple and uses `HAM_CLASSES[predicted_index]` to name the argmax output.
5. `src/external_eval.py` maps only common nevus→`nv`, melanoma→`mel`, atypical nevus→excluded (`None`). The prediction-to-binary step preserves `nv`/`mel`; other HAM outputs are labeled `other`.

The saved checkpoint was loaded with `weights_only=True` for metadata inspection; its embedded epoch is 6, split is `split_leakage_aware.csv`, class list agrees with source, and its SHA-256 matches the verified value `f96ebe86ffaa984333105c9092ac16e8cc2137190a36411cc0b6445620960848`.

No `LabelEncoder` or `ImageFolder.class_to_idx` is used in the audited training path. The mapping path is internally consistent; no evidence of an index permutation was found.

Evidence: `src/dataset.py`, `src/models/baseline_effnet.py`, `src/run_baseline.py`, `src/external_eval.py`, `experiments/efficientnetv2s_leakage_aware/config.json`, checkpoint metadata, `experiments/ph2_external_eval/evaluation_config.json`.

## 4. Loss and imbalance handling

The exact run config specifies focal loss with `alpha=0.25`, `gamma=2.0`, Adamax at learning rate 0.001, batch size 16, and 25 epochs. `src/train.py:focal_loss` computes softmax probabilities, gathers each sample's true-class probability, computes unreduced cross-entropy, then averages `alpha * (1 - p_t)^gamma * cross_entropy`.

The focal-loss alpha is one shared scalar; it is **not a per-class weight**. The training runner uses a shuffled `DataLoader` without a sampler. There is no weighted sampler, oversampling, undersampling, class-specific loss weight, or label smoothing in the audited code/config.

Thus melanoma receives ordinary per-example focal-loss signal when it is the target, but there is no explicit mechanism to compensate for the `nv`/`mel` frequency difference. The available evidence establishes that setup; it does not establish whether that alone caused zero PH² melanoma predictions.

Evidence: `src/train.py`, `src/run_baseline.py`, `experiments/efficientnetv2s_leakage_aware/config.json`.

## 5. Training/validation evidence

The best checkpoint is epoch 6 because the runner saves the checkpoint when **validation loss** reaches a new minimum. The run's lowest saved validation loss is 0.07601257 at epoch 6; checkpoint selection is not based on melanoma recall, macro-F1, or external performance.

Saved epoch-6 validation evidence:

- Accuracy: 0.8073; macro-F1: 0.6517.
- Melanoma recall: **51/107 = 0.4766**; melanoma precision from the saved confusion matrix: **51/87 = 0.5862**.
- Melanoma confusion row: 4 predicted `akiec`, 1 `bcc`, 1 `bkl`, 0 `df`, **51 `mel`**, 46 `nv`, 4 `vasc`.
- `vasc` recall: **14/15 = 0.9333**. Its validation prediction column contains 20 outputs, 14 of which are correct (`vasc` precision 0.70).

The checkpoint already had limited melanoma sensitivity on internal validation, and many validation melanomas were assigned to `nv`. However, it did predict 51 of 107 melanomas correctly, so the saved validation evidence does not show a complete inability to produce the melanoma class.

The file `experiments/efficientnetv2s_leakage_aware/test_metrics.json` reports melanoma recall 55/107 = 0.5140, but the training runner writes that file after the final epoch-25 model, not the epoch-6 best-validation checkpoint. It must not be attributed to the verified PH² checkpoint. A separate existing uncertainty analysis is keyed to the verified checkpoint hash and records 45/107 melanoma test recall; it stores HAM10000 test probabilities only.

Evidence: `src/run_baseline.py` checkpoint selection and final test path; `experiments/efficientnetv2s_leakage_aware/training_history.json`, `test_metrics.json`; `experiments/uncertainty/uncertainty_summary.json`.

## 6. Preprocessing comparison

| Operation | HAM10000 training/evaluation | PH² external evaluation | Status |
|---|---|---|---|
| Image color mode | `PIL.Image.open(...).convert("RGB")` | `PIL.Image.open(...).convert("RGB")` | **EXACT MATCH** |
| Resize | `Resize((384, 384))` | `Resize((384, 384))` | **EXACT MATCH** |
| Resize interpolation | torchvision default for PIL resize (bilinear) | Same torchvision transform and default | **EXACT MATCH** |
| Tensor and normalization | `ToTensor`; ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` | Same | **EXACT MATCH** |
| Crop | None | None | **EXACT MATCH** |
| Augmentation at evaluation | None | None | **EXACT MATCH** |
| Training-only augmentation | horizontal/vertical flips, rotation up to 15° | None at external evaluation | **EXPECTED DIFFERENCE** |
| Before model transform | HAM images go through hair removal, Gray World, Retinex, and are saved as processed JPGs | Original PH² dermoscopic BMPs are used directly | **POTENTIAL DISTRIBUTION SHIFT** |
| Background/ROI | No ROI/mask crop in the HAM dataset loader | Full dermoscopic image; masks and ROI files are not passed to model | **MATCH IN POLICY** |

The resize policy forces both sources to a square and may distort geometry, but it is the same operation in both paths. HAM processed images are 450×600; PH² originals are approximately 4:3 (for example, 767×576). Both are resized to 384×384. The major verified preprocessing discrepancy is that the model was trained on preprocessed HAM images while PH² images bypass the frozen preprocessing pipeline.

This discrepancy is a confirmed distribution difference, not proof that it caused the zero-melanoma result. The saved quality table and a read-only descriptive calculation show much higher average grayscale brightness for raw PH² than processed HAM test images; those measurements are not a controlled causal test.

Evidence: `src/train.py:make_transforms`, `src/dataset.py`, `src/preprocessing.py`, `src/external_eval.py:build_eval_transform` and PH² dataset loader, `experiments/image_quality/image_quality.csv`.

## 7. PH² image pipeline

The audited path is:

```text
PH² clinical metadata / ph2_manifest.csv
    -> inclusion and label validation
    -> included row's image_id
    -> data/external/ph2/images/{image_id}.bmp
    -> PIL Image.open
    -> convert("RGB")
    -> Resize((384, 384))
    -> ToTensor()
    -> ImageNet Normalize(mean, std)
    -> EfficientNetV2S
    -> argmax over seven output indices
    -> HAM_CLASSES[index]
    -> binary overlap mapping: nv/mel, else other
```

Read-only integrity checks found:

- The manifest contains 200 unique IDs and its ID set matches the preserved dermoscopic-image IDs.
- Every manifest image basename agrees with the evaluator's generated `{image_id}.bmp` path.
- All 200 images exist; the copied BMP for each ID is byte-identical to its preserved original dermoscopic BMP.
- All 200 copied inputs decode as RGB. The input path selects files in `data/external/ph2/images/`; code does not select lesion-mask or ROI-mask paths.
- The labels are validated against `PH2_dataset.txt`; the evaluator only includes rows with approved `nv` or `mel` mapping.

No wrong extension, missing image, manifest-to-image ID association, grayscale conversion, BGR conversion, mask/ROI substitution, or image scaling error was found in the inspected local artifacts/source. This verifies the file/pipeline association, not that PH² and HAM10000 have equivalent image distributions.

Evidence: `src/external_eval.py:read_manifest`, `audit_manifest_rows`, `audit_project`, and `run_evaluation`; `data/external/ph2/metadata/ph2_manifest.csv`; `data/external/ph2/metadata/verification_report.json`.

## 8. VASC investigation

The training split contains 109 `vasc` images, compared with 899 melanomas and 5,366 nevi. This is a small class, but not the smallest training class (`df` has 95). On epoch-6 internal validation, `vasc` recall was high at 14/15, with 20 total `vasc` predictions and 14 correct. The classifier output row for `vasc` has L2 norm 1.065 and bias −0.027; the `mel` row has L2 norm 0.625 and bias −0.012. These are descriptive parameter statistics only; without a defined reference distribution or activation/logit evidence, they do **not** establish an anomalous classifier row or explain a prediction.

From existing PH² predictions, 15 of 18 `vasc` outputs are actual common nevi and 3 are melanomas. This suggests `vasc` is not specific to the melanoma misses. The source/index mapping is consistent, so a simple class-index mismatch is not supported. Because the PH² artifact stores no scores and no PH² logits were generated during this investigation, it is not possible to tell how close those outputs were to `mel` or whether `vasc` won by a large margin.

Evidence: frozen split counts, epoch-6 validation confusion matrix in `training_history.json`, saved checkpoint classifier row parameters, and `experiments/ph2_external_eval/predictions.csv`.

## 9. Logit/confidence availability

The current PH² evaluation artifacts contain only `image_id`, `true_label`, `predicted_ham10000_label`, and `binary_prediction`; they do **not** contain logits, probabilities, top-1/top-2 confidence, or melanoma/nevus/vasc scores. The evaluator computes logits to obtain `argmax` and then discards them.

`experiments/uncertainty/predictions.csv` does contain probabilities, but these are for HAM10000 test images, not PH². No PH² probability values can be inferred from the saved hard labels.

No PH² diagnostic forward pass was run. Obtaining PH² top-1/top-2 probabilities and `mel`, `nv`, `vasc` scores requires a forward pass on those PH² images (and saving diagnostic outputs in a new, non-overwriting artifact). Per instruction, that experiment is not performed here.

## 10. Root-cause candidates

### CONFIRMED

- The checkpoint had materially limited internal melanoma recall: 51/107 (47.7%) on its epoch-6 validation split, with 46 validation melanomas predicted as `nv`.
- The evaluation used original PH² dermoscopic images, while HAM training images had hair removal, Gray World, and Retinex preprocessing.
- Training was class-imbalanced and used no class-specific weighting or sampler.
- All 40 PH² melanomas received non-`mel` argmax predictions in the saved result.

These are confirmed conditions/findings, **not individually confirmed causal explanations**.

### STRONGLY SUPPORTED

- The model's melanoma-vs-nevus decision boundary was already weak on the in-domain validation set; the PH² zero-recall result is consistent with that weakness becoming worse under an external dataset.
- Preprocessing/domain shift is a strong candidate contributor because PH² bypasses the three image operations used to create the HAM training inputs. The local descriptive brightness comparison is consistent with this difference, but is not causal evidence.

### POSSIBLE

- The roughly 6:1 `nv`-to-`mel` training frequency, without class-specific loss weighting or resampling, may have contributed to a decision preference for `nv`.
- Other PH²-to-HAM domain differences (image capture, lesion appearance, framing, color distribution, or dataset composition) may contribute; no matched domain analysis was performed.
- The small, imbalanced PH² direct-overlap sample may make the observed external metric unstable.
- The comparatively high `vasc` classifier row norm or its internal-validation behavior could relate to some `vasc` outputs, but row norms alone cannot identify a causal mechanism.

### NOT SUPPORTED

- A class-index or PH² label-mapping permutation: the source mapping, dataset targets, checkpoint class order, and evaluator class order agree.
- Wrong PH² file/path association or use of mask/ROI files: manifest IDs match the preserved source images, copied input bytes match, and the evaluator reads only `{image_id}.bmp` from the dermoscopic-image directory.
- RGB/BGR or grayscale channel handling error: both paths use PIL RGB conversion; the PH² files decode as RGB.
- A claim that the `vasc` outputs are caused by a class-index mismatch.

### UNKNOWN

- PH² per-image logits/probabilities, including melanoma probability and the margins between `mel`, `nv`, and `vasc`.
- Whether the PH² melanoma examples have consistently low melanoma scores or merely lose argmax narrowly.
- The causal share of preprocessing shift, broader dataset/domain shift, and training class imbalance.
- Whether the saved classifier row norms are unusual relative to relevant initialization/training references.

## 11. Recommended next diagnostic experiment

Run one **paired, same-checkpoint diagnostic on the same 120 included PH² cases**: preserve the current raw-BMP input path as the reference, and evaluate a second condition where those same images are passed through the frozen HAM preprocessing operations (hair removal → Gray World → Retinex) before the existing evaluation transform. Keep image IDs, labels, checkpoint hash, RGB conversion, resize, tensor conversion, normalization, and evaluation mode fixed. Save per-image seven-class logits/probabilities for both conditions in a new output location, then compare `mel` probability distributions, top-1/top-2 margins, prediction counts, and the 2×3 confusion matrices.

This is the next experiment only; it is not executed by this report. The repository's external evaluator requires CUDA, and matching the verified evaluation environment means running it on the Colab Tesla T4. Do not overwrite the verified PH² artifacts.
