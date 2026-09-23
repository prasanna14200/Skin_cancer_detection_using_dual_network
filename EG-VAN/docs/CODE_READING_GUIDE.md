# EG-VAN Code Reading Guide

## End-to-End Map

```text
prepare_metadata.py
  -> data/processed/metadata_clean.csv
build_splits.py
  -> data/splits/split_naive.csv
  -> data/splits/split_leakage_aware.csv
src/preprocessing.py
  -> data/processed/images/
src/run_baseline.py
  -> src/train.py
  -> src/dataset.py
  -> src/models/baseline_effnet.py
  -> training loop
  -> checkpoints and metrics
src/evaluate.py
  -> checkpoint-only test evaluation
src/external_eval.py
  -> PH2 inference-only evaluation, not yet run
```

## `prepare_metadata.py`

Why it exists: Validates HAM10000 metadata and confirms image files correspond to metadata IDs.

Where execution starts: `main()`.

Important functions:

`load_metadata(path)`: receives a metadata path, returns CSV rows, and raises if required columns are absent.

`image_file_exists(image_id)`: receives one image ID and returns whether an image file exists in `data/raw/images/`.

`count_missing_values(rows)`: receives metadata rows and returns missing counts per column.

Dependencies: Python `csv`, `Counter`, `Path`, and the raw HAM10000 files.

Called by: Run directly with `python prepare_metadata.py`; wrapper `src/prepare_metadata.py` also executes it.

Example: `python prepare_metadata.py` reads `data/raw/HAM10000_metadata.csv`, verifies IDs, and writes `data/processed/metadata_clean.csv`.

## `build_splits.py`

Why it exists: Creates deterministic naive and lesion-level split CSVs.

Where execution starts: `main()`.

Important functions:

`load_rows(path)`: receives `metadata_clean.csv`, returns metadata rows.

`build_naive(rows)`: receives metadata rows and returns image-level stratified rows with a `split` value.

`build_leakage_aware(rows)`: receives metadata rows and returns rows grouped by `lesion_id` before split assignment.

`count_lesion_crossing(rows, mode)`: receives split rows and reports whether lesion IDs cross partitions.

`write_split_csv(path, rows)`: writes only `image_id`, `lesion_id`, `dx`, and `split`.

Dependencies: `data/processed/metadata_clean.csv`.

Called by: Run directly with `python build_splits.py`; wrapper `src/build_splits.py` also executes it.

Example: `python build_splits.py` writes `data/splits/split_naive.csv` and `data/splits/split_leakage_aware.csv`.

## `src/preprocessing.py`

Why it exists: Implements deterministic Phase 4B preprocessing.

Where execution starts: `main()`.

Important functions:

`remove_hair(image_bgr)`: receives a BGR image array and returns an inpainted BGR image.

`gray_world(image_bgr)`: receives a BGR image and returns a color-balanced BGR image.

`retinex(image_bgr)`: receives a BGR image and returns the Retinex-normalized output.

`preprocess_image(image_bgr)`: receives a BGR image and applies hair removal, Gray World, and Retinex in order.

`discover_images()`: returns original JPG paths from both HAM10000 image directories.

`process_one(input_path, output_path)`: receives an input/output path pair and returns a `ProcessingResult`.

Dependencies: OpenCV, NumPy, raw image directories.

Called by: Run directly with `python src/preprocessing.py`.

Example: One raw JPG is decoded, validated, preprocessed, written to `data/processed/images/`, reopened, and logged.

## `src/dataset.py`

Why it exists: Loads processed HAM10000 images using frozen split CSVs.

Where execution starts: There is no script entry point; classes/functions are imported by training/evaluation code.

Important objects:

`CLASS_NAMES`: the frozen class order `akiec,bcc,bkl,df,mel,nv,vasc`.

`CLASS_TO_INDEX`: maps class labels to integer targets.

`validate_split_integrity(split_csv, require_lesion_isolation)`: receives a split CSV and a boolean; returns row, lesion, and crossing-lesion counts.

`HAM10000Dataset(images_dir, split_csv, split, transform)`: receives image root, split CSV, partition name, and transform; returns `(image_tensor, class_index)` samples.

Dependencies: PIL, PyTorch Dataset, split CSV, processed images.

Called by: `src/train.py`, `src/run_baseline.py`, and `src/evaluate.py`.

Example: The training runner creates `HAM10000Dataset(..., "train", train_transform)` and passes it to a `DataLoader`.

## `src/models/baseline_effnet.py`

Why it exists: Defines the current baseline model.

Where execution starts: Imported function `build_model()`.

Important function:

`build_model(num_classes=7, pretrained=True)`: receives class count and pretraining flag; returns `(model, weights)`.

What it returns: A torchvision EfficientNetV2S with only the final classifier replaced.

Dependencies: `torchvision.models.efficientnet_v2_s`.

Called by: `src/train.py`, `src/evaluate.py`, and `src/external_eval.py`.

Example: `model, weights = build_model(pretrained=True)` creates the ImageNet-pretrained baseline for training.

## `src/train.py`

Why it exists: Holds approved training configuration and shared helpers.

Where execution starts: There is no training loop here; it is imported by `src/run_baseline.py`.

Important functions:

`set_seed(seed)`: receives an integer seed and sets Python, NumPy, and PyTorch seeds.

`focal_loss(logits, targets, alpha, gamma)`: receives model logits and labels, returns scalar focal loss.

`make_transforms(weights)`: receives torchvision weights, returns train and evaluation transforms.

`build_run_components(project_root, split_name)`: receives project root and split filename, returns model plus train/val/test datasets.

`configuration(device)`: returns a serializable config dictionary.

`make_optimizer_and_scheduler(model)`: returns Adamax optimizer and ReduceLROnPlateau scheduler.

Dependencies: `src/dataset.py`, `src/models/baseline_effnet.py`, PyTorch, torchvision transforms.

Called by: `src/run_baseline.py`.

Example: `run_baseline.py` imports `BATCH_SIZE`, `EPOCHS`, transforms, loss, and optimizer helpers from this file.

## `src/run_baseline.py`

Why it exists: Runs one approved CUDA baseline experiment.

Where execution starts: `main()`.

Important functions:

`metrics_from_confusion(matrix)`: receives a torch confusion matrix and returns accuracy, macro-F1, per-class recall, and matrix values.

`run_epoch(model, loader, optimizer, scaler, device, training)`: receives model/runtime objects and a training flag; returns average loss and metrics.

`write_json(path, value)`: writes dictionaries as indented JSON.

`main()`: parses arguments, validates split integrity, builds datasets/loaders, trains 25 epochs, saves checkpoints, writes metrics.

Dependencies: `src/train.py`, `src/dataset.py`, `src/evaluate.py`, CUDA runtime.

Called by: Run directly in Colab/GPU.

Example:

```bash
python src/run_baseline.py --project-root . --split-csv split_leakage_aware.csv --run-dir experiments/efficientnetv2s_leakage_aware
```

## `src/evaluate.py`

Why it exists: Evaluates an existing checkpoint without training.

Where execution starts: `main()` for command-line use, or `evaluate_checkpoint()` for imports.

Important function:

`evaluate_checkpoint(checkpoint_path, project_root, split_csv_name, split="test", batch_size=16)`: loads a checkpoint, rebuilds the baseline, evaluates one split, and returns metrics.

Dependencies: checkpoint file, split CSV, processed images, baseline model definition.

Called by: Command line; imported by `src/run_baseline.py`.

Example:

```bash
python src/evaluate.py experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt . split_leakage_aware.csv --output metrics.json
```

## `src/external_eval.py`

Why it exists: Performs inference-only PH2 evaluation using the frozen HAM10000 checkpoint.

Where execution starts: Command-line block at the bottom calls `evaluate_ph2()`.

Important functions:

`load_verified_manifest(path)`: receives a PH2 manifest path, returns rows, and rejects mappings outside common nevus -> `nv` and melanoma -> `mel`.

`build_manifest_template(output_path)`: writes an empty manifest template.

`evaluate_ph2(manifest_path, checkpoint_path, output_dir)`: loads mapped PH2 rows, loads the checkpoint, runs CUDA inference, and writes metrics/predictions/config.

Dependencies: PH2 manifest, PH2 image files, leakage-aware checkpoint, CUDA runtime.

Called by: Run directly after the runtime blocker is removed.

Example:

```bash
python src/external_eval.py data/external/ph2/metadata/ph2_manifest.csv experiments/efficientnetv2s_leakage_aware/best_checkpoint.pt experiments/ph2_external_eval
```

## Simple Baseline Example

1. `src/run_baseline.py` parses `--split-csv split_leakage_aware.csv`.
2. It calls `validate_split_integrity()` to confirm zero lesion crossing.
3. It calls `build_run_components()` from `src/train.py`.
4. `build_run_components()` creates `HAM10000Dataset` objects from `src/dataset.py`.
5. It creates EfficientNetV2S through `build_model()` in `src/models/baseline_effnet.py`.
6. The training loop calls `model(images)` to get logits.
7. `focal_loss()` computes loss.
8. Backpropagation and Adamax update model weights.
9. Validation loss controls the best checkpoint.
10. Test evaluation writes `test_metrics.json`.
