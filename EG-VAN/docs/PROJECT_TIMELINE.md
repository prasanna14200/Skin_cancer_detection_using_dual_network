# EG-VAN Project Timeline

## Phase 1 - Project Setup / Research Understanding

Goal: Understand the EG-VAN paper and define a careful reproduction/validation path.

Work completed: Local paper PDF and Phase 4 experimental protocol are present. Deviation tracking was created in `docs/DEVIATIONS.md`.

Files created/modified: `EG-VAN_Phase4_Experimental_Protocol.md`, `docs/DEVIATIONS.md`.

Result: The repository distinguishes paper components from locally implemented baseline work.

Status: COMPLETED for current baseline scope.

Next dependency: Keep deviations updated whenever a paper detail is underspecified or a local choice differs.

## Phase 2 - Dataset / Data Understanding

Goal: Verify HAM10000 metadata, classes, images, and lesion grouping.

Work completed: Verified 10,015 metadata rows, 10,015 JPG files, 7,470 unique lesions, and seven diagnostic classes.

Files created/modified: `prepare_metadata.py`, `src/prepare_metadata.py`, `data/processed/metadata_clean.csv`, `docs/PHASE_4A_DATASET_REPORT.md`.

Result: Dataset foundation is verified. H-MNIST files are present but not used.

Status: COMPLETED.

Next dependency: Preserve raw metadata and image IDs.

## Phase 3 - Preprocessing

Goal: Implement deterministic paper-informed preprocessing before model input.

Work completed: Hair removal, Gray World correction, and multi-scale Retinex were implemented and run for all 10,015 images.

Files created/modified: `src/preprocessing.py`, `data/processed/images/`, `data/processed/preprocessing_log.csv`, `data/processed/preprocessing_samples/`, `docs/PHASE_4B_PREPROCESSING_REPORT.md`.

Result: 10,015 processed JPG outputs; zero failed outputs in the Phase 4B report.

Status: COMPLETED.

Next dependency: Do not rerun or replace processed images casually.

## Phase 4A - Dataset and Split Validation

Goal: Create and audit image-level and lesion-level split foundations.

Work completed: Built `split_naive.csv` and `split_leakage_aware.csv`.

Files created/modified: `build_splits.py`, `src/build_splits.py`, `data/splits/split_naive.csv`, `data/splits/split_leakage_aware.csv`.

Result: Naive split has 764 cross-partition lesions. Leakage-aware split has 0 cross-partition lesions.

Status: COMPLETED.

Next dependency: Use leakage-aware split for the main honest baseline.

## Phase 4B - Preprocessing Validation

Goal: Validate preprocessing mechanics and output integrity.

Work completed: Deterministic sample gate, full run, output log, output reopening checks, and before/after samples.

Files created/modified: `src/preprocessing.py`, `data/processed/preprocessing_log.csv`, `docs/PHASE_4B_PREPROCESSING_REPORT.md`.

Result: Cached processed images are ready for baseline model input transforms.

Status: COMPLETED.

Next dependency: Keep model resize/normalization separate from cached preprocessing.

## Phase 4C - GPU Smoke Test

Goal: Confirm the Colab GPU baseline mechanics.

Work completed: Forward pass, focal loss, backward pass, optimizer step, mixed precision, and transform audit passed on Tesla T4.

Files created/modified: `src/train.py`, `src/dataset.py`, `src/models/baseline_effnet.py`, `notebooks/phase4c_gpu_smoke_test.ipynb`, `docs/PHASE_4C_GPU_SMOKE_TEST_REPORT.md`.

Result: Runtime mechanics were validated; the historical smoke-test report predates the later completed 25-epoch runs.

Status: COMPLETED.

Next dependency: Keep environment details with reproducibility notes.

## Phase A / Stage A - Naive vs Leakage-Aware Baseline

Goal: Compare EfficientNetV2S baseline performance under image-level and lesion-level split protocols.

Work completed: Both 25-epoch baseline runs completed with the same model/configuration except split file.

Files created/modified: `src/run_baseline.py`, `src/evaluate.py`, `experiments/efficientnetv2s_naive/`, `experiments/efficientnetv2s_leakage_aware/`, `docs/PHASE_4C_STAGE_A_REPORT.md`.

Result: Naive accuracy 0.883813, macro-F1 0.807350. Leakage-aware accuracy 0.841223, macro-F1 0.711408.

Status: COMPLETED, with the limitation that repeated runs were not performed.

Next dependency: External validation and later full EG-VAN architecture work.

## Phase 5 - PH2 External Validation

Goal: Evaluate the frozen leakage-aware HAM10000 model on PH2 overlap classes without training on PH2.

Work completed: PH2 secondary-mirror files are locally present. A 200-row manifest exists. Label distribution is common nevus 80, atypical nevus 80, melanoma 40. Mapping policy excludes atypical nevus.

Files created/modified: `src/external_eval.py`, `data/external/ph2/`, `docs/PHASE_5_PH2_PLAN.md`, `docs/PHASE_5_PH2_EXTERNAL_VALIDATION_REPORT.md`.

Result: Pre-run audit completed. No PH2 inference metrics were created.

Status: BLOCKED before inference.

Next dependency: CUDA-enabled Colab runtime with compatible torchvision.

## Planned Research Extensions

Goal: Add image quality assessment, uncertainty estimation, and full EG-VAN components only after explicit approval.

Work completed: No implementation found in the current repository.

Files created/modified: None verified.

Result: These remain future work.

Status: PLANNED / NOT YET IMPLEMENTED.

Next dependency: Separate approved implementation plan.
