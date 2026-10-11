# Stage 23 CUDA Phase B artifact inventory

This inventory describes the downloaded **original** CUDA experiment directory. New interpretation files reside only in `final_research_results/`. The machine-readable `original_artifact_inventory.csv` lists every original file, byte count, and SHA256.

## Complete file set

There are **883 original files** totaling about 4.92 MB: 15 top-level files and 868 files in committed batch directories. Each of the seven conditions has 62 committed batch directories, each containing exactly `records.csv` and `metadata.json` (124 files per condition, 434 committed directories overall). No incomplete committed batch or duplicate image-condition record was found. Temporary `__pycache__` files are excluded from the research inventory.

| Original top-level file | Purpose |
|---|---|
| `run_cuda_full_validation.py` | Frozen CUDA FP16 batch-16 inference runner and batch integrity logic |
| `analyze_cuda_full_validation.py` | Saved-prediction analyzer and full-table integrity gate |
| `COLAB_CUDA_FULL_VALIDATION.md` | Execution instructions |
| `BASELINE_PIXEL_IDENTITY_DIAGNOSIS.md` | Baseline input-path correction audit |
| `CUDA_ANALYSIS_INTEGRITY_AUDIT.md` | CSV representation mismatch audit |
| `baseline_gate.json` | Complete baseline equivalence gate |
| `cuda_full_validation_predictions.csv` | 6,902 original saved prediction records |
| `cuda_execution_manifest.json` | Runtime, frozen input, source, and output hashes |
| `cuda_summary.csv` | Seven-condition performance and paired changes |
| `cuda_melanoma_analysis.csv` | MEL-specific saved-record analysis |
| `cuda_statistical_results.json` | Classwise and lesion-cluster bootstrap results |
| `cuda_robustness_curves.png` | Accuracy, macro-F1, MEL recall, entropy figure |
| `cuda_prediction_transitions.png` | Paired correct/incorrect transitions figure |
| `cuda_melanoma_robustness.png` | MEL outcome figure |
| `FINAL_CUDA_PHASE_B_REPORT.md` | Original Colab analysis report |

The seven batch directories are `batches/baseline/`, `batches/blur_r1/`, `batches/blur_r2/`, `batches/underexposure_070/`, `batches/overexposure_130/`, `batches/contrast_070/`, and `batches/jpeg_q40/`. Within each, directories `batch_0000` through `batch_0061` hold the committed records and metadata. Each condition has 986 unique validation image IDs: 61 batches of 16 and one batch of 10.

## Original research anchors

| Artifact | SHA256 |
|---|---|
| Frozen Stage 23 checkpoint | `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab` |
| Frozen validation split | `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` |
| Frozen CUDA Phase B protocol | `df94205ec017fde3c4cc7b8f4094b7432fef0ce2a8cd2b812464d6ccb8c8c90c` |
| Original Stage 23 validation predictions | `b3f5d2e43b26fb63916d4e29ab597fde4183b3c76e75d4d0a7caeaccc88a368a` |
| Complete CUDA prediction table | `ff7c7a454bd896c54628bd4aeb61949789b82d1a69ae3473878690ca11d0d221` |
| Baseline gate | `f20be2817c9f8242b10c50257a344660899f153aa6c5f17f0393b18082ddb99d` |
| CUDA execution manifest | `0b0ce6adb2ecca37775a0c7a1e868d42d82296ab44bc70a55b13ce4746f6fc81` |

`cuda_execution_manifest.json` contains individual hashes for the original summary, statistical JSON, three plots, MEL analysis CSV, and original report. All listed hashes were independently checked and matched. The derived results have their own `derived_results_manifest.json`; no original artifact was rewritten.
