# Stage 23 CUDA Phase B reproducibility audit

## Audit result

**PASS for saved-artifact completeness and internal integrity.** The downloaded CUDA experiment reports `COMPLETE_FULL_VALIDATION`: 986 validation images from 743 lesions, seven fixed conditions, 6,902 unique image-condition records, and 434 complete committed batch directories. The read-only analyzer `--check-only` returned `INTEGRITY_PASS`. It checked every committed batch SHA256 and metadata, ordering, labels, all seven class probabilities, summary scalar consistency, the final table, and the baseline gate. The prior float-serialization issue is handled by an exact probability comparison plus a machine-precision bound on redundant derived scalars; it did not represent different predictions.

The checkpoint SHA256 is `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`. The class order is `akiec, bcc, bkl, df, mel, nv, vasc`. Split SHA256 is `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`; frozen protocol SHA256 is `df94205ec017fde3c4cc7b8f4094b7432fef0ce2a8cd2b812464d6ccb8c8c90c`. The frozen Stage 23 validation prediction SHA256 is `b3f5d2e43b26fb63916d4e29ab597fde4183b3c76e75d4d0a7caeaccc88a368a`. The execution manifest records Tesla T4, PyTorch `2.11.0+cu130`, CUDA FP16 autocast, physical batch 16, and the frozen selective FP32 nonlocal3 q@k policy. Runner and analyzer hashes matched the manifest. The downloaded images themselves were not reprocessed or inferred on in this audit.

## Baseline and statistical checks

The complete baseline gate passed: all 986 predicted labels matched the frozen Stage 23 validation reference, with **maximum absolute seven-class probability difference 0.0** and no class disagreements. The baseline was a saved-JPEG/PIL replay; no degradation was applied. All six fixed degradation conditions were applied to the same validation IDs and batch boundaries, after baseline clearance. The final table has 986 unique IDs per condition, one record per image-condition, with original split labels and lesion IDs. Each condition has 107 melanoma images.

Independent saved-record recomputation reproduced the reported accuracy, macro-F1, melanoma recall, flip count, MEL→NV count, and mean entropy for every condition to within `1e-12`. The saved statistical JSON specifies 1,000 lesion-cluster bootstrap resamples, seed 2310 plus condition index, and 2.5/97.5 percentile intervals for paired accuracy, macro-F1, MEL recall, flip-rate and mean-entropy changes. These intervals were accepted from the hash-verified statistical artifact after source-code and source-table verification; they are not external-validation confidence intervals. Multiple condition/class comparisons are exploratory and unadjusted.

The previously completed 21-image pilot is a class-balanced sample from 21 lesions. Its manifest and key output hashes passed. All 147 paired pilot/full records (21 images × seven conditions) have matching true labels and predicted classes. This does not make the 21-image estimates representative of the full class distribution; the CPU pilot and CUDA full run also differ in execution mode.

## Figures and derived results

The three original figures listed in the execution manifest pass SHA256 checks and are approximately 300 DPI: `cuda_robustness_curves.png` is 4200×1500 pixels, while the transition and melanoma figures are 2700×1500 pixels. New 300-DPI figures in this results directory show lesion-cluster intervals for paired changes and descriptive entropy error-discrimination AUROC. Titles specify Stage 23 validation and synthetic degradation; no clinical-quality inference is shown.

`build_phase_b_results.py` performs a full read-only source check before generating four comparison tables and two figures in this separate directory. `derived_results_manifest.json` records original manifest hashes, checkpoint/split/protocol/table hashes, validation counts, builder hash, and SHA256 of every derived table/figure. The original CUDA outputs and batch commits remain unchanged. No training, model inference, HAM test outcome access, or PH² access occurred during this audit.

## Scope of reproducibility

The complete saved-result analysis is reproducible from the downloaded files without rerunning CUDA inference. This audit verifies files, provenance, and calculations; it does not independently replay the full T4 model execution. The baseline equivalence and runtime configuration are supported by the committed batch metadata, manifest, and exact saved probability comparison. Synthetic perturbations do not substitute for real acquisition artifacts, labeled image-quality judgments, external-domain testing, or clinical validation.
