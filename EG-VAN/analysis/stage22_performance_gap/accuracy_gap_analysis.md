# Accuracy-gap analysis: paper report versus frozen EG-VAN+

**Direct percentage subtraction is not an effect size.** The [paper](https://doi.org/10.1109/ACCESS.2025.3561240) reports 98.20% for an aggregated nine-class set and 97.80% for its seven-class experiment; our frozen recovered Stage15 epoch16 model achieved 82.347140% on 1,014 lesion-isolated HAM10000 test images. Paper split membership, augmentation-family isolation and evaluation semantics cannot be reconstructed fully. No evidence establishes what its result would be on our partition.

| Candidate explanation | Evidence grade | Observation versus inference |
|---|---|---|
| Number of classes / aggregated dataset | HIGH | The 98.20% task has seven HAM diagnoses plus two ISIC 2017 labels; ours has only seven HAM diagnoses. Direction of accuracy effect is unknown. |
| Different source images | HIGH | Paper Table 2 has 13,312 originals including 3,297 ISIC 2017 images; ours uses HAM10000 only. |
| Different split/evaluation | HIGH | Paper: 1,332 original test images; 37,883 train and 6,686 validation, both involving augmented images. Ours: 8,015/986/1,014 lesion-isolated, unaugmented validation/test. Original-level separation of paper validation is not specified. |
| Possible image-level versus lesion-level leakage | MEDIUM as protocol risk; UNSUPPORTED as occurrence | Paper does not state lesion or patient isolation, and augmentation precedes train/validation division in its narrative. No partition IDs prove shared lesions or transformed siblings; do not assert leakage occurred. |
| Augmentation | HIGH difference; MEDIUM causal hypothesis | Paper claims 20 unspecified transforms and class-specific augmentation counts; ours uses train flips/rotation. Broader augmentation could regularize, but benefit remains unmeasured. |
| Preprocessing | MEDIUM difference; LOW causal attribution | Named hair/Gray World/Retinex methods are implemented approximately; crop absent and many constants unspecified. No isolated comparison supports a size/direction of effect. |
| Architecture reconstruction | MEDIUM uncertainty; LOW causal attribution | All major named modules exist, but MFF tap/carry/fusion details are assumptions. No evidence that a specific alternative recovers accuracy. |
| Training configuration | HIGH difference; MEDIUM causal hypothesis | Batch 32 vs 16, random ResNet initialization, Stage15 MEL multiplier/sampler, validation gates and no early stop differ. Individual effects unmeasured. |
| Class imbalance and minority failures | HIGH current limitation | HAM test NV support 676/1014; BKL recall 58/104, MEL 56/107, AKIEC 24/40. 41/51 MEL misses went to NV. |
| Numerical stability policy | HIGH existence; UNSUPPORTED accuracy cause | FP32 `nonlocal3` q@k prevents observed FP16 affinity overflow; no controlled accuracy-effect estimate. |
| Insufficient convergence | LOW | Train focal loss 0.138999→0.016910 through saved original epoch24 while validation plateaus/rises after minimum 0.081272 at epoch7. More epochs alone are not supported. |
| Overfitting / generalization gap | HIGH descriptive, not unique cause | At original epoch24 train 0.016910 vs validation 0.088597; LR fell to 1.95e-6. Loss scale also depends on sampling/objective, so train/validation losses are not perfectly like-for-like. |
| Domain shift | HIGH for PH² follow-up; LOW for paper–HAM gap | PH² MEL recall 11/40 vs HAM 56/107, under a two-label mapping with OTHER errors and prior PH² use. This does not explain paper versus HAM directly. |
| Reconstruction uncertainty | HIGH existence; LOW quantitative attribution | Paper omits operational dimensions, constants, exact architecture wiring, seed and split IDs; reproduction cannot be certified. |

## Current saved-evidence synthesis

Stage15 original history CSV persists epochs 1–24; its recovered epoch16 was the selected eligible checkpoint and matches original observable epoch16 validation artifacts. At epoch16, validation accuracy was 0.8194726166, macro F1 0.6611314671, MEL recall 63/107 and NV recall 0.9396681750. The original validation focal loss minimum was epoch7 (0.0812721586) but was not eligible. Epoch16 validation loss was 0.0838739128; epoch24 was 0.0885974073 while training loss continued down. These observations fit **generalization/regularization pressure and class trade-offs**, not plain undertraining. The original checkpoint recovery limits byte-identification to the missing original epoch16 file; observable replay was exact for available comparators.

Frozen Stage16 confusion matrix: MEL→NV 41, MEL→BKL 5, BKL→MEL 13, NV→MEL 18; MEL precision 56/(56+34)=0.6222 and recall 56/107=0.5234. MEL→NV accounts for 41/51=80.39% of MEL false negatives and 41/107=38.32% of all test melanomas. The aggregate test accuracy is dominated by NV support (676/1014), so report macro F1 and MEL sensitivity alongside accuracy. The Stage20 entropy review sends 23/51 MEL false negatives to review, leaving 28 unflagged; this is not a substitute for melanoma accuracy.

Stage20 calibration/review and processed-image quality proxies are observational. ECE validation/HAM/PH² is 0.019574/0.029678/0.060178; entropy review retains 805/1014 at 90.93% conditional accuracy. Low-sharpness validation tercile error 22.19% versus 14.33% middle and 17.63% high is non-monotonic and does not identify a causal preprocessing defect.

**Conclusion:** protocol non-comparability is the strongest explanation for the apparent 98.20% versus 82.35% gap. Within our protocol, minority-class confusion and a widening train/validation gap are the strongest measured limitations. The contribution of each architecture/preprocessing/training difference remains unquantified until a prespecified controlled experiment on training/validation data. No existing evidence warrants a promise of 95% full-cohort accuracy.
