# Experiment #5 Domain-Shift Analysis - Stage 1

## Scope and safeguards

This Stage 1 analysis compares saved Experiment #5 test and PH² follow-up artifacts with image-domain features measured on frozen model-input representations. No model inference was run. No threshold or preprocessing setting was tuned.

PH² is treated as an external follow-up dataset and is not used for model selection, hyperparameter tuning, threshold optimization, or training.

The Experiment #5 checkpoint remains frozen at epoch 15. Checkpoint SHA256: `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`. Frozen HAM split SHA256: `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. PH² has prior evaluation and preprocessing-diagnostic exposure; this analysis is follow-up evidence, not an untouched independent validation.

## Cohorts and representation

- HAM: Experiment #5 TEST images with true class `nv` or `mel`: 783 total (676 nv, 107 mel).
- PH²: existing direct-overlap saved follow-up cases: 120 total (80 nv, 40 mel). Atypical nevi were excluded.
- Native geometry is measured from raw source images. Image-quality and RGB features are measured after frozen preprocessing and the same 384×384 bilinear model-input resize: Phase 4B processed HAM JPGs versus PH² passed through the approved HAM-matched pipeline. The PH² transform reuses `src/preprocessing.py`, in-memory JPEG round-trip, RGB conversion, and the same fixed evaluation resize. No masks/ROI, augmentation, or test-time augmentation are used.

## Native geometry and matched-input feature differences

Native geometry differs in scale: the HAM originals in this test cohort are 600×450 (aspect ratio 1.333) throughout. PH² originals average about 766.5×575.5, with widths 761–769 and heights 572–577 (median aspect ratio about 1.332). The model-input images in both domains are resized to 384×384. Because the HAM native dimensions have zero variance, SMDs for native width/height/aspect are not meaningful; those geometry SMD entries are left blank in `domain_feature_summary.csv` and are not ranked with appearance features.

Class-stratified matched-input appearance features with the largest absolute SMDs (HAM minus PH²) include:

- **Melanoma / blue-channel mean:** HAM 59.227, PH² 108.017, SMD -2.229.
- **Melanoma / green-channel mean:** HAM 57.657, PH² 100.471, SMD -1.831.
- **Melanoma / brightness:** HAM 57.105, PH² 94.694, SMD -1.666.
- **Melanoma / dark-pixel ratio:** HAM 0.410, PH² 0.105, SMD +1.218.
- **Melanoma / red-channel mean:** HAM 55.224, PH² 78.274, SMD -0.999.
- **Melanoma / sharpness:** HAM 159.951, PH² 273.354, SMD -0.982.
- **Nevus / saturation:** HAM 0.364, PH² 0.456, SMD -1.153.
- **Nevus / red-channel mean:** HAM 58.186, PH² 38.983, SMD +0.837.

Laplacian sharpness is resolution- and resampling-dependent. Features are computed after each domain's frozen preprocessing and common resize, but they remain image proxies. These differences are consistent with possible image-domain shift; they do not establish clinical importance or causality.

## Saved prediction comparison

Metrics below are restricted to the shared true classes `nv` and `mel`; predictions outside these classes remain `other`.

| Dataset | n | Accuracy | Balanced accuracy | Mel recall | NV recall |
|---|---:|---:|---:|---:|---:|
| HAM10000 TEST overlap-only | 783 | 0.8799 | 0.7417 | 0.5514 | 0.9320 |
| PH² follow-up | 120 | 0.6250 | 0.5437 | 0.3000 | 0.7875 |

Saved confidence, entropy, top-two margin, and melanoma-probability distributions are summarized in `prediction_shift_summary.json` and shown in `plots/saved_prediction_distributions.png`. Overall mean confidence is 0.8336 on the HAM overlap-only test cohort versus 0.6899 on PH²; mean entropy is 0.4510 versus 0.7737; mean top-two margin is 0.7019 versus 0.4721. These pooled values reflect different class mixes, so class-conditional comparisons are more informative.

For true melanoma, mean confidence is 0.7105 (HAM) versus 0.6496 (PH²), mean entropy 0.7191 versus 0.8785, mean margin 0.4955 versus 0.4133, and mean `p_mel` 0.4923 versus 0.3060. For true nevus, mean confidence is 0.8530 versus 0.7101, mean entropy 0.4086 versus 0.7213, mean margin 0.7346 versus 0.5016, and mean `p_mel` 0.1033 versus 0.1933. These saved scores indicate lower confidence and larger entropy on PH², alongside a higher mean melanoma probability for PH² nevi; they do not establish why these differences occur.

Outcome proportions by true class are in `plots/outcome_proportions.png`.

HAM true melanoma predictions (n=107): mel=59 (55.1%), nv=34 (31.8%), other=14 (13.1%).

PH² true melanoma predictions (n=40): mel=12 (30.0%), nv=15 (37.5%), other=13 (32.5%).

HAM true nevus predictions (n=676): nv=630 (93.2%), mel=28 (4.1%), other=18 (2.7%).

PH² true nevus predictions (n=80): nv=63 (78.8%), mel=8 (10.0%), other=9 (11.2%).

These differences show that the saved PH² predictions have lower overlap-class performance and different confidence/outcome distributions. They do not identify a cause; population, image acquisition, resolution, and preprocessing representation all differ.

## Limitations and interpretation

The PH² cohort has 40 melanoma and 80 common-nevus cases. The HAM internal test is a different population and includes multiple images per lesion; PH² has no patient-linkage table in the audited package. The class-stratified image summaries use uneven sample sizes and descriptive SMDs; no p-value establishes clinical importance or causality. Phase 6 image-quality metrics describe processed HAM and are not raw-HAM capture measurements. Saved prediction summaries are conditional on this single frozen checkpoint and its saved evaluation outputs.

No Grad-CAM was generated in Stage 1. The planned 12-case comparison is a separate, later approved analysis; Grad-CAM will not prove causal reasoning. No model changes are justified by this descriptive stage.

## Reproducibility artifacts

- `domain_features.csv`: native dimensions, matched-input features, class labels, saved probabilities, confidence, entropy, and top-two margin per image.
- `domain_feature_summary.csv`: count, mean, median, standard deviation, quartiles, IQR, mean difference, and SMD by domain and true-class stratum.
- `prediction_shift_summary.json`: saved probability summaries, predicted-class distributions, and class-specific outcome counts/proportions.
- `domain_shift_metrics.json`: frozen hashes, cohort counts, overlap metrics, and largest class-stratified feature SMDs.
- `plots/feature_boxplots.png`, `plots/selected_feature_ecdfs.png`, `plots/saved_prediction_distributions.png`, and `plots/outcome_proportions.png`.

No inference, training, fine-tuning, PH² Grad-CAM, threshold tuning, or model selection was performed.
