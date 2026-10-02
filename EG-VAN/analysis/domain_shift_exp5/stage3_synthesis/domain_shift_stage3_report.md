# Domain Shift Analysis

## Experimental provenance

This Stage 3 synthesis combines saved artifacts from the frozen Controlled Experiment #5 and its follow-up analyses. The checkpoint is `experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt`, selected epoch 15 under `validation_threshold_constrained_min_loss`. Checkpoint SHA256: `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`. Frozen split SHA256: `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. Grad-CAM target: `model.features[-1]`.

PH? is treated as an external follow-up dataset and is not used for model selection, hyperparameter tuning, threshold optimization, or training. PH? had previously been used in project analyses, so it is not described as a completely untouched independent external validation. Experiment #5 remains frozen.

## Internal HAM evidence

On the full seven-class HAM10000 TEST partition (n=1014), the saved Experiment #5 checkpoint has accuracy 0.8333, seven-class macro-F1 0.6994, melanoma recall 0.5514, melanoma F1 0.5592, and nevus recall 0.9320.

For a class-support-matched descriptive comparison, the Stage 1 overlap-only internal subset contains 783 true `nv`/`mel` cases (676 nv, 107 mel). It has accuracy 0.8799, balanced accuracy 0.7417, melanoma recall 0.5514, and nevus recall 0.9320. Model predictions to the other five HAM classes remain counted as `other` rather than being dropped.

The existing internal HAM Grad-CAM set contains 24 preselected validation cases, with 120 saved PNGs and target layer `model.features[-1]`. It is a curated review set, not a random sample of the HAM test cohort.

## PH? external follow-up evidence

The saved PH? follow-up cohort contains 120 direct-overlap cases: 80 common nevi mapped to `nv` and 40 melanomas mapped to `mel`; 80 atypical nevi are excluded. Saved follow-up metrics are accuracy 0.6250, balanced accuracy 0.5437, two-class macro-F1 0.5987, melanoma sensitivity 0.3000, melanoma precision 0.6000, melanoma F1 0.4000, nevus recall 0.7875, and melanoma AUROC 0.6047.

PH? had previously been used for preprocessing diagnostics and an earlier model evaluation before Experiment #5. Therefore this result is an external follow-up evaluation, not a completely untouched independent external validation.

## Stage 1 image-domain differences

Stage 1 measured 783 HAM internal-test overlap images and 120 PH? follow-up images using the same quality-feature definitions on matched model-input representations. Native source geometry was reported separately: HAM originals in this cohort are 600?450, while PH? originals average approximately 766.5?575.5; both are resized to 384?384. Native-dimension standardized effects are not ranked because HAM dimensions have zero within-cohort variance.

The largest matched-input appearance differences were class-stratified. For true melanomas, PH? had higher blue/green channel means and brightness and a lower dark-pixel ratio than HAM; the leading reported absolute SMDs were 2.229 (blue mean), 1.831 (green mean), and 1.666 (brightness). For true nevi, saturation had an absolute SMD of 1.153 and red-channel mean 0.837. These measured differences are consistent with possible domain shift and warrant further investigation; they do not identify a cause.

## Prediction-confidence and entropy shift

Saved probabilities show lower mean maximum confidence on PH? than the HAM overlap-only test subset (0.8336 vs 0.6899) and higher mean predictive entropy (0.4510 vs 0.7737). Mean top-two margin is also smaller (0.7019 vs 0.4721).

For true melanoma, mean `p_mel` is 0.4923 on HAM and 0.3060 on PH?. For true nevus, mean `p_mel` is 0.1033 on HAM and 0.1933 on PH?. These are descriptive shifts in saved predictions, not calibrated confidence guarantees.

## Melanoma outcome comparison

| Dataset | n true melanoma | predicted mel | predicted nv | predicted other |
|---|---:|---:|---:|---:|
| HAM internal TEST | 107 | 59 (55.1%) | 34 (31.8%) | 14 (13.1%) |
| PH? follow-up | 40 | 12 (30.0%) | 15 (37.5%) | 13 (32.5%) |

The PH? saved argmax outcomes contain more melanoma-to-other predictions and fewer correct melanoma predictions than this internal cohort. This difference is consistent with a possible generalization gap; it does not identify its cause.

## Nevus outcome comparison

| Dataset | n true nevus | predicted nv | predicted mel | predicted other |
|---|---:|---:|---:|---:|
| HAM internal TEST | 676 | 630 (93.2%) | 28 (4.1%) | 18 (2.7%) |
| PH? follow-up | 80 | 63 (78.8%) | 8 (10.0%) | 9 (11.2%) |

The PH? cohort has fewer correct nevus predictions and larger proportions predicted as melanoma or another class. The sample sizes and different source populations limit direct generalization.

## Stage 2 Grad-CAM methodology

Stage 2 used the frozen Experiment #5 checkpoint, the same HAM-matched PH? preprocessing, 384?384 input, and `model.features[-1]`. Reproduction was checked for only the 12 approved cases before maps: 12/12 top-1 agreement and maximum probability difference 0 under tolerance `5e-4`. The reproduction matched the saved batch shape of 16 by repeating four already-approved inputs for padding; duplicate outputs were discarded.

Twelve predicted-class maps were computed; eight true-class maps were separately computed for incorrect predictions; four identical-target cases reused the predicted map. The saved summary reports zero invalid maps. Existing maps were included in category and comparison panels; no Grad-CAM was recomputed for Stage 3.

## Correct-case evidence

The saved case group contains two correct melanoma and two correct nevus examples in each dataset's selected map set. However, the existing `gradcam_review.csv` marks all PH? attention judgments `UNCERTAIN`, and the HAM qualitative CSV is a prior small curated assessment, not a matched blinded review. This Stage 3 report therefore does not claim that either domain's correct-case maps focus more on lesion interiors, borders, or background. Those case-level judgments remain `NEEDS_HUMAN_REVIEW` in `stage3_case_comparison.csv`.

## Failure-case evidence

The selected PH? errors include melanoma?nv (2), melanoma?other (2), nevus?melanoma (2), and nevus?other (2). Their saved predictions establish the outcome categories, but the saved Stage 2 worksheet does not contain completed human attention labels. Accordingly, no case-level statement is made about lesion-centered, border, background, artifact, or diffuse attention, nor whether predicted- and true-class maps visibly differ.

## HAM versus PH? qualitative comparison

The figures folder contains six PH? category panels covering all 12 approved cases and four HAM-versus-PH? panels: correct melanoma, melanoma?nv, correct nevus, and nevus?melanoma. For each panel, all available cases in the selected preexisting HAM category and the fixed PH? category were included, ordered by image ID. No map was selected for visual attractiveness. The preselected HAM Grad-CAM set has no exact matched group for PH? melanoma?other or nevus?other, so these are shown as PH?-only category panels rather than matched comparisons.

Panels are review aids, not findings. Without completed human review, the comparison does not support a statement that activation moved to or away from lesions, borders, hair, or background across domains.

## Interpretation

Stage 1 supplies measurable image-feature differences and saved-prediction shifts; Stage 2 supplies technically valid exploratory maps for 12 fixed follow-up cases. Together they are consistent with possible domain shift and motivate careful visual review. They do not establish that brightness, color, image geometry, attention, or any other feature caused the observed performance differences.

## Limitations

- PH? was previously used in project analyses; this is an external follow-up evaluation, not an untouched independent validation.
- PH? includes only 40 melanoma and 80 common-nevus cases and uses a secondary mirror with documented provenance limitations.
- HAM test and PH? are different populations; the internal test includes repeated lesions, while the PH? package has no patient-linkage table.
- The HAM Grad-CAM cases were preselected validation examples, not a random test sample. PH? cases were fixed by the approved outcome categories.
- All PH? qualitative review fields remain `NEEDS_HUMAN_REVIEW`; no lesion masks or quantitative localization ground truth support localization accuracy claims.
- Grad-CAM is an exploratory visualization of model sensitivity and does not establish causal reasoning or clinical validity.
- No clinical claims are made. No thresholds or model settings were tuned from PH?.

## What the evidence supports

The saved artifacts support a descriptive observation that the PH? follow-up differs from the internal HAM overlap cohort in measured image proxies, confidence/entropy summaries, and class-outcome proportions. The selected Grad-CAM figures are ready for human review under a fixed, documented selection rule.

## What the evidence does not support

The artifacts do not prove that domain shift is the cause of the performance change; do not prove what the model reasons about; do not establish clinical interpretability, clinical validity, or localization accuracy; and do not justify changing Experiment #5 based on PH?.

## Implications for EG-VAN+

The measured image and prediction shifts motivate testing, in a future separately approved and independently evaluated EG-VAN+ study, whether explicit attention and robustness objectives improve performance across acquisition domains. This is a hypothesis/design motivation only. The current evidence does not show that an attention branch will improve results.
