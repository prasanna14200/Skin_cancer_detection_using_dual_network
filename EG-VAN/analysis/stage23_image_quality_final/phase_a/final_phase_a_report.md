# EG-VAN+ Stage 23 image quality: Phase A completion

## Scope and provenance

This is **exploratory validation-only quality characterization**, not a technical-adequacy classifier. The frozen Stage 23 epoch-14 checkpoint SHA256 is `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`. The script verifies it and the frozen split, saved predictions, image-quality table and grouped out-of-fold score table against the historical manifest before analysis. It reads 986 unique Stage 23 validation IDs from 743 lesions, all matched to the frozen validation partition; 168 predictions are incorrect. No training, inference, HAM test outcome, or PH2 outcome was used. The raw, processed and 384×384 model-view measurements are preserved in the historical source table; eight raw-image proxies were the primary features here. Their exact definitions and limitations are in [quality_metric_definitions.md](quality_metric_definitions.md).

## Class mix and adjusted associations

| True class | Images | Errors | Error rate |
|---|---:|---:|---:|
| AKIEC | 30 | 15 | 50.0% |
| BCC | 58 | 22 | 37.9% |
| BKL | 104 | 49 | 47.1% |
| DF | 9 | 4 | 44.4% |
| MEL | 107 | 37 | 34.6% |
| NV | 663 | 38 | 5.7% |
| VASC | 15 | 3 | 20.0% |

[Classwise distributions and effects](class_adjusted_quality_analysis.csv) provide, for each feature and true class, quartiles, correct/error medians, and rank-biserial error effect with a 400-draw lesion-cluster percentile interval. Separate single-feature ridge logistic models adjust each raw feature for true class; odds ratios are per one full-cohort SD. The class-adjusted estimates are exploratory associations, not effects of image defects. They do not adjust for lesion difficulty, acquisition center, skin tone or other confounders. Tiny DF and VASC event counts make their subgroup effects especially imprecise.

| Raw proxy | Class-adjusted error OR per SD | Lesion-bootstrap 95% interval |
|---|---:|---:|
| Brightness | 1.158 | 0.934–1.416 |
| Contrast | 1.277 | 1.005–1.614 |
| Sharpness | 1.203 | 0.623–1.534 |
| Saturation | 0.573 | 0.440–0.711 |
| Dark-pixel fraction | 1.178 | 0.986–1.684 |
| Bright-pixel fraction | 1.241 | 1.128–1.609 |
| Histogram entropy | 1.193 | 0.930–1.563 |
| Illumination variation | 1.184 | 0.962–1.458 |

Several adjusted intervals exclude one, but these eight proxies are correlated and multiply examined; this is not evidence that changing brightness, contrast, saturation or pixel fractions will improve accuracy. For example, lesion color and morphology can affect saturation and contrast. The prior aggregate dark-pixel gradient was strongly class-confounded, and this adjusted dark-pixel interval includes one. [Adjusted-association figure](class_adjusted_quality_or.png) is descriptive.

## Melanoma subgroup

There are 70 correct MEL predictions, 37 false negatives, and 26 MEL→NV errors. Median raw Laplacian variance is 91.89 for correct MEL and 53.75 for false negatives; the rank-biserial effect for *higher sharpness among errors* is −0.432 (lesion-bootstrap 95% interval −0.661 to −0.215). Equivalently, **lower** sharpness ranks false negatives above correct MEL cases with apparent subgroup AUROC 0.716. This is a within-cohort association, not a validated screening score or evidence that blur caused the errors: the Laplacian also responds to hair, noise, lesion texture and framing. The raw dark-pixel fraction effect is −0.268 (−0.481 to −0.039); this particularly illustrates why darkness cannot be equated with poor quality. Other subgroup intervals mostly cross zero. All eight feature effects and MEL→NV medians are in [melanoma_quality_analysis.csv](melanoma_quality_analysis.csv).

Saved maximum-softmax confidence has median 0.720 for correct MEL versus 0.689 for false negatives; predictive entropy medians are 0.681 and 0.716 nats. Two melanoma false negatives have confidence ≥0.9 (`ISIC_0026296`, `ISIC_0032862`), so a review policy based on uncertainty can miss confidently wrong cases. These softmax values are not calibrated clinical probabilities. Subgroup estimates are based on only 37 false negatives.

## Does quality add value beyond entropy?

The historical OOF analysis used five `StratifiedGroupKFold` folds, with zero lesion overlap; `StandardScaler` and fixed balanced logistic regression (`C=1`) were fitted only in each training fold. The eight raw features were fixed, with no feature selection. This Phase A script independently checks all 986 OOF IDs, lesion/fold consistency and scores, then recomputes the metrics and paired lesion-cluster intervals from the **saved scores**. It does not refit or tune the models. AURC is the mean prefix error rate over all retained counts; lower is better. Ties are broken by image ID.

| Error ranking | AUROC | Average precision | AURC | Errors referred at 80% coverage | MEL false negatives referred |
|---|---:|---:|---:|---:|---:|
| Saved entropy | 0.8349 | 0.4660 | 0.04994 | 96/168 | 15/37 |
| Raw quality only, OOF | 0.7239 | 0.2936 | 0.07295 | 61/168 | 4/37 |
| Entropy + raw quality, OOF | 0.8403 | 0.4540 | 0.04660 | 93/168 | 10/37 |

Combined minus entropy AUROC is +0.00544 (paired lesion-bootstrap 95% interval −0.00795 to +0.01755); average precision is −0.01204 (−0.04312 to +0.02705), and AURC is −0.00334 (−0.00838 to +0.00085). At exploratory 80% coverage, the error-capture difference is −0.01786 (−0.06453 to +0.03848); the MEL false-negative capture difference is −5/37, with a paired interval for the fraction of −0.273 to 0. The [comparison JSON](quality_entropy_comparison.json) includes risk–coverage points from 100% to 50% and the [curve](quality_entropy_risk_coverage.png). The modest AURC point estimate does not overcome uncertain AUROC/AP differences or worse melanoma error capture. **No reliable added value beyond entropy is established.** These intervals condition on fixed OOF scores and omit refit uncertainty; this is not an independent unbiased validation of a selected feature combination. The OOF ranking scores are not interpreted as calibrated risks; calibration of these exploratory error scores would not establish image quality. No review threshold or quality gate was selected.

## Human-label feasibility and disposition

The [blinded review rubric](reviewer_quality_rubric.md) specifies focus, exposure, obstruction, visibility and adequacy ratings from at least two independent reviewers, adjudication, and agreement statistics. Suitable raw validation images exist, but no Stage 23 human quality labels or documented pair of qualified reviewers is available; no reviewer study was performed. The [gap report](quality_validation_gap_report.md) explains the remaining validation requirements. The earlier seven-case synthetic pilot is bounded perturbation evidence, not a population-level acquisition-quality study. Phase B can investigate prespecified degradation behavior, but cannot create a validated gate without external adequacy labels and a separate evaluation boundary. No dark or low-saturation image is designated unacceptable.

Reproduce with `python analysis/stage23_image_quality_final/phase_a/analyze_phase_a.py` and verify with `python -m unittest discover -s tests -p test_stage23_quality_phase_a.py`. This recomputes only saved-image-quality/prediction statistics and figures; it does not run model inference.
