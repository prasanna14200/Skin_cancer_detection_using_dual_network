# Stage 23 final image-quality assessment: validation development analysis

**Status: observational analysis and bounded synthetic-degradation pilot complete; no validated image-quality gate.** The classifier remains frozen at Stage 23 epoch 14, SHA256 `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`. The analysis uses only the 986 saved HAM validation predictions and corresponding validation images. No classifier training, threshold tuning, HAM test outcome, or PH2 image/outcome was used.

## Mapping and measurement audit

All **986** Stage 23 selected prediction IDs joined one-to-one to the frozen split and cleaned metadata; every true label and lesion ID agreed. All **986 raw JPEGs** and **986 paper-preprocessed JPEGs** were present, JPEG-decodable, and exactly 600×450 pixels. Individual raw and processed SHA256 values are in [image_quality_metrics.csv](image_quality_metrics.csv); no missing or duplicate image was dropped. Recomputed processed quality features agreed with the existing historical `experiments/image_quality/image_quality.csv` table to absolute tolerance `1e-8`. That historical table contains quality rows for all splits; only validation IDs were joined or analyzed. No HAM test image or outcome was opened. The Stage 23 prediction, checkpoint, split, registry and metadata hashes are recorded in [quality_error_analysis.json](quality_error_analysis.json) and [quality_analysis_manifest.json](quality_analysis_manifest.json).

Three measurement views are kept distinct:

1. **Raw source:** `data/raw/images/{image_id}.jpg`, before EG-VAN paper preprocessing.
2. **Processed source:** `data/processed/images/{image_id}.jpg`, after the paper-informed preprocessing but before Stage 23 resize.
3. **Model view:** processed image converted to RGB and bilinearly resized to 384×384, before tensor conversion and ImageNet normalization. Pixel-level quality proxies after normalization would be hard to interpret as brightness/contrast, so they were not used.

The eight reused `src/image_quality.py` proxies are grayscale mean brightness (0–255), grayscale SD contrast, variance of the OpenCV grayscale Laplacian as an edge/sharpness proxy, mean HSV saturation divided by 255, grayscale fraction below 30, grayscale fraction above 225, histogram entropy in bits, and 16×16-cell spatial illumination variation. Bounds 30 and 225 define pixel fractions; they are **not** accept/reject criteria. No defensible hair, ruler, occlusion, or clinical artifact detector was available, so no artifact label was invented.

## Association with Stage 23 classification errors

There were **168/986** validation errors. The cohort is imbalanced: NV 663, MEL 107, BKL 104, BCC 58, AKIEC 30, VASC 15, and DF 9. Exploratory equal-count terciles (`329/328/329` images) were formed from raw-feature ranks; image ID breaks equal-value ties. Error-rate intervals are Wilson 95% descriptive intervals, with no multiplicity correction.

| Raw proxy | Low tercile errors | Middle | High | Interpretation |
|---|---:|---:|---:|---|
| Sharpness | 56/329 = 17.02% | 56/328 = 17.07% | 56/329 = 17.02% | No aggregate error-rate gradient. |
| Brightness | 47/329 = 14.29% | 65/328 = 19.82% | 56/329 = 17.02% | Non-monotonic. |
| Contrast | 61/329 = 18.54% | 49/328 = 14.94% | 58/329 = 17.63% | Non-monotonic. |
| Dark-pixel fraction | 40/329 = 12.16% | 41/328 = 12.50% | 87/329 = 26.44% | Highest band has more errors, but strong class-mix confounding. |
| Bright-pixel fraction | 44/329 = 13.37% | 35/328 = 10.67% | 89/329 = 27.05% | Highest band has more errors; not a validated overexposure cutoff. |
| Saturation | 83/329 = 25.23% | 76/328 = 23.17% | 9/329 = 2.74% | Strong class-mix confounding; high band includes 303 NV cases. |

For the highest dark-pixel band, the error-rate Wilson interval is **21.97–31.46%**, versus **9.06–16.13%** in the lowest band. Yet the high-dark band includes **70 MEL cases** versus 15 in the low band; higher error rate cannot be attributed to darkness alone. The highest-saturation band contains 303 NV cases, which are easier for this model, versus 187 in the low band. Full classwise true/predicted distributions, correct/incorrect feature summaries, band ranges and intervals are in the JSON. [Distributions](quality_error_distributions.png), [quality–entropy scatter](quality_vs_entropy.png), and [band error rates](quality_error_rate_analysis.png) show these exploratory associations.

Raw saturation correlates with confidence (Spearman **+0.436**) and inversely with entropy (**−0.431**); raw dark-pixel fraction shows the reverse (confidence **−0.405**, entropy **+0.390**). Raw sharpness has weak correlations with confidence (**−0.073**) and entropy (**+0.064**). Associations may reflect class mix, lesion properties, imaging conditions and preprocessing interactions. They are not causal evidence that changing one proxy will change classification accuracy.

### Melanoma-specific observation

Stage 23 correctly classified **70/107** melanomas; **37** were false negatives, including **26 MEL→NV**. Median raw sharpness was **91.89** for correctly classified MEL, **53.75** for MEL false negatives, and **49.36** for MEL→NV cases. This subgroup pattern differs from the flat aggregate sharpness tercile pattern. It is descriptive only: no image-quality ground truth, random assignment, or adjustment for lesion difficulty supports a causal blur explanation. The subgroup contains only 37 false negatives. Within the raw sharpness terciles, melanoma false-negative counts were 11, 15 and 11; denominator/class mix differs by band.

## Does raw-image quality add error-ranking information beyond entropy?

The prespecified comparison used the eight existing **raw** technical features for a quality-only score. Two fixed, untuned error-ranking models were fitted with `StandardScaler` and balanced logistic regression (`C=1`) in five `StratifiedGroupKFold` folds, grouped by lesion ID. Each image received a prediction only from a model trained outside its fold; no lesion appeared in both fold train and held-out partitions. The combined model uses saved entropy plus the same eight raw features. Direct saved entropy is the no-fit baseline. The [OOF table](quality_ranking_oof.csv) records fold and score per image; [comparison JSON](quality_uncertainty_comparison.json) records every fold, metric and 80%-coverage result.

| Error ranking | AUROC | Average precision | AURC (lower better) | Errors reviewed at exploratory 80% coverage |
|---|---:|---:|---:|---:|
| Saved entropy alone | **0.8349** | **0.4660** | 0.04994 | **96/168** |
| Raw quality only, grouped OOF | 0.7239 | 0.2936 | 0.07295 | 61/168 |
| Entropy + raw quality, grouped OOF | 0.8403 | 0.4540 | **0.04660** | 93/168 |

The combined AUROC difference from entropy alone is **+0.00544**. A 1,000-repetition lesion-cluster bootstrap of the fixed out-of-fold scores gave a 95% percentile interval of **−0.00795 to +0.01755** for that difference. This interval includes zero; average precision fell and the 80%-coverage error capture was slightly worse. **No clear additional error-detection value is established.** AURC is the mean retained error rate across every coverage prefix `k=1…986`; tied scores use ascending image ID. These are development-set out-of-fold comparisons, not independent external validation or a fitted clinical review rule.

## Quality-warning decision

The current evidence supports displaying **descriptive quality indicators** and, at most, clearly labeled exploratory risk ranking. It does **not** justify a validated image accept/reject gate or an automatic clinical warning threshold. The raw-image feature bands were derived from this same validation set, contain strong class-mix differences, and have no dermatologist quality labels. No threshold was promoted to the local app or frozen model.

## Synthetic-degradation pilot

The fixed [protocol](degradation_protocol.json) selected the lexicographically smallest validation ID from each of seven true classes without viewing correctness or confidence. It specifies identity, Gaussian blur radii 1 and 2 pixels, brightness factors 0.7 and 1.3, contrast factor 0.7, and JPEG quality 40. Perturbations were applied to **already processed** images before the unchanged Stage 23 resize/normalization; they do not model raw acquisition quality. The bounded CPU run completed **49/49 forwards** on seven cases. All seven CPU baseline argmax labels matched the saved validation labels; the largest absolute probability difference was **0.000940**, so the CPU replay is not byte-identical to the original CUDA predictions. The frozen checkpoint hash remained unchanged.

| Synthetic condition | Correct of 7 | Mean max-softmax | Mean entropy, nats | Mean Laplacian sharpness |
|---|---:|---:|---:|---:|
| Baseline | 4 | 0.7509 | 0.6576 | 1216.26 |
| Mild blur, radius 1 px | 3 | 0.6767 | 0.9067 | 30.95 |
| Moderate blur, radius 2 px | 2 | 0.5754 | 1.1768 | 4.59 |
| Brightness ×0.7 | 4 | 0.7200 | 0.7071 | 598.30 |
| Brightness ×1.3 | 5 | 0.7279 | 0.6507 | 1968.62 |
| Contrast ×0.7 | 4 | 0.7047 | 0.7123 | 598.19 |
| JPEG quality 40 | 4 | 0.7022 | 0.7609 | 503.49 |

Blur reduced the technical sharpness proxy and, in these seven examples, lowered mean confidence and raised mean entropy. Moderate blur changed four argmax labels relative to baseline; the single selected MEL case changed from MEL to NV. These observations **do not estimate population robustness or justify a quality threshold**. The brighter condition's 5/7 versus baseline's 4/7 must not be called a performance improvement: one case changed to correct in a nonrepresentative pilot. Full per-case results, condition parameters and hashes are in [degradation_results.csv](degradation_results.csv), [degradation_summary.json](degradation_summary.json), and the [figure](degradation_robustness.png). The run took roughly 13 minutes on the available CPU, so no larger validation degradation sweep was attempted.

## Day 3 and research boundary

The [Day 3 plan](../final_three_objectives/DAY3_EXTERNAL_VALIDATION_PLAN.md) fixes the Stage 23 external follow-up cohort, mapping, preprocessing, metrics, overlap checks and uncertainty analysis before PH2 inference. PH2 had prior project use, so it cannot be called untouched external validation. Day 2 made no Stage 23 model change or new threshold, and did not read HAM test or PH2 outcomes.

Reproduce the observational outputs with `python analysis/stage23_image_quality_final/analyze_quality.py`; run `python analysis/stage23_image_quality_final/run_degradation_pilot.py --check` to verify the fixed pilot. The completed bounded pilot used `--run`. Source and output hashes are in `quality_analysis_manifest.json` and `degradation_summary.json`.
