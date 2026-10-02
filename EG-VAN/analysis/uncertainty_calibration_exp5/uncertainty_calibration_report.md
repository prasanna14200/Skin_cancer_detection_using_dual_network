# Experiment #5 Uncertainty and Calibration Analysis

## 1. Objective

Evaluate confidence and probabilistic reliability of frozen Experiment #5 saved predictions on the internal HAM10000 TEST set and PH2 external follow-up set. No model forward pass or calibration fitting was performed.

## 2. Frozen Model Provenance

- Checkpoint: `experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt` (SHA256 `81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a`).
- Frozen split SHA256: `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`; selected epoch: 15.
- Class order: `akiec, bcc, bkl, df, mel, nv, vasc`.

## 3. Data Sources

- HAM: saved Experiment #5 TEST probabilities, 1014 samples across seven classes.
- PH2: saved external follow-up probabilities, 120 samples (80 nv, 40 mel); atypical nevi excluded.
- PH2 was previously used in project analyses and is treated as external follow-up, not an untouched independent validation.

## 4. Methods

- Confidence is the maximum of the seven saved class probabilities; true-class probability is the saved probability for the true label.
- Entropy is `-sum(p_k * ln(max(p_k, 1e-12)))` in natural-log units; saved probabilities are unchanged.
- Top-1/top-2 margin is the largest probability minus the second largest.
- ECE uses 10 fixed equal-width bins over [0,1]: `sum_b (n_b/N) * abs(empirical_accuracy_b - mean_confidence_b)`. Empty bins are retained and contribute zero.
- Brier is the per-sample sum of squared errors across all seven classes, averaged over samples; it is not divided by seven.
- NLL is negative mean natural log of true-class probability, clipped below at `1e-12` only for log stability.
- Confidence bands are descriptive: LOW <0.50; MODERATE [0.50,0.75); HIGH [0.75,0.90); VERY_HIGH [0.90,1.00]. No threshold is optimized.

## 5. HAM10000 Internal Results

- N=1014; correct=845; incorrect=169; accuracy=0.8333.
- ECE=0.0390; Brier=0.2408; NLL=0.4757.
- Mean/median confidence=0.8099/0.8751; Q25/Q75=0.6798/0.9581.
- Mean confidence correct/incorrect=0.8486/0.6167; medians=0.9123/0.5979.
- Mean entropy correct/incorrect=0.4364/0.9022; mean margin correct/incorrect=0.7326/0.3499.
- HIGH+VERY_HIGH confidence errors: 38.

## 6. PH² External Follow-Up Results

- N=120; correct=75; incorrect=45; accuracy=0.6250.
- ECE=0.0741; Brier=0.5171; NLL=0.9838.
- Mean/median confidence=0.6899/0.7042; Q25/Q75=0.5815/0.8116.
- Mean confidence correct/incorrect=0.7232/0.6345; medians=0.7437/0.6556.
- Mean entropy correct/incorrect=0.7125/0.8757; mean margin correct/incorrect=0.5294/0.3767.
- HIGH+VERY_HIGH confidence errors: 11.

## 7. Internal vs External Calibration Comparison

PH2 ECE is higher than HAM (0.0741 vs 0.0390); Brier is 0.5171 vs 0.2408; NLL is 0.9838 vs 0.4757. These cross-dataset contrasts are descriptive and not causal.

## 8. Correct vs Incorrect Prediction Confidence

HAM mean confidence correct/incorrect is 0.8486/0.6167; mean entropy correct/incorrect is 0.4364/0.9022.
PH2 mean confidence correct/incorrect is 0.7232/0.6345; mean entropy correct/incorrect is 0.7125/0.8757. These describe saved samples; confidence is not a guarantee of correctness.

## 9. High-Confidence Misclassifications

HIGH+VERY_HIGH error counts are 38 for HAM and 11 for PH2. Band counts and error percentages are in `confidence_band_error_summary.csv`; all errors are in `high_confidence_errors.csv`, sorted by descending confidence. These are high-confidence misclassifications, not clinical-risk labels.

## 10. Class-Level Observations

`class_uncertainty_summary.csv` reports class N, accuracy, mean confidence, entropy, and true-class probability. `error_transitions.csv` reports error counts and mean error confidence. PH2 has 40 melanoma cases; smaller PH2 error groups are descriptive and uncertain.

## 11. Relationship With the 12 PH² Grad-CAM Cases

`ph2_gradcam_uncertainty_join.csv` joins exactly 12 fixed IDs to saved uncertainty features and preserves the prior human-review fields. The visual judgments were not reinterpreted. No Grad-CAM was regenerated, and no causal relationship is inferred.

## 12. Interpretation

ECE, Brier, and NLL describe different aspects of saved probability behavior. Results are computed separately for HAM and PH2. Differences may be consistent with changed reliability on the external follow-up distribution but are limited by cohort size, dataset differences, and ECE binning.

## 13. Limitations

- PH2 is an external follow-up dataset previously used in project analyses, not a completely untouched independent validation.
- PH2 contains 120 included cases and only 40 melanoma cases; only true nv and mel classes are represented.
- Calibration estimates depend on sample size; ECE is bin-dependent.
- HAM and PH2 are different populations and acquisition domains.
- Grad-CAM is exploratory and does not establish causal reasoning or clinical validity.
- No temperature scaling, calibration fitting, or confidence-threshold tuning was performed.

## 14. Conclusion

These values describe reliability metrics of the saved Experiment #5 probabilities. No calibration correction or model adjustment was fitted from PH2.
