# Accuracy and performance gap audit

## Target and comparison boundary

**Binding numerical target: UNVERIFIED.** The original [project documentation](../../docs/PROJECT_DOCUMENTATION.md) states a seven-class recognition goal and leakage-aware baseline comparison, but no numerical pass threshold. The later [Stage 22 audit](../stage22_performance_gap/stage22_audit.md) discusses an aspirational 95–98% full-cohort range and the source paper's 98.20% nine-class and 97.80% seven-class scores. Neither is shown to be a preregistered Stage 23 threshold or a same-split comparator. Therefore no scientifically valid *target gap* is computed. The measured Stage 23 validation accuracy is **818/986 = 0.8296146045** (82.96%), macro F1 **0.6712537991**, MEL recall **70/107 = 0.6542056075**. “Close to target” is unanswerable until a target and comparable evaluation protocol are defined.

## Comparable validation results

The machine-readable [comparison audit](MODEL_COMPARISON_AUDIT.csv) lists source files, architecture, partition, split protocol, selected status, seven class recalls where available, and checkpoint identity. All five controlled EfficientNetV2S runs, Stage 9, Stage 15, Stage 23 and Stage 25 use the frozen 986-image lesion-aware validation split, though objectives, initialization and selection policies differ. Stage 13C A, Stage 14 and Stage 24 have **no eligible selected checkpoint**; their diagnostic epochs in the CSV are explicitly not candidate results. Historical naive/leakage-aware EfficientNet test values are on a different partition and must not be ranked against Stage 23 validation.

| Run, selected epoch | Validation accuracy | Macro F1 | MEL recall | Interpretation |
|---|---:|---:|---:|---|
| EfficientNetV2S controlled #5, 15 | 0.827586 | 0.668408 | 60/107 = 0.560748 | Historical single-backbone reference; different model. |
| Stage 9 dual EG-VAN, 15 | 0.822515 | 0.670912 | 62/107 = 0.579439 | Earlier dual-network control. |
| Stage 15 dual EG-VAN, 16 | 0.819473 | 0.661131 | 63/107 = 0.588785 | Recovered selected checkpoint; original weight byte identity unprovable. |
| **Stage 23**, 14 | **0.829615** | **0.671254** | **70/107 = 0.654206** | Current frozen scientific candidate. |
| Stage 25, 12 | 0.831643 | 0.686166 | 63/107 = 0.588785 | Valid within-run checkpoint, but registered Stage 23 comparison is mixed: +2 correct overall, −7 MEL true positives. |

Stage 23 versus Stage 25 is the clearest matched comparison: same partition, 986 images, seven classes, and shared scientific configuration except the registered backbone learning-rate policy. Stage 25's higher accuracy (+0.203 percentage points), macro F1 (+0.014912), and lower focal loss coexist with a **6.54 percentage-point MEL recall reduction** and 10 more MEL→NV errors (26→36). The [Stage 25 scientific audit](../stage25_conservative_backbone_lr_exp1/final_stage25_scientific_audit.md) correctly calls this `MIXED_VALIDATION_RESULT`; higher aggregate scores alone do not replace Stage 23 under the melanoma-sensitive objective.

The historical naive EfficientNet test achieved 0.883813 accuracy and 0.807350 macro F1; the leakage-aware test achieved 0.841223 and 0.711408 ([saved metrics](../../experiments/efficientnetv2s_leakage_aware/test_metrics.json)). Different split protocols and partitions preclude a direct Stage 23 validation ranking. The source-paper 98.20% nine-class claim and 97.80% seven-class claim lack an equivalent lesion-isolated protocol ([Stage 22 reconstruction](../stage22_performance_gap/paper_pipeline_reconstruction.md)); subtracting either from 82.96% would be misleading.

## Stage 23 class behavior

From [selected validation metrics](../../experiments/stage23_resnet50_imagenet_init_exp1/run/validation_metrics.json):

| True class | Support | Recall / true positives |
|---|---:|---:|
| AKIEC | 30 | 0.5000 (15) |
| BCC | 58 | 0.6207 (36) |
| BKL | 104 | 0.5288 (55) |
| DF | 9 | 0.5556 (5) |
| MEL | 107 | 0.6542 (70) |
| NV | 663 | 0.9427 (625) |
| VASC | 15 | 0.8000 (12) |

Confusion matrix, true rows and predicted columns in `AKIEC BCC BKL DF MEL NV VASC` order:

```text
AKIEC  15  2  7 1  3   2 0
BCC     2 36  4 1  0  15 0
BKL     6  2 55 2 14  25 0
DF      1  2  0 5  0   1 0
MEL     0  3  3 1 70  26 4
NV      0  2  9 0 27 625 0
VASC    0  0  0 0  0   3 12
```

NV makes up 663/986 = 67.24% of validation cases. High NV recall and prevalence help overall accuracy; macro F1 exposes lower performance in smaller classes. MEL→NV accounts for 26/37 = 70.27% of MEL false negatives; NV→MEL counts 27. DF support is only nine, so single-case changes strongly move its recall. These are observed error counts, not a proven cause of the errors.

## Bottlenecks: observation versus hypothesis

| Topic | Observation | Hypothesis or limit |
|---|---|---|
| MEL/NV confusion and imbalance | 26 MEL→NV and 27 NV→MEL on Stage 23 validation; MEL sensitivity is well below NV sensitivity. | Class prevalence and visual similarity may contribute; neither causal contribution has been isolated. |
| Preprocessing and image quality | [Phase A](../stage23_image_quality_final/phase_a/final_phase_a_report.md) reports lower median raw sharpness among MEL false negatives, but quality proxies are confounded. [Phase B](../stage23_image_quality_final/phase_b_full_validation_cuda/final_research_results/PHASE_B_RESULTS_SUMMARY.md) shows synthetic blur radius 1 cuts MEL recall 70→16/107 and all six degradations reduce overall accuracy. | Real acquisition blur may matter, but synthetic perturbation cannot establish prevalence, clinical adequacy, or benefit of a new preprocessing pipeline. Resolution (384 pixels) was not independently varied here. |
| Training and loss | Stage 24 staged fine-tuning had no eligible epoch; Stage 25 lower trunk LR improved aggregates but lost MEL sensitivity. Stage 14 MEL-weighted run had no eligible epoch. | No single mechanism is proved; further searches on the same validation set risk selecting noise. |
| Calibration/uncertainty | [Stage 23 uncertainty report](../stage23_uncertainty_final/uncertainty_report.md): ECE 0.039768, entropy error AUROC 0.834869, MEL-only miss AUROC 0.5927. Phase B entropy often fell as degradation increased errors. | Softmax uncertainty can rank many errors but is an unreliable quality detector and may miss confident MEL errors; no clinical probability claim. |
| Domain shift/external | Stage 16 prior Stage 15 model: PH2 external follow-up MEL recall 11/40 (27.5%); PH2 had prior project use and excluded atypical nevi. | Suggests external transfer risk for that *older checkpoint*, not measured Stage 23 external performance. |

## Selection and leakage risks

The frozen split groups by lesion and Stage 23 validation IDs match it, reducing within-split repeated-lesion leakage. It does not eliminate acquisition/site or cross-dataset confounding. The original naive image-level split had related-lesion leakage risk. Stage 23, Stage 24, Stage 25, uncertainty, and quality analyses reuse the same validation cohort; repeated model and cutoff inspection makes apparent gains optimistic. Historical project-wide HAM test and PH2 exposure also means those datasets should not be repurposed as fresh development sets. No Stage 23 held-out or external performance is inferred from Stage 15/16 outcomes. The current evidence supports a **VALIDATE** decision before another accuracy search.
