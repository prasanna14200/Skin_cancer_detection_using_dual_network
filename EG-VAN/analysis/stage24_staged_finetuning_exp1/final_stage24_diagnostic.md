# Stage 24 final diagnostic: staged fine-tuning versus Stage 23

**Scope:** Frozen HAM validation and training records only. This is a diagnostic audit, not checkpoint selection, training, or test/external evaluation. The Stage 23 selected epoch 14 is the reference. Stage 24 completed 25 epochs and produced `NO_CANDIDATE_SELECTED`.

## Evidence and integrity

- Read the Stage 23 selected `run/validation_metrics.json` and `run/training_history.csv`; Stage 23 `best_checkpoint.pt` SHA256 is `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`, matching the Stage 24 manifest reference.
- Read the Stage 24 frozen `selection_rule.json`, `finetuning_policy.json`, `config.json`, numerical protocol, 25-row history, all 25 validation metric JSONs and prediction CSVs, and `run/experiment_manifest.json`. All 61 manifest-listed artifact hashes match their files. The Stage 24 manifest SHA256 is `b8776cf3140aa9dfb3d60d63b7fca6643405b97ff64e6115ad20f6bf0b318526`.
- Stage 24 epochs are exactly 1–25. All recorded train and validation losses and validation metrics are finite. Each 986-row validation prediction file reproduces its saved confusion matrix; all 25 files have the same ordered image IDs and true labels. Confusion matrices independently reproduce accuracy, balanced accuracy, macro F1, MEL precision/recall/F1, and NV recall. Recomputed gate failures match the saved metric sidecars and history for every epoch.
- Stage 24 manifest records `selected_epoch: null`, `best_validation_loss: null`, `NO_CANDIDATE_SELECTED`, and no HAM test or PH2 access. `best_checkpoint.pt` is absent as required when no epoch is eligible. The Stage 23 selected checkpoint remains the current candidate. No checkpoint was loaded or changed for this audit.

## Frozen eligibility decision

Every condition must hold: MEL true positives at least **63/107**, MEL recall at least **0.5887850467**, MEL F1 at least **0.5757731959**, macro F1 at least **0.6509124105**, accuracy at least **0.8025152130**, and NV recall at least **0.9172096531**. Only then does minimum common unweighted validation focal loss select a checkpoint, with earlier epoch breaking an exact loss tie. **None of the 25 Stage 24 epochs passes all six gates.** The lower-loss epoch 7 cannot be selected.

| Reference / epoch | Train loss | Val loss | Accuracy | Balanced accuracy | Macro F1 | MEL precision | MEL TP/107 | MEL F1 | NV recall | MEL→NV | NV→MEL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage 23 selected 14 | 0.02718 | 0.07411 | 0.82961 | 0.65743 | 0.67125 | 0.61404 | 70/107 | 0.63348 | 0.94268 | 26 | 27 |
| Stage 24 epoch 6 | 0.05060 | 0.07286 | 0.81237 | 0.66133 | 0.66052 | 0.54545 | 54/107 | 0.52427 | 0.93665 | 37 | 27 |
| Stage 24 epoch 7 | 0.04290 | **0.07138** | 0.82150 | 0.66792 | **0.68186** | 0.63158 | **48/107** | **0.52459** | 0.94268 | **41** | 16 |
| Stage 24 epoch 8 | 0.03936 | 0.07953 | 0.79209 | 0.64475 | 0.65918 | 0.45652 | 63/107 | 0.51429 | 0.90196 | 30 | 51 |
| Stage 24 epoch 9 | 0.03759 | 0.07625 | 0.81947 | 0.68002 | 0.67697 | 0.56863 | 58/107 | 0.55502 | 0.94419 | 34 | 25 |
| Stage 24 epoch 16 | 0.01362 | 0.08412 | **0.82657** | 0.66954 | 0.66995 | 0.65116 | 56/107 | 0.58031 | 0.95173 | 38 | 17 |
| Stage 24 epoch 19 | 0.01368 | 0.08649 | 0.82252 | 0.65953 | 0.66143 | 0.62766 | 59/107 | **0.58706** | 0.95023 | 37 | 21 |

**Epoch 7:** validation loss is 0.002729 lower than Stage 23 epoch 14; macro F1 is 0.010601 higher and balanced accuracy is 0.010493 higher. Accuracy is 0.008114 lower. MEL recall falls from 70/107 (0.654206) to 48/107 (0.448598), a loss of 22 correctly identified melanomas, and MEL F1 falls from 0.633484 to 0.524590. MEL precision rises slightly (0.614035 to 0.631579), but this does not compensate for the sensitivity loss. Epoch 7 fails **MEL TP, MEL recall, and MEL F1**; it passes the aggregate and NV gates. Its 41 MEL→NV errors represent 41/59 = 69.49% of MEL false negatives, compared with Stage 23's 26/37 = 70.27%; the absolute MEL→NV count increased by 15. Epoch 8 reaches the 63-MEL-TP threshold, but has 75 MEL false positives and fails MEL F1, accuracy, and NV recall. No alternative Stage 24 epoch supplies a valid checkpoint.

### All Stage 24 validation epochs

Decimals below are display-rounded only; gate checks used saved full-precision values. `MEL TP` is the number of correctly recalled melanomas out of 107.

|Epoch|Phase|Train loss|Val loss|Acc|Bal acc|Macro F1|MEL P|MEL TP/107|MEL F1|NV R|MEL→NV|NV→MEL|
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|A|0.12238|0.10968|0.649|0.474|0.466|0.284|84/107|0.417|0.725|17|131|
|2|A|0.08544|0.09969|0.776|0.536|0.538|0.476|49/107|0.467|0.938|35|31|
|3|A|0.07616|0.08197|0.774|0.597|0.598|0.429|57/107|0.475|0.888|29|50|
|4|A|0.07058|0.09171|0.765|0.553|0.585|0.413|50/107|0.439|0.926|44|38|
|5|A|0.06770|0.08654|0.768|0.614|0.583|0.453|39/107|0.404|0.902|41|24|
|6|B|0.05060|0.07286|0.812|0.661|0.661|0.545|54/107|0.524|0.937|37|27|
|7|B|0.04290|0.07138|0.822|0.668|0.682|0.632|48/107|0.525|0.943|41|16|
|8|B|0.03936|0.07953|0.792|0.645|0.659|0.457|63/107|0.514|0.902|30|51|
|9|B|0.03759|0.07625|0.819|0.680|0.677|0.569|58/107|0.555|0.944|34|25|
|10|B|0.02877|0.08573|0.817|0.674|0.673|0.667|50/107|0.549|0.956|44|15|
|11|B|0.02482|0.08392|0.801|0.656|0.661|0.518|57/107|0.525|0.944|41|27|
|12|B|0.02093|0.07590|0.820|0.676|0.679|0.586|51/107|0.526|0.955|39|19|
|13|B|0.01855|0.08370|0.812|0.662|0.662|0.564|57/107|0.548|0.941|38|26|
|14|B|0.01697|0.08055|0.817|0.670|0.661|0.591|55/107|0.550|0.946|38|21|
|15|B|0.01531|0.08235|0.817|0.685|0.664|0.563|58/107|0.552|0.938|36|23|
|16|B|0.01362|0.08412|0.827|0.670|0.670|0.651|56/107|0.580|0.952|38|17|
|17|B|0.01322|0.08714|0.815|0.660|0.651|0.582|57/107|0.556|0.941|38|24|
|18|B|0.01427|0.08816|0.820|0.657|0.663|0.632|55/107|0.567|0.959|41|15|
|19|B|0.01368|0.08649|0.823|0.660|0.661|0.628|59/107|0.587|0.950|37|21|
|20|B|0.01313|0.08707|0.820|0.660|0.657|0.644|56/107|0.577|0.953|38|17|
|21|B|0.01281|0.08588|0.816|0.660|0.654|0.592|58/107|0.566|0.941|35|25|
|22|B|0.01204|0.08515|0.819|0.673|0.663|0.622|56/107|0.569|0.941|36|22|
|23|B|0.01248|0.08706|0.816|0.654|0.648|0.611|58/107|0.574|0.943|35|25|
|24|B|0.01230|0.08760|0.823|0.666|0.661|0.662|53/107|0.567|0.956|40|16|
|25|B|0.01147|0.08500|0.815|0.666|0.657|0.570|57/107|0.551|0.938|35|26|

## Classwise and confusion-matrix diagnosis

|Class (validation support)|Stage 23 epoch 14 P / R / F1|Stage 24 epoch 7 P / R / F1|F1 change|
|---|---|---|---:|
|AKIEC (30)|0.625 / 0.500 / 0.556|0.545 / 0.400 / 0.462|−0.094|
|BCC (58)|0.766 / 0.621 / 0.686|0.750 / 0.621 / 0.679|−0.006|
|BKL (104)|0.705 / 0.529 / 0.604|0.622 / 0.663 / 0.642|+0.037|
|DF (9)|0.500 / 0.556 / 0.526|0.857 / 0.667 / 0.750|+0.224|
|MEL (107)|0.614 / 0.654 / 0.633|0.632 / 0.449 / 0.525|−0.109|
|NV (663)|0.897 / 0.943 / 0.919|0.890 / 0.943 / 0.916|−0.003|
|VASC (15)|0.750 / 0.800 / 0.774|0.700 / 0.933 / 0.800|+0.026|

Epoch 7's higher macro F1 reflects larger BKL, DF, and VASC gains averaging over classes, while MEL and AKIEC worsen. DF has only nine validation cases, so its F1 swing deserves caution. Stage 23 MEL→NV was 26 and NV→MEL 27; Stage 24 epoch 7 was 41 and 16. At epoch 8, the MEL recall gate is met, but NV→MEL rises to 51; MEL precision drops to 0.457 and NV recall to 0.902. The classifier's MEL/NV tradeoff changes across epochs rather than improving both directions. Stage 24 epoch 19 has the highest MEL F1 (0.587) but only 59/107 MEL true positives. Stage 24 epoch 16 has the highest accuracy (0.827), yet only 56/107 MEL true positives. Stage 23's selected 70/107 MEL true positives is not matched by any Stage 24 epoch that also satisfies the other gates.

## Losses, phases, and learning rates

Stage 23 had the entire ResNet trunk trainable from epoch 1 at a common initial LR of 0.001. Its train loss decreased from 0.12311 at epoch 1 to 0.02718 at selected epoch 14 and 0.01257 at epoch 25. Validation loss was 0.10240 at epoch 1, reached its selected minimum of 0.07411 at epoch 14, and was 0.08139 at epoch 25. Stage 24 train loss decreased from 0.12238 at epoch 1 to 0.06770 at epoch 5, 0.05060 at epoch 6, 0.04290 at epoch 7, and 0.01147 at epoch 25. Its validation loss went 0.10968 → 0.08654 (epochs 1→5), improved to 0.07286/0.07138 at epochs 6/7, then fluctuated and finished at 0.08500. The widening late train–validation gap and no sustained validation improvement support late fitting without useful validation gain; they do not establish a specific mechanism.

**Phase A (epochs 1–5):** `conv1`, `bn1`, `layer1`, and `layer2` were frozen; `layer3`/`layer4` were trainable at 0.0001 initially, versus 0.001 for non-ResNet/new modules. The lowest Phase A validation loss was 0.08197 (epoch 3); its highest macro F1 was 0.59836 (epoch 3). Phase A never passed the gates. Epoch 1's 84 MEL true positives came with MEL precision 0.284, NV recall 0.725, and poor overall accuracy; high isolated recall cannot establish good discrimination. By epoch 5 MEL TP had fallen to 39.

**Phase B (epochs 6–25):** `layer2` became trainable at epoch 6; `conv1`, `bn1`, and `layer1` remained frozen. The 5→6 transition had train loss 0.06770→0.05060, val loss 0.08654→0.07286, accuracy 0.768→0.812, macro F1 0.583→0.661, and MEL TP 39→54. This is a measurable coincident transition, but it cannot isolate the causal effect of unfreezing: one more training epoch, ongoing updates to other modules, and the existing scheduler also changed the trajectory. Epoch 7 was the loss/macro-F1 peak; after epoch 9, no sustained MEL recovery occurred. Three recoverable AMP skips were logged (epochs 9, 21, 25); all saved losses/metrics remain finite, so the validation outcome is not explained by an observed numerical collapse.

The history reports LRs **after each epoch's scheduler step**. `layer3` and `layer4` were equal throughout; `layer2` retained half their LR even while frozen in Phase A. All four groups were halved together by `ReduceLROnPlateau`, preserving 1:0.1:0.1:0.05 ratios.

|Epoch after step|Non-ResNet/new|Layer3|Layer4|Layer2|Stage 23 common LR|
|---:|---:|---:|---:|---:|---:|
|1|0.001|0.0001|0.0001|0.00005|0.001|
|5|0.0005|0.00005|0.00005|0.000025|0.001|
|9|0.00025|0.000025|0.000025|0.0000125|0.001|
|11|0.000125|0.0000125|0.0000125|0.00000625|0.0005|
|13|0.0000625|0.00000625|0.00000625|0.000003125|0.00025|
|14|0.0000625|0.00000625|0.00000625|0.000003125|0.00025|
|15|0.00003125|0.000003125|0.000003125|0.0000015625|0.00025|
|17|0.000015625|0.0000015625|0.0000015625|0.00000078125|0.000125|
|19|0.0000078125|0.00000078125|0.00000078125|0.000000390625|0.0000625|
|21|0.00000390625|0.000000390625|0.000000390625|0.0000001953125|0.00003125|
|23|0.000001953125|0.0000001953125|0.0000001953125|0.00000009765625|0.000015625|
|25|0.0000009765625|0.00000009765625|0.00000009765625|0.000000048828125|0.0000078125|

The exact post-step reductions occurred at Stage 24 epochs 5, 9, 11, 13, 15, 17, 19, 21, 23, and 25. By epoch 14, `layer3`/`layer4` LR was 40× below Stage 23's common LR and `layer2` LR was 80× below. By epoch 25, those ratios were 80× and 160×. This is strong evidence that the registered policy provided much smaller **potential** late backbone updates. Actual parameter-update norms or representation drift were not recorded, so “negligible updates” cannot be verified directly. The scheduler was responding to validation-loss plateauing, not acting as an independent randomized cause. The policy combines initial freezing, permanently frozen shallow layers/BN handling, and lower discriminative LRs; this experiment cannot disentangle them.

## Interpretation and future hypotheses

Stage 24 is a completed, numerically finite experiment with a clear early Phase B aggregate response but a **validation-inferior melanoma operating point** relative to selected Stage 23. It is consistent with an over-constrained/under-adapted pretrained trunk, particularly given frozen early layers and rapidly shrinking layer2–4 LRs. It also shows late train loss continuing downward while validation loss rises. Neither observation alone proves under-adaptation or classic overfitting. There is no evidence of persistent instability. Stage 23's full trunk updates from epoch 1, at substantially higher LR, plausibly helped retain more MEL-sensitive features; this is a hypothesis, not a causal conclusion, because Stage 24 changed several coupled parts of the fine-tuning policy.

Ranked hypotheses for a **separately preregistered future study**, all based solely on this training/validation comparison:

1. **The low discriminative backbone LR, independent of staged freezing, limited MEL adaptation.** Isolate one change relative to Stage 23: keep its full trainability from epoch 1 and all other settings, but set the ResNet trunk initial LR to **0.0001** while keeping non-ResNet/new modules at **0.001** and the same `ReduceLROnPlateau` rule. This is one fixed LR-policy comparison, with no sweep or freeze phase. Expected information value **high** for separating LR from freezing; compute cost **one 25-epoch T4 run**; validation-overfitting risk **moderate** because this same 986-case validation partition has already informed multiple experiments. A failure would support retaining Stage 23, not justify retuning the LR after inspection.
2. **The early freeze interval itself reduced MEL discrimination.** Isolate only a preregistered freeze interval relative to Stage 23 while keeping the Stage 23 LR policy. Expected value **moderate**, compute **one full run**, overfitting risk **moderate/high**. Stage 24's phase transition makes this plausible but does not prove it.
3. **Repeated validation selection is amplifying apparent small aggregate gains.** A future, independently sourced validation study could test robustness without further optimizer changes. Expected scientific value **high**, compute **data collection plus evaluation**, overfitting risk **lower if genuinely independent**. This is not a license to revisit current held-out test or PH2 outcomes for configuration.

**Recommendation:** If another T4 experiment is funded, preregister only hypothesis 1 as a controlled LR-only comparison to Stage 23, with the fixed 0.0001 ResNet/0.001 other initial LRs and unchanged eligibility gates. The scientific purpose is to isolate the LR component of Stage 24's coupled policy, not to claim an expected performance gain. Do not create Stage 25 code or launch a run from this audit; the Stage 23 epoch-14 checkpoint remains the current candidate. If the objective is immediate deployment rather than mechanism testing, stopping model development and retaining Stage 23 is the lower-risk decision. No HAM test or PH2 result was consulted.

## Required disposition

STAGE24_FINAL_STATUS: COMPLETE; NO_CANDIDATE_SELECTED
CANDIDATE_SELECTED: NO
WHY_STAGE24_FAILED: No epoch passed all frozen gates; epoch 7 had lower loss and higher macro F1 but only 48/107 MEL TP and MEL F1 0.524590, failing three MEL gates.
STAGE23_REMAINS_CURRENT_CANDIDATE: YES; selected epoch 14
PHASE_A_FINDING: Loss fell, but no eligible epoch; MEL discrimination remained weak/variable under frozen lower ResNet layers.
PHASE_B_FINDING: Epoch-6 unfreezing coincided with aggregate gains, but no eligible epoch or sustained MEL recovery.
LR_SCHEDULER_FINDING: Ten post-epoch halving events reduced late layer2–4 LRs to very small values; actual update norms were not recorded.
MELANOMA_FINDING: Stage23 selected 70/107 MEL TP and 26 MEL→NV; Stage24 epoch 7 selected none, with 48/107 MEL TP and 41 MEL→NV.
OVERALL_DIAGNOSIS: Finite and complete but validation-inferior on the preregistered melanoma objective; over-constrained/under-adapted policy is plausible, not causally established.
NEXT_HYPOTHESES: (1) backbone LR limitation; (2) freeze interval effect; (3) validation-selection fragility on an independent cohort.
RECOMMENDED_NEXT_EXPERIMENT: Only if funded and separately preregistered, one Stage23-relative LR-only ablation (ResNet 0.0001; other 0.001; full trainability from epoch 1); no code or run prepared here.
HAM_TEST_ACCESSED: NO
PH2_ACCESSED: NO
TRAINING_PERFORMED: NO
NEXT_ACTION: Review this diagnostic and decide whether a single controlled future study is warranted; otherwise retain Stage23.
