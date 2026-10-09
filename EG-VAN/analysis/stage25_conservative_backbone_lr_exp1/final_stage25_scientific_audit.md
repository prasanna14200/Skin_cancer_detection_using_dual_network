# Stage 25 final scientific audit: conservative ResNet LR versus Stage 23

**Scope:** Read-only audit of saved HAM train/validation evidence. No training, resume, model inference, HAM test performance, PH2 performance, or external evaluation was used. Stage 23 selected epoch 14 is the frozen comparator. Stage 24 artifacts were not altered.

## 1. Artifact integrity and completion

Stage 25 `run/experiment_manifest.json` records `CANDIDATE_SELECTED_VALIDATION_ONLY`, 25 epochs, selected epoch 12, best validation loss **0.07156147242808983**, and validation decision `MIXED_VALIDATION_RESULT`. Its 63 listed artifacts all exist and independently match their recorded SHA256 hashes. The run snapshots of configuration, selection rule, numerical protocol, LR policy, and initialization specification match their preregistered files. The manifest's source hashes match the current Stage 25 `train.py`, `initialization.py`, and `comparison.py`; split SHA256 is `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`.

`training_history.csv` contains exactly epochs 1–25 without gaps or duplicates. Every train/validation loss and recorded validation metric is finite. All 25 saved prediction files contain the same ordered 986 validation image IDs and true labels. Their predicted labels reproduce the corresponding confusion matrices and recorded metrics. Independent application of all frozen gates identifies **epoch 12 alone** as eligible. Its validation loss is also the unconditional minimum among the 25 epochs; checkpoint selection is therefore consistent. Stage 25 epochs 14 and 15 had 67/107 and 64/107 MEL true positives, respectively, but each missed the macro-F1 gate. They cannot replace epoch 12.

| Checkpoint | SHA256 | Embedded epoch | History rows | Selected epoch / loss |
|---|---|---:|---:|---|
| Stage 25 `best_checkpoint.pt` | `0c80944672852bccc3fad07841c4b8a265afeb1af02a33e96d21a19c614dffb5` | 12 | 12 | 12 / 0.07156147242808983 |
| Stage 25 `last_checkpoint.pt` | `a2269b49b65e4fd68db34dd200d848fbc078b697a74ab593ee68457d89b5961a` | 25 | 25 | 12 / 0.07156147242808983 |
| Stage 23 selected `best_checkpoint.pt` | `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab` | 14 | 14 | 14 / 0.07410776640362206 |

Both Stage 25 checkpoint histories match the CSV prefix exactly. Their embedded per-epoch validation artifact hashes match the corresponding saved prediction and metric files. Model parameters **and buffers** (1,277 state entries) and optimizer state tensors (766 parameter slots) are finite in both checkpoints. Their scheduler epoch, LR groups, and last-LR values agree with history; GradScaler states have finite positive scales (65,536 at epoch 12 and 131,072 at epoch 25). Python, NumPy, torch, CUDA, and sampler RNG states are present. The final checkpoint has two optimizer groups: 607 new/non-ResNet tensors and 159 pretrained trunk tensors. No model/optimizer corruption is evidenced.

The run logs two recoverable AMP skipped steps: epoch 21 batch 178 and epoch 25 batch 225, each at scale 262,144 with a backoff to 131,072. The event log matches the history skip counts and the checkpoint event records. Both checkpoints remain finite, and all epochs completed. These late skipped steps do not explain the selected epoch-12 comparison.

### Configuration and provenance

The initialization report states `LOADED` from official `ResNet50_Weights.IMAGENET1K_V2`, using torchvision hash checking, for 159 trunk parameter tensors and 159 trunk buffer tensors; ImageNet classifier `fc.weight`/`fc.bias` were intentionally excluded. No missing, unexpected, or shape-mismatched backbone keys are reported. The initializer source requests this official state directly and does not load the trained Stage 23 checkpoint. The independent CPU parameter-group test confirms that all ResNet `conv1`, `bn1`, and `layer1`–`layer4` parameters are trainable and assigned exactly once to the pretrained group. The source has no staged freezing or forced pretrained BatchNorm eval mode. Epoch 1 history records 501 optimizer steps with LRs **0.001** for new/non-ResNet modules and **0.0001** for the pretrained trunk; every saved epoch preserves the 10:1 ratio. Checkpoint optimizer groups and scheduler states corroborate the split. These records strongly support the registered policy; a final checkpoint alone cannot independently reconstruct every historical `requires_grad` flag at every batch.

Stage 25 and Stage 23 saved configurations have the same architecture, seed, split, class order, 384-pixel preprocessing/augmentation path, batch 16, weighted sampler, focal loss with true-MEL multiplier, numerical policy, 25-epoch maximum, and Adamax/scheduler settings apart from the registered two-group LR allocation. Both runners call the same guarded validation function for the **common unweighted validation focal loss**. The recorded loss values are comparable as saved; no loss was recalculated by inference in this audit. Stage 25's frozen gates and tie rule match Stage 23's, including the literal registered thresholds.

### Metadata clarification

`best_val_loss` is **not** a Stage 25 checkpoint key. Both checkpoints and the manifest store the value under **`best_validation_loss`**, consistently at 0.07156147242808983. `comparison.json` is **not** required by this runner or its manifest. The expected, hash-verified output is `run/stage15_stage23_stage25_validation_comparison.json`, which exists and records `MIXED_VALIDATION_RESULT`. The absence of `comparison.json` is not an artifact-integrity failure.

## 2. Independent selected-epoch validation comparison

The Stage 23 and Stage 25 selected prediction CSVs have identical ordered image IDs and true labels (986 cases; class order `akiec, bcc, bkl, df, mel, nv, vasc`). Recomputed confusion matrices reproduce every aggregate and classwise metric below. Recorded focal losses come from the selected validation metric files, whose SHA256 hashes agree with each experiment's manifest. A separately marked [derived validation comparison](stage23_vs_stage25_validation_comparison.json) contains the full-precision recomputation and source hashes; it is **not** an original training artifact.

| Metric | Stage 23 epoch 14 | Stage 25 epoch 12 | Stage 25 − Stage 23 |
|---|---:|---:|---:|
| Validation focal loss | 0.0741077664 | **0.0715614724** | −0.0025462940 |
| Accuracy | 0.8296146045 | **0.8316430020** | +0.0020283976 (2 additional correct cases) |
| Balanced accuracy / macro recall | 0.6574259626 | **0.6667058236** | +0.0092798610 |
| Macro precision | 0.6938315547 | **0.7138027185** | +0.0199711637 |
| Macro F1 | 0.6712537991 | **0.6861662373** | +0.0149124382 |
| Weighted F1 | 0.8243435992 | **0.8263818969** | +0.0020382977 |
| MEL precision | 0.6140350877 | **0.6428571429** | +0.0288220551 |
| MEL recall | **0.6542056075 (70/107)** | 0.5887850467 (63/107) | **−0.0654205607 (7 fewer MEL TP)** |
| MEL F1 | **0.6334841629** | 0.6146341463 | **−0.0188500166** |
| NV recall | 0.9426847662 (625/663) | **0.9472096531 (628/663)** | +0.0045248869 |

| Class (support) | Stage 23 P / R / F1 | Stage 25 P / R / F1 | TP change |
|---|---|---|---:|
| AKIEC (30) | 0.625 / 0.500 / 0.556 | 0.571 / 0.533 / 0.552 | +1 |
| BCC (58) | 0.766 / 0.621 / 0.686 | 0.735 / 0.621 / 0.673 | 0 |
| BKL (104) | 0.705 / 0.529 / 0.604 | 0.690 / 0.577 / 0.628 | +5 |
| DF (9) | 0.500 / 0.556 / 0.526 | 0.545 / 0.667 / 0.600 | +1 |
| MEL (107) | 0.614 / 0.654 / 0.633 | 0.643 / 0.589 / 0.615 | **−7** |
| NV (663) | 0.897 / 0.943 / 0.919 | 0.896 / 0.947 / 0.921 | +3 |
| VASC (15) | 0.750 / 0.800 / 0.774 | 0.917 / 0.733 / 0.815 | −1 |

Stage 23 confusion matrix (true rows, predicted columns in frozen class order):

```text
       AK  BCC BKL DF MEL NV  VA
AK     15   2   7  1   3  2   0
BCC     2  36   4  1   0 15   0
BKL     6   2  55  2  14 25   0
DF      1   2   0  5   0  1   0
MEL     0   3   3  1  70 26   4
NV      0   2   9  0  27 625  0
VA      0   0   0  0   0  3  12
```

Stage 25 confusion matrix:

```text
       AK  BCC BKL DF MEL NV  VA
AK     16   2   6  0   3  3   0
BCC     3  36   4  2   2 11   0
BKL     7   7  60  2  10 18   0
DF      1   1   0  6   0  1   0
MEL     1   1   4  1  63 36   1
NV      0   2  13  0  20 628  0
VA      0   0   0  0   0  4  11
```

## 3. Melanoma error trade-off

Stage 25 made **44** MEL false negatives versus Stage 23's **37**. MEL→NV increased **26→36** (+10), while NV→MEL decreased **27→20** (−7). MEL false positives decreased **44→35**, improving MEL precision even as recall fell. BKL true positives increased by five and NV by three, helping aggregate accuracy and macro F1. Thus the aggregate gains and lower loss coexist with a more conservative MEL prediction pattern on this validation partition. The selected predictions differ on 96 of 986 cases. This is a descriptive association with the lower trunk LR, not proof that a particular feature or arithmetic mechanism caused the shift. The nine-case DF and fifteen-case VASC supports also limit interpretation of their classwise changes.

## 4. Frozen decision and candidate recommendation

Stage 25 epoch 12 satisfies **all** within-run gates: MEL TP 63/107, MEL recall 0.5887850467, MEL F1 0.6146341463, macro F1 0.6861662373, accuracy 0.8316430020, and NV recall 0.9472096531. It is a valid selected Stage 25 **validation-only** checkpoint. The registered Stage 25-versus-Stage 23 comparison requires no regression across aggregates/classwise metrics and retention of Stage 23 MEL recall and F1 for `CLEAR_VALIDATION_IMPROVEMENT`. Both MEL measures regress; some other classwise measures also regress. Because primary aggregate metrics improve, the exact registered outcome is **`MIXED_VALIDATION_RESULT`**, matching the saved comparison sidecar and manifest. It cannot be called a clear improvement.

The scientific question asked whether the lower backbone LR could preserve or improve melanoma discrimination **and** aggregate validation performance. Stage 25 improved several aggregates but **did not preserve Stage 23 melanoma recall or F1**. For a melanoma-sensitive classifier objective, Stage 23 epoch 14 remains the more defensible current scientific candidate; Stage 25 remains a valid within-run selected checkpoint with a mixed comparative result. This recommendation uses validation evidence only and makes no claim about unseen outcomes.

## 5. Missing evidence, limitations, and next action

No required original Stage 25 artifact is missing. The two apparent metadata gaps are naming differences resolved above. We cannot independently prove the exact historical pretrained tensor values or every per-batch trainability flag solely from final checkpoints, although the hash-verified initializer source, initialization report, optimizer-group coverage/state, and epoch-1 LR/step records support the registered execution. No replicate run or independent validation cohort is available to quantify the stability of the small aggregate gains. Repeated use of the same validation set raises overfitting risk for further post-hoc LR searches.

**No additional training run is justified on the present evidence and time constraint.** Stage 25 demonstrates a trade-off rather than a win on the combined objective; choosing another LR from these outcomes would be a new validation-driven search. Retain Stage 23 epoch 14 as the current scientific candidate and stop model-development decisions based on this validation comparison. Do not substitute HAM test or PH2 outcomes. CPU-only validation completed: ten Stage 25 preflight tests passed and the read-only `--check` preflight passed. No inference or training occurred during this audit.
