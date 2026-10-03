# Stage 13B Variant B numerical failure audit

**Disposition: `INVALID_NUMERICAL_FAILURE`. Clean B rerun from initialization required.** This is a read-only audit of the failed training artifacts. No training or inference was performed.

## Evidence and provenance

The downloaded failed run is preserved at `colab_artifacts/stage13_failed_B/`. It contains exactly `config.json`, `experiment_manifest.json`, `last_checkpoint.pt`, and `training_history.csv`. There is no best checkpoint, selected validation metrics/predictions, or separate resume metadata. The downloaded `config.json` exactly matches the preregistered `experiments/egvan_melanoma_ablation_exp/config_B.json`. The manifest reports `FAIL_NO_ELIGIBLE_CHECKPOINT`, 25 epochs, no selected epoch, Tesla T4, PyTorch 2.11.0+cu130, and the frozen Stage 9 checkpoint/split hashes.

| Preserved file | SHA256 |
|---|---|
| `config.json` | `ddb17729184cc143582ec19f2de25c6af685978e7ed421fa9e3068d5b141135b` |
| `experiment_manifest.json` | `bfabad9a51534094d452352643ae41618211ec98b67b57984eab39a8e2b9b1ef` |
| `training_history.csv` | `0b0bd752217bde41a47d51c980289c59c651da56147c23b84f5520924da4688e` |
| `last_checkpoint.pt` | `2c95e409c15dd4aec411820dbced9a89748a24a437de963d0e6d2ed85ea330f2` |

The checkpoint SHA256 matches the downloaded manifest. A recursive, hidden-inclusive filename search of this repository found no other Stage 13 B `.pt`, `.pth`, or `.ckpt` file. In particular, no epoch-18/pre-resume checkpoint exists locally. This does not rule out a Colab Drive version or a copy outside the repository.

## Epoch history and interruption

`training_history.csv` has exactly epochs 1–25 in order. The checkpoint embeds the same first and last epoch records and has `epoch=25`, `best_epoch=None`, and `best_validation_loss=+Inf`. No epoch is eligible. **The first non-finite epoch is 1; there is no last finite/valid B epoch.** Every epoch's train and validation loss is NaN. The pre-interruption rows 1–18 are therefore already non-finite; the failure did not first appear on resume.

For **each** epoch 1 through 18, the saved values are identical:

| Epochs | Train loss | Val loss | Val accuracy | Val macro F1 | MEL recall | MEL F1 | NV recall | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1–18, every epoch | NaN | NaN | 0.030425963488843813 | 0.00843644544431946 | 0 | 0 | 0 | False |

Epochs 19–25 have the same NaN losses and listed metrics. Their reported zero MEL recall is a continuation of the epoch-1 pattern, not evidence that the resume operation introduced it. The user's chronology says epoch 18 was completed before interruption and 19–25 followed resume. The final artifacts contain no timestamped resume event or saved pre-resume checkpoint, so the exact interruption boundary cannot be independently proved from them. The last **recorded** pre-interruption epoch is 18 according to that chronology; none of epochs 1–18 is a valid finite result.

## Epoch-25 checkpoint tensor audit

The checkpoint was loaded on CPU for inspection only; no forward pass was run.

| State | Finding |
|---|---|
| Model state | **Non-finite.** 1,100 of 1,277 tensors contain NaNs; 58,295,425 floating elements are NaN. No Inf elements were found. This includes learned parameters and running statistics. |
| Adamax optimizer state | **Corrupt.** 1,532 of 2,298 tensors contain NaNs; 116,174,818 floating elements are NaN. No Inf elements were found. `exp_avg` and `exp_inf` buffers are affected. |
| ReduceLROnPlateau state | Structurally present, but no finite best validation loss was ever recorded: `best=+Inf`, `last_epoch=25`, last LR `2.44140625e-07`. `mode_worse=+Inf` is a normal scheduler sentinel; `best=+Inf` here reflects the all-NaN validation history. It is not a usable continuation state. |
| AMP GradScaler state | Structurally present but **invalid for continuation**: `scale=0.0`; the scale must be positive. Growth factor 2, backoff factor 0.5, growth interval 2000, and growth tracker 0 are present. |
| Resume RNG fields | CPU sampler generator, Python, NumPy, CPU PyTorch, and CUDA RNG states are present. |

These are findings for the **epoch-25** checkpoint. The file was atomically overwritten after each epoch, so it is not the checkpoint loaded at the reported epoch-18 resume. Without an independent pre-resume copy, the epoch-18 model, optimizer, scheduler, and scaler tensors cannot be inspected directly. Nevertheless, the recorded epoch-1 through epoch-18 losses prove the original run had already failed numerically before interruption.

## Resume implementation and root-cause limits

`train_ablation.py` restores the exact configuration, model state, optimizer state, scheduler state, AMP GradScaler state, sampler generator state, Python/NumPy/CPU PyTorch/CUDA RNG states, epoch, best loss/epoch, and history. It resumes at saved epoch + 1. Static inspection found no omitted required restore field. It cannot make a NaN-contaminated checkpoint scientifically valid. The current epoch-25 checkpoint is unusable for `--resume` because model and optimizer tensors are corrupted, the scaler is zero, and no finite eligible checkpoint exists.

The immediate trigger of the first NaN cannot be identified from epoch-level records and the final checkpoint alone. Possible batch-level numerical behavior cannot be distinguished without the original logs or an earlier finite checkpoint. The code does have an **error-handling defect**: it neither rejects non-finite batch/epoch losses nor checks model/optimizer finiteness before replacing `last_checkpoint.pt`, allowing the failed run to continue and erase an earlier checkpoint. That defect explains why 25 failed epochs were saved; it does not establish what first caused the NaN. No change to the registered loss, AMP, optimizer, clipping policy, or selection thresholds is justified by this audit. The B multiplier and sampler weights in the downloaded config are exactly the preregistered values.

## Clean rerun preparation; do not execute locally

Keep `colab_artifacts/stage13_failed_B/` unchanged as downloaded evidence. On Colab, preserve the original `experiments/egvan_melanoma_ablation_exp/B/` directory by moving it to `experiments/egvan_melanoma_ablation_exp/B_failed_original_20261003/`. The runner hardcodes its B output to a **new** `experiments/egvan_melanoma_ablation_exp/B/` directory; after the move, that new directory is separate from both failure archives. Do not use `--resume`. This preserves the exact registered B configuration and requires no runner or protocol edit.

Exact Colab cell, using the existing mounted repository path:

```python
from pathlib import Path
import shutil, subprocess, sys

ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
RUNNER = ROOT / 'experiments/egvan_melanoma_ablation_exp/train_ablation.py'
OLD_B = ROOT / 'experiments/egvan_melanoma_ablation_exp/B'
FAILED_ARCHIVE = ROOT / 'experiments/egvan_melanoma_ablation_exp/B_failed_original_20261003'
assert RUNNER.is_file() and OLD_B.is_dir() and (OLD_B / 'last_checkpoint.pt').is_file()
assert not FAILED_ARCHIVE.exists()
shutil.move(str(OLD_B), str(FAILED_ARCHIVE))
assert FAILED_ARCHIVE.is_dir() and not OLD_B.exists()
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT), '--variant', 'B', '--train'], check=True)
```

The cell archives the original Colab B folder first, then starts a clean B run from initialization. Run it only on the intended Tesla T4 environment after confirming the Stage 13 preflight and memory-probe conditions. The runner uses seed 42, the frozen HAM train/validation split, Stage 8B architecture and initialization, 25 epochs, physical/effective batch 16, Adamax at LR 0.001 with weight decay 0.0001, ReduceLROnPlateau, CUDA AMP, original focal loss with true-MEL multiplier **1.5630495442733532**, original MEL sampler weight **1.5630495442733532**, and no gradient clipping. If the clean run again has a non-finite first epoch, stop and collect batch-level evidence; do not tune the registered configuration in response.

The later candidate decision must apply `selection_rule.json` unchanged. Variant C's lack of an eligible checkpoint means the present finalizer's all-three-complete requirement also needs a validation-only workflow resolution before finalization; this does not permit changing the candidate eligibility or ranking rule.

## Boundaries and modifications

Only this audit report was updated. No downloaded failed-B file, A/C artifact, checkpoint, frozen split, config, training runner, or selection rule was modified. HAM test and PH2 files/performance were not accessed. No HAM test, PH2, or other inference was run. No training or Stage 14 work was started.
