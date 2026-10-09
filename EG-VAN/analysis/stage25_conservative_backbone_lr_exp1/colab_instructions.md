# Stage 25 Colab commands

Use the existing Drive shortcut with a Tesla T4 runtime. The read-only check must pass before explicitly starting the single preregistered run. These commands do not access HAM test or PH2 evaluation data.

## Exact Colab check command

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage25_conservative_backbone_lr_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --check
```

## Exact Colab training command

Run only after reviewing the preflight and confirming `experiments/stage25_conservative_backbone_lr_exp1/run/` does not already exist.

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage25_conservative_backbone_lr_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --train
```

## Exact Colab resume command

Use only for an interrupted, unfinalized Stage25 run with its own valid `run/last_checkpoint.pt`. The command restores all saved state and resumes at saved epoch + 1; it never initializes a fresh run on a missing checkpoint.

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage25_conservative_backbone_lr_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --resume --train
```

The Stage23 trained checkpoint is a hash-verified validation reference only. Stage25 initialization loads the official torchvision ImageNet-V2 ResNet50 state. `--check` is read-only and does not download weights or create `run/`; training and resume require a T4. No Stage25 training command was executed in local preparation.
