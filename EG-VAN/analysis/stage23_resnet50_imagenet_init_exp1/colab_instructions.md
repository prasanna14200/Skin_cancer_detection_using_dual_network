# Stage 23 Colab instructions

The local Windows environment is for source checks and CPU tests only. Do not run `--train` locally. Stage23 requires a Tesla T4, CUDA, and the existing Drive shortcut used by Stage15.

The Stage15 eligibility gates and selection rule are printed by both the read-only check and training startup. Review them before starting the run. The check does not download ImageNet weights; the training command loads the hash-checked official torchvision ResNet50 V2 state before epoch 1 and records the runtime initialization report.

## Exact commands

`EXACT_COLAB_CHECK_COMMAND`

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage23_resnet50_imagenet_init_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --check
```

`EXACT_COLAB_TRAIN_COMMAND`

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage23_resnet50_imagenet_init_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --train
```

`EXACT_COLAB_RESUME_COMMAND`

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage23_resnet50_imagenet_init_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --resume --train
```

Resume is valid only when `run/last_checkpoint.pt` exists and passes provenance, state, RNG, history, and saved-artifact checks. It starts from the next unfinished epoch; it never falls back to a fresh run. Do not resume after a finalized manifest. Do not start Stage24 or evaluate HAM test/PH2 automatically.
