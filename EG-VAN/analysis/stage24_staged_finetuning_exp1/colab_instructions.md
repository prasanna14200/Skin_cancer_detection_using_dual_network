# Stage 24 Colab Instructions

Use the existing verified Drive shortcut and Tesla T4 runtime. The local Windows machine is for CPU tests and read-only preflight only. Run the check first; it does not download pretrained weights, infer, train, or create a run directory. Review the printed preregistration before explicitly starting the single run.

`EXACT_COLAB_CHECK_COMMAND`

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage24_staged_finetuning_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --check
```

`EXACT_COLAB_TRAIN_COMMAND`

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage24_staged_finetuning_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --train
```

`EXACT_COLAB_RESUME_COMMAND`

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage24_staged_finetuning_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --resume --train
```

Resume is allowed only from a valid unfinalized `run/last_checkpoint.pt`. It validates the checkpoint against history, RNGs, phase, optimizer group layout, scheduler ratios, numerical protocol, initialization provenance, and sidecar hashes. An ahead or conflicting sidecar stops resume. No command automatically evaluates HAM test/PH2 or starts another stage.

Expected runtime: the `--check` command is a CPU/read-only metadata and hash audit and may take minutes because it hashes large frozen checkpoints. The 25-epoch `--train` command requires a Tesla T4 and may take hours; exact duration depends on Drive throughput and Colab availability. `--resume --train` runs only epochs after the saved checkpoint, never a fresh fallback. Review the `--check` result and verify that `run/` does not exist before explicitly choosing the training command. No training command was executed during Stage24 preparation.
