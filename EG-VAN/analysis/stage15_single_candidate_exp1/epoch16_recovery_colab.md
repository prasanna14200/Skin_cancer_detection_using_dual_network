# Stage 15 bounded epoch-16 recovery on Tesla T4

Use the existing Google Drive shortcut and select a Tesla T4. The original `run/` directory must contain the exact audited epoch-13 checkpoint and original validation files; the separate `recovery_epoch16/` directory must not yet exist. Do not use the edited `train.py --resume` for this recovery: its source hash intentionally differs from the epoch-13 checkpoint. The recovery script imports the hash-matched archived source.

Optional **read-only** Colab preflight:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage15_single_candidate_exp1/recover_epoch16.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --check
```

Exact **bounded** T4 recovery command (epochs 14–16 only):

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage15_single_candidate_exp1/recover_epoch16.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --recover
```

Expected separate output: `experiments/stage15_single_candidate_exp1/recovery_epoch16/` with `epoch_014_checkpoint.pt`, `epoch_015_checkpoint.pt`, `epoch_016_checkpoint.pt`, three prediction/metric pairs under `validation_epochs/`, `comparison.json`, and—**only if all strict comparisons pass and epoch 16 remains eligible**—`best_checkpoint.pt` and `recovery_manifest.json`. A mismatch writes `recovery_failure.json` and stops. No original Stage 15 artifact is overwritten. Download and audit hashes after the Colab run before treating any recovered checkpoint as valid.
