# Stage 14 single-candidate Colab commands — preparation only

The local `--check` and CPU unit suite passed. **Do not run `--train` until the preregistration and runner have been reviewed.** No extra bounded T4 diagnostic is currently required: the prior audited selective-FP32 B 16-batch gate used the same scientific training objective, sampler, seed, model initialization, and numerical policy.

After mounting the existing Google Drive shortcut, use these exact commands in a Colab terminal. The first is read-only:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage14_single_mel_objective_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --check
```

Only after reviewing the preregistration and selecting a Tesla T4, start the **one** candidate run:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage14_single_mel_objective_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --train
```

If that exact run is interrupted **after** an epoch checkpoint, continue it without changing config/policy:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage14_single_mel_objective_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --resume --train
```

Expected future output directory: `experiments/stage14_single_mel_objective_exp1/run/`. A successful 25-epoch run writes `last_checkpoint.pt`, `training_history.csv`, config/rule/numerical snapshots, `numerical_events.json`, and `experiment_manifest.json`; eligible runs also write `best_checkpoint.pt`, `validation_metrics.json`, and `validation_predictions.csv`. An invalid run writes `failure.json` and stops. The runner refuses to overwrite an existing fresh run.
