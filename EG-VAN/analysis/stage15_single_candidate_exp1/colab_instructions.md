# Stage 15 single midpoint candidate: Colab commands

The local CPU suite and read-only preflight must pass before the one T4 run. Review the preregistration and runner before starting training. The following path uses the existing Google Drive shortcut already used for Stage 14.

Read-only Colab preflight:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage15_single_candidate_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --check
```

After review, start **one** Tesla T4 candidate run:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage15_single_candidate_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --train
```

If this exact run is interrupted after a completed epoch, resume its existing checkpoint:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage15_single_candidate_exp1/train.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --resume --train
```

Expected output: `experiments/stage15_single_candidate_exp1/run/`, including one `last_checkpoint.pt`, history, numerical events, a hashed manifest, and **25 pairs** of compact per-epoch validation files if all epochs complete. `best_checkpoint.pt` and selected validation files exist only if at least one epoch meets all frozen gates. No HAM test or PH2 loader is used.
