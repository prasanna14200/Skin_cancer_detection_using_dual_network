# Stage 16 Colab T4 final evaluation

Review `pre_evaluation_audit.md` and both evaluators before either one-time run. Sync the exact audited **nested** Stage 15 candidate path to the Drive root below. A missing path or hash mismatch must stop the run. Execute HAM first; keep its result evaluation-only. PH² is a separate mapped external follow-up and must not be interpreted as untouched validation.

Read-only preflight (no inference):

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage16_final_evaluation/evaluate_ham.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --preflight
```

One-time HAM test evaluation after review:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage16_final_evaluation/evaluate_ham.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --run
```

Read-only PH² protocol preflight:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage16_final_evaluation/evaluate_ph2.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --preflight
```

Separate PH² external follow-up, only after reviewing the frozen mapped protocol:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/experiments/stage16_final_evaluation/evaluate_ph2.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --run
```

All output files are written to `analysis/stage16_final_evaluation/` under the Drive project root. Download the outputs and independently audit hashes and metrics before reporting conclusions.
