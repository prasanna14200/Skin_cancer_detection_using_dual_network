# Stage 13 Colab Tesla T4 handoff

The local HAM train/validation preflight and five synthetic/unit checks passed. No full training has run. Keep the A/B/C configurations and `selection_rule.json` fixed. The runner uses only HAM train/validation and refuses local CPU training.

Mount Drive and point to the existing repository:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
RUNNER = ROOT / 'experiments/egvan_melanoma_ablation_exp/train_ablation.py'
assert RUNNER.is_file()
```

Run read-only preflight and the batch-16 T4 memory probe. The probe performs one synthetic optimizer step and saves no research checkpoint:

```python
import subprocess, sys, torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT), '--preflight'], check=True)
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT), '--memory-probe'], check=True)
```

After the probe reports `PASS`, run each **pre-registered** configuration once. These are full 25-epoch HAM train/validation runs, with no HAM test or PH² access:

```python
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT), '--variant', 'A', '--train'], check=True)
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT), '--variant', 'B', '--train'], check=True)
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT), '--variant', 'C', '--train'], check=True)
```

If one of these runs is interrupted, use the same command with `--resume` for that variant only. Do not rerun a completed variant. The runner checks configuration identity before restoring model, optimizer, scheduler, scaler, sampler and RNG state.

After A/B/C have all completed, apply the frozen validation-only final rule:

```python
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT), '--finalize'], check=True)
```

Inspect `analysis/stage13_melanoma_ablation/selected_candidate.json`. It will report either `CANDIDATE_SELECTED` with a frozen checkpoint SHA256 or `NO_CANDIDATE_SELECTED`. Do not change the rule after seeing validation results. Stop before HAM test, PH², or Stage 14.
