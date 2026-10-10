# Stage 10 Colab Tesla T4 handoff

Stage 10 preflight passed locally. No test inference ran because the local environment has CPU only. The Colab evaluator reruns preflight and prints it before inference; it refuses to run without a CUDA Tesla T4 or if test outputs already exist.

Mount Drive and locate the existing repository:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
RUNNER = ROOT / 'analysis/egvan_internal_test_exp1/evaluate_stage10.py'
assert RUNNER.is_file()
```

Run the read-only preflight:

```python
import subprocess, sys, torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT), '--preflight'], check=True)
```

After preflight reports `PASS`, perform the **single** frozen HAM test evaluation:

```python
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT),
                '--batch-size', '16', '--num-workers', '2', '--run'], check=True)
```

The runner writes predictions, metrics, class metrics, confusion matrices, the descriptive Experiment #5 comparison, report and final manifest to `analysis/egvan_internal_test_exp1/`. It verifies saved predictions before marking the manifest complete. Do not rerun the command after outputs exist. It never reads PH² or changes either checkpoint.
