# Stage 11 Colab Tesla T4 handoff

The local preflight passed. Inference was not run locally because CUDA is unavailable. The evaluator uses the frozen epoch-15 EG-VAN checkpoint and the exact 120 IDs in the saved Experiment #5 PH² cohort manifest. It reproduces the existing HAM-matched BMP preprocessing, including the in-memory JPEG round trip. No images, checkpoint, or previous results are modified.

Mount Drive and locate the existing repository:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
RUNNER = ROOT / 'analysis/egvan_ph2_external_followup_exp1/evaluate_stage11.py'
assert RUNNER.is_file()
```

Run read-only preflight:

```python
import subprocess, sys, torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT), '--preflight'], check=True)
```

After it prints `PASS`, run the single frozen 120-image follow-up inference:

```python
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT),
                '--batch-size', '16', '--num-workers', '2', '--run'], check=True)
```

The runner verifies saved prediction rows against the fixed cohort and independently recomputes the saved metrics. It writes results only to `analysis/egvan_ph2_external_followup_exp1/`, refuses to overwrite existing result files, and never reruns HAM or Experiment #5 inference. Its final report calls PH² an **external follow-up** because the dataset had earlier project exposure.
