# Stage 13C bounded multistep AMP diagnostic

This diagnostic runs **Variant A only**, from fresh seed-42 initialization, for 16 HAM **train** batches at each initial scale 8192, 4096, and 2048. It uses the frozen split, existing augmentation, sampler, architecture, focal loss, Adamax settings, batch size 16, CUDA autocast, and dynamic GradScaler. Each scale gets a fresh model and the same deterministic data sequence. It records per-batch hashes, loss and gradient finiteness, actual optimizer-step execution, GradScaler backoff, and model/optimizer finiteness. A detected gradient overflow is allowed to pass through `scaler.step()` and `scaler.update()` so a normal skipped update can be distinguished from state corruption. There is no validation/test/PH2 inference, scheduler step, research checkpoint, or full epoch.

Run this **single Colab cell** after mounting the existing Drive repository and selecting a Tesla T4 runtime:

```python
from pathlib import Path
import subprocess, sys, torch

ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
SCRIPT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_multistep_amp_diagnostic.py'
OUTPUT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_multistep_amp_trace.json'
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
assert SCRIPT.is_file() and not OUTPUT.exists(), 'Keep any existing diagnostic trace; use a new output name'
subprocess.run([sys.executable, str(SCRIPT), '--project-root', str(ROOT),
                '--output', str(OUTPUT)], check=True)
```

Download `stage13c_multistep_amp_trace.json` into the same local analysis directory for independent audit. The script refuses to overwrite an existing output. A finite 16-batch path is evidence only for that bounded window; it does not establish whole-run stability. Do not start full training after this cell. Review the trace first, including any skipped optimizer steps and post-step state checks.
