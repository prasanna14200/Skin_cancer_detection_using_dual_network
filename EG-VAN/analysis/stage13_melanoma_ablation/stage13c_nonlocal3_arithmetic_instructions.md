# Stage 13C bounded nonlocal3 arithmetic localization

The verified memory-safe probe established finite C batch-7 FP32 forward and non-finite CUDA FP16 AMP forward from the same pre-batch-7 state. It did not record the internal operation that first produced a non-finite value before `resnet.nonlocal3.project`.

This **first-stage localization only** replays C train batches 1–6 from seed 42, verifies the same data hashes/step pattern and pre-batch-7 model hash, and runs batch 7 under AMP and FP32 from a CPU model snapshot. It traces the operations in the repository's actual `NonLocalBlock.forward`: pooled key/value source, `theta`, `phi`, `g`, reshapes/transposes, `torch.bmm(q, k)`, `torch.softmax`, `torch.bmm(attention, v)`, `project`, and residual addition. The implementation has **no attention scaling operation**, and the trace records that explicitly. Each intermediate records dtype, shape, finiteness, NaN/Inf counts, and finite min/max. The diagnostic temporarily replaces only this instance's `forward` method in memory; it restores it after each pass and **does not edit the architecture source**.

No batch-7 optimizer step, research checkpoint, full epoch, HAM validation/test loader, or PH2 loader is used. This script does **not** test a selective-FP32 policy. Review its JSON before creating that second bounded diagnostic.

Run this one cell in the mounted Google Drive repository on a **Tesla T4**:

```python
from pathlib import Path
import subprocess, sys, torch

ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
SCRIPT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_nonlocal3_arithmetic_probe.py'
REFERENCE = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_C_batch7_memory_safe_probe.json'
OUTPUT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_nonlocal3_arithmetic_trace.json'
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
assert SCRIPT.is_file() and REFERENCE.is_file() and not OUTPUT.exists()
subprocess.run([sys.executable, str(SCRIPT), '--project-root', str(ROOT),
                '--output', str(OUTPUT)], check=True)
```

Exact Colab terminal command after Drive is mounted:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/analysis/stage13_melanoma_ablation/stage13c_nonlocal3_arithmetic_probe.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --output /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/analysis/stage13_melanoma_ablation/stage13c_nonlocal3_arithmetic_trace.json
```

Download `stage13c_nonlocal3_arithmetic_trace.json` for independent audit. Do not start a selective-FP32 test or A/B/C gate until the first non-finite arithmetic operation has been established from this trace.
