# Stage 13C memory-safe C batch-7 forward probe

The original `stage13c_C_batch7_forward_probe.json` is preserved as failed diagnostic evidence. It reproduced C's first six batches and the non-finite **AMP** batch-7 forward, then ran out of Tesla T4 memory while attempting FP32. It contains **no FP32 numerical result**.

The replacement script writes a **new** file, `stage13c_C_batch7_memory_safe_probe.json`. It replays the same seed-42 C train batches 1–6 and checks their hashes, losses, GradScaler scales, steps/skips, and finite state against the audited 8192 gate. Before batch 7 it stores the model state on **CPU**, then releases the last training graph, gradient tensors, DataLoader iterator, optimizer and its GPU state. AMP and FP32 forward-only passes run sequentially under `torch.no_grad()` from the same saved model and RNG state. Hooks are removed after each pass and inspect activation slices to limit temporary mask memory. The JSON records allocated/reserved/free CUDA memory around each phase and records any CUDA OOM explicitly. It never saves a research checkpoint or accesses validation, HAM test, or PH2.

Use the same mounted Google Drive repository in a **Tesla T4** Colab runtime. Run this one cell:

```python
from pathlib import Path
import subprocess, sys, torch

ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
SCRIPT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_c_batch7_memory_safe_probe.py'
GATE = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_abc_8192_gate.json'
OUTPUT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_C_batch7_memory_safe_probe.json'
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
assert SCRIPT.is_file() and GATE.is_file() and not OUTPUT.exists()
subprocess.run([sys.executable, str(SCRIPT), '--project-root', str(ROOT),
                '--output', str(OUTPUT)], check=True)
```

Exact Colab terminal command after Drive is mounted:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/analysis/stage13_melanoma_ablation/stage13c_c_batch7_memory_safe_probe.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --output /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/analysis/stage13_melanoma_ablation/stage13c_C_batch7_memory_safe_probe.json
```

Download the new JSON for audit. If it reports `CUDA_OOM_IN_DIAGNOSTIC`, treat that mode's numerical status as undetermined. Do not start any 25-epoch run or another scale gate from this diagnostic alone.
