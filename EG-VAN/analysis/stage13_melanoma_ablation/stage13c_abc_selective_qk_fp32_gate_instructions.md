# Stage 13C selective q@k FP32, 16-batch A/B/C gate

This is a bounded **HAM train only** numerical gate on Tesla T4. It runs exactly 16 deterministic batches for each variant unless a failure stops a variant early. It does not write a research checkpoint or run a full epoch. It uses seed 42, the registered scientific configurations, initial GradScaler scale 8192, and the same in-memory `resnet.nonlocal3` q@k FP32 policy for A, B, and C. The policy changes only the precision of the demonstrated overflowing affinity matrix multiplication. Production model source remains untouched.

The gate verifies the audited selective C batch-7 trace hash, frozen split, configurations, model initialization, and reference data-sequence hashes. Per batch it records loss, logits, scaled/unscaled gradients, GradScaler scale and step/skip, model parameters/buffers, optimizer state, and input/label hashes. A safe overflow with a skipped step and scaler backoff can continue; any non-finite forward or state, unexplained skip, invalid step, or incomplete 16-batch variant fails the gate. `PASS_BOUNDED_WINDOW_ALL_VARIANTS` means only bounded numerical clearance, not permission to start 25-epoch training automatically.

After mounting the same Google Drive shortcut and selecting a Tesla T4, run this **one Colab cell**:

```python
from pathlib import Path
import subprocess, sys, torch

ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
SCRIPT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.py'
REFERENCE = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_selective_qk_fp32_trace.json'
OUTPUT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.json'
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
assert SCRIPT.is_file() and REFERENCE.is_file() and not OUTPUT.exists()
subprocess.run([sys.executable, str(SCRIPT), '--project-root', str(ROOT),
                '--output', str(OUTPUT)], check=True)
```

Exact terminal command after Drive is mounted:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --output /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.json
```

Download `stage13c_abc_selective_qk_fp32_gate.json` for independent audit before any further experiment.
