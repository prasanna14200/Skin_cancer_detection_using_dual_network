# Stage 13C C batch-7 forward probe

The audited AMP-8192 A/B/C gate passed A and B for 16 batches each. C had a **recoverable GradScaler overflow** on batch 2: the optimizer step was skipped, scale backed off to 4096, and model/optimizer state remained finite. C then took finite steps on batches 3–6. On batch 7, its forward logits became non-finite and a BatchNorm running-mean buffer was non-finite before any batch-7 backward or optimizer step. Lowering the initial scale alone is therefore not yet an evidence-based repair.

This probe reproduces C's first six HAM **train** batches from seed 42 and checks their hashes, losses, step/skip events, scales, and state against the audited gate. It snapshots the post-batch-6 model **in CPU memory only**, then runs the same batch-7 input twice from that state: once with CUDA FP16 autocast and once in FP32. The two passes use the same RNG state. Forward hooks record the first leaf-module input/output with a non-finite value and separately scan parameters and buffers after each pass. It performs **no batch-7 optimizer update** and writes only `stage13c_C_batch7_forward_probe.json`, never a research checkpoint. It creates no validation, HAM test, or PH2 loader.

On the same mounted Google Drive repository in a **Tesla T4** Colab runtime, run this single cell:

```python
from pathlib import Path
import subprocess, sys, torch

ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
SCRIPT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_c_batch7_forward_probe.py'
GATE = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_abc_8192_gate.json'
OUTPUT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_C_batch7_forward_probe.json'
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
assert SCRIPT.is_file() and GATE.is_file() and not OUTPUT.exists()
subprocess.run([sys.executable, str(SCRIPT), '--project-root', str(ROOT),
                '--output', str(OUTPUT)], check=True)
```

Exact Colab terminal command after Drive is mounted:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/analysis/stage13_melanoma_ablation/stage13c_c_batch7_forward_probe.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --output /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/analysis/stage13_melanoma_ablation/stage13c_C_batch7_forward_probe.json
```

Download the JSON for audit. Do not run another scale gate or full training on the basis of this probe alone.
