# Stage 13C bounded selective q@k FP32 probe

Run only after copying the current repository and all referenced Stage 13C JSON files to Colab. Use a Tesla T4 and the same mounted project root used for the preceding arithmetic probe. The script refuses to overwrite its output. It verifies the audited gate and arithmetic-trace hashes, seed-42 C batch hashes, replay behavior for batches 1–6, and pre-batch-7 model hash. It uses HAM train only. Batch 7 is evaluated first under ordinary AMP, then with only `resnet.nonlocal3`'s `torch.bmm(q, k)` in FP32. A CPU model snapshot and RNG restore keep both paths at the same starting state. The selective path includes backward and GradScaler unscale, but performs **no batch-7 optimizer step** and saves no research checkpoint.

```python
from pathlib import Path
import subprocess, sys, torch

ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
SCRIPT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_selective_qk_fp32_probe.py'
REFERENCE = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_nonlocal3_arithmetic_trace.json'
OUTPUT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_selective_qk_fp32_trace.json'
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
assert SCRIPT.is_file() and REFERENCE.is_file() and not OUTPUT.exists()
subprocess.run([sys.executable, str(SCRIPT), '--project-root', str(ROOT),
                '--output', str(OUTPUT)], check=True)
```

Exact Colab terminal command after Drive is mounted:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/analysis/stage13_melanoma_ablation/stage13c_selective_qk_fp32_probe.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --output /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/analysis/stage13_melanoma_ablation/stage13c_selective_qk_fp32_trace.json
```

Download and provide `stage13c_selective_qk_fp32_trace.json` for audit. A result of `SELECTIVE_BATCH7_PASS` clears only this bounded C batch-7 probe; it is not training clearance. A later bounded 16-batch A/B/C gate can be prepared after this JSON is audited.
