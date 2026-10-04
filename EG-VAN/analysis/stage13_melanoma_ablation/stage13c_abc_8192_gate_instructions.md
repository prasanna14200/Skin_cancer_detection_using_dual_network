# Stage 13C bounded A/B/C gate at initial AMP scale 8192

The audited A-only T4 diagnostic completed 16 batches at 8192, 4096, and 2048 without a skipped optimizer update or non-finite model/optimizer state. The next bounded check is A/B/C at **8192**, the highest scale that passed that 16-batch window. This is a numerical execution test, not a change to the registered A/B/C scientific factors.

The gate uses seed 42, the frozen HAM **train** partition, 16 batches per variant, batch size 16, the registered A/B/C samplers and focal losses, Adamax, CUDA autocast, and dynamic GradScaler. It records every batch's data hashes, loss, scaled and unscaled gradient status, actual optimizer-step execution or skip, scale before and after update, and model/optimizer state. A recoverable GradScaler skip is recorded distinctly from corruption, but does **not** clear the gate. The gate stops before the next variant if one variant has a numerical failure, a skip, or a model/data provenance mismatch. It saves one JSON trace and no research checkpoint. It never constructs a validation, HAM test, or PH2 loader.

On the existing mounted Google Drive repository with a Tesla T4 runtime, run this **one Colab cell**:

```python
from pathlib import Path
import subprocess, sys, torch

ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
SCRIPT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_abc_8192_gate.py'
TRACE = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_multistep_amp_trace.json'
OUTPUT = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_abc_8192_gate.json'
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
assert SCRIPT.is_file() and TRACE.is_file() and not OUTPUT.exists()
subprocess.run([sys.executable, str(SCRIPT), '--project-root', str(ROOT),
                '--output', str(OUTPUT)], check=True)
```

Exact shell command, if running in a Colab terminal after Drive is mounted:

```bash
python /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/analysis/stage13_melanoma_ablation/stage13c_abc_8192_gate.py --project-root /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN --output /content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN/analysis/stage13_melanoma_ablation/stage13c_abc_8192_gate.json
```

Download the resulting `stage13c_abc_8192_gate.json` for independent audit. **Do not start 25-epoch training on the basis of the command's exit status alone.** Its per-batch trace and source hashes must be reviewed first. Even a passing 16-batch A/B/C gate is bounded evidence, not proof of whole-run stability.
