# Colab Tesla T4: Stage 23 PH2 external follow-up

The PH2 study is **external follow-up, not untouched independent validation**. Run only after reviewing the [frozen protocol](STAGE23_EXTERNAL_VALIDATION_PROTOCOL.md) and deciding that the qualified follow-up answers the intended question. No training is included.

In a Google Colab notebook with **Tesla T4 GPU** runtime, mount the same Drive shortcut used for the repository:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
root = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
assert root.is_dir(), root
assert (root / 'analysis/stage23_external_validation/evaluate_stage23_ph2.py').is_file()
assert (root / 'data/external/ph2/metadata/PH2_dataset.txt').is_file()
assert (root / 'experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt').is_file()
print(root)
```

Verify the runtime dependencies and every top-level input before the preflight. The preflight subsequently checks the full frozen source registry and all 200 images. Do not install or switch packages mid-run after a failed check without recording the environment change.

```python
import torch, torchvision, cv2, numpy, PIL, sklearn
print({'torch': torch.__version__, 'torchvision': torchvision.__version__,
       'opencv': cv2.__version__, 'numpy': numpy.__version__,
       'pillow': PIL.__version__, 'sklearn': sklearn.__version__,
       'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
required = [
    'analysis/stage23_external_validation/phase23_ph2_protocol.json',
    'analysis/ph2_external_validation_exp5/ph2_manifest.csv',
    'data/external/ph2/metadata/PH2_dataset.txt',
    'data/external/ph2/images',
    'data/external/ph2/original/PH2 Dataset images',
    'data/splits/split_leakage_aware.csv',
    'models/frozen_stage23/model_registry.json',
    'models/frozen_stage23/load_frozen.py',
    'src/preprocessing.py',
    'src/train.py',
    'analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.py',
]
for relative in required:
    assert (root / relative).exists(), relative
output = root / 'analysis/stage23_external_validation/run'
partials = list(output.parent.glob('.run_inprogress_*'))
assert not output.exists() and not partials, (output, partials)
print('Inputs and empty output destination: PASS')
```

Read-only preflight, which must print `PASS_QUALIFIED_EXTERNAL_FOLLOWUP`:

```python
import subprocess
script = root / 'analysis/stage23_external_validation/evaluate_stage23_ph2.py'
subprocess.run(['python', str(script), '--project-root', str(root), '--preflight'], check=True)
```

**Single frozen follow-up execution command; do not repeat it after an output directory exists:**

```python
subprocess.run(['python', str(script), '--project-root', str(root), '--run',
                '--accept-prior-ph2-use'], check=True)
```

Expected new Drive directory: `analysis/stage23_external_validation/run/` containing `predictions.csv` (120 rows), `metrics.json` (including melanoma recall, binary melanoma specificity and mapped balanced accuracy), `confusion_matrix.csv`, and `manifest.json` with hashes, runtime, cohort and source-image provenance. The runner stages files in a distinct temporary directory before the final rename, refuses overwrite or rerun with a partial directory, hashes the checkpoint both before and after inference, and never reads HAM test outcomes. If preflight fails, **stop**; do not alter the frozen cohort or tolerance to force a pass. If the run fails, preserve its temporary diagnostic directory and investigate before any rerun. Send back the complete `run/` directory for a read-only result audit; do not interpret console numbers alone.
