# Stage 13B Colab batch diagnostic

The local artifact copy is incomplete: it currently contains `A/last_checkpoint.pt`, `B/last_checkpoint.pt`, and `C/last_checkpoint.pt` only. The downloaded B checkpoint is epoch 18 from the original failed run, as shown by its embedded history and bitwise state comparison with the original epoch-25 checkpoint. It is not the fresh B run or the bounded diagnostic trace. Obtain the complete A/C/fresh-B folders and the already generated `stage13b_t4_first_epoch_trace.json` separately. The first failed B download at `colab_artifacts/stage13_failed_B/` is preserved. Do not use loose Drive checkpoints of unknown provenance.

## Preserve the fresh failed B run

Mount the same Drive repository used for Stage 13. Inspect the `B/` folder before moving it. This cell refuses to overwrite an existing archive and verifies the checkpoint hash after the move. It does not train:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import hashlib, shutil

ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
FRESH_B = ROOT / 'experiments/egvan_melanoma_ablation_exp/B'
FRESH_ARCHIVE = ROOT / 'experiments/egvan_melanoma_ablation_exp/B_failed_fresh_epoch1_20261003'
assert FRESH_B.is_dir() and not FRESH_ARCHIVE.exists()
assert (FRESH_B / 'training_history.csv').is_file() and (FRESH_B / 'last_checkpoint.pt').is_file()

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

before = sha256(FRESH_B / 'last_checkpoint.pt')
shutil.move(str(FRESH_B), str(FRESH_ARCHIVE))
assert FRESH_ARCHIVE.is_dir() and not FRESH_B.exists()
assert sha256(FRESH_ARCHIVE / 'last_checkpoint.pt') == before
print('Archived fresh failed B:', FRESH_ARCHIVE, 'checkpoint SHA256:', before)
```

If `B/` is absent, stop and locate the actual fresh-run folder using its config, manifest, history, and checkpoint metadata. Do not substitute a loose checkpoint based on its filename.

## Run the bounded diagnostic on the T4

The script recreates the registered seed-42 A/B train pipeline, starts B first, and stops at the first non-finite quantity or after **one train epoch at most**. It then compares A through the same batch. It writes only a diagnostic JSON, no model checkpoint. It never constructs a HAM test or PH2 loader.

```python
import subprocess, sys, torch
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
DIAGNOSTIC = ROOT / 'analysis/stage13_melanoma_ablation/diagnose_variant_b_colab.py'
TRACE = ROOT / 'analysis/stage13_melanoma_ablation/stage13b_t4_first_epoch_trace.json'
assert DIAGNOSTIC.is_file() and not TRACE.exists()
subprocess.run([sys.executable, str(DIAGNOSTIC), '--project-root', str(ROOT),
                '--max-batches', '502', '--output', str(TRACE)], check=True)
```

Inspect the JSON's `B.first_nonfinite`, `A.first_nonfinite`, and `comparison` fields. Share the trace and the complete A/C/fresh-B archives for the final root-cause and artifact audit. Do not start another 25-epoch B run or change the registered configuration in response to a diagnostic result.
