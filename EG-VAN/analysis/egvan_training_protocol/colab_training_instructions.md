# Colab Tesla T4 handoff

Use a Colab notebook with a Tesla T4 runtime. Mount Drive and locate the existing repository; the path below is the Experiment #5 Drive location recorded in its config. If Drive exposes it elsewhere, set `ROOT` to that existing EG-VAN directory.

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
assert (ROOT / 'experiments/egvan_reconstruction_controlled_exp1/train_colab.py').is_file()
print(ROOT)
```

Verify the two protected hashes and CUDA/T4:

```python
import hashlib, torch
def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
assert sha256(ROOT / 'experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt') == '81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a'
assert sha256(ROOT / 'data/splits/split_leakage_aware.csv') == 'f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883'
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
```

Run preflight, then the memory probe. These commands perform no full training:

```python
import subprocess, sys, json
RUNNER = ROOT / 'experiments/egvan_reconstruction_controlled_exp1/train_colab.py'
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT), '--preflight'], check=True)
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT), '--memory-probe'], check=True)
PROBE = json.loads((ROOT / 'experiments/egvan_reconstruction_controlled_exp1/memory_probe.json').read_text())
print('Recommended physical batch:', PROBE['recommended_physical_batch_size'])
print('Accumulation:', PROBE['gradient_accumulation_steps'])
print('Effective batch:', PROBE['effective_batch_size'])
assert PROBE['recommended_physical_batch_size'] is not None
assert PROBE['exact_batch_equivalence']
```

Review the measured result and obtain explicit approval before the full run. The following is the later full training command; do not execute it during this Stage 9 handoff:

```python
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT),
    '--batch-size', str(PROBE['recommended_physical_batch_size']),
    '--gradient-accumulation', str(PROBE['gradient_accumulation_steps']),
    '--epochs', '25', '--num-workers', '2', '--train'], check=True)
```

After an interruption, use the same command with `--resume`. The runner refuses a different configuration and restores model, optimizer, scheduler, scaler, sampler and RNG states:

```python
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT),
    '--batch-size', str(PROBE['recommended_physical_batch_size']),
    '--gradient-accumulation', str(PROBE['gradient_accumulation_steps']),
    '--epochs', '25', '--num-workers', '2', '--resume', '--train'], check=True)
```

Verify generated artifacts after the approved run. If no epoch meets all three frozen validation gates, the manifest records `FAIL_NO_ELIGIBLE_CHECKPOINT` and there is no best checkpoint or selected validation CSV. Do not infer success from `last_checkpoint.pt` alone.

```python
OUT = ROOT / 'experiments/egvan_reconstruction_controlled_exp1'
manifest = json.loads((OUT / 'experiment_manifest.json').read_text())
for name in ('config.json', 'training_history.csv', 'last_checkpoint.pt', 'experiment_manifest.json'):
    assert (OUT / name).is_file(), name
if manifest['status'] == 'TRAINED_VALIDATION_ONLY':
    for name in ('best_checkpoint.pt', 'validation_metrics.csv', 'validation_predictions.csv'):
        assert (OUT / name).is_file(), name
print(manifest)
assert sha256(ROOT / 'experiments/efficientnetv2s_controlled_exp5/best_checkpoint.pt') == '81d567f533d08286d5e599b90512d96e92c413ad74c7f4f941d81fc7bb46de3a'
assert sha256(ROOT / 'data/splits/split_leakage_aware.csv') == 'f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883'
```

Keep HAM test and PH² evaluation for the later evaluation stage.
