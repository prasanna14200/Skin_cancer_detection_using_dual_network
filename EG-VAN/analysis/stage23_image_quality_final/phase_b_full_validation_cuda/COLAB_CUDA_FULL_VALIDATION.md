# Stage 23 Phase B: original-policy CUDA full-validation workflow

The research checkpoint, Stage 23 saved predictions, split, and seven Phase B conditions remain frozen. This workflow writes to **`analysis/stage23_image_quality_final/phase_b_full_validation_cuda/` only**. The 473 CPU baseline records and six one-case CPU degradation records in the adjacent `phase_b_full_validation/` directory are separate diagnostic evidence and are never imported.

The runner uses the original validation split order, 16 images per forward (last batch contains 10), `model.eval()`, CUDA FP16 autocast, selective FP32 `resnet.nonlocal3` q@k from the frozen loader, and softmax of FP32-cast logits. For baseline, it verifies each saved processed JPEG against the SHA256 recorded in the prior 986-image quality audit, decodes that exact file with PIL as the original Stage 23 validation loader did, and asserts exact RGB pixel identity of the input. It does not regenerate or re-encode the baseline. The six frozen degradations still start from raw RGB and run the frozen preprocessing. Each condition retains identical batch boundaries and image order. A batch is committed as a directory containing `records.csv` and hash-checked `metadata.json`; the directory rename is atomic on the same filesystem. Resume validates every existing committed batch and skips it. It never overwrites a committed batch. A leftover hidden `.batch_*` staging directory has no research status and is ignored; leave it in place until separately inspected. The final 6,902-row CSV is written only after all 434 batches (62 × 7) validate.

## Colab setup and checks

Ensure the updated `phase_b_full_validation_cuda/` scripts are present in the Google Drive repository. Select a **Tesla T4** runtime. In a Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import os, torch, hashlib, json
project = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
assert project.is_dir(), project
assert torch.cuda.is_available() and torch.cuda.get_device_name() == 'Tesla T4'
assert torch.__version__ == '2.11.0+cu130', torch.__version__
checkpoint = project / 'experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt'
split = project / 'data/splits/split_leakage_aware.csv'
reference = project / 'experiments/stage23_resnet50_imagenet_init_exp1/run/validation_predictions.csv'
protocol = project / 'analysis/stage23_image_quality_final/phase_b_full_validation/phase_b_full_validation_protocol.json'
for path in (checkpoint, split, reference, protocol):
    assert path.is_file(), path
digest = hashlib.sha256()
with checkpoint.open('rb') as stream:
    for block in iter(lambda: stream.read(8*1024*1024), b''):
        digest.update(block)
assert digest.hexdigest() == '42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab'
os.chdir(project)
print('READY', project, torch.__version__, torch.cuda.get_device_name())
```

Run the read-only preflight (also checks any existing **CUDA** batch commits):

```bash
!python analysis/stage23_image_quality_final/phase_b_full_validation_cuda/run_cuda_full_validation.py --check
```

## Baseline first

For an empty CUDA output directory, run **only** the 986-image baseline:

```bash
!python analysis/stage23_image_quality_final/phase_b_full_validation_cuda/run_cuda_full_validation.py --run --baseline-only
```

If Colab disconnects during the baseline, check commits and resume without redoing valid batches:

```bash
!python analysis/stage23_image_quality_final/phase_b_full_validation_cuda/run_cuda_full_validation.py --check
!python analysis/stage23_image_quality_final/phase_b_full_validation_cuda/run_cuda_full_validation.py --run --resume --baseline-only
```

Proceed only when the console prints `BASELINE_GATE_PASS` and `baseline_gate.json` records **986 images, zero class disagreements, and maximum seven-class probability delta ≤0.005**. A failure stops before any degradation. Do not change tolerance or delete valid batches to force passage.

## Six frozen conditions and analysis

After baseline passes, run all six conditions. This command also resumes safely after a disconnect:

```bash
!python analysis/stage23_image_quality_final/phase_b_full_validation_cuda/run_cuda_full_validation.py --run --resume
```

After it prints `COMPLETE 6902 rows`, generate statistics and figures from the saved CSV only:

```bash
!python analysis/stage23_image_quality_final/phase_b_full_validation_cuda/analyze_cuda_full_validation.py
```

The persistent output directory is `analysis/stage23_image_quality_final/phase_b_full_validation_cuda/` under the mounted Drive project. It contains `batches/`, `baseline_gate.json`, `cuda_full_validation_predictions.csv`, condition summaries, lesion-cluster bootstrap results, figures, `FINAL_CUDA_PHASE_B_REPORT.md`, and `cuda_execution_manifest.json`. The analysis script rejects incomplete or altered batch sets. No training, HAM test, or PH2 access occurs.

The original Stage 23 test was trained/evaluated on a Tesla T4 with PyTorch `2.11.0+cu130`. If this exact runtime is unavailable, stop and report it. Do not alter the runtime guard, batch size, or scientific parameters within this execution.
