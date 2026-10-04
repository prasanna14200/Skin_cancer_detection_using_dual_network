# Prepared Stage 13C full A/B/C runner — do not execute yet

The audited 16-batch selective q@k FP32 gate gives bounded numerical clearance only. The separate full runner was **not run by this audit**. A future run requires a deliberate training decision. The runner refuses to start if the gate JSON, source/config hashes, frozen split, Stage 9 control checkpoint, runtime, or output location differs. It requires a Tesla T4. Fresh training writes to new `experiments/egvan_melanoma_ablation_exp/stage13c_selective_qk_fp32_runs/{A,B,C}` directories and refuses to overwrite any existing variant directory. Explicit `--resume --train` requires a validated existing `last_checkpoint.pt` and continues that variant; see `stage13c_variant_a_resume_audit.md` for the interrupted A run.

The only numerical policy is the same in-memory `resnet.nonlocal3` q@k FP32 matmul used by the bounded gate; all other EG-VAN operations remain under normal CUDA AMP. Dynamic GradScaler starts at 8192. Recoverable gradient overflow is skipped and logged; non-finite forward, loss, model parameters/buffers, optimizer state, unexplained skip, invalid step, or invalid scaler fails fast with `failure.json`. Scientific A/B/C configurations, 25 epochs, Adamax, scheduler, focal loss, sampler, Stage 9 validation eligibility, minimum eligible validation loss, and validation-only candidate-selection rule remain unchanged. No HAM test or PH2 loader is used. The runner does **not** perform final A/B/C candidate selection; that must be audited after all runs.

When full training is explicitly authorized later, mount the same Google Drive shortcut and select Tesla T4. This exact Colab cell runs A, then B, then C; it stops if any run fails:

```python
from pathlib import Path
import subprocess, sys, torch

ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
SCRIPT = ROOT / 'experiments/egvan_melanoma_ablation_exp/stage13c_selective_qk_fp32_train.py'
GATE = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_abc_selective_qk_fp32_gate.json'
RUNS = ROOT / 'experiments/egvan_melanoma_ablation_exp/stage13c_selective_qk_fp32_runs'
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
assert SCRIPT.is_file() and GATE.is_file()
assert all(not (RUNS / variant).exists() for variant in 'ABC')
for variant in 'ABC':
    subprocess.run([sys.executable, str(SCRIPT), '--project-root', str(ROOT),
                    '--variant', variant, '--train'], check=True)
```

Expected per-variant research outputs after a *future* successful run: `config.json`, `numerical_protocol.json`, `last_checkpoint.pt`, `training_history.csv`, and `experiment_manifest.json`; `best_checkpoint.pt`, `validation_predictions.csv`, and `validation_metrics.json` only if an eligible epoch exists; `numerical_events.json` if the scaler skips at least one step. On failure, `failure.json` records the stop. Do not make any HAM test, PH2, or candidate-selection decision until the full runs and validation artifacts have been independently audited.
