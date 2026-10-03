# Stage 13B CUDA AMP backward diagnostic

## Evidence available now

The user reports that the bounded seed-42 T4 A/B trace found identical initial model, first input batch, and labels. Both A and B have finite recorded quantities through the scaled loss, then the first non-finite tensor is `gradient_before_optimizer_step/resnet.conv1.weight.grad` in batch 1 with initial GradScaler scale 65,536. A has 216 NaNs and 685 Infs; B has 295 NaNs and 840 Infs in that tensor. Finite values approach approximately ±65,000. The diagnostic stopped **before `optimizer.step()`**. These figures are recorded as **user-provided trace findings**: `stage13b_t4_first_epoch_trace.json` is not present at its stated local path, so its full JSON and hashes have not been independently read here.

The observation rules out input/label invalidity, the recorded forward/logit and focal-loss stages, and the optimizer **step** as the first detected source. Because A fails on the identical first batch with the unmodified Stage 9 loss, B's MEL multiplier alone cannot cause the initial non-finite gradient. The higher B NaN/Inf counts show greater gradient contamination in that run, but counts alone do not establish the first overflow mechanism. The pattern is **consistent with CUDA FP16 AMP scaling/backward overflow**, especially with finite gradient values near FP16's representable limit. It remains unproven until the same initial batch is checked at lower scales and in FP32. A non-finite *scaled* gradient is not necessarily a non-finite unscaled mathematical gradient.

## Diagnostic prepared; no run performed here

`diagnose_amp_scaling_colab.py` uses the frozen HAM **train** partition only. It seeds 42, builds the exact registered A/B architecture and first batch, and **requires the prior bounded trace** to verify that its initial model and first input/label hashes match both A and B before any backward path runs. It snapshots the initial model and RNG states, then restores them for every path. The A/B configuration comparison rejects any difference outside the registered true-MEL loss multiplier. It performs no optimizer step, saves no checkpoint, and runs no full epoch or validation/test/external inference.

For A, it checks AMP GradScaler initial scales **65,536, 32,768, 16,384, 8,192, 4,096** and full FP32 with autocast disabled. Each AMP path records the same backward gradients **before and after `scaler.unscale_(optimizer)`**. It then checks B at 65,536, at the highest A-tested finite scale if one exists, and in full FP32. For every path it records the first non-finite operation, forward/loss stage finiteness, gradient finiteness before/after unscale, first affected parameter, NaN/Inf counts, finite min/max and maximum absolute gradient, AMP scale, and the common input/label hashes. The script catches CUDA out-of-memory in a path and reports it rather than treating it as a numerical result. Full FP32 at physical batch 16 may be memory limited on a T4; an OOM result does not imply FP32 numerical failure.

The intended interpretation is prespecified for this diagnostic:

| Observation | Interpretation |
|---|---|
| A FP32 finite; A AMP 65,536 non-finite; a lower AMP scale finite | Strong evidence that loss scaling/FP16 backward range is causal at the original scale. Report the finite scales without changing the registered experiment. |
| A FP32 finite; all tested AMP scales non-finite | FP16 autocast/backward instability remains possible; scaling alone is not shown to fix it. |
| A FP32 non-finite | Failure is not confined to AMP scaling; localize the first FP32 stage. |
| A FP32 OOM | FP32 comparison is unresolved at the exact batch size; do not infer finiteness. |
| B still fails at an A-finite scale | B multiplier may narrow the stable scale range, but the shared A failure still predates a B-only explanation. |

## Exact Colab cell

Run only after confirming the T4 and that the output path is unused. This performs **backward-only diagnostics on one batch**, no optimizer update:

```python
from pathlib import Path
import subprocess, sys, torch
ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
SCRIPT = ROOT / 'analysis/stage13_melanoma_ablation/diagnose_amp_scaling_colab.py'
OUTPUT = ROOT / 'analysis/stage13_melanoma_ablation/stage13b_amp_t4_run.json'
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
REFERENCE = ROOT / 'analysis/stage13_melanoma_ablation/stage13b_t4_first_epoch_trace.json'
assert SCRIPT.is_file() and REFERENCE.is_file() and not OUTPUT.exists()
subprocess.run([sys.executable, str(SCRIPT), '--project-root', str(ROOT),
                '--output', str(OUTPUT)], check=True)
```

The local machine has no CUDA T4. The diagnostic script passed syntax checks but has **not** produced `stage13b_amp_t4_run.json`. `stage13b_amp_diagnostic.json` records the reported first trace and all pending comparisons. Do not alter A/B/C research configurations based on this diagnostic. If scale stabilization is established, the smallest scientifically valid next experiment is a **new, separately named and preregistered** run changing only the AMP initial scale; do not relabel it as the original B or execute it as part of this investigation.

No full training, optimizer update, HAM test access, PH2 access, or Stage 14 work occurred here.
