# Stage 13C numerical stability protocol — audit and bounded gate

## Independently checked T4 evidence

Both source JSON files are now present. The first trace SHA256 is `cd84f34d9a832ee7ef7698673690e0704bc3ef232e4715c2b6da83de8735a899`; the AMP trace SHA256 is `ab79c7c8b4efdccca34d53b95e6e8905941bf6a9a5cc821bd2adc6227b9dba09`. The AMP trace embeds the exact first-trace hash. Its initial model hash (`3151c00f259baa031ac123944b5f9688a4dfe68957e11d350912edc496577973`), first input hash (`9444dd6b46ce85903643cfaae88de0e713dcfee2988b238d526cc700a49be4e3`), and label hash (`a4c62cc9f6d0419b5f12d981163fcf48f9c6fdccf17f7ff6266aa82f623e2632`) match A, B, and every AMP diagnostic path. The saved A/B configurations exactly match `config_A.json` and `config_B.json`; their only active training-path difference is B's registered true-MEL loss multiplier. Programmatic checks of all nine paths passed; details are in `stage13c_trace_validation.json`.

| First-batch backward path | Gradient before unscale | Gradient after unscale | First bad parameter |
|---|---|---|---|
| A AMP scale 65,536 | 216 NaNs, 685 Infs | Same 216 NaNs, 685 Infs | `resnet.conv1.weight.grad` |
| A AMP scale 32,768 | 0 NaNs, 178 Infs | Same 178 Infs | `resnet.conv1.weight.grad` |
| A AMP scale 16,384 | Finite | Finite | None |
| A AMP scale 8,192 | Finite | Finite | None |
| A AMP scale 4,096 | Finite | Finite | None |
| A full FP32 | Finite | Unscale inapplicable | None |
| B AMP scale 65,536 | 295 NaNs, 840 Infs | Same 295 NaNs, 840 Infs | `resnet.conv1.weight.grad` |
| B AMP scale 16,384 | Finite | Finite | None |
| B full FP32 | Finite | Unscale inapplicable | None |

All recorded inputs, labels, logits, softmax/log-softmax, cross entropy, true-class probability, focal terms, final per-example and mean losses, and scaled losses were finite. At 65,536, A's mean loss was 0.3791278600692749 and B's was 0.3999216556549072; the scaled losses were finite at 24,846.5234375 and 26,209.265625 respectively. The first **recorded** non-finite value occurs during scaled backward before unscale, not in an optimizer update. Unscaling cannot restore gradients that are already Inf/NaN. No diagnostic path performed an optimizer step. The lowest *numerical* scale tested and finite for A is **4,096**; the highest tested passing scale for both A and B is **16,384**. The threshold between 16,384 and 32,768 was not finely searched.

**Supported conclusion:** the initial first-batch AMP scale of 65,536 causes a scale-dependent FP16 backward overflow on the tested T4 path. A itself fails at that scale and at 32,768, while A/B FP32 and A/B AMP 16,384 backward paths are finite. B's multiplier is not a sufficient explanation for the initial failure. **Unresolved mechanism:** the exact internal backward operation that first overflows was not traced; `resnet.conv1.weight.grad` is the first *parameter gradient scanned*, not necessarily the originating layer or operation. One finite backward batch does not establish that 25 epochs at scale 16,384 will be stable. The original B checkpoint's later model/optimizer NaNs are established, but the exact step that first corrupted them is still unknown.

## Stage 9 versus failed Stage 13

Stage 9 `train_colab.py` and original Stage 13 `train_ablation.py` both create `torch.amp.GradScaler("cuda")` with the default initial scale. The inspected local PyTorch 2.11 signature gives that default as **65,536**, with growth factor 2, backoff factor 0.5, and growth interval 2,000. Both use CUDA autocast, scaled backward, `scaler.step(optimizer)`, `scaler.update()`, and `zero_grad(set_to_none=True)`. Stage 9 zeroes after a step; Stage 13 zeroes before the next batch. With accumulation 1 this is equivalent for ordinary batches. Both use the same verified Stage 8B architecture, seed 42, 384-pixel images, physical/effective batch 16, Adamax/LR/weight decay, scheduler, focal alpha/gamma, sampler for A, and initialization policy. The A last-checkpoint model tensors, scaler state, train/validation loss history, and CPU RNG state are bitwise identical to Stage 9's last checkpoint; the serialized checkpoint files themselves have different hashes/metadata.

PyTorch documents that `GradScaler.step()` unscales gradients and **skips `optimizer.step()` if Inf/NaN gradients are detected**, then `update()` adjusts the scale ([official AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)). Stage 9's selected epoch-15 checkpoint has scale **32,768**, below the inspected default of 65,536. Under that default, at least one backoff occurred; its epoch-25 scale is 131,072, showing later dynamic growth. All saved Stage 9 epoch losses are finite. This supports the possibility that Stage 9 tolerated and skipped overflowing early steps. It does **not** identify which batch skipped, because no per-step scale/skip log exists. Stage 13's original runner also calls `scaler.step/update`, so the failed B run cannot be explained by an omitted automatic skip call. The B checkpoint has model/Adamax NaNs and scale zero; the first state-corrupting step or forward buffer update remains unlocalized. Stage 9's actual PyTorch/CUDA runtime was not recorded in its manifest; failed B's manifest reports PyTorch 2.11.0+cu130 on Tesla T4, so a runtime difference cannot be excluded or asserted as causal.

The original Stage 13 protocol used default dynamic scale 65,536, no explicit gradient finiteness check before `scaler.step`, and no model/optimizer finiteness check before checkpoint replacement. It let epoch-1 NaNs persist through 25 recorded epochs in B. The existing failed artifacts remain invalid and frozen as evidence.

## Proposed corrected numerical protocol

`experiments/egvan_melanoma_ablation_exp/stage13c_guarded.py` is a **new, separately named** numerical protocol. It starts CUDA GradScaler at **16,384**, the highest tested finite scale for both A and B and the smallest reduction in the prespecified halving series that passed A's first batch. Normal dynamic GradScaler growth/backoff remains enabled. This is a change in numerical execution policy, not a retrospective repair of the failed runs. The scientific A/B/C files and definitions are imported unchanged: A has Stage 9 loss/sampler; B changes only the true-MEL loss multiplier; C changes only the MEL sampler weight. Alpha 0.25, gamma 2, B multiplier 1.5630495442733532, A/B sampler weight 1.5630495442733532, C sampler weight 2.4431238778531372, architecture, initialization, split, seed, Adamax LR 0.001/weight decay 0.0001, ReduceLROnPlateau, 25 epochs, batch 16, checkpoint eligibility gates, and candidate-selection rule are unchanged. No clipping is added.

The new runner checks finite inputs, labels, logits, each per-example focal loss, reduced loss, and scaled loss. After `scaler.unscale_(optimizer)`, it checks **every parameter gradient** and aborts with variant, epoch, batch, tensor, finite min/max, NaN/Inf counts, target-class distribution, MEL presence, and AMP scale if any are non-finite. It then checks model tensors, optimizer tensors, and positive finite GradScaler scale after each optimizer step. Validation inputs/logits/per-example and reduced losses are checked. Train/validation epoch losses are checked before scheduler, history, or checkpoint writes. A checkpoint is written only after a second model/optimizer/scaler/history guard. Research output goes to a new `stage13c_guarded_runs/` directory; the old A/B/C folders are untouched. The runner has no resume mode, so an interruption cannot silently restore a corrupted or incomplete state.

## Mandatory bounded T4 stability gate

The same new file exposes `--bounded-gate`, which performs **eight prespecified real HAM train batches** for each of A, B, and C with optimizer steps enabled at initial scale 16,384 and the complete guards. It saves a gate JSON only, no research checkpoint. It stops and records the first failure. A later full run is programmatically refused unless the gate JSON says all three passed and its runner/config/split hashes match; **no full run was started here**. The gate is necessary but not sufficient for whole-run stability; later batches remain subject to the same fail-fast checks. C's first-batch AMP behavior has not been tested by the existing A/B scaling trace.

Exact Colab cell, after mounting the existing Drive repository on a Tesla T4:

```python
from pathlib import Path
import subprocess, sys, torch
ROOT = Path('/content/drive/.shortcut-targets-by-id/1eBbnRyG1adUFX4ajjJJERTGKKaHZ2Dxo/EG-VAN')
RUNNER = ROOT / 'experiments/egvan_melanoma_ablation_exp/stage13c_guarded.py'
GATE = ROOT / 'analysis/stage13_melanoma_ablation/stage13c_bounded_t4_gate.json'
assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
assert RUNNER.is_file() and not GATE.exists()
subprocess.run([sys.executable, str(RUNNER), '--project-root', str(ROOT),
                '--bounded-gate', '--output', str(GATE)], check=True)
```

Only CPU-safe/static/synthetic checks were run locally: five new numerical-guard tests passed, covering frozen factors, focal-loss equivalence, non-finite loss/scale rejection, model/optimizer corruption detection, and checkpoint write refusal. All ten Stage 13 tests passed. The gate has **not** run locally or on Colab as part of this work. Full training remains disallowed until its result is audited.
