# Stage 4E readiness

## Decision

**READY — implementation and CPU/read-only verification complete. This is not training authorization.** No Stage 4E training, Stage 4D selection inference, or threshold optimization was run. Before training, the intended Colab T4 operator must complete the initialization receipt check and obtain separate explicit authorization. The trained Stage 23 checkpoint is not part of Stage 4E's model-loading path.

## Architecture discrepancy resolution

The frozen Stage 23 checkpoint is:

`experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt`

SHA256: `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`

Read-only inspection of that checkpoint's metadata and tensor keys, together with the frozen Stage 23 configuration, freeze manifest/model registry, initialization report, and source definitions, confirms:

- Architecture identifier: `stage8b_egvan_four_mff_serial_terminal_branches`.
- It is a **dual-branch EG-VAN** classifier, not a ResNet-only model.
- The `efficient.*` tensors represent the EfficientNetV2-S branch, initialized with `EfficientNet_V2_S_Weights.DEFAULT`.
- The `resnet.*` tensors represent the modified ResNet50 branch, initialized with torchvision `ResNet50_Weights.IMAGENET1K_V2`; the initialization report records hash-checked transfer of ResNet backbone tensors.
- The checkpoint contains `fusions.*` (four MFFs), `aggregate.*`, and `classifier.*` weights as well as both backbones. Architecture metadata specifies EfficientNet taps `[2, 3, 5, 7]`, four 128-channel MFF outputs, SCGA after ResNet stages 1 and 2, and non-local blocks after stages 3 and 4.
- The checkpoint is the trained epoch-14 model. Stage 4E never loads it. Stage 4E freshly initializes the same model family from the two ImageNet sources and seeded defaults; its initialization receipt is designed to compare the initial state hashes of the two arms.

This resolves the conflicting characterization: descriptions of Stage 23 as the full dual-branch EG-VAN model are supported by the frozen artifact; any description of it as a standalone ResNet50 is inaccurate.

## Verification completed

- Stage 4D v2 independent read-only audit: `PASS_STAGE4D_V2_INTEGRITY`.
  - 6,477 development-training images / 4,804 lesions.
  - 1,538 internal-selection images / 1,169 lesions, including 163 melanoma images / 95 melanoma lesions.
  - All 40 frozen refined candidate edges contained within development training.
  - 5,189 dHash-only cross-role candidates remain; zero cross-role pairs passed the refined pHash cutoff.
  - No patient IDs or human pair adjudication are available. The boundary must not be described as patient-level independence.
- Stage 4E `train.py --check`: `PREFLIGHT_PASS`; all raw and processed hashes for the Stage 4D v2 manifests were checked.
- Frozen Stage 23 configuration and source hashes, Stage 4D v2 manifest/artifact hashes, and Stage 4E frozen config/source-catalog hashes passed their preflight checks.
- Syntax compilation passed for `common.py`, `train.py`, `evaluate_pair.py`, and `test_integrity.py`.
- Focused CPU-only Stage 4E tests: **11 passed**. They cover split/preflight, paired configuration, treatment-only training blur, common fresh initialization fingerprint behavior, sampler, paired lesion bootstrap and candidate gate, checkpoint resume/RNG restoration, atomic epoch commits and artifact-hash rejection, incomplete-epoch recovery, and exclusion of Stage 23 checkpoint inference.
- The paired evaluator's runner import was made explicit after testing exposed a module-name collision: `import train` could resolve to the unrelated Stage 23 `train.py` because Stage 4E inserts that source directory into `sys.path`. It now loads the sibling Stage 4E runner directly; a test asserts the resolved module path.
- Stage 4E's preflight configuration comparison now checks each frozen seed/epoch/input/batch factor explicitly rather than using chained inequality.

## Scientific and operational controls

- Gaussian blur is the only arm difference: treatment-only `RandomApply(GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)), p=0.25)` after rotation and before tensor conversion. It exists only in the training transform; selection transforms match and contain no blur.
- Control and treatment share the same Stage 4D v2 manifests, config, seed, architecture, class mapping, optimizer, loss, scheduler, sampler, and initialization hash requirement.
- Metrics use the fixed seven-class order `akiec, bcc, bkl, df, mel, nv, vasc`; comparison resamples lesion groups within true class and pairs both arms on the same selection images.
- Within-arm epoch choice is minimum finite common selection loss, earliest epoch on ties. The treatment-candidate decision requires a positive melanoma-recall difference, paired 95% interval lower bound strictly above zero, and no point decrease in macro-F1 or MEL specificity.
- `--check` is read-only. `--check-initialization` is a T4-only model initialization/fingerprint step that writes a receipt, but performs no training or image inference. Training requires an additional, separate user authorization.
- A committed epoch directory appears only after checkpoint, prediction, metrics, history, numerical-event files, and their hash manifest are written. Resume accepts only complete, hash-verified sequential epoch directories; an interrupted pending directory is not a completed epoch.

## Colab T4 setup and commands

Use a Colab **T4 GPU** runtime with the repository and permitted HAM development files already available in Google Drive. Keep `experiments/stage4e_control_vs_blur_exp1/runs/` on Drive so epoch commits and the initialization receipt persist across runtime interruption. Do not upload data to a new service or copy it to an unapproved location.

In a Colab cell:

```python
from google.colab import drive
from pathlib import Path
import os

drive.mount("/content/drive")
PROJECT_ROOT = Path("/content/drive/MyDrive/EG-VAN")
assert (PROJECT_ROOT / "experiments/stage4e_control_vs_blur_exp1/train.py").is_file()
os.chdir(PROJECT_ROOT)
```

Confirm the framework/runtime and required modules before running the project:

```python
import torch, torchvision, numpy, PIL
print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("numpy:", numpy.__version__, "Pillow:", PIL.__version__)
```

Keep Colab's preinstalled CUDA-matched `torch`/`torchvision`; do not replace them with CPU wheels or independently upgrade one without the other. The Stage 4E path requires Python 3.10+, PyTorch with `torch.amp.GradScaler("cuda", ...)`, matching torchvision, NumPy, and Pillow. The independent Stage 4D image-hash audit additionally requires OpenCV (`cv2`). Stop if required imports or CUDA checks fail; do not install a different PyTorch build into the T4 runtime as an ad hoc fix.

Read-only preflight and T4 initialization check:

```bash
python experiments/stage4e_control_vs_blur_exp1/train.py --check
nvidia-smi
python experiments/stage4e_control_vs_blur_exp1/train.py --check-initialization
```

The first command verifies source, split, and image hashes and performs no training or inference. The second command checks the accelerator. The third downloads/uses the official pretrained ImageNet weights as needed, constructs the two fresh models sequentially, compares their state fingerprints, and writes `experiments/stage4e_control_vs_blur_exp1/runs/shared_initialization.json`; it does not train or evaluate images. Keep sufficient Drive space available: budget at least approximately 35 GB for both arms' 25 full epoch checkpoints plus prediction/evidence artifacts, with additional free space for filesystem overhead.

**Only after separate explicit authorization**, start a fresh arm:

```bash
python experiments/stage4e_control_vs_blur_exp1/train.py --train --arm control
python experiments/stage4e_control_vs_blur_exp1/train.py --train --arm treatment
```

For an interrupted, incomplete arm only, resume that same arm and Drive output:

```bash
python experiments/stage4e_control_vs_blur_exp1/train.py --train --arm control --resume
```

Use `--arm treatment --resume` only for an existing incomplete treatment run. Do not use resume on a complete arm or delete/replace run artifacts to restart it. Run `evaluate_pair.py --analyze` only after both arm manifests and all 25 epoch commits verify. This report itself does not authorize any of those training or analysis commands.

## Remaining limitations

- Actual control/treatment fresh-model fingerprints have not yet been recorded on the registered T4; the CPU test covers the deterministic fingerprint mechanism with test fixtures.
- No Stage 4E training, selection predictions, paired result, or candidate decision exists.
- The v2 split is a containment protocol rather than human-adjudicated or patient-linked independence.
- This exploratory selection-set comparison cannot establish external generalization or justify replacement of frozen Stage 23.
