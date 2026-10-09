# Frozen EG-VAN+ classifier: Stage 23 epoch 14

`EGVAN_STAGE23_FINAL` is the **final validation-selected research classifier** for the melanoma-sensitive objective. It references the original Stage 23 checkpoint at `experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt`, SHA256 `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`. No checkpoint was copied or modified. The [registry](model_registry.json) and [freeze manifest](freeze_manifest.json) pin its provenance and validation evidence.

## Why Stage 23 was retained

Stage 23 epoch 14 passed all frozen validation gates and had the lowest common validation focal loss among eligible epochs 14, 15, and 19. Stage 24 had no eligible epoch. Stage 25 completed 25 epochs and selected epoch 12 within its own run, but its registered comparison with Stage 23 is `MIXED_VALIDATION_RESULT`: accuracy and macro F1 were modestly higher while MEL recall fell from 70/107 to 63/107 and MEL F1 fell from 0.633484 to 0.614634. Retaining Stage 23 is a **validation-based research decision** for melanoma sensitivity, not evidence that it performs better on an untouched test set.

## Verify without changing the freeze

From the repository root:

```bash
python models/frozen_stage23/verify_freeze.py
```

This hashes the original checkpoint, registry, frozen source files, selection evidence, and experiment manifests. The optional CPU-only synthetic smoke test also strict-loads the checkpoint and checks the deterministic transform and seven-class output:

```bash
python models/frozen_stage23/verify_freeze.py --smoke
```

Both commands are read-only. They never download pretrained weights, train, open HAM test images, or open PH2 images. The registry and manifest are also marked read-only on the local filesystem; their hashes provide the substantive integrity check. The freeze process is idempotent: verification does not rewrite model weights, registry, manifest, or experiment artifacts.

## Loading and inference contract

```python
import sys
sys.path.insert(0, "models/frozen_stage23")
from load_frozen import load_classifier, predict_rgb_image

model, transform, class_order = load_classifier("cpu")
# Given a PIL image that is authorized for inference:
result = predict_rgb_image(model, transform, image, class_order)
```

The loader verifies all pinned hashes, constructs `EGVAN(num_classes=7, pretrained_efficient=False, pretrained_resnet=False)` **without downloading weights**, strict-loads the frozen state dictionary, installs the Stage 23 selective FP32 `resnet.nonlocal3` q@k policy, and calls `eval()`. `predict_rgb_image` uses `torch.inference_mode()`; on CUDA it uses FP16 autocast around the model while q@k remains FP32. CPU inference is full FP32. Output is seven raw logits converted to softmax probabilities and an argmax label in this fixed order:

`akiec, bcc, bkl, df, mel, nv, vasc` (indices 0–6).

Mandatory validation/evaluation preprocessing is: RGB conversion; resize to **384×384**; `ToTensor()`; normalize channels with mean **[0.485, 0.456, 0.406]** and standard deviation **[0.229, 0.224, 0.225]** from `EfficientNet_V2_S_Weights.DEFAULT.transforms()`. The evaluation pipeline has **no random augmentation**. Do not add quality filtering, test-time augmentation, thresholds, calibration, or label remapping to this frozen classifier. The prior Stage 20 entropy-review threshold belongs to the older Stage 15 model's exploratory reliability study; it is **not** bundled with this Stage 23 classifier freeze.

Changing the checkpoint bytes, class order, model implementation, preprocessing, numerical policy, output semantics, or pinned source files invalidates this freeze. Changing the registry or manifest to point to another checkpoint is a new decision and must not be represented as this model.

## Evaluation boundary

Stage 23's selected **validation** values are accuracy 0.829615, macro F1 0.671254, MEL recall 70/107, MEL F1 0.633484, and common validation focal loss 0.0741077664 on 986 validation cases. They are not held-out test results. The repository's Stage 16 HAM test and PH2 external follow-up results belong to the earlier Stage 15 recovered checkpoint; they must not be attributed to Stage 23. Stage 16/20 results were already viewed before later research planning, and PH2 had prior project use. Neither is an untouched confirmation set for the newly frozen Stage 23 classifier. See the [evaluation-boundary audit](../../analysis/final_model_freeze/evaluation_boundary_audit.md).
