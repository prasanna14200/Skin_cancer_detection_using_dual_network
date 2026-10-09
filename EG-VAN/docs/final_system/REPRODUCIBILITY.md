# Reproducing local frozen-model inference

The exact model is `EGVAN_STAGE23_FINAL`, original Stage 23 checkpoint at `experiments/stage23_resnet50_imagenet_init_exp1/run/best_checkpoint.pt`, selected epoch 14, SHA256 `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`. The [registry](../../models/frozen_stage23/model_registry.json) and [freeze manifest](../../models/frozen_stage23/freeze_manifest.json) pin source and decision hashes. No copied checkpoint is used.

From repository root:

```powershell
python models/frozen_stage23/verify_freeze.py --smoke
python -m pytest tests/test_final_local_app.py -q
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

The loader hashes pinned artifacts **before** deserializing the model and strict-loads all weights with no pretrained download. The transform is the hash-pinned `src/train.py:make_transforms` evaluation path. Inference runs in evaluation mode and under `torch.inference_mode()`; optional Grad-CAM temporarily enables gradients, removes its hook, and zeroes gradients afterward. On CPU the model runs FP32; on CUDA its surrounding forward uses FP16 autocast while `resnet.nonlocal3` q@k uses FP32. Results can vary slightly across CPU and CUDA arithmetic.

The integration test uses only a synthetic uniform image and a tiny synthetic Grad-CAM model. It checks checkpoint hash immutability, preprocessing shape/normalization, seven probabilities, entropy, quality proxies and Grad-CAM map shape/finiteness. It is an execution test, not an accuracy evaluation. Run the app only on images supplied with permission; no data are transmitted to external APIs and no image is saved by the integration path.
