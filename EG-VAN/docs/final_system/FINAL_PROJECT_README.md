# EG-VAN+ local research viewer

This is a locally runnable interface for the **frozen, validation-selected Stage 23 epoch-14 EG-VAN classifier**. It is a research demonstrator, not a medical device or diagnostic tool. It accepts one user-supplied PNG/JPEG image, shows seven softmax scores, top score, predictive entropy, descriptive technical image-quality measurements, and optional predicted-class Grad-CAM. It makes no accept/review or image-rejection decision.

## Architecture and pipeline

The EG-VAN model combines an EfficientNet-V2-S branch, a modified ImageNet-V2 ResNet50 branch, paired multi-scale feature fusion, attention modules, and a seven-output classifier. The frozen numerical policy executes `resnet.nonlocal3` q@k affinity in FP32 under CUDA FP16 autocast. The local app uses the existing strict frozen loader and exact evaluation transform: RGB, resize to 384×384, tensor conversion, ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. Class order: `akiec, bcc, bkl, df, mel, nv, vasc`.

```text
user image → RGB/384×384/normalization → frozen Stage 23 EG-VAN
           → seven softmax scores → argmax + entropy
           → descriptive quality measurements (resized view)
           → optional predicted-class Grad-CAM overlay
```

The original upload is displayed unchanged. Quality indicators are measured on a resized RGB view using existing `src/image_quality.py` proxies: brightness, contrast, Laplacian variance, saturation, dark/bright pixel fractions, histogram entropy, and spatial illumination variation. These have no validated clinical quality cutoff. Grad-CAM uses `model.resnet.nonlocal3`; it is a qualitative model-attention visualization and is not lesion-localization ground truth.

## Local setup and launch

Use a Python environment with PyTorch, torchvision, Pillow, NumPy, pandas, OpenCV, pytest and Streamlit. The frozen checkpoint is referenced in the original experiment folder; no weights are downloaded or copied. On this machine, Python 3.10.11 with CPU PyTorch 2.11.0, torchvision 0.16.0, OpenCV 5.0.0 and Streamlit were available for local checks. A matching PyTorch/torchvision pair is preferable when setting up a fresh environment.

From `D:\Cancerdetection\EG-VAN` in PowerShell:

```powershell
python models/frozen_stage23/verify_freeze.py --smoke
python -m pytest tests/test_final_local_app.py -q
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

The app verifies the checkpoint and pinned sources before loading. A failed hash check blocks inference. The optional Grad-CAM pass is slower and uses gradients only to generate a heatmap; it does not train or update weights.

## Interpretation

Stage 23 was selected from HAM **validation**, not an untouched independent cohort. Its validation accuracy was 0.829615, MEL recall 70/107, and macro F1 0.671254. The older Stage 15 checkpoint's HAM test, PH2, calibration, and selective-review outcomes do not measure this Stage 23 model. The Stage 20 entropy threshold is deliberately not applied here. Softmax scores are not calibrated clinical probabilities. See [results](EXPERIMENT_RESULTS_SUMMARY.md), [reproducibility](REPRODUCIBILITY.md), and [evaluation limits](EVALUATION_LIMITATIONS.md).
