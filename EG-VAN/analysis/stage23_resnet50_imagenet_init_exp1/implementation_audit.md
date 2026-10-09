# Stage 23 implementation audit

## Read-only confirmation before Stage23 creation

The Stage15 configuration states `initialization.resnet50 = "random"`. The executable path confirms it:

1. Stage15 `train.py` imports `stage13c_guarded.build_components` and calls `guard.build_components(root, "B")`.
2. That builder calls `EGVAN(num_classes=7, pretrained_efficient=True, pretrained_resnet=False)`.
3. `EGVAN` passes this flag to `ModifiedResNet50(pretrained_resnet)`.
4. `ModifiedResNet50` invokes `resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)`. With the Stage15 false flag, torchvision receives `weights=None`.

Therefore Stage22's statement that the Stage15 ResNet50 trunk is randomly initialized is **CONFIRMED**. Stage15 EfficientNetV2S remains pretrained using `EfficientNet_V2_S_Weights.DEFAULT`.

## Frozen reference fingerprints

| Artifact | SHA256 |
|---|---|
| Stage15 `config.json` (root and nested copies identical) | `3ad8c205e06cde8203a869be0a6cc9643ed0ca5b6beb553b57d53d4dce9d25f2` |
| Stage15 `selection_rule.json` (copies identical) | `daf07a1bac3538b5a35ed59121bd5561aba4797a6889e852fd0ca07ad88be735` |
| Stage15 `numerical_protocol.json` (copies identical) | `f59010b7e2509c0bf1633b7126ff1d8d629afab39b515f57e95012fb78eb0196` |
| Stage15 actual nested runner | `391769f3993e1593064cf9f6460feeB8846e0ca2cc955e3fc469700420d9fdb9` |
| Stage15 config-B source | `9f69e862c8d63a2375291343f732c73f5f4bbc4339029b4afced844c52cf7562` |
| Stage13 guarded component builder | `93e2318afafff8ad1c6907752781d61856cdf3bbde3a614b29686686dac0340d` |
| Stage13 base training/data helpers | `56a71821c1f944028874a76febf00ea68b3b152e481e6450ddd5bd94606e5389` |
| selective q@k FP32 training helper | `d5cdfbade60e834e56acf9d67fbb3f83d90f0c6f96473dff9b9d3dd4c399ef1a` |
| Stage15 saved training history | `2df7edd11a23658e56dc723ee57638476579cb3620d9c5d10039eefa07941fd0` |
| leakage-aware split CSV | `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883` |
| recovered Stage15 checkpoint (reference only; not loaded) | `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5` |

Stage15 split integrity requires 10,015 rows, 7,470 lesions, and zero crossing lesions. Its data preflight uses only train/validation datasets and fixed Stage9 validation evidence. Stage15 history contains epochs 1-24; Stage23 remains capped at the registered 25 epochs.

Stage23 preregistration hashes: config `67a9d77d8464ef51d94f5861004084b179c3c42159e172c680d85952514d7afa`; selection rule `a03213d5db62af968a425f3557773aa14c85adeabdbaeb8e6d46bf4d5481111d`; numerical protocol `06222c374635f15ad99642e1bdbeaf4c6c69959dd753d6b332cdb753b87abdfb`; initialization spec `145758267bb655d21f70d69c8c384259e4b4074a7596dcf99d2dcb0a2f7a9d76`.

## Stage23 weight-loading policy

The installed project interpreter reports torchvision `0.29.0+cpu`; `ResNet50_Weights.IMAGENET1K_V2` exists, and the installed enum URL is `https://download.pytorch.org/models/resnet50-11ad3fa6.pth`. The runner uses this explicit enum and its hash-checked state-dict API rather than `DEFAULT` or a guessed filename.

To preserve exact seeded initialization of the modified architecture's new modules, Stage23 constructs the same EGVAN with `pretrained_resnet=False`, using the same seed/order as Stage15, then copies only shape-compatible official tensors for `conv1`, `bn1`, and `layer1` through `layer4` into the existing ResNet branch. `fc.weight`/`fc.bias` are intentionally ignored because the branch has no ImageNet classifier. SCGA and Non-Local modules retain their seeded fresh initialization. All unexpected source keys, missing expected backbone keys, or shape mismatches fail before training. This changes initial backbone tensor values only; no shared model source file is changed.

At actual Colab initialization, the runner writes an atomic machine-readable `run/initialization_report.json` before epoch 1 with runtime versions, enum and URL, loaded keys and tensor/parameter counts, ignored classifier keys, newly initialized module keys, mismatch lists, and report hash. The checked-in `initialization_spec.json` is only a preregistered specification, not evidence that weights have already loaded.

## Split and forbidden-data boundary

The data code creates only `train` and `val` datasets. The runner preflight verifies the frozen split SHA/lesion isolation, counts, processed-image existence for train/validation, and reference hashes; no `test` or PH2 dataset object is created. Static safety tests assert the Stage23 runner has no test/PH2 paths or evaluator calls. Stage16/20 outputs are used only as historical context in this audit, never by the Stage23 runner.

No training, HAM test inference, PH2 inference, or ImageNet weight download was performed during preparation.
