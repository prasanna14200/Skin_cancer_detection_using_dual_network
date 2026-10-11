"""Frozen Stage 4E data, transformation, initialization and metric helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import EfficientNet_V2_S_Weights


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
STAGE23 = ROOT / "experiments/stage23_resnet50_imagenet_init_exp1"
V2 = ROOT / "analysis/stage4d_conservative_split"
for entry in (str(SRC), str(STAGE23)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from initialization import RESNET_WEIGHTS, apply_resnet_state  # noqa: E402
from models.egvan import EGVAN  # noqa: E402


CONFIG_SHA = "281e24f8d09d825c7f7c08ac2d31a10c298adf8fedd5fbe7689df5da8b6b95ae"
SOURCE_HASHES_SHA = "99364c150c95dbc521e59be61de0280706fca29ffe04d01d1b95c55129c0b536"
STAGE23_CONFIG_SHA = "67a9d77d8464ef51d94f5861004084b179c3c42159e172c680d85952514d7afa"
V2_MANIFEST_SHA = "e218c9c2772452738cc53753d3b8857f1d321722a3e64b0b0e389442fcdadc93"
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
LABEL = {name: index for index, name in enumerate(CLASSES)}
TRAIN_NAME = "stage4d_development_train_v2.csv"
SELECT_NAME = "stage4d_internal_selection_v2.csv"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_config() -> dict:
    if sha256(HERE / "config.json") != CONFIG_SHA:
        raise ValueError("Stage 4E config hash differs from registration")
    return json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def arm_config(cfg: dict, arm: str) -> dict:
    if arm not in ("control", "treatment"):
        raise ValueError("Unknown arm")
    return {**cfg, "arm": arm, "blur_enabled": arm == "treatment"}


def preflight(root: Path, *, hash_images: bool = True) -> dict:
    if root.resolve() != ROOT.resolve():
        raise ValueError("Use the pinned EG-VAN repository")
    cfg = read_config()
    sources_path = HERE / "source_hashes.json"
    if sha256(sources_path) != SOURCE_HASHES_SHA:
        raise ValueError("Stage 4E pinned source-hash catalog changed")
    for relative, expected in json.loads(sources_path.read_text(encoding="utf-8")).items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Frozen architecture/training source changed: {relative}")
    manifest_path = V2 / "stage4d_split_manifest_v2.json"
    if sha256(manifest_path) != V2_MANIFEST_SHA or cfg["stage4d_manifest_sha256"] != V2_MANIFEST_SHA:
        raise ValueError("Stage 4D v2 manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, digest in manifest["artifact_sha256"].items():
        if sha256(V2 / name) != digest:
            raise ValueError(f"Stage 4D artifact changed: {name}")
    if sha256(STAGE23 / "config.json") != STAGE23_CONFIG_SHA:
        raise ValueError("Stage 23 reference config changed")
    old = json.loads((STAGE23 / "config.json").read_text(encoding="utf-8"))
    if (tuple(cfg["class_order"]) != CLASSES or tuple(old["classes"]) != CLASSES or
            cfg["seed"] != old["seed"] or cfg["seed"] != 42 or
            cfg["epochs"] != old["epochs"] or cfg["epochs"] != 25 or
            cfg["image_size"] != old["image_size"] or cfg["image_size"] != 384 or
            cfg["physical_batch_size"] != old["physical_batch_size"] or cfg["physical_batch_size"] != 16 or
            cfg["gradient_accumulation_steps"] != old["gradient_accumulation_steps"] or
            cfg["gradient_accumulation_steps"] != 1 or
            cfg["num_workers"] != old["num_workers"] or
            cfg["sampler"]["mel_weight"] != old["sampler"]["mel_weight"] or
            cfg["sampler"]["replacement"] != old["sampler"]["replacement"] or
            cfg["sampler"]["num_samples"] != manifest["counts"]["development_train"]["images"] or
            cfg["optimizer"]["lr"] != old["optimizer"]["learning_rate"] or
            cfg["optimizer"]["betas"] != old["optimizer"]["betas"] or
            cfg["optimizer"]["eps"] != old["optimizer"]["eps"] or
            cfg["optimizer"]["weight_decay"] != old["optimizer"]["weight_decay"] or
            cfg["scheduler"]["factor"] != old["scheduler"]["factor"] or
            cfg["scheduler"]["patience"] != old["scheduler"]["patience"] or
            cfg["loss"]["alpha"] != old["loss"]["alpha"] or
            cfg["loss"]["gamma"] != old["loss"]["gamma"] or
            cfg["loss"]["true_mel_multiplier"] != old["loss"]["mel_multiplier"] or
            cfg["numerical_policy"]["initial_grad_scaler_scale"] != 8192 or
            cfg["numerical_policy"]["selective_fp32"] != "resnet.nonlocal3 q@k affinity only" or
            cfg["selection_bootstrap_replicates"] != 2000 or
            cfg["treatment_only_transform"] != {"position": "after_RandomRotation_before_ToTensor",
                "type": "RandomApply_GaussianBlur", "probability": 0.25,
                "kernel_size": 3, "sigma_min": 0.1, "sigma_max": 0.5}):
        raise ValueError("Stage 4E common training factors differ from frozen Stage 23/4B policy")
    train, selection = read_csv(V2 / TRAIN_NAME), read_csv(V2 / SELECT_NAME)
    by_id = {r["image_id"]: r for r in train + selection}
    if (len(train) != 6477 or len(selection) != 1538 or len(by_id) != 8015 or
            {r["lesion_id"] for r in train} & {r["lesion_id"] for r in selection} or
            {r["raw_image_sha256"] for r in train} & {r["raw_image_sha256"] for r in selection} or
            {r["processed_image_sha256"] for r in train} & {r["processed_image_sha256"] for r in selection} or
            {r["dx"] for r in train} != set(CLASSES) or {r["dx"] for r in selection} != set(CLASSES) or
            sum(r["dx"] == "mel" for r in selection) != 163):
        raise ValueError("Stage 4D image/lesion/class separation failed")
    split = read_csv(ROOT / "data/splits/split_leakage_aware.csv")
    original_train = {r["image_id"]: r for r in split if r["split"] == "train"}
    heldout = [r for r in split if r["split"] != "train"]
    if (set(by_id) != set(original_train) or
            {r["lesion_id"] for r in train + selection} & {r["lesion_id"] for r in heldout} or
            any((r["lesion_id"], r["dx"]) != (original_train[r["image_id"]]["lesion_id"], original_train[r["image_id"]]["dx"])
                for r in train + selection)):
        raise ValueError("Stage 4D membership differs from eligible original train pool")
    if hash_images:
        processed = ROOT / cfg["processed_images"]
        for row in train + selection:
            image_id = row["image_id"]
            if (sha256(processed / f"{image_id}.jpg") != row["processed_image_sha256"] or
                    sha256(ROOT / "data/raw/images" / f"{image_id}.jpg") != row["raw_image_sha256"]):
                raise ValueError(f"Stage 4D source image changed: {image_id}")
    return {"status": "PREFLIGHT_PASS", "train_images": len(train), "selection_images": len(selection),
            "selection_mel_images": 163, "selection_mel_lesions": 95,
            "stage4d_manifest_sha256": V2_MANIFEST_SHA, "config_sha256": CONFIG_SHA,
            "image_hashes_checked": hash_images, "training_started": False}


class ManifestDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], transform):
        self.rows = rows
        self.images_dir = root / "data/processed/images"
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.images_dir / f"{row['image_id']}.jpg") as image:
            return self.transform(image.convert("RGB")), LABEL[row["dx"]]


def transforms_for_arm(arm: str):
    preprocess = EfficientNet_V2_S_Weights.DEFAULT.transforms()
    core = [transforms.Resize((384, 384)), transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), transforms.RandomRotation(15)]
    if arm == "treatment":
        core.append(transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.25))
    elif arm != "control":
        raise ValueError("Unknown arm")
    core.extend([transforms.ToTensor(), transforms.Normalize(mean=preprocess.mean, std=preprocess.std)])
    evaluate = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(),
                                   transforms.Normalize(mean=preprocess.mean, std=preprocess.std)])
    return transforms.Compose(core), evaluate


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii") + b"\0")
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def fresh_model():
    set_seed(42)
    model = EGVAN(num_classes=7, pretrained_efficient=True, pretrained_resnet=False)
    source = RESNET_WEIGHTS.get_state_dict(progress=True, check_hash=True)
    report = apply_resnet_state(model.resnet, source)
    if report["status"] != "LOADED" or RESNET_WEIGHTS.name != "IMAGENET1K_V2":
        raise ValueError("Official ResNet V2 initialization failed")
    return model, {"resnet": report, "resnet_weight_enum": RESNET_WEIGHTS.name,
                   "resnet_weight_url": RESNET_WEIGHTS.url,
                   "efficientnet_weight_enum": EfficientNet_V2_S_Weights.DEFAULT.name,
                   "initial_state_sha256": model_fingerprint(model)}


def sampler_for(rows, generator):
    weights = torch.tensor([1.5630495442733532 if row["dx"] == "mel" else 1.0 for row in rows], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True, generator=generator)


def metrics_from_prediction_rows(rows: list[dict]):
    matrix = [[0] * 7 for _ in range(7)]
    for row in rows:
        matrix[LABEL[row["true_label"]]][LABEL[row["predicted_label"]]] += 1
    per = {}
    for index, name in enumerate(CLASSES):
        tp, support = matrix[index][index], sum(matrix[index])
        predicted = sum(line[index] for line in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[name] = {"support": support, "tp": tp, "fp": predicted - tp,
                     "fn": support - tp, "precision": precision, "recall": recall, "f1": f1}
    total = len(rows)
    non_mel = total - per["mel"]["support"]
    non_mel_predicted_mel = per["mel"]["fp"]
    if total == 0 or non_mel == 0 or any(x["support"] == 0 for x in per.values()):
        raise ValueError("Missing class or MEL specificity denominator")
    return {"sample_count": total, "accuracy": sum(matrix[i][i] for i in range(7)) / total,
            "balanced_accuracy": sum(x["recall"] for x in per.values()) / 7,
            "macro_f1": sum(x["f1"] for x in per.values()) / 7,
            "mel_specificity": (non_mel - non_mel_predicted_mel) / non_mel,
            "mel_specificity_denominator": non_mel,
            "per_class": per, "confusion_matrix": matrix}


def validate_predictions(rows: list[dict], expected: list[dict]):
    if len(rows) != len(expected):
        raise ValueError("Selection prediction count mismatch")
    seen = set()
    for row, source in zip(rows, expected):
        if (row["image_id"] != source["image_id"] or row["true_label"] != source["dx"] or
                row["image_id"] in seen or row["predicted_label"] not in CLASSES):
            raise ValueError("Selection prediction ID/order/class mismatch")
        seen.add(row["image_id"])
        probabilities = json.loads(row["probabilities"])
        if (len(probabilities) != 7 or not all(math.isfinite(p) and p >= 0 for p in probabilities) or
                abs(sum(probabilities) - 1) > 1e-3 or
                CLASSES[max(range(7), key=lambda j: probabilities[j])] != row["predicted_label"]):
            raise ValueError("Selection probability vector invalid")
    return metrics_from_prediction_rows(rows)
