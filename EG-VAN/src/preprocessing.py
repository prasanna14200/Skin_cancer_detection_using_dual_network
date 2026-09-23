#!/usr/bin/env python3
"""Paper-informed, deterministic Phase 4B preprocessing for HAM10000.

The paper specifies the operation order but leaves several numerical parameters
unspecified. Those choices are explicit constants below and documented in the
Phase 4B report and deviation log.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
OUTPUT_ROOT = ROOT / "data" / "processed" / "images"
LOG_PATH = ROOT / "data" / "processed" / "preprocessing_log.csv"
SAMPLE_ROOT = ROOT / "data" / "processed" / "preprocessing_samples"
SEED = 42

# The paper does not specify these values. Keep them centralized and auditable.
HAIR_KERNEL_SIZE = 17
HAIR_THRESHOLD = 10
INPAINT_RADIUS = 1.0
RETINEX_SIGMAS = (15.0, 80.0, 250.0)
RETINEX_WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


@dataclass(frozen=True)
class ProcessingResult:
    image_id: str
    input_path: str
    output_path: str
    status: str
    reason: str
    duration_seconds: float


def _validate_image(image: np.ndarray, label: str) -> None:
    """Reject empty, non-RGB/BGR, or numerically invalid image arrays."""
    if image is None or image.size == 0:
        raise ValueError(f"{label} is empty")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{label} must be a three-channel image")
    if not np.isfinite(image.astype(np.float32)).all():
        raise ValueError(f"{label} contains non-finite values")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"{label} has invalid dimensions")


def remove_hair(
    image_bgr: np.ndarray,
    kernel_size: int = HAIR_KERNEL_SIZE,
    threshold: int = HAIR_THRESHOLD,
    inpaint_radius: float = INPAINT_RADIUS,
) -> np.ndarray:
    """Remove dark hair-like structures with blackhat masking and Telea inpainting.

    The paper describes grayscale morphology, thresholding, and Telea inpainting.
    OpenCV's standard blackhat is used: closing minus the grayscale image.
    """
    _validate_image(image_bgr, "hair-removal input")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")
    # Hair appears as thin dark structures, so the blackhat response highlights
    # dark pixels surrounded by brighter lesion/skin context.
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    hair_response = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)
    # The mask is intentionally binary because Telea inpainting expects a clear
    # region to replace from neighboring pixels.
    hair_mask = cv2.threshold(hair_response, threshold, 255, cv2.THRESH_BINARY)[1]
    return cv2.inpaint(image_bgr, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)


def gray_world(image_bgr: np.ndarray) -> np.ndarray:
    """Apply the paper's pairwise Gray World channel gains (equation 9)."""
    _validate_image(image_bgr, "Gray World input")
    image = image_bgr.astype(np.float32)
    # OpenCV stores channels as BGR, so the channel averages are read in that
    # order even though the paper describes color channels conceptually as RGB.
    b_avg, g_avg, r_avg = image.mean(axis=(0, 1))
    epsilon = 1e-6
    gains = np.array(
        [
            (g_avg + r_avg) / (2.0 * max(b_avg, epsilon)),
            (r_avg + b_avg) / (2.0 * max(g_avg, epsilon)),
            (g_avg + b_avg) / (2.0 * max(r_avg, epsilon)),
        ],
        dtype=np.float32,
    )
    balanced = np.clip(image * gains, 0.0, 255.0)
    return balanced.astype(np.uint8)


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    # Percentile clipping avoids letting a few extreme pixels dominate the
    # Retinex output range.
    low, high = np.percentile(channel, (1.0, 99.0))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise ValueError("Retinex channel normalization received non-finite values")
    if high <= low:
        return np.zeros(channel.shape, dtype=np.uint8)
    normalized = (channel - low) * (255.0 / (high - low))
    return np.clip(normalized, 0.0, 255.0).astype(np.uint8)


def retinex(
    image_bgr: np.ndarray,
    sigmas: Iterable[float] = RETINEX_SIGMAS,
    weights: Iterable[float] = RETINEX_WEIGHTS,
) -> np.ndarray:
    """Apply a multi-scale logarithmic Retinex and normalize per channel.

    The paper defines Gaussian filtering, logarithmic subtraction, weighted
    multi-scale summation, and Gray World gains. Gaussian scales, weights, and
    output normalization are not specified in the paper and are fixed here.
    """
    _validate_image(image_bgr, "Retinex input")
    sigma_values = tuple(float(value) for value in sigmas)
    weight_values = tuple(float(value) for value in weights)
    if not sigma_values or len(sigma_values) != len(weight_values):
        raise ValueError("Retinex sigmas and weights must have equal nonzero length")
    if not np.isclose(sum(weight_values), 1.0):
        raise ValueError("Retinex weights must sum to one")

    image = image_bgr.astype(np.float32) / 255.0
    log_image = np.log(np.maximum(image, 1e-6))
    output = np.zeros_like(image, dtype=np.float32)
    for sigma, weight in zip(sigma_values, weight_values):
        # Each Gaussian scale captures illumination variation at a different
        # spatial size; the weighted sum forms the multi-scale Retinex response.
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        filtered_log = np.log(np.maximum(blurred, 1e-6))
        # This follows equation (11) as printed in the paper: log(filtered) - log(input).
        output += weight * (filtered_log - log_image)

    channels = [_normalize_channel(output[:, :, index]) for index in range(3)]
    result = cv2.merge(channels)
    _validate_image(result, "Retinex output")
    return result


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    """Run the frozen Phase 4B order: hair removal, Gray World, Retinex."""
    return retinex(gray_world(remove_hair(image_bgr)))


def discover_images() -> list[Path]:
    """Find original JPGs in both official extracted image directories."""
    paths = sorted(
        path
        for directory in ("HAM10000_images_part_1", "HAM10000_images_part_2")
        for path in (RAW_ROOT / directory).rglob("*.jpg")
    )
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Duplicate image filenames found in original image directories")
    return paths


def process_one(input_path: Path, output_path: Path) -> ProcessingResult:
    started = time.perf_counter()
    image_id = input_path.stem
    try:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("input could not be decoded")
        _validate_image(image, "input")
        processed = preprocess_image(image)
        _validate_image(processed, "output")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), processed):
            raise OSError("output could not be written")
        reopened = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if reopened is None:
            raise ValueError("written output could not be reopened")
        _validate_image(reopened, "reopened output")
        return ProcessingResult(image_id, str(input_path), str(output_path), "success", "", time.perf_counter() - started)
    except Exception as exc:
        return ProcessingResult(image_id, str(input_path), str(output_path), "failed", str(exc), time.perf_counter() - started)


def validate_sample(paths: list[Path], sample_size: int) -> list[ProcessingResult]:
    """Run a deterministic sample before allowing the full preprocessing pass."""
    rng = random.Random(SEED)
    sample = rng.sample(paths, min(sample_size, len(paths)))
    results = []
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    for path in sample:
        output_path = SAMPLE_ROOT / path.name
        result = process_one(path, output_path)
        results.append(result)
        if result.status == "success":
            original = cv2.imread(str(path), cv2.IMREAD_COLOR)
            processed = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            comparison = np.hstack((original, processed))
            cv2.imwrite(str(SAMPLE_ROOT / f"{path.stem}_before_after.jpg"), comparison)
    return results


def write_log(results: list[ProcessingResult]) -> None:
    """Persist per-image success/failure records for later auditing."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def process_one_task(task: tuple[str, str]) -> ProcessingResult:
    """Worker entry point for safe, independent per-image processing."""
    input_path, output_path = task
    return process_one(Path(input_path), Path(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args()

    paths = discover_images()
    if len(paths) != 10015:
        raise RuntimeError(f"Expected 10015 original images, found {len(paths)}")
    # The sample gate catches obvious preprocessing failures before spending
    # time rewriting the full processed image cache.
    sample_results = validate_sample(paths, args.sample_size)
    sample_failures = [result for result in sample_results if result.status != "success"]
    print(json.dumps({"sample_total": len(sample_results), "sample_success": len(sample_results) - len(sample_failures), "sample_failed": len(sample_failures)}, indent=2))
    if sample_failures:
        for result in sample_failures:
            print(f"SAMPLE FAILURE: {result.image_id}: {result.reason}")
        return 1
    if args.sample_only:
        return 0

    started = time.perf_counter()
    worker_count = min(4, max(1, os.cpu_count() or 1))
    tasks = [(str(path), str(OUTPUT_ROOT / path.name)) for path in paths]
    # Each image is independent, so a small process pool improves throughput
    # without introducing dataset-wide state or statistics.
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(process_one_task, tasks))
    write_log(results)
    failures = [result for result in results if result.status != "success"]
    elapsed = time.perf_counter() - started
    print(json.dumps({"total": len(results), "successful": len(results) - len(failures), "failed": len(failures), "duration_seconds": round(elapsed, 3), "log": str(LOG_PATH)}, indent=2))
    for result in failures:
        print(f"FAILURE: {result.image_id}: {result.reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
