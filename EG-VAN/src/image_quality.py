"""Deterministic image-quality proxy measurements.

These measurements are computational proxies, not clinical quality labels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

QUALITY_FEATURES = (
    "brightness",
    "contrast",
    "sharpness",
    "saturation",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "entropy",
    "illumination_variation",
)

# Fixed descriptive thresholds, not medical-quality cutoffs.
DARK_THRESHOLD = 30
BRIGHT_THRESHOLD = 225


def compute_image_quality(image_or_path: np.ndarray | str | Path) -> dict[str, float]:
    """Return deterministic quality proxies for one RGB array or image path."""
    if isinstance(image_or_path, (str, Path)):
        image = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {image_or_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.asarray(image_or_path)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected a non-empty three-channel image")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]
    grid = 16
    local_means = []
    for y in np.array_split(gray, grid, axis=0):
        for cell in np.array_split(y, grid, axis=1):
            local_means.append(float(cell.mean()))
    local_mean = float(np.mean(local_means))
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "saturation": float(hsv[:, :, 1].mean() / 255.0),
        "dark_pixel_ratio": float(np.mean(gray < DARK_THRESHOLD)),
        "bright_pixel_ratio": float(np.mean(gray > BRIGHT_THRESHOLD)),
        "entropy": float(-(probabilities * np.log2(probabilities)).sum()),
        "illumination_variation": float(np.std(local_means) / local_mean) if local_mean else 0.0,
    }
