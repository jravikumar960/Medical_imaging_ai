"""Unvalidated localization helpers for X-ray Grad-CAM audit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects


def border_mask(shape, margin=16):
    mask = np.zeros(shape, dtype=bool)
    if margin <= 0:
        return mask
    margin = min(margin, min(shape) // 2)
    mask[:margin, :] = True
    mask[-margin:, :] = True
    mask[:, :margin] = True
    mask[:, -margin:] = True
    return mask


def mass_fraction(cam, mask):
    cam = np.asarray(cam, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    total = float(cam.sum())
    if total <= 0 or cam.shape != mask.shape:
        return 0.0
    return float(cam[mask].sum() / total)


def border_mass_fractions(cam, widths=(8, 16, 24)):
    result = {}
    for width in widths:
        result[str(width)] = mass_fraction(cam, border_mask(np.asarray(cam).shape, width))
    return result


def region_mass_fractions(cam, lung_mask, border_width=16):
    cam = np.asarray(cam, dtype=np.float64)
    lung_mask = np.asarray(lung_mask, dtype=bool)
    border = border_mask(cam.shape, border_width)
    total = float(cam.sum())
    if total <= 0 or cam.shape != lung_mask.shape:
        return {}
    return {
        "inside_lung": float(cam[lung_mask].sum() / total),
        "outside_lung": float(cam[~lung_mask].sum() / total),
        "border": float(cam[border].sum() / total),
        "non_border": float(cam[~border].sum() / total),
        "border_and_lung": float(cam[border & lung_mask].sum() / total),
        "border_and_non_lung": float(cam[border & ~lung_mask].sum() / total),
        "non_border_lung": float(cam[~border & lung_mask].sum() / total),
        "non_border_non_lung": float(cam[~border & ~lung_mask].sum() / total),
    }


def estimate_unvalidated_lung_mask(image):
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError("Lung mask estimation requires a 2D grayscale image.")
    body = array > 5
    if not body.any():
        return np.zeros_like(body, dtype=bool)
    threshold = threshold_otsu(array[body])
    candidate = body & (array < threshold)
    candidate = remove_small_objects(candidate, min_size=max(1, int(0.001 * array.size)))
    labeled = label(candidate)
    if labeled.max() == 0:
        return candidate
    areas = [(int((labeled == component).sum()), component) for component in range(1, labeled.max() + 1)]
    areas.sort(reverse=True)
    keep = [component for _, component in areas[:2]]
    return np.isin(labeled, keep)


def occlude_image(image, mask, method="zero", fill_value=0, sigma=8.0):
    if isinstance(image, Image.Image):
        output = image.copy().convert("L")
        array = np.asarray(output, dtype=np.float64).copy()
    else:
        array = np.asarray(image, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=bool)
    if array.shape != mask.shape:
        raise ValueError("Mask shape must match image shape.")
    if method == "zero":
        array[mask] = float(fill_value)
    elif method == "local_mean":
        if np.any(~mask):
            replacement = float(array[~mask].mean())
        else:
            replacement = float(fill_value)
        array[mask] = replacement
    elif method == "gaussian_blur":
        blurred = gaussian_filter(array, sigma=sigma)
        array[mask] = blurred[mask]
    else:
        raise ValueError(f"Unknown occlusion method: {method}")
    result = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")
    return result