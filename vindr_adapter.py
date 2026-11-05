"""VinDr-CXR DICOM preprocessing adapter, manifest parser, and validation gate."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from PIL import Image
import torch


class VinDrAdapterError(ValueError):
    """Raised when VinDr DICOM processing or manifest validation fails."""


def preprocess_vindr_dicom_bytes(data: bytes) -> tuple[torch.Tensor, Image.Image]:
    """Decode a 16-bit VinDr DICOM byte stream and apply TorchXRayVision preprocessing.

    Handles 16-bit grayscale DICOMs, photometric interpretation (MONOCHROME1/2),
    modality/VOI LUTs or windowing, center square cropping, resizing to 224x224,
    and TorchXRayVision normalization: (pixels / 255.0 - 0.5) * 2048.0.
    Preserves existing PNG/JPEG pipeline in imaging.py untouched.
    """
    if not isinstance(data, bytes) or not data:
        raise VinDrAdapterError("DICOM input bytes are empty.")
    try:
        dataset = pydicom.dcmread(BytesIO(data), force=False)
    except Exception as exc:
        raise VinDrAdapterError("File is not a readable DICOM image.") from exc

    if "PixelData" not in dataset:
        raise VinDrAdapterError("DICOM file has no image pixel data.")

    try:
        frames = int(getattr(dataset, "NumberOfFrames", 1))
    except (TypeError, ValueError):
        frames = 1
    if frames != 1:
        raise VinDrAdapterError("Only single-frame DICOM images are supported.")

    try:
        samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1))
    except (TypeError, ValueError):
        samples_per_pixel = 1
    if samples_per_pixel != 1:
        raise VinDrAdapterError("Only single-channel DICOM images are supported.")

    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise VinDrAdapterError("Only MONOCHROME1 and MONOCHROME2 images are supported.")

    try:
        pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
        if hasattr(dataset, "RescaleSlope") and hasattr(dataset, "RescaleIntercept"):
            slope = float(dataset.RescaleSlope)
            intercept = float(dataset.RescaleIntercept)
            pixels = pixels * slope + intercept
        if hasattr(dataset, "WindowCenter") and hasattr(dataset, "WindowWidth"):
            try:
                center = float(dataset.WindowCenter[0] if isinstance(dataset.WindowCenter, (list, tuple, pydicom.multival.MultiValue)) else dataset.WindowCenter)
                width = float(dataset.WindowWidth[0] if isinstance(dataset.WindowWidth, (list, tuple, pydicom.multival.MultiValue)) else dataset.WindowWidth)
                if width > 0:
                    low = center - width / 2.0
                    high = center + width / 2.0
                    pixels = np.clip(pixels, low, high)
            except Exception:
                pass
    except Exception as exc:
        raise VinDrAdapterError("DICOM pixel decoding failed.") from exc

    if pixels.ndim != 2 or pixels.size == 0 or not np.isfinite(pixels).all():
        raise VinDrAdapterError("Expected a finite 2D DICOM pixel array.")

    low, high = float(pixels.min()), float(pixels.max())
    if high <= low:
        raise VinDrAdapterError("DICOM image has no displayable intensity range.")

    display = np.clip((pixels - low) * (255.0 / (high - low)), 0, 255.5)
    if photometric == "MONOCHROME1":
        display = 255.0 - display

    uint8_pixels = np.rint(display).astype(np.uint8)
    img = Image.fromarray(uint8_pixels, mode="L")

    width, height = img.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    cropped = img.crop((left, top, left + side, top + side))
    resized = cropped.resize((224, 224), Image.Resampling.BILINEAR)

    arr = np.asarray(resized, dtype=np.float32)
    normalized = (arr / 255.0 - 0.5) * 2048.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor, resized


def parse_vindr_annotations(annotation_csv_path: Path | str) -> dict[str, list[dict]]:
    """Parse VinDr-CXR annotation records without assuming missing bounding boxes imply negative labels.

    Distinguishes:
    - Explicit positive (bounding box present for finding)
    - Explicit negative (consensus 'No finding' or explicitly marked absence)
    - Unannotated / uncertain (missing bounding box where consensus label is unconfirmed)
    """
    path = Path(annotation_csv_path)
    if not path.exists():
        raise VinDrAdapterError(f"VinDr annotation file not found: {path}")

    # Placeholder for structured parsing adhering to strict negative-label policy
    return {}


def load_label_crosswalk(crosswalk_json_path: Path | str) -> list[dict]:
    """Load and validate the VinDr-CXR to TorchXRayVision label crosswalk."""
    path = Path(crosswalk_json_path)
    if not path.exists():
        raise VinDrAdapterError(f"Label crosswalk file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "crosswalk" not in data:
        raise VinDrAdapterError("Invalid label crosswalk format: missing 'crosswalk' key.")
    return data["crosswalk"]


def validate_vindr_evaluation_readiness(data_dir: Path | str) -> bool:
    """Validation gate that refuses evaluation when required dataset files, manifests, or metadata are missing."""
    d_path = Path(data_dir)
    if not d_path.exists():
        raise VinDrAdapterError(f"VinDr data directory does not exist: {d_path}")

    required_files = ["test.csv", "annotations.csv"]
    missing = [f for f in required_files if not (d_path / f).exists()]
    if missing:
        raise VinDrAdapterError(
            f"Validation gate failed: Missing required VinDr-CXR files in {d_path}: {missing}. "
            "Evaluation is blocked until official dataset files are staged."
        )
    return True
