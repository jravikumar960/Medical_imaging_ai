"""DICOM pixel decoding for local research previews; patient metadata is discarded."""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from input_suitability import require_cxr_dicom_metadata


class DicomInputError(ValueError):
    """Raised when an uploaded DICOM is outside the supported preview subset."""


def _to_uint8_pixels(dataset) -> np.ndarray:
    if str(getattr(dataset, "BurnedInAnnotation", "")).strip().upper() == "YES":
        raise DicomInputError("DICOM indicates burned-in annotations; upload a de-identified image without them.")
    try:
        frames = int(getattr(dataset, "NumberOfFrames", 1))
    except (TypeError, ValueError):
        frames = 0
    if frames != 1:
        raise DicomInputError("Only single-frame DICOM images are supported.")
    try:
        samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1))
    except (TypeError, ValueError):
        samples_per_pixel = 0
    if samples_per_pixel != 1:
        raise DicomInputError("Only single-channel DICOM images are supported.")
    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise DicomInputError("Only MONOCHROME1 and MONOCHROME2 images are supported.")
    try:
        pixels = np.asarray(dataset.pixel_array)
        pixels = apply_modality_lut(pixels, dataset)
        if hasattr(dataset, "VOILUTSequence") or (
            hasattr(dataset, "WindowCenter") and hasattr(dataset, "WindowWidth")
        ):
            pixels = apply_voi_lut(pixels, dataset)
    except Exception:
        raise DicomInputError("DICOM pixel data could not be decoded or transformed.") from None
    if pixels.ndim != 2 or pixels.size == 0 or not np.isfinite(pixels).all():
        raise DicomInputError("Expected a finite, non-empty 2D DICOM pixel array.")
    pixels = pixels.astype(np.float32, copy=False)
    low, high = np.percentile(pixels, (0.5, 99.5))
    low, high = float(low), float(high)
    if high <= low:
        raise DicomInputError("DICOM image has no displayable intensity range.")
    display = np.clip((pixels - low) * (255.0 / (high - low)), 0, 255)
    if photometric == "MONOCHROME1":
        display = 255.0 - display
    return np.rint(display).astype(np.uint8)


def dicom_bytes_to_image(data: bytes) -> Image.Image:
    """Decode pixels from one DICOM byte stream without retaining its metadata."""
    dataset = _read_dataset(data)
    if "PixelData" not in dataset:
        raise DicomInputError("DICOM file has no image pixel data.")
    pixels = _to_uint8_pixels(dataset)
    return Image.fromarray(pixels)


def dicom_bytes_to_cxr_candidate(data: bytes) -> Image.Image:
    """Decode a DICOM candidate after checking declared modality and body region.

    The DICOM tags are a basic input filter only. They do not prove the pixels
    depict a chest X-ray and can be inaccurate.
    """
    dataset = _read_dataset(data)
    require_cxr_dicom_metadata(dataset)
    if "PixelData" not in dataset:
        raise DicomInputError("DICOM file has no image pixel data.")
    pixels = _to_uint8_pixels(dataset)
    return Image.fromarray(pixels)


def _read_dataset(data: bytes):
    """Read DICOM bytes while keeping metadata local to this call."""
    if not isinstance(data, bytes) or not data:
        raise DicomInputError("DICOM input is empty.")
    try:
        dataset = pydicom.dcmread(BytesIO(data), force=False)
    except Exception:
        raise DicomInputError("File is not a readable DICOM image.") from None
    return dataset
