"""Task-specific 3D BraTS brain-tumor segmentation using MONAI."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


MODEL_PATH = Path(__file__).resolve().parent / "bundles" / "brats_mri_segmentation" / "models" / "model.pt"
MODEL_BYTES = 18_840_620
MODEL_SHA256 = "860ccb3f1c21c99d0410ad8a1ac4ef6b8fab60cec0a503b0ba42675741a750ae"
OFFICIAL_ROI = (240, 240, 160)
MRI_CHANNELS = ("t1c", "t1", "t2", "flair")
MRI_LABELS = {0: "background", 1: "tumor_core", 2: "whole_tumor_remainder", 4: "enhancing_tumor"}
MAX_VOXELS = 12_000_000


class MRISegmentationError(ValueError):
    """Unsupported MRI input or invalid pretrained-model output."""


class MRIDependencyError(RuntimeError):
    """A required MRI workflow dependency is unavailable."""


def _nib_only():
    try:
        import nibabel as nib
    except ImportError as exc:
        raise MRIDependencyError(
            "BraTS MRI segmentation requires nibabel for NIfTI I/O."
        ) from exc
    return nib


def _modules():
    try:
        import nibabel as nib
        from monai.inferers import SlidingWindowInferer
        from monai.networks.nets import SegResNet
        from monai.transforms import NormalizeIntensity
    except ImportError as exc:
        raise MRIDependencyError(
            "BraTS MRI segmentation requires the isolated MONAI dependencies in requirements-segmentation.txt."
        ) from exc
    return nib, SlidingWindowInferer, SegResNet, NormalizeIntensity


def load_model(checkpoint_path: Path = MODEL_PATH):
    if not checkpoint_path.is_file():
        raise FileNotFoundError("The verified BraTS MRI checkpoint is missing.")
    if checkpoint_path.stat().st_size != MODEL_BYTES:
        raise MRISegmentationError("The BraTS MRI checkpoint has an unexpected file size.")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        raise MRISegmentationError("The BraTS MRI checkpoint SHA-256 does not match the published MONAI artifact.")

    segresnet = _modules()[2]
    model = segresnet(
        spatial_dims=3,
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _validated_inputs(paths: Mapping[str, Path]):
    nib = _nib_only()
    try:
        from monai.transforms import NormalizeIntensity
    except ImportError:
        NormalizeIntensity = None
    if set(paths) != set(MRI_CHANNELS):
        raise MRISegmentationError("Exactly T1c, T1, T2, and FLAIR NIfTI files are required.")

    images = {}
    arrays = []
    reference_shape = None
    reference_affine = None
    for channel in MRI_CHANNELS:
        try:
            image = nib.as_closest_canonical(nib.load(str(paths[channel])))
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
            raise MRISegmentationError(f"Could not read the {channel.upper()} NIfTI volume.") from exc
        shape = tuple(int(value) for value in image.shape)
        if len(shape) != 3:
            raise MRISegmentationError(f"{channel.upper()} must be a finite 3D NIfTI volume.")
        voxel_count = math.prod(shape)
        if voxel_count == 0 or voxel_count > MAX_VOXELS:
            raise MRISegmentationError(
                f"{channel.upper()} volume must contain between 1 and {MAX_VOXELS:,} voxels."
            )
        affine = np.asarray(image.affine, dtype=np.float64)
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        if not np.isfinite(affine).all() or not np.isfinite(spacing).all():
            raise MRISegmentationError(f"{channel.upper()} NIfTI geometry is invalid.")
        if not np.allclose(spacing, (1.0, 1.0, 1.0), atol=0.01, rtol=0.0):
            raise MRISegmentationError(
                f"{channel.upper()} voxel spacing must be 1 mm isotropic, as required by the BraTS bundle."
            )
        direction = affine[:3, :3] / spacing
        if not np.allclose(direction, np.eye(3), atol=0.01, rtol=0.0):
            raise MRISegmentationError(
                f"{channel.upper()} must use axis-aligned canonical RAS orientation; oblique inputs are unsupported."
            )
        if reference_shape is None:
            reference_shape, reference_affine = shape, affine
        elif shape != reference_shape or not np.allclose(affine, reference_affine, atol=1e-4, rtol=1e-5):
            raise MRISegmentationError(
                "T1c, T1, T2, and FLAIR must be co-registered NIfTI volumes with identical shapes and affines."
            )
        try:
            array = np.asarray(image.dataobj, dtype=np.float32)
        except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
            raise MRISegmentationError(f"Could not decode the {channel.upper()} NIfTI voxels.") from exc
        if array.shape != shape or not np.isfinite(array).all():
            raise MRISegmentationError(f"{channel.upper()} must contain finite voxel values matching its header.")
        if not np.any(array != 0):
            raise MRISegmentationError(f"{channel.upper()} volume contains no non-zero voxels.")
        images[channel] = image
        arrays.append(array)

    channels = np.stack(arrays, axis=0)
    if NormalizeIntensity is not None:
        normalized = NormalizeIntensity(nonzero=True, channel_wise=True)(channels)
    else:
        normalized = _normalize_channels_numpy(channels)
    tensor = torch.as_tensor(np.asarray(normalized, dtype=np.float32)).unsqueeze(0)
    return images["flair"], tensor


def _normalize_channels_numpy(channels: np.ndarray) -> np.ndarray:
    """Numpy fallback for MONAI NormalizeIntensity(nonzero=True, channel_wise=True)."""
    result = np.zeros_like(channels, dtype=np.float32)
    for c in range(channels.shape[0]):
        channel = channels[c]
        nonzero = channel[channel != 0]
        if len(nonzero) > 0:
            mean = float(nonzero.mean())
            std = float(nonzero.std())
            if std > 0:
                result[c] = (channel - mean) / std
    return result


def _roi_for_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(
        min(limit, max(8, (dimension // 8) * 8))
        for dimension, limit in zip(shape, OFFICIAL_ROI)
    )


def run_segmentation(
    paths: Mapping[str, Path],
    output_path: Path,
    model=None,
    model_loader: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run the documented BraTS model on four aligned sequences and write a NIfTI label map."""
    nib = _nib_only()
    flair_image, input_tensor = _validated_inputs(paths)
    if model is None:
        model = (model_loader or load_model)()
    if input_tensor.shape[1] != 4:
        raise MRISegmentationError("Preprocessed MRI input must have exactly four channels.")

    try:
        from monai.inferers import SlidingWindowInferer
    except ImportError as exc:
        raise MRIDependencyError(
            "BraTS MRI segmentation requires the isolated MONAI dependencies in requirements-segmentation.txt."
        ) from exc

    roi_size = _roi_for_shape(tuple(int(value) for value in input_tensor.shape[2:]))
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=1,
        overlap=0.5,
        mode="constant",
    )
    with torch.inference_mode():
        output = inferer(input_tensor, model)
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != (
        1,
        3,
        *input_tensor.shape[2:],
    ):
        raise MRISegmentationError("The BraTS MRI model returned an invalid output tensor shape.")
    if not torch.isfinite(output).all():
        raise MRISegmentationError("The BraTS MRI model returned non-finite values.")

    probabilities = torch.sigmoid(output[0]).cpu().numpy()
    channels = probabilities >= 0.5
    tumor_core, whole_tumor, enhancing_tumor = channels
    labels = np.zeros(tumor_core.shape, dtype=np.uint8)
    labels[whole_tumor] = 2
    labels[tumor_core] = 1
    labels[enhancing_tumor] = 4

    output_image = nib.Nifti1Image(labels, flair_image.affine, header=flair_image.header.copy())
    output_image.set_data_dtype(np.uint8)
    nib.save(output_image, str(output_path))

    flair = np.asarray(flair_image.dataobj, dtype=np.float32)
    nonzero = flair[flair != 0]
    low, high = np.percentile(nonzero, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise MRISegmentationError("FLAIR intensity range is unsuitable for display.")
    preview = np.clip((flair - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return {
        "mask": labels,
        "preview": preview,
        "shape": tuple(int(value) for value in labels.shape),
        "affine": np.asarray(flair_image.affine, dtype=np.float64),
        "spacing_mm": tuple(float(value) for value in np.linalg.norm(flair_image.affine[:3, :3], axis=0)),
        "roi_size": roi_size,
        "label_counts": {
            str(label): int(np.count_nonzero(labels == label))
            for label in (0, 1, 2, 4)
        },
    }
