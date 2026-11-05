"""Separate MONAI 3D spleen segmentation for attested NIfTI CT volumes.

This module is optional and is not imported by the chest X-ray application.
It never downloads model weights. Callers must provide a checkpoint or
explicitly opt into random, untrained weights for software experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any, Callable, Sequence

import nibabel as nib
import numpy as np
import torch

import ct_provenance
import orientation_review

MAX_VOLUME_VOXELS = 150_000_000


class SegmentationError(ValueError):
    """Base exception for safe segmentation input/configuration errors."""


class GeometryError(SegmentationError):
    """Raised when volume geometry is absent, ambiguous, or untrusted."""


class SegmentationDependencyError(RuntimeError):
    """Raised when optional NIfTI/MONAI packages are missing."""


@dataclass(frozen=True)
class UNetConfig:
    """MONAI 3D U-Net shape, matching the official spleen bundle by default."""

    channels: tuple[int, ...] = (16, 32, 64, 128, 256)
    strides: tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2

    def __post_init__(self) -> None:
        if len(self.channels) != len(self.strides) + 1:
            raise SegmentationError("UNet channels must have exactly one more entry than strides.")
        if any(not isinstance(value, int) or value < 1 for value in self.channels + self.strides):
            raise SegmentationError("UNet channels and strides must contain positive integers.")
        if not isinstance(self.num_res_units, int) or self.num_res_units < 0:
            raise SegmentationError("num_res_units must be a non-negative integer.")


@dataclass(frozen=True)
class VolumeGeometry:
    shape: tuple[int, int, int]
    affine: np.ndarray
    spacing: tuple[float, float, float]
    axis_codes: tuple[str | None, str | None, str | None]


@dataclass(frozen=True)
class SegmentationResult:
    output_path: Path
    mask_shape: tuple[int, int, int]
    labels: tuple[int, ...]
    model_status: str


def _optional_modules():
    try:
        import nibabel as nib
        from monai.inferers import sliding_window_inference
        from monai.transforms import (
            Compose,
            EnsureChannelFirstd,
            EnsureTyped,
            Invertd,
            LoadImaged,
            Orientationd,
            ScaleIntensityRanged,
            Spacingd,
        )
        from monai.networks.nets import UNet
    except ImportError as exc:
        raise SegmentationDependencyError(
            "3D segmentation requires the optional dependencies in "
            "requirements-segmentation.txt. Install them in a separate environment."
        ) from exc
    return nib, sliding_window_inference, Compose, EnsureChannelFirstd, EnsureTyped, Invertd, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd, UNet


def _validate_numeric_geometry(
    shape: Sequence[int],
    affine: np.ndarray,
    *,
    qform: np.ndarray | None = None,
    sform: np.ndarray | None = None,
) -> VolumeGeometry:
    if len(shape) != 3 or any(not isinstance(v, (int, np.integer)) or int(v) <= 0 for v in shape):
        raise SegmentationError("Input must be a non-empty 3D volume; 2D images and 4D volumes are unsupported.")
    if prod(int(v) for v in shape) > MAX_VOLUME_VOXELS:
        raise SegmentationError(f"Volume exceeds the supported {MAX_VOLUME_VOXELS:,}-voxel limit.")
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise GeometryError("NIfTI affine must be a finite 4×4 matrix.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise GeometryError("NIfTI affine has an invalid homogeneous-coordinate row.")
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 1e-6):
        raise GeometryError("NIfTI affine has zero or invalid voxel spacing.")
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-8:
        raise GeometryError("NIfTI affine is singular; physical orientation is ambiguous.")
    for name, candidate in (("qform", qform), ("sform", sform)):
        if candidate is not None:
            candidate = np.asarray(candidate, dtype=np.float64)
            if candidate.shape != (4, 4) or not np.isfinite(candidate).all():
                raise GeometryError(f"NIfTI {name} is malformed.")
    if qform is not None and sform is not None and not np.allclose(qform, sform, atol=1e-4, rtol=1e-5):
        raise GeometryError("NIfTI qform and sform disagree; physical geometry is ambiguous.")
    return VolumeGeometry(
        tuple(int(v) for v in shape),
        matrix.copy(),
        tuple(float(v) for v in spacing),
        (None, None, None),
    )


def validate_geometry(
    shape: Sequence[int],
    affine: np.ndarray,
    *,
    geometry_attested: bool,
    qform: np.ndarray | None = None,
    sform: np.ndarray | None = None,
) -> VolumeGeometry:
    """Validate numeric geometry and require independent confirmation of anatomy.

    Numeric affine checks cannot reveal a semantically mislabeled header such
    as the reported MSD LPS anatomy with RAS-looking affine. Consequently,
    inference callers must attest that anatomical directions were checked
    against source DICOM or an authoritative conversion before transforms run.
    """
    if not geometry_attested:
        raise GeometryError(
            "Volume orientation is not attested. Verify laterality and physical directions "
            "against source DICOM or an authoritative viewer, then explicitly attest geometry."
        )
    return _validate_numeric_geometry(shape, affine, qform=qform, sform=sform)


def inspect_nifti_geometry(input_path: str | Path) -> dict[str, Any]:
    """Inspect a NIfTI header without attesting anatomy or reading voxel data."""
    source = Path(input_path)
    if source.suffix.lower() != ".nii" and not source.name.lower().endswith(".nii.gz"):
        raise SegmentationError("Only NIfTI .nii and .nii.gz volumes are supported.")
    if not source.is_file():
        raise SegmentationError("Input NIfTI volume does not exist as a file.")
    try:
        image = nib.load(str(source))
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        geometry = _validate_numeric_geometry(
            image.shape,
            image.affine,
            qform=qform if qcode else None,
            sform=sform if scode else None,
        )
        axis_codes = nib.aff2axcodes(geometry.affine)
        if len(axis_codes) != 3 or any(code is None for code in axis_codes) or len(set(axis_codes)) != 3:
            raise GeometryError("NIfTI affine does not define three unique anatomical axis directions.")
        return {
            "shape": list(geometry.shape),
            "affine": geometry.affine.tolist(),
            "spacing_mm": list(geometry.spacing),
            "axis_codes": list(axis_codes),
            "qform": np.asarray(qform).tolist() if qcode else None,
            "sform": np.asarray(sform).tolist() if scode else None,
            "qform_code": int(qcode),
            "sform_code": int(scode),
            "datatype": str(image.header.get_data_dtype()),
            "voxel_count": int(prod(geometry.shape)),
        }
    except SegmentationError:
        raise
    except Exception as exc:
        raise SegmentationError("Input is not a readable, valid 3D NIfTI header.") from exc


def validate_volume_array(volume: np.ndarray) -> np.ndarray:
    """Validate a scalar 3D CT array and return float32 values without rescaling."""
    array = np.asarray(volume)
    if array.ndim != 3 or any(size <= 0 for size in array.shape):
        raise SegmentationError("Input must be a non-empty 3D CT volume.")
    if not np.issubdtype(array.dtype, np.number):
        raise SegmentationError("CT volume must contain numeric voxel values.")
    if not np.isfinite(array).all():
        raise SegmentationError("CT volume contains NaN or infinite voxel values.")
    return array.astype(np.float32, copy=False)


def build_unet(config: UNetConfig = UNetConfig()):
    """Create a randomly initialized MONAI UNet. This model is UNTRAINED."""
    modules = _optional_modules()
    UNet = modules[-1]
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=config.channels,
        strides=config.strides,
        num_res_units=config.num_res_units,
        norm="batch",
    )


def load_pretrained_unet(checkpoint_path: str | Path, config: UNetConfig = UNetConfig()):
    """Load a local compatible state dict; never fetches weights from a network."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise SegmentationError("Configured pretrained checkpoint does not exist as a local file.")
    model = build_unet(config)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SegmentationError("Checkpoint could not be safely loaded as a tensor-only state dict.") from exc
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    if not isinstance(payload, dict) or not payload or not all(torch.is_tensor(value) for value in payload.values()):
        raise SegmentationError("Checkpoint must contain a model state dict of tensors.")
    try:
        model.load_state_dict(payload, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise SegmentationError("Checkpoint does not match the configured 3D U-Net architecture.") from exc
    return model


def _validate_logits(logits: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    if not torch.is_tensor(logits) or logits.ndim != 5:
        raise SegmentationError("Model must return logits shaped [batch, 2, depth, height, width].")
    if logits.shape[0] != 1 or logits.shape[1] != 2 or tuple(logits.shape[2:]) != tuple(spatial_shape):
        raise SegmentationError("Model logits do not match the one-volume, two-class spatial contract.")
    if not torch.isfinite(logits).all():
        raise SegmentationError("Model returned non-finite segmentation logits.")
    return logits


def infer_preprocessed_volume(
    volume: np.ndarray,
    model: torch.nn.Module,
    inferer: Callable[..., torch.Tensor],
    *,
    device: str | torch.device = "cpu",
    roi_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
) -> torch.Tensor:
    """Run a small testable core on a single-channel preprocessed volume."""
    array = validate_volume_array(volume)
    if len(roi_size) != 3 or any(not isinstance(v, int) or v <= 0 for v in roi_size):
        raise SegmentationError("Sliding-window ROI must contain three positive integers.")
    if sw_batch_size < 1 or not 0.0 <= overlap < 1.0:
        raise SegmentationError("Invalid sliding-window batch size or overlap.")
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = inferer(inputs=tensor, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, overlap=overlap)
    logits = _validate_logits(logits, array.shape)
    probabilities = torch.softmax(logits, dim=1)
    if not torch.isfinite(probabilities).all():
        raise SegmentationError("Model produced non-finite segmentation probabilities.")
    return probabilities[0]


def probabilities_to_mask(probabilities: torch.Tensor, expected_shape: Sequence[int]) -> np.ndarray:
    if probabilities.ndim != 4 or probabilities.shape[0] != 2:
        raise SegmentationError("Restored probabilities must have shape [2, depth, height, width].")
    if tuple(probabilities.shape[1:]) != tuple(expected_shape):
        raise SegmentationError("Restored mask dimensions do not match the original NIfTI volume.")
    if not torch.isfinite(probabilities).all():
        raise SegmentationError("Restored probabilities contain non-finite values.")
    mask = probabilities.argmax(dim=0).detach().cpu().numpy().astype(np.uint8, copy=False)
    if not np.isin(mask, (0, 1)).all():
        raise SegmentationError("Segmentation mask contains labels outside background/spleen (0/1).")
    return mask


def save_mask_nifti(nib: Any, output_path: str | Path, mask: np.ndarray, geometry: VolumeGeometry, header: Any) -> None:
    """Write a discrete mask using the original NIfTI affine and header."""
    if tuple(mask.shape) != geometry.shape or mask.dtype != np.uint8:
        raise SegmentationError("Mask must be uint8 and match the original NIfTI dimensions.")
    image = nib.Nifti1Image(mask, geometry.affine, header=header.copy())
    image.set_data_dtype(np.uint8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))


def _preprocessing_transforms(modules):
    _, _, Compose, EnsureChannelFirstd, EnsureTyped, _, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd, _ = modules
    return Compose([
        LoadImaged(keys="image", image_only=False),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        Spacingd(keys="image", pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys="image", a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        # Keep MONAI's MetaTensor operation history so Invertd can restore
        # predictions to the original voxel grid after Orientationd/Spacingd.
        EnsureTyped(keys="image", dtype=torch.float32, track_meta=True),
    ])


def segment_nifti(
    input_path: str | Path,
    output_path: str | Path,
    *,
    geometry_attested: bool = False,
    orientation_review_manifest: dict[str, Any] | None = None,
    source_metadata_path: str | Path | None = None,
    conversion_record_path: str | Path | None = None,
    nifti_origin_record_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    model: torch.nn.Module | None = None,
    model_status: str | None = None,
    allow_untrained: bool = False,
    config: UNetConfig = UNetConfig(),
    device: str | torch.device = "cpu",
    roi_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
) -> SegmentationResult:
    """Segment one attested 3D CT NIfTI and save a uint8 mask in source geometry."""
    source = Path(input_path)
    target = Path(output_path)
    if source.suffix.lower() != ".nii" and not source.name.lower().endswith(".nii.gz"):
        raise SegmentationError("Only NIfTI .nii and .nii.gz volumes are supported.")
    if target.suffix.lower() != ".nii" and not target.name.lower().endswith(".nii.gz"):
        raise SegmentationError("Output path must end in .nii or .nii.gz.")
    if not source.is_file():
        raise SegmentationError("Input NIfTI volume does not exist as a file.")
    if source.resolve() == target.resolve():
        raise SegmentationError("Output mask path must differ from the source volume path.")
    if target.exists():
        raise SegmentationError("Output mask already exists; choose a new output path to avoid overwriting data.")
    if model is not None and checkpoint_path is not None:
        raise SegmentationError("Provide either a model object or a checkpoint path, not both.")
    if checkpoint_path is not None and model_status not in {None, "pretrained"}:
        raise SegmentationError("A checkpoint-backed model must have model_status='pretrained'.")
    if model is None and checkpoint_path is None and model_status == "pretrained":
        raise SegmentationError("A model cannot be marked pretrained without a local checkpoint or supplied model.")
    if model is not None and model_status not in {"pretrained", "untrained"}:
        raise SegmentationError("A supplied model object requires explicit model_status='pretrained' or 'untrained'.")
    if model_status == "untrained" and not allow_untrained:
        raise SegmentationError("Untrained model inference requires allow_untrained=True.")
    if model is None and checkpoint_path is None and not allow_untrained:
        raise SegmentationError("No pretrained checkpoint configured. Untrained inference requires allow_untrained=True.")
    modules = _optional_modules()
    nib, sliding_window_inference = modules[0], modules[1]
    try:
        original = nib.load(str(source))
        qform, qcode = original.get_qform(coded=True)
        sform, scode = original.get_sform(coded=True)
        geometry = validate_geometry(
            original.shape,
            original.affine,
            geometry_attested=geometry_attested,
            qform=qform if qcode else None,
            sform=sform if scode else None,
        )
        axis_codes = nib.aff2axcodes(geometry.affine)
        if len(axis_codes) != 3 or any(code is None for code in axis_codes) or len(set(axis_codes)) != 3:
            raise GeometryError("NIfTI affine does not define three unique anatomical axis directions.")
        geometry = VolumeGeometry(geometry.shape, geometry.affine, geometry.spacing, tuple(axis_codes))
        original_array = validate_volume_array(original.get_fdata(dtype=np.float32))
    except SegmentationError:
        raise
    except Exception as exc:
        raise SegmentationError("Input is not a readable, valid 3D NIfTI volume.") from exc
    if tuple(original_array.shape) != geometry.shape:
        raise SegmentationError("Decoded voxel shape does not match NIfTI header dimensions.")
    effective_model_status = model_status or ("pretrained" if checkpoint_path is not None else "untrained")
    if effective_model_status == "pretrained":
        if orientation_review_manifest is None:
            raise GeometryError(
                "Pretrained inference requires a trusted, signed, volume-specific orientation review; "
                "geometry_attested alone is insufficient."
            )
        if orientation_review_manifest.get("schema") == orientation_review.NIFTI_ORIGIN_ORIENTATION_REVIEW_SCHEMA:
            if nifti_origin_record_path is None or source_metadata_path is not None or conversion_record_path is not None:
                raise GeometryError(
                    "NIfTI-origin inference requires its generated origin record and must not use DICOM provenance records."
                )
            geometry_record = inspect_nifti_geometry(source)
            volume_sha256 = orientation_review.sha256_file(source)
            try:
                origin_bytes = Path(nifti_origin_record_path).read_bytes()
                origin_record = json.loads(origin_bytes.decode("utf-8"))
                orientation_review.verify_nifti_origin_orientation_review(
                    orientation_review_manifest,
                    volume_sha256,
                    geometry_record,
                    hashlib.sha256(origin_bytes).hexdigest(),
                )
                ct_provenance.verify_nifti_origin_provenance(
                    origin_record,
                    geometry_record,
                    orientation_review_manifest,
                    expected_volume_sha256=volume_sha256,
                    expected_record_sha256=hashlib.sha256(origin_bytes).hexdigest(),
                )
            except (OSError, UnicodeError, json.JSONDecodeError):
                raise GeometryError("NIfTI-origin provenance record is missing or unreadable.") from None
            except orientation_review.OrientationReviewError as exc:
                raise GeometryError(str(exc)) from None
            except ct_provenance.ProvenanceError as exc:
                raise GeometryError(str(exc)) from None
        else:
            if nifti_origin_record_path is not None:
                raise GeometryError("DICOM-origin inference cannot use a NIfTI-origin record.")
            if source_metadata_path is None or conversion_record_path is None:
                raise GeometryError(
                    "Pretrained inference also requires the actual source DICOM metadata and conversion-record files."
                )
            geometry_record = inspect_nifti_geometry(source)
            volume_sha256 = orientation_review.sha256_file(source)
            # Numeric NIfTI headers are only a geometry fingerprint. A trusted,
            # independently controlled reviewer key is required to verify the
            # source-series/conversion evidence manifest before loading weights.
            try:
                orientation_review.verify_orientation_review(
                    orientation_review_manifest,
                    volume_sha256,
                    geometry_record,
                )
            except orientation_review.OrientationReviewError as exc:
                raise GeometryError(str(exc)) from None
            try:
                ct_provenance.verify_source_provenance(
                    source, geometry_record, orientation_review_manifest,
                    source_metadata_path, conversion_record_path,
                )
            except ct_provenance.ProvenanceError as exc:
                raise GeometryError(str(exc)) from None
    if model is None:
        model = load_pretrained_unet(checkpoint_path, config) if checkpoint_path else build_unet(config)
    model_status = model_status or ("pretrained" if checkpoint_path is not None else "untrained")
    preprocess = _preprocessing_transforms(modules)
    try:
        transformed = preprocess({"image": str(source)})
        image = transformed["image"]
        probabilities = infer_preprocessed_volume(
            image[0].detach().cpu().numpy(),
            model,
            sliding_window_inference,
            device=device,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
        )
        # Invertd copies the recorded transform history from ``image`` when
        # ``pred`` is a MetaTensor. Plain tensors take a different code path
        # that expects a legacy ``image_transforms`` dictionary entry.
        # EnsureTyped(track_meta=True) makes this the MONAI MetaTensor class in
        # the real pipeline, while retaining compatibility with test adapters.
        transformed["pred"] = type(image)(probabilities)
        Invertd = modules[5]
        invert = Invertd(
            keys="pred",
            transform=preprocess,
            orig_keys="image",
            nearest_interp=False,
            to_tensor=True,
        )
        restored = invert(transformed)["pred"]
        if restored.ndim == 5 and restored.shape[0] == 1:
            restored = restored[0]
        mask = probabilities_to_mask(restored, geometry.shape)
        save_mask_nifti(nib, target, mask, geometry, original.header)
    except SegmentationError:
        raise
    except Exception as exc:
        raise RuntimeError("MONAI segmentation or geometry restoration failed; no mask was reported.") from exc
    return SegmentationResult(target, tuple(mask.shape), tuple(int(v) for v in np.unique(mask)), model_status)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run separate MONAI 3D spleen segmentation on a NIfTI CT volume.")
    parser.add_argument("input", help="Input 3D CT volume (.nii or .nii.gz)")
    parser.add_argument("output", help="Output mask (.nii or .nii.gz)")
    parser.add_argument("--checkpoint", help="Local compatible pretrained MONAI U-Net checkpoint")
    parser.add_argument("--geometry-attested", action="store_true", help="Confirm anatomy/orientation was checked against source DICOM or an authoritative viewer")
    parser.add_argument("--orientation-review", help="Path to a reviewer-signed volume-specific orientation evidence JSON manifest")
    parser.add_argument("--source-metadata", help="Path to the exact source DICOM series metadata JSON")
    parser.add_argument("--conversion-record", help="Path to the geometry-preserving NIfTI conversion record JSON")
    parser.add_argument("--nifti-origin-record", help="Path to the server-generated NIfTI-origin record JSON")
    parser.add_argument("--allow-untrained", action="store_true", help="Explicitly run random weights for software experiments only")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0")
    args = parser.parse_args(argv)
    try:
        review_manifest = None
        if args.orientation_review:
            import json
            try:
                with Path(args.orientation_review).open("r", encoding="utf-8") as stream:
                    review_manifest = json.load(stream)
            except (OSError, json.JSONDecodeError) as exc:
                parser.error(f"Cannot read orientation review JSON: {exc}")
        result = segment_nifti(
            args.input,
            args.output,
            geometry_attested=args.geometry_attested,
            orientation_review_manifest=review_manifest,
            source_metadata_path=args.source_metadata,
            conversion_record_path=args.conversion_record,
            nifti_origin_record_path=args.nifti_origin_record,
            checkpoint_path=args.checkpoint,
            allow_untrained=args.allow_untrained,
            device=args.device,
        )
    except (SegmentationError, SegmentationDependencyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Saved {result.model_status} model mask to {result.output_path}; labels={result.labels}, shape={result.mask_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
