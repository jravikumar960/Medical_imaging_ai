"""Input gates for the 2D chest X-ray research workflow.

DICOM tags are only a preliminary filter and do not authenticate pixel content.
Decoded images must also pass the fail-closed pixel-modality verifier.
"""
from __future__ import annotations

from pydicom.uid import (
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
    DigitalXRayImageStorageForProcessing,
)


class InputSuitabilityError(ValueError):
    """Raised when the available input evidence is insufficient for CXR inference."""


def require_cxr_dicom_metadata(dataset) -> None:
    """Require explicit radiography SOP class and chest-region tags; not content proof."""
    modality = str(getattr(dataset, "Modality", "")).strip().upper()
    body_part = str(getattr(dataset, "BodyPartExamined", "")).strip().upper()
    sop_class = str(getattr(dataset, "SOPClassUID", "")).strip()
    media_sop_class = str(
        getattr(getattr(dataset, "file_meta", None), "MediaStorageSOPClassUID", "")
    ).strip()
    valid_sop_classes = {
        "CR": {str(ComputedRadiographyImageStorage)},
        "DX": {
            str(DigitalXRayImageStorageForPresentation),
            str(DigitalXRayImageStorageForProcessing),
        },
    }
    if (
        modality not in valid_sop_classes
        or sop_class not in valid_sop_classes.get(modality, set())
        or media_sop_class != sop_class
        or body_part not in {"CHEST", "THORAX"}
    ):
        raise InputSuitabilityError(
            "Chest X-ray suitability cannot be established: DICOM must declare "
            "Modality CR/DX, matching CR/DX radiography SOP class values in the dataset "
            "and file meta, and "
            "BodyPartExamined CHEST/THORAX. Metadata is not proof of image content."
        )


def require_verified_cxr_modality_image(image, content_type: str | None = None):
    """Verify decoded image content with the validated modality classifier.

    This is fail-closed: a missing or invalid classifier rejects the request.
    """
    from modality_verifier import load_modality_verifier

    verifier = load_modality_verifier()
    if verifier is None:
        if content_type == "image/png" or content_type == "image/jpeg":
            raise InputSuitabilityError(
                "Chest X-ray finding inference is disabled for PNG/JPEG uploads because no "
                "validated image-modality classifier is available. File format cannot establish "
                "that pixels depict a chest X-ray; no scores or Grad-CAM were generated."
            )
        if content_type == "application/dicom":
            raise InputSuitabilityError(
                "Chest X-ray finding inference is disabled for DICOM uploads because metadata "
                "cannot prove that pixel data depict a chest X-ray and no validated "
                "image-modality classifier is available; no scores or Grad-CAM were generated."
            )
        raise InputSuitabilityError(
            "Chest X-ray finding inference is disabled until a validated image-modality "
            "classifier is available; no scores or Grad-CAM were generated."
        )

    try:
        result = verifier.verify(image)
    except (TypeError, ValueError, RuntimeError):
        raise InputSuitabilityError(
            "The modality classifier produced an invalid result; no scores or "
            "Grad-CAM were generated."
        ) from None
    if not result.is_cxr:
        raise InputSuitabilityError(
            "The modality verifier did not confirm this image as a chest X-ray "
            f"(CXR score {result.score:.6f}; required threshold {result.threshold:.6f}). "
            "The verifier score is not a calibrated probability. No finding scores or "
            "Grad-CAM were generated."
        )
    return result
