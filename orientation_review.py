"""Validation for reviewer-signed, volume-specific CT orientation evidence.

This verifies evidence-manifest integrity and binding. It cannot prove that the
reviewer's claims or cited source records are truthful.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any


ORIENTATION_REVIEW_SCHEMA = "medical-imaging-ct-orientation-review/v1"
NIFTI_ORIGIN_ORIENTATION_REVIEW_SCHEMA = "medical-imaging-nifti-origin-orientation-review/v1"
MAX_ORIENTATION_REVIEW_VALIDITY = timedelta(days=7)


class OrientationReviewError(ValueError):
    """A missing, invalid, expired, or mismatched independent review record."""


class OrientationReviewKeyUnavailable(OrientationReviewError):
    """No trusted reviewer verification key is configured by the service owner."""


def review_key_from_environment() -> bytes:
    configured = os.environ.get("CT_ORIENTATION_REVIEW_HMAC_KEY", "")
    return _validate_review_key(configured.encode("utf-8"))


def _validate_review_key(key: bytes) -> bytes:
    if not isinstance(key, bytes) or len(key) < 32:
        raise OrientationReviewKeyUnavailable(
            "Trusted CT orientation review is not configured. Set the reviewer-controlled HMAC key before inference."
        )
    return key


def geometry_sha256(geometry: dict[str, Any]) -> str:
    canonical = json.dumps(geometry, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def review_signing_bytes(review: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in review.items() if key != "signature_hmac_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return canonical.encode("utf-8")


def _review_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise OrientationReviewError(f"Orientation review {field} must be an ISO-8601 timestamp.")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result.astimezone(timezone.utc)
    except ValueError:
        raise OrientationReviewError(f"Orientation review {field} must include a timezone.") from None


def verify_orientation_review(
    review: dict[str, Any],
    file_sha256: str,
    geometry: dict[str, Any],
    key: bytes | None = None,
) -> None:
    """Require a trusted signature and evidence bound to one file and geometry."""
    if not isinstance(review, dict):
        raise OrientationReviewError("Orientation review manifest must be a JSON object.")
    key = _validate_review_key(key) if key is not None else review_key_from_environment()
    signature = review.get("signature_hmac_sha256")
    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
        raise OrientationReviewError("Orientation review signature is missing or invalid.")
    expected_signature = hmac.new(key, review_signing_bytes(review), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise OrientationReviewError("Orientation review signature verification failed.")

    if review.get("schema") != ORIENTATION_REVIEW_SCHEMA:
        raise OrientationReviewError("Unsupported orientation review manifest schema.")
    if review.get("volume_sha256") != file_sha256:
        raise OrientationReviewError("Orientation review is for a different NIfTI file.")
    if review.get("geometry_sha256") != geometry_sha256(geometry):
        raise OrientationReviewError("Orientation review is for different NIfTI geometry.")

    for field in ("source_metadata_sha256", "conversion_record_sha256"):
        if not isinstance(review.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", review[field]):
            raise OrientationReviewError(f"Orientation review {field} must be a SHA-256 digest.")
    for field, maximum in (("series_instance_uid", 128), ("frame_of_reference_uid", 128),
                           ("conversion_software", 120), ("conversion_version", 80),
                           ("reviewer_id", 128), ("review_notes", 2000)):
        value = review.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            raise OrientationReviewError(f"Orientation review {field} is missing or invalid.")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", review["series_instance_uid"]):
        raise OrientationReviewError("Orientation review SeriesInstanceUID is malformed.")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", review["frame_of_reference_uid"]):
        raise OrientationReviewError("Orientation review FrameOfReferenceUID is malformed.")
    if review.get("laterality_verified") is not True or review.get("slice_direction_verified") is not True:
        raise OrientationReviewError("Reviewer must explicitly verify laterality and physical slice direction.")
    if review.get("nifti_axis_codes") != geometry.get("axis_codes"):
        raise OrientationReviewError("Orientation review axis codes differ from the inspected NIfTI geometry.")

    orientation = review.get("image_orientation_patient")
    first_position = review.get("first_image_position_patient")
    last_position = review.get("last_image_position_patient")
    if (not isinstance(orientation, list) or len(orientation) != 6 or
            not isinstance(first_position, list) or len(first_position) != 3 or
            not isinstance(last_position, list) or len(last_position) != 3):
        raise OrientationReviewError("Orientation review must include DICOM direction cosines and first/last slice positions.")
    try:
        numbers = [float(value) for value in orientation + first_position + last_position]
    except (TypeError, ValueError):
        numbers = []
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise OrientationReviewError("Orientation review DICOM geometry must contain finite numbers.")
    row = numbers[0:3]
    column = numbers[3:6]
    row_norm = math.sqrt(sum(value * value for value in row))
    column_norm = math.sqrt(sum(value * value for value in column))
    dot = sum(row[index] * column[index] for index in range(3))
    if abs(row_norm - 1.0) > 0.02 or abs(column_norm - 1.0) > 0.02 or abs(dot) > 0.02:
        raise OrientationReviewError("Orientation review DICOM direction cosines are not orthonormal.")
    normal = [row[1] * column[2] - row[2] * column[1],
              row[2] * column[0] - row[0] * column[2],
              row[0] * column[1] - row[1] * column[0]]
    delta = [numbers[9 + index] - numbers[6 + index] for index in range(3)]
    slice_projection = sum(delta[index] * normal[index] for index in range(3))
    order = review.get("slice_order")
    if abs(slice_projection) <= 1e-3 or order not in {"ascending", "descending"}:
        raise OrientationReviewError("Orientation review slice positions/order are insufficient to establish direction.")
    if (order == "ascending" and slice_projection < 0) or (order == "descending" and slice_projection > 0):
        raise OrientationReviewError("Orientation review slice order conflicts with the supplied DICOM positions.")

    reviewed_at = _review_time(review.get("reviewed_at"), "reviewed_at")
    expires_at = _review_time(review.get("expires_at"), "expires_at")
    now = datetime.now(timezone.utc)
    if reviewed_at > now + timedelta(minutes=5) or expires_at <= now:
        raise OrientationReviewError("Orientation review is stale or has a future review time.")
    if expires_at <= reviewed_at or expires_at - reviewed_at > MAX_ORIENTATION_REVIEW_VALIDITY:
        raise OrientationReviewError("Orientation review expiry must be within seven days of review.")


def verify_nifti_origin_orientation_review(
    review: dict[str, Any],
    file_sha256: str,
    geometry: dict[str, Any],
    origin_record_sha256: str,
    key: bytes | None = None,
) -> None:
    """Verify independent human orientation attestation for a NIfTI-origin record."""
    if not isinstance(review, dict):
        raise OrientationReviewError("NIfTI-origin orientation review must be a JSON object.")
    key = _validate_review_key(key) if key is not None else review_key_from_environment()
    signature = review.get("signature_hmac_sha256")
    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
        raise OrientationReviewError("NIfTI-origin orientation review signature is missing or invalid.")
    expected_signature = hmac.new(key, review_signing_bytes(review), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise OrientationReviewError("NIfTI-origin orientation review signature verification failed.")
    if review.get("schema") != NIFTI_ORIGIN_ORIENTATION_REVIEW_SCHEMA:
        raise OrientationReviewError("Unsupported NIfTI-origin orientation review schema.")
    if set(review) != {
        "schema", "origin_record_sha256", "volume_sha256", "geometry_sha256",
        "nifti_axis_codes", "reviewer_id", "reviewer_independent",
        "laterality_verified", "slice_direction_verified", "physical_slice_direction",
        "orientation_reference", "orientation_method", "reviewed_at", "expires_at",
        "review_notes", "signature_hmac_sha256",
    }:
        raise OrientationReviewError("NIfTI-origin review contains unsupported or DICOM-origin fields.")
    if review.get("volume_sha256") != file_sha256:
        raise OrientationReviewError("NIfTI-origin orientation review is for a different NIfTI file.")
    if review.get("geometry_sha256") != geometry_sha256(geometry):
        raise OrientationReviewError("NIfTI-origin orientation review is for different NIfTI geometry.")
    if not isinstance(origin_record_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", origin_record_sha256):
        raise OrientationReviewError("NIfTI-origin record SHA-256 is missing or invalid.")
    if review.get("origin_record_sha256") != origin_record_sha256:
        raise OrientationReviewError("NIfTI-origin orientation review is for a different provenance record.")
    if review.get("nifti_axis_codes") != geometry.get("axis_codes"):
        raise OrientationReviewError("NIfTI-origin orientation review axis codes differ from inspected geometry.")
    for field, maximum in (
        ("reviewer_id", 128),
        ("orientation_reference", 2000),
        ("orientation_method", 2000),
        ("review_notes", 2000),
        ("physical_slice_direction", 80),
    ):
        value = review.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            raise OrientationReviewError(f"NIfTI-origin orientation review {field} is missing or invalid.")
    if review.get("reviewer_independent") is not True:
        raise OrientationReviewError("An independent human reviewer must attest to independence.")
    if review.get("laterality_verified") is not True or review.get("slice_direction_verified") is not True:
        raise OrientationReviewError("Reviewer must explicitly verify laterality and physical slice direction.")
    if review.get("physical_slice_direction") not in {"inferior_to_superior", "superior_to_inferior"}:
        raise OrientationReviewError("Physical slice direction must be explicitly reviewed.")

    reviewed_at = _review_time(review.get("reviewed_at"), "reviewed_at")
    expires_at = _review_time(review.get("expires_at"), "expires_at")
    now = datetime.now(timezone.utc)
    if reviewed_at > now + timedelta(minutes=5) or expires_at <= now:
        raise OrientationReviewError("NIfTI-origin orientation review is stale or has a future review time.")
    if expires_at <= reviewed_at or expires_at - reviewed_at > MAX_ORIENTATION_REVIEW_VALIDITY:
        raise OrientationReviewError("NIfTI-origin review expiry must be within seven days of review.")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sign_orientation_review(review: dict[str, Any], key: bytes | None = None) -> dict[str, Any]:
    """Sign an already human-verified orientation review manifest."""
    signing_key = _validate_review_key(key) if key is not None else review_key_from_environment()
    signature = hmac.new(signing_key, review_signing_bytes(review), hashlib.sha256).hexdigest()
    review["signature_hmac_sha256"] = signature
    return review


def _sorted_source_slices(source_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    slices = source_metadata.get("slices")
    if not isinstance(slices, list) or not slices:
        raise OrientationReviewError("Source metadata does not contain CT slices.")
    orientation = slices[0].get("image_orientation_patient")
    try:
        orientation_values = [float(value) for value in orientation]
    except (TypeError, ValueError):
        raise OrientationReviewError("Source metadata has invalid ImageOrientationPatient.") from None
    if len(orientation_values) != 6:
        raise OrientationReviewError("Source metadata has invalid ImageOrientationPatient.")
    row_cosine = orientation_values[:3]
    column_cosine = orientation_values[3:]
    normal = [
        row_cosine[1] * column_cosine[2] - row_cosine[2] * column_cosine[1],
        row_cosine[2] * column_cosine[0] - row_cosine[0] * column_cosine[2],
        row_cosine[0] * column_cosine[1] - row_cosine[1] * column_cosine[0],
    ]
    def projection(item: dict[str, Any]) -> float:
        position = item.get("image_position_patient")
        try:
            values = [float(value) for value in position]
        except (TypeError, ValueError) as exc:
            raise OrientationReviewError("Source metadata has invalid ImagePositionPatient.") from exc
        if len(values) != 3:
            raise OrientationReviewError("Source metadata has invalid ImagePositionPatient.")
        return sum(values[axis] * normal[axis] for axis in range(3))
    return sorted(slices, key=projection)


def prepare_orientation_review(
    *,
    nifti_path: str | os.PathLike[str],
    geometry: dict[str, Any],
    source_metadata_path: str | os.PathLike[str],
    conversion_record_path: str | os.PathLike[str],
    reviewer_id: str,
    laterality_verified: bool = False,
    slice_direction_verified: bool = False,
    review_notes: str = "",
    reviewed_at: str | None = None,
    expires_at: str | None = None,
    sign: bool = False,
    signing_key: bytes | None = None,
) -> dict[str, Any]:
    """Prepare reviewer orientation evidence without ever auto-approving it.

    ``laterality_verified`` and ``slice_direction_verified`` must be explicitly
    supplied as ``True`` by a human reviewer.  The helper never changes a false
    value to true and never invents source metadata, hashes, or signatures.
    """
    if laterality_verified is not True or slice_direction_verified is not True:
        raise OrientationReviewError(
            "Reviewer must explicitly verify anatomy/laterality and physical slice direction."
        )
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise OrientationReviewError("Orientation review reviewer_id is required.")
    if not isinstance(review_notes, str) or not review_notes.strip():
        raise OrientationReviewError("Orientation review notes are required.")

    source_path = Path(source_metadata_path)
    conversion_path = Path(conversion_record_path)
    source_metadata, _ = _read_json_like(source_path)
    conversion_record, _ = _read_json_like(conversion_path)

    series_uid = source_metadata.get("series_instance_uid")
    frame_uid = source_metadata.get("frame_of_reference_uid")
    if not isinstance(series_uid, str) or not series_uid.strip():
        raise OrientationReviewError("Source metadata is missing SeriesInstanceUID.")
    if not isinstance(frame_uid, str) or not frame_uid.strip():
        raise OrientationReviewError("Source metadata is missing FrameOfReferenceUID.")

    sorted_slices = _sorted_source_slices(source_metadata)
    conversion_order = conversion_record.get("source_sop_instance_uids_by_slice_index")
    sorted_uids = [item.get("sop_instance_uid") for item in sorted_slices]
    if conversion_order == sorted_uids:
        slice_order = "ascending"
    elif conversion_order == list(reversed(sorted_uids)):
        slice_order = "descending"
    else:
        raise OrientationReviewError("Conversion slice order does not match source DICOM slices.")
    if slice_order == "ascending":
        first_slice, last_slice = sorted_slices[0], sorted_slices[-1]
    else:
        first_slice, last_slice = sorted_slices[-1], sorted_slices[0]

    def position_values(item: dict[str, Any], field: str) -> list[float]:
        try:
            values = [float(value) for value in item.get(field)]
        except (TypeError, ValueError) as exc:
            raise OrientationReviewError(f"Invalid {field} in source metadata.") from exc
        if len(values) != 3:
            raise OrientationReviewError(f"Invalid {field} in source metadata.")
        return values

    orientation_values = [float(value) for value in first_slice.get("image_orientation_patient")]
    if len(orientation_values) != 6:
        raise OrientationReviewError("Invalid ImageOrientationPatient in source metadata.")
    first_position = position_values(first_slice, "image_position_patient")
    last_position = position_values(last_slice, "image_position_patient")

    now = datetime.now(timezone.utc)
    if reviewed_at is None:
        reviewed_at_value = now
    else:
        reviewed_at_value = _review_time(reviewed_at, "reviewed_at")
    if expires_at is None:
        expires_at_value = reviewed_at_value + timedelta(days=7)
    else:
        expires_at_value = _review_time(expires_at, "expires_at")
    reviewed_at_text = reviewed_at_value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    expires_at_text = expires_at_value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    volume_sha = sha256_file(nifti_path)
    geometry_sha = geometry_sha256(geometry)
    source_metadata_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    conversion_record_sha = hashlib.sha256(conversion_path.read_bytes()).hexdigest()
    axis_codes = geometry.get("axis_codes")
    if not isinstance(axis_codes, list) or len(axis_codes) != 3:
        raise OrientationReviewError("Geometry axis codes are required.")

    review = {
        "schema": ORIENTATION_REVIEW_SCHEMA,
        "volume_sha256": volume_sha,
        "geometry_sha256": geometry_sha,
        "source_metadata_sha256": source_metadata_sha,
        "conversion_record_sha256": conversion_record_sha,
        "series_instance_uid": series_uid,
        "frame_of_reference_uid": frame_uid,
        "image_orientation_patient": orientation_values,
        "first_image_position_patient": first_position,
        "last_image_position_patient": last_position,
        "slice_order": slice_order,
        "conversion_software": conversion_record.get("conversion_software"),
        "conversion_version": conversion_record.get("conversion_version"),
        "reviewer_id": reviewer_id.strip(),
        "reviewed_at": reviewed_at_text,
        "expires_at": expires_at_text,
        "laterality_verified": True,
        "slice_direction_verified": True,
        "nifti_axis_codes": axis_codes,
        "review_notes": review_notes.strip(),
    }
    # The manifest is intentionally unsigned until a reviewer-controlled key is
    # available.  ``sign=True`` is an explicit request, not an automatic approval.
    if sign:
        key_to_use = signing_key
        if key_to_use is None:
            key_to_use = review_key_from_environment()
        sign_orientation_review(review, key_to_use)
    return review

def _read_json_like(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise OrientationReviewError("Expected a JSON object.")
    return value, hashlib.sha256(raw).hexdigest()
