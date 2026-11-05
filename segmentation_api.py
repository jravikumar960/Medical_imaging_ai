"""Optional loopback FastAPI for the isolated pretrained MONAI CT workflow.

Run this module only with ``.venv-seg``. It does not load the checkpoint until
the inspection token, exact file hash and geometry, and signed review manifest
are validated. Signature verification binds supplied claims; it cannot inspect
whether supplied source records were honestly extracted or prove anatomical truth.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

import segmentation
import ct_provenance
import orientation_review


MAX_REQUEST_BYTES = 256 * 1024 * 1024
MAX_API_VOXELS = 25_000_000
INSPECTION_TTL_SECONDS = 15 * 60
ORIENTATION_REVIEW_SCHEMA = "medical-imaging-ct-orientation-review/v1"
NIFTI_ORIGIN_ORIENTATION_REVIEW_SCHEMA = "medical-imaging-nifti-origin-orientation-review/v1"
MAX_ORIENTATION_REVIEW_HEADER = 16_384
MAX_ORIENTATION_REVIEW_VALIDITY = timedelta(days=7)
MAX_PROVENANCE_BODY_BYTES = 3 * 1024 * 1024
MAX_PROVENANCE_RECORD_BYTES = 1024 * 1024
MAX_PENDING_PROVENANCE_RECORDS = 8
CT_SESSION_COOKIE = "ct_workflow_session"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "bundles" / "spleen_ct_segmentation" / "models" / "model.pt"
CHECKPOINT_BYTES = 19_297_197
CHECKPOINT_SHA256 = "502c312893994ec071a85c4f2e7d83a43f7789969faa73dae62f1291177f50fe"
_TOKEN_SECRET = os.urandom(32)
_PROVENANCE_CACHE: dict[str, dict[str, Any]] = {}
_PROVENANCE_CACHE_LOCK = threading.Lock()

app = FastAPI(title="Local MONAI Spleen CT Segmentation API", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=[
        "Content-Type", "X-Volume-Filename", "X-Geometry-Token", "X-Orientation-Review",
        "X-Provenance-Token",
    ],
)

CT_API_URL = "http://127.0.0.1:8002"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _issue_token(
    file_sha256: str, geometry: dict[str, Any], session_id: str, filename: str
) -> str:
    payload = {
        "sha256": file_sha256,
        "filename": filename,
        "geometry": geometry,
        "geometry_sha256": _geometry_sha256(geometry),
        "session_sha256": hashlib.sha256(session_id.encode("ascii")).hexdigest(),
        "inspection_id": secrets.token_urlsafe(24),
        "expires": int(time.time()) + INSPECTION_TTL_SECONDS,
    }
    encoded = _b64url(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    signature = _b64url(hmac.new(_TOKEN_SECRET, encoded.encode("ascii"), hashlib.sha256).digest())
    return f"{encoded}.{signature}"


def _geometry_sha256(geometry: dict[str, Any]) -> str:
    canonical = json.dumps(geometry, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _review_key() -> bytes | None:
    configured = os.environ.get("CT_ORIENTATION_REVIEW_HMAC_KEY", "")
    encoded = configured.encode("utf-8")
    return encoded if len(encoded) >= 32 else None


def _review_signing_bytes(review: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in review.items() if key != "signature_hmac_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return canonical.encode("utf-8")


def _decode_orientation_review(header: str) -> dict[str, Any]:
    if not header:
        raise HTTPException(
            status_code=422,
            detail="A reviewer-signed, volume-specific orientation evidence manifest is required.",
        )
    if len(header) > MAX_ORIENTATION_REVIEW_HEADER:
        raise HTTPException(status_code=413, detail="Orientation review manifest exceeds the header size limit.")
    try:
        raw = base64.urlsafe_b64decode(header + "=" * (-len(header) % 4))
        review = json.loads(raw.decode("utf-8"))
        if not isinstance(review, dict):
            raise ValueError
        return review
    except Exception:
        raise HTTPException(status_code=422, detail="Orientation review manifest is malformed or unreadable.") from None


def _parse_review_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise HTTPException(status_code=422, detail=f"Orientation review {field} must be an ISO-8601 timestamp.")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result.astimezone(timezone.utc)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Orientation review {field} must include a timezone.") from None


def _validate_orientation_review(
    review: dict[str, Any], file_sha256: str, geometry: dict[str, Any]
) -> None:
    key = _review_key()
    if key is None:
        raise HTTPException(
            status_code=503,
            detail="CT orientation review verification is not configured. Set CT_ORIENTATION_REVIEW_HMAC_KEY out of band; this shared secret verifies integrity but does not independently authenticate a reviewer.",
        )

    signature = review.get("signature_hmac_sha256")
    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
        raise HTTPException(status_code=422, detail="Orientation review signature is missing or invalid.")
    expected_signature = hmac.new(key, _review_signing_bytes(review), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=422, detail="Orientation review signature verification failed.")

    if review.get("schema") != ORIENTATION_REVIEW_SCHEMA:
        raise HTTPException(status_code=422, detail="Unsupported orientation review manifest schema.")
    if review.get("volume_sha256") != file_sha256:
        raise HTTPException(status_code=409, detail="Orientation review is for a different NIfTI file.")
    geometry_sha256 = _geometry_sha256(geometry)
    if review.get("geometry_sha256") != geometry_sha256:
        raise HTTPException(status_code=409, detail="Orientation review is for different NIfTI geometry.")

    for field in ("source_metadata_sha256", "conversion_record_sha256"):
        if not isinstance(review.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", review[field]):
            raise HTTPException(status_code=422, detail=f"Orientation review {field} must be a SHA-256 digest.")
    for field, maximum in (("series_instance_uid", 128), ("frame_of_reference_uid", 128),
                           ("conversion_software", 120), ("conversion_version", 80),
                           ("reviewer_id", 128), ("review_notes", 2000)):
        value = review.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            raise HTTPException(status_code=422, detail=f"Orientation review {field} is missing or invalid.")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", review["series_instance_uid"]):
        raise HTTPException(status_code=422, detail="Orientation review SeriesInstanceUID is malformed.")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", review["frame_of_reference_uid"]):
        raise HTTPException(status_code=422, detail="Orientation review FrameOfReferenceUID is malformed.")
    if review.get("laterality_verified") is not True or review.get("slice_direction_verified") is not True:
        raise HTTPException(status_code=422, detail="Reviewer must explicitly verify laterality and physical slice direction.")
    if review.get("nifti_axis_codes") != geometry.get("axis_codes"):
        raise HTTPException(status_code=409, detail="Orientation review axis codes differ from the inspected NIfTI geometry.")

    orientation = review.get("image_orientation_patient")
    first_position = review.get("first_image_position_patient")
    last_position = review.get("last_image_position_patient")
    if (not isinstance(orientation, list) or len(orientation) != 6 or
            not isinstance(first_position, list) or len(first_position) != 3 or
            not isinstance(last_position, list) or len(last_position) != 3):
        raise HTTPException(status_code=422, detail="Orientation review must include DICOM direction cosines and first/last slice positions.")
    try:
        numbers = [float(value) for value in orientation + first_position + last_position]
    except (TypeError, ValueError):
        numbers = []
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise HTTPException(status_code=422, detail="Orientation review DICOM geometry must contain finite numbers.")
    row = numbers[0:3]
    column = numbers[3:6]
    row_norm = math.sqrt(sum(value * value for value in row))
    column_norm = math.sqrt(sum(value * value for value in column))
    dot = sum(row[index] * column[index] for index in range(3))
    if abs(row_norm - 1.0) > 0.02 or abs(column_norm - 1.0) > 0.02 or abs(dot) > 0.02:
        raise HTTPException(status_code=422, detail="Orientation review DICOM direction cosines are not orthonormal.")
    normal = [row[1] * column[2] - row[2] * column[1],
              row[2] * column[0] - row[0] * column[2],
              row[0] * column[1] - row[1] * column[0]]
    delta = [numbers[9 + index] - numbers[6 + index] for index in range(3)]
    slice_projection = sum(delta[index] * normal[index] for index in range(3))
    order = review.get("slice_order")
    if abs(slice_projection) <= 1e-3 or order not in {"ascending", "descending"}:
        raise HTTPException(status_code=422, detail="Orientation review slice positions/order are insufficient to establish direction.")
    if (order == "ascending" and slice_projection < 0) or (order == "descending" and slice_projection > 0):
        raise HTTPException(status_code=422, detail="Orientation review slice order conflicts with the supplied DICOM positions.")

    reviewed_at = _parse_review_time(review.get("reviewed_at"), "reviewed_at")
    expires_at = _parse_review_time(review.get("expires_at"), "expires_at")
    now = datetime.now(timezone.utc)
    if reviewed_at > now + timedelta(minutes=5) or expires_at <= now:
        raise HTTPException(status_code=422, detail="Orientation review is stale or has a future review time.")
    if expires_at <= reviewed_at or expires_at - reviewed_at > MAX_ORIENTATION_REVIEW_VALIDITY:
        raise HTTPException(status_code=422, detail="Orientation review expiry must be within seven days of review.")


def _validate_nifti_origin_orientation_review(
    review: dict[str, Any],
    file_sha256: str,
    geometry: dict[str, Any],
    origin_record_sha256: str,
) -> None:
    key = _review_key()
    if key is None:
        raise HTTPException(
            status_code=503,
            detail="CT orientation review verification is not configured. Set CT_ORIENTATION_REVIEW_HMAC_KEY out of band; this shared secret verifies integrity but does not independently authenticate a reviewer.",
        )
    try:
        orientation_review.verify_nifti_origin_orientation_review(
            review, file_sha256, geometry, origin_record_sha256, key=key
        )
    except orientation_review.OrientationReviewError as exc:
        message = str(exc)
        status = 409 if "different NIfTI" in message or "different provenance record" in message else 422
        raise HTTPException(status_code=status, detail=message) from None


async def _read_bounded_body(request: Request, limit: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > limit:
                raise HTTPException(status_code=413, detail="Provenance upload exceeds the configured request limit.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid provenance request size.") from None
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail="Provenance upload exceeds the configured request limit.")
        chunks.append(chunk)
    if not size:
        raise HTTPException(status_code=400, detail="Provenance request body is empty.")
    return b"".join(chunks)


def _decode_record(payload: Any, field: str) -> bytes:
    if not isinstance(payload, str):
        raise HTTPException(status_code=422, detail=f"{field} must be base64-encoded JSON file bytes.")
    if len(payload) > ((MAX_PROVENANCE_RECORD_BYTES + 2) // 3) * 4:
        raise HTTPException(status_code=413, detail=f"{field} exceeds the configured decoded file limit.")
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception:
        raise HTTPException(status_code=422, detail=f"{field} is not valid base64.") from None
    if not raw or len(raw) > MAX_PROVENANCE_RECORD_BYTES:
        raise HTTPException(status_code=413 if raw else 422, detail=f"{field} is empty or exceeds the configured decoded file limit.")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise HTTPException(status_code=422, detail=f"{field} must contain UTF-8 JSON.") from None
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail=f"{field} must contain a JSON object.")
    return raw


def _review_digest(review: dict[str, Any]) -> str:
    return hashlib.sha256(_review_signing_bytes(review) + review.get("signature_hmac_sha256", "").encode("ascii")).hexdigest()


def _cleanup_provenance_cache() -> None:
    now = time.time()
    with _PROVENANCE_CACHE_LOCK:
        for key, entry in list(_PROVENANCE_CACHE.items()):
            if entry["expires"] <= now:
                _PROVENANCE_CACHE.pop(key)


def _store_provenance_evidence(entry: dict[str, Any]) -> str:
    _cleanup_provenance_cache()
    with _PROVENANCE_CACHE_LOCK:
        if len(_PROVENANCE_CACHE) >= MAX_PENDING_PROVENANCE_RECORDS:
            raise HTTPException(status_code=429, detail="Too many pending CT evidence uploads; retry after existing tokens expire.")
        token = secrets.token_urlsafe(32)
        _PROVENANCE_CACHE[token] = entry
    return token


def _take_provenance_evidence(
    token: str,
    file_sha256: str,
    geometry_sha256: str,
    review: dict[str, Any],
    session_sha256: str,
    inspection_token: str,
    provenance_type: str = "dicom",
) -> dict[str, Any]:
    _cleanup_provenance_cache()
    with _PROVENANCE_CACHE_LOCK:
        entry = _PROVENANCE_CACHE.get(token)
        if entry is None:
            raise HTTPException(status_code=422, detail="CT provenance token is missing, expired, or already used.")
        if entry["expires"] <= time.time():
            _PROVENANCE_CACHE.pop(token)
            raise HTTPException(status_code=422, detail="CT provenance token is missing, expired, or already used.")
        if (entry["file_sha256"] != file_sha256 or entry["geometry_sha256"] != geometry_sha256 or
                entry["review_sha256"] != _review_digest(review) or
                entry["session_sha256"] != session_sha256 or
                entry["inspection_token_sha256"] != hashlib.sha256(inspection_token.encode("ascii")).hexdigest() or
                entry.get("provenance_type", "dicom") != provenance_type):
            raise HTTPException(status_code=409, detail="CT provenance token is bound to a different session, inspection, file, geometry, or review.")
        return _PROVENANCE_CACHE.pop(token)


def _read_token(token: str) -> dict[str, Any]:
    try:
        if not isinstance(token, str) or len(token) > 4096:
            raise ValueError
        encoded, supplied_signature = token.split(".", 1)
        expected_signature = _b64url(hmac.new(_TOKEN_SECRET, encoded.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise ValueError
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if int(payload["expires"]) <= int(time.time()):
            raise ValueError
        if not re.fullmatch(r"[0-9a-f]{64}", payload["sha256"]):
            raise ValueError
        if not re.fullmatch(r"[0-9a-f]{64}", payload["geometry_sha256"]):
            raise ValueError
        if not re.fullmatch(r"[0-9a-f]{64}", payload["session_sha256"]):
            raise ValueError
        if not isinstance(payload.get("inspection_id"), str) or not payload["inspection_id"]:
            raise ValueError
        if not isinstance(payload.get("geometry"), dict) or not hmac.compare_digest(
            _geometry_sha256(payload["geometry"]), payload["geometry_sha256"]
        ):
            raise ValueError
        return payload
    except Exception:
        raise HTTPException(status_code=422, detail="CT inspection token is invalid or expired. Inspect the same file again.") from None


def _session_id_from_cookie(request: Request, *, create: bool = False) -> str:
    session_id = request.cookies.get(CT_SESSION_COOKIE, "")
    if re.fullmatch(r"[A-Za-z0-9_-]{43}", session_id):
        return session_id
    if create:
        return secrets.token_urlsafe(32)
    raise HTTPException(status_code=403, detail="CT workflow session is missing or invalid. Inspect the volume again in this browser session.")


def _validate_token_session(payload: dict[str, Any], session_id: str) -> str:
    session_sha256 = hashlib.sha256(session_id.encode("ascii")).hexdigest()
    if not hmac.compare_digest(session_sha256, payload["session_sha256"]):
        raise HTTPException(status_code=403, detail="CT inspection token belongs to a different browser session.")
    return session_sha256


def _validate_filename(filename: str) -> str:
    if not filename or len(filename) > 255 or "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Provide a simple NIfTI filename without a path.")
    lower = filename.lower()
    if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
        raise HTTPException(status_code=415, detail="CT input must be a 3D NIfTI .nii or .nii.gz volume.")
    return "volume.nii.gz" if lower.endswith(".nii.gz") else "volume.nii"


async def _stage_request(request: Request, filename: str, directory: Path) -> tuple[Path, str, int]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in {"application/octet-stream", "application/nifti", "application/gzip", "application/x-nifti"}:
        raise HTTPException(status_code=415, detail="Send the NIfTI file as a binary request body.")
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="Compressed CT volume exceeds the 256 MiB request limit.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid request size.") from None

    target = directory / filename
    digest = hashlib.sha256()
    size = 0
    try:
        with target.open("wb") as output:
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_REQUEST_BYTES:
                    raise HTTPException(status_code=413, detail="Compressed CT volume exceeds the 256 MiB request limit.")
                digest.update(chunk)
                output.write(chunk)
    except HTTPException:
        raise
    except OSError:
        raise HTTPException(status_code=500, detail="Could not stage the CT volume for temporary processing.") from None
    if size == 0:
        raise HTTPException(status_code=400, detail="The uploaded CT volume is empty.")
    return target, digest.hexdigest(), size


def _inspect_path(path: Path) -> dict[str, Any]:
    try:
        geometry = segmentation.inspect_nifti_geometry(path)
    except segmentation.SegmentationDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except segmentation.SegmentationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if geometry["voxel_count"] > MAX_API_VOXELS:
        raise HTTPException(status_code=413, detail=f"CT volume exceeds the API limit of {MAX_API_VOXELS:,} voxels.")
    return geometry


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "workflow": "pretrained-spleen-ct",
        "checkpoint_present": CHECKPOINT_PATH.is_file(),
        "trusted_orientation_review_configured": _review_key() is not None,
    }


@app.post("/ct/inspect")
async def inspect_ct(request: Request, response: Response):
    session_id = _session_id_from_cookie(request, create=True)
    requested_filename = request.headers.get("x-volume-filename", "")
    staged_name = _validate_filename(requested_filename)
    with tempfile.TemporaryDirectory(prefix="medical-imaging-ct-") as temp_directory:
        path, digest, size = await _stage_request(request, staged_name, Path(temp_directory))
        geometry = _inspect_path(path)
        response.set_cookie(
            CT_SESSION_COOKIE,
            session_id,
            max_age=INSPECTION_TTL_SECONDS,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            path="/",
        )
        return {
            "status": "inspection_only",
            "file_sha256": digest,
            "file_bytes": size,
            "geometry": geometry,
            "attestation_token": _issue_token(digest, geometry, session_id, requested_filename),
            "geometry_sha256": _geometry_sha256(geometry),
            "attestation_required": True,
            "orientation_verified": False,
            "model_loaded": False,
            "note": "Numeric header checks do not establish anatomical orientation. No inference was run.",
        }


@app.post("/ct/evidence")
async def upload_ct_evidence(request: Request):
    """Bind bounded metadata/conversion JSON files to a prior inspection token."""
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
        raise HTTPException(status_code=415, detail="CT evidence must be sent as an application/json request.")
    body = await _read_bounded_body(request, MAX_PROVENANCE_BODY_BYTES)
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise HTTPException(status_code=422, detail="CT evidence request must be UTF-8 JSON.") from None
    if not isinstance(payload, dict) or set(payload) != {"source_metadata_base64", "conversion_record_base64"}:
        raise HTTPException(status_code=422, detail="CT evidence requires exactly source_metadata_base64 and conversion_record_base64.")

    inspection_token = request.headers.get("x-geometry-token", "")
    if not inspection_token:
        raise HTTPException(status_code=422, detail="Inspect this exact NIfTI volume before uploading provenance records.")
    signed_inspection = _read_token(inspection_token)
    session_id = _session_id_from_cookie(request)
    session_sha256 = _validate_token_session(signed_inspection, session_id)
    review = _decode_orientation_review(request.headers.get("x-orientation-review", ""))
    _validate_orientation_review(review, signed_inspection["sha256"], signed_inspection["geometry"])
    source_bytes = _decode_record(payload["source_metadata_base64"], "source_metadata_base64")
    conversion_bytes = _decode_record(payload["conversion_record_base64"], "conversion_record_base64")

    directory = Path(tempfile.mkdtemp(prefix="medical-imaging-ct-evidence-check-"))
    metadata_path = directory / "source_metadata.json"
    conversion_path = directory / "conversion_record.json"
    try:
        metadata_path.write_bytes(source_bytes)
        conversion_path.write_bytes(conversion_bytes)
        ct_provenance.verify_source_provenance(
            None,
            signed_inspection["geometry"],
            review,
            metadata_path,
            conversion_path,
            expected_nifti_sha256=signed_inspection["sha256"],
        )
        expires = min(time.time() + INSPECTION_TTL_SECONDS, float(signed_inspection["expires"]))
        provenance_token = _store_provenance_evidence({
            "source_metadata_bytes": source_bytes,
            "conversion_record_bytes": conversion_bytes,
            "provenance_type": "dicom",
            "file_sha256": signed_inspection["sha256"],
            "geometry_sha256": signed_inspection["geometry_sha256"],
            "review_sha256": _review_digest(review),
            "session_sha256": session_sha256,
            "inspection_token_sha256": hashlib.sha256(inspection_token.encode("ascii")).hexdigest(),
            "expires": expires,
        })
        return {
            "status": "provenance_checked",
            "provenance_token": provenance_token,
            "expires_at_unix": int(expires),
            "warning": "Record hashes and geometric consistency were checked. This does not establish source authenticity or independently prove anatomical orientation.",
        }
    except ct_provenance.ProvenanceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    finally:
        shutil.rmtree(directory, ignore_errors=True)


async def _nifti_origin_request(
    request: Request,
) -> tuple[dict[str, Any], bytes, dict[str, Any], str, str]:
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
        raise HTTPException(status_code=415, detail="NIfTI-origin evidence must be sent as application/json.")
    body = await _read_bounded_body(request, MAX_PROVENANCE_BODY_BYTES)
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise HTTPException(status_code=422, detail="NIfTI-origin request must be UTF-8 JSON.") from None
    if not isinstance(payload, dict) or set(payload) != {"dataset_json_base64", "image_path"}:
        raise HTTPException(status_code=422, detail="NIfTI-origin request requires exactly dataset_json_base64 and image_path.")
    dataset_bytes = _decode_record(payload["dataset_json_base64"], "dataset_json_base64")
    image_path = payload["image_path"]
    if not isinstance(image_path, str):
        raise HTTPException(status_code=422, detail="NIfTI-origin image_path must be a string.")
    token = request.headers.get("x-geometry-token", "")
    if not token:
        raise HTTPException(status_code=422, detail="Inspect this exact NIfTI volume before creating origin evidence.")
    inspection = _read_token(token)
    session_id = _session_id_from_cookie(request)
    session_sha256 = _validate_token_session(inspection, session_id)
    if inspection.get("filename") != "spleen_1.nii.gz":
        raise HTTPException(status_code=422, detail="NIfTI-origin Task09 evidence requires the inspected file spleen_1.nii.gz.")
    try:
        record = ct_provenance.build_nifti_origin_record(
            dataset_bytes, image_path, inspection["sha256"], inspection["geometry"]
        )
    except ct_provenance.ProvenanceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    record_bytes = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    return record, record_bytes, inspection, session_sha256, record_sha256


@app.post("/ct/nifti-origin/record")
async def create_nifti_origin_record(request: Request):
    """Create a Task09 NIfTI-origin record from dataset.json and server inspection."""
    record, _record_bytes, inspection, _session_sha256, record_sha256 = await _nifti_origin_request(request)
    return {
        "status": "nifti_origin_record_created",
        "provenance_type": "nifti_origin",
        "origin_record": record,
        "origin_record_sha256": record_sha256,
        "expires_at_unix": int(inspection["expires"]),
        "warning": "Dataset declaration and exact NIfTI identity are bound. Original DICOM metadata and conversion history were unavailable; header axis codes do not verify anatomical orientation.",
    }


@app.post("/ct/nifti-origin/evidence")
async def upload_nifti_origin_evidence(request: Request):
    """Verify a separately signed review and issue a one-use NIfTI-origin token."""
    record, record_bytes, inspection, session_sha256, record_sha256 = await _nifti_origin_request(request)
    review = _decode_orientation_review(request.headers.get("x-orientation-review", ""))
    _validate_nifti_origin_orientation_review(
        review, inspection["sha256"], inspection["geometry"], record_sha256
    )
    geometry_sha256 = inspection["geometry_sha256"]
    try:
        ct_provenance.verify_nifti_origin_provenance(
            record,
            inspection["geometry"],
            review,
            expected_volume_sha256=inspection["sha256"],
            expected_record_sha256=record_sha256,
        )
    except ct_provenance.ProvenanceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    expires = min(time.time() + INSPECTION_TTL_SECONDS, float(inspection["expires"]))
    provenance_token = _store_provenance_evidence({
        "nifti_origin_record_bytes": record_bytes,
        "provenance_type": "nifti_origin",
        "file_sha256": inspection["sha256"],
        "geometry_sha256": geometry_sha256,
        "review_sha256": _review_digest(review),
        "session_sha256": session_sha256,
        "inspection_token_sha256": hashlib.sha256(
            request.headers["x-geometry-token"].encode("ascii")
        ).hexdigest(),
        "expires": expires,
    })
    return {
        "status": "provenance_checked",
        "provenance_type": "nifti_origin",
        "provenance_token": provenance_token,
        "origin_record_sha256": record_sha256,
        "expires_at_unix": int(expires),
        "warning": "The dataset manifest declaration, server-inspected geometry, exact volume hash, and signed human review were bound. This does not independently authenticate dataset membership or reviewer identity.",
    }


def run_pretrained_segmentation(
    input_path: Path, output_path: Path, orientation_review_manifest: dict[str, Any],
    source_metadata_path: Path | None = None, conversion_record_path: Path | None = None,
    nifti_origin_record_path: Path | None = None,
) -> dict[str, Any]:
    """Execute only the fixed local checkpoint and package validated slice arrays."""
    if nifti_origin_record_path is None and (source_metadata_path is None or conversion_record_path is None):
        raise segmentation.GeometryError(
            "Verified source metadata and conversion-record files are required before pretrained inference."
        )
    if nifti_origin_record_path is not None and (
        source_metadata_path is not None or conversion_record_path is not None
    ):
        raise segmentation.GeometryError("Use either DICOM provenance records or one NIfTI-origin record, not both.")
    if not CHECKPOINT_PATH.is_file() or CHECKPOINT_PATH.stat().st_size != CHECKPOINT_BYTES:
        raise segmentation.SegmentationError("The verified local spleen checkpoint is missing or has an unexpected size.")
    checkpoint_hash = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise segmentation.SegmentationError("The local spleen checkpoint integrity check failed.")

    result = segmentation.segment_nifti(
        input_path,
        output_path,
        geometry_attested=True,
        orientation_review_manifest=orientation_review_manifest,
        source_metadata_path=source_metadata_path,
        conversion_record_path=conversion_record_path,
        nifti_origin_record_path=nifti_origin_record_path,
        checkpoint_path=CHECKPOINT_PATH,
        model_status="pretrained",
        device="cpu",
    )
    modules = segmentation._optional_modules()
    nib = modules[0]
    original = nib.load(str(input_path))
    mask_image = nib.load(str(result.output_path))
    if tuple(mask_image.shape) != tuple(original.shape) or not np.allclose(mask_image.affine, original.affine, atol=1e-5, rtol=1e-5):
        raise segmentation.SegmentationError("Exported mask geometry does not match the uploaded CT volume.")

    volume = segmentation.validate_volume_array(np.asarray(original.get_fdata(dtype=np.float32)))
    mask = np.asarray(mask_image.dataobj)
    if tuple(mask.shape) != tuple(original.shape) or not np.isin(mask, (0, 1)).all():
        raise segmentation.SegmentationError("Exported mask has invalid dimensions or labels.")
    preview = np.rint(np.clip((volume + 57.0) / 221.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    preview_gzip = gzip.compress(np.asfortranarray(preview).tobytes(order="F"), mtime=0)
    mask_gzip = gzip.compress(np.asfortranarray(mask.astype(np.uint8)).tobytes(order="F"), mtime=0)
    return {
        "volume_preview_gzip_base64": base64.b64encode(preview_gzip).decode("ascii"),
        "mask_gzip_base64": base64.b64encode(mask_gzip).decode("ascii"),
        "mask_sha256": hashlib.sha256(mask_gzip).hexdigest(),
        "shape": list(original.shape),
        "affine": np.asarray(original.affine, dtype=np.float64).tolist(),
        "model_status": result.model_status,
        "labels": list(result.labels),
        "window": {"center": 53.5, "width": 221},
    }


@app.post("/ct/segment")
async def segment_ct(request: Request):
    token = request.headers.get("x-geometry-token", "")
    review = _decode_orientation_review(request.headers.get("x-orientation-review", ""))
    provenance_token = request.headers.get("x-provenance-token", "")
    if not token:
        raise HTTPException(status_code=422, detail="Inspect this exact CT file before submitting signed orientation review evidence.")
    signed_inspection = _read_token(token)
    session_id = _session_id_from_cookie(request)
    session_sha256 = _validate_token_session(signed_inspection, session_id)
    requested_filename = request.headers.get("x-volume-filename", "")
    staged_name = _validate_filename(requested_filename)
    with tempfile.TemporaryDirectory(prefix="medical-imaging-ct-") as temp_directory:
        directory = Path(temp_directory)
        input_path, digest, _size = await _stage_request(request, staged_name, directory)
        if not hmac.compare_digest(digest, signed_inspection["sha256"]):
            raise HTTPException(status_code=409, detail="Uploaded CT bytes differ from the inspected file. Inspect the selected file again.")
        geometry = _inspect_path(input_path)
        geometry_hash = _geometry_sha256(geometry)
        if not hmac.compare_digest(geometry_hash, signed_inspection["geometry_sha256"]):
            raise HTTPException(status_code=409, detail="CT geometry changed after inspection. Inspect the selected file again.")
        if review.get("schema") == NIFTI_ORIGIN_ORIENTATION_REVIEW_SCHEMA:
            provenance_type = "nifti_origin"
            if signed_inspection.get("filename") != requested_filename:
                raise HTTPException(status_code=409, detail="NIfTI-origin filename differs from the inspected Task09 image.")
            _validate_nifti_origin_orientation_review(
                review, digest, geometry, review.get("origin_record_sha256")
            )
        else:
            provenance_type = "dicom"
            _validate_orientation_review(review, digest, geometry)
        if not provenance_token:
            if provenance_type == "nifti_origin":
                raise HTTPException(status_code=422, detail="Generate and verify NIfTI-origin evidence before segmentation.")
            raise HTTPException(status_code=422, detail="Upload and verify source metadata/conversion records before segmentation.")
        evidence = _take_provenance_evidence(
            provenance_token,
            digest,
            geometry_hash,
            review,
            session_sha256,
            token,
            provenance_type=provenance_type,
        )
        output_path = directory / "predicted_spleen_mask.nii.gz"
        try:
            if provenance_type == "nifti_origin":
                origin_record_path = directory / "nifti_origin_record.json"
                origin_record_path.write_bytes(evidence["nifti_origin_record_bytes"])
                output = await run_in_threadpool(
                    run_pretrained_segmentation,
                    input_path,
                    output_path,
                    review,
                    nifti_origin_record_path=origin_record_path,
                )
            else:
                metadata_path = directory / "source_metadata.json"
                conversion_path = directory / "conversion_record.json"
                metadata_path.write_bytes(evidence["source_metadata_bytes"])
                conversion_path.write_bytes(evidence["conversion_record_bytes"])
                output = await run_in_threadpool(
                    run_pretrained_segmentation,
                    input_path,
                    output_path,
                    review,
                    metadata_path,
                    conversion_path,
                )
        except segmentation.GeometryError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except segmentation.SegmentationDependencyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except Exception:
            # Attestation text is intentionally not logged or included in errors.
            raise HTTPException(status_code=503, detail="Pretrained CT segmentation failed; no mask was returned.") from None
        # The signed reviewer manifest is validated in memory only and is not persisted or logged.
        return {
            "status": "completed",
            "model": "MONAI spleen_ct_segmentation",
            "provenance_type": provenance_type,
            "model_status": output["model_status"],
            "input_sha256": digest,
            "mask_sha256": output["mask_sha256"],
            "geometry": {"shape": output["shape"], "affine": output["affine"], **{k: geometry[k] for k in ("spacing_mm", "axis_codes", "qform", "sform", "qform_code", "sform_code")}},
            "volume_preview_gzip_base64": output["volume_preview_gzip_base64"],
            "mask_gzip_base64": output["mask_gzip_base64"],
            "labels": output["labels"],
            "display_window": output["window"],
            "warning": (
                "Research demonstration only. The NIfTI-origin dataset declaration and exact-volume checks passed; original DICOM metadata and conversion history were unavailable. The service cannot independently authenticate dataset membership or prove anatomical orientation."
                if provenance_type == "nifti_origin" else
                "Research demonstration only. DICOM source-record hashes and geometric consistency passed. This service cannot establish source authenticity or independently prove anatomical orientation."
            ),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("segmentation_api:app", host="127.0.0.1", port=8002, reload=False)
