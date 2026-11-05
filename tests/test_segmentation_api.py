"""Synthetic HTTP contract tests; no medical CT or real segmentation is used."""
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import hmac
import importlib.util
import json
import shutil
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request
from fastapi import HTTPException

import segmentation
import segmentation_api


SYNTHETIC_NIFTI_BYTES = b"generated synthetic nifti fixture bytes"
GEOMETRY = {
    "shape": [4, 5, 6],
    "affine": [[1.0, 0, 0, -2], [0, 1.0, 0, -2], [0, 0, 2.0, -5], [0, 0, 0, 1.0]],
    "spacing_mm": [1.0, 1.0, 2.0],
    "axis_codes": ["R", "A", "S"],
    "qform": None,
    "sform": None,
    "qform_code": 0,
    "sform_code": 0,
    "datatype": "int16",
    "voxel_count": 120,
}
TEST_REVIEW_KEY = "test-review-key-with-at-least-thirty-two-bytes"
PROVENANCE_GEOMETRY = {
    "shape": [4, 3, 2],
    "affine": [[-1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1.0]],
    "spacing_mm": [1.0, 1.0, 2.0], "axis_codes": ["L", "P", "S"],
    "qform": None, "sform": None, "qform_code": 0, "sform_code": 0,
    "datatype": "int16", "voxel_count": 24,
}


@pytest.fixture(autouse=True)
def _clear_cached_synthetic_provenance():
    yield
    with segmentation_api._PROVENANCE_CACHE_LOCK:
        entries = list(segmentation_api._PROVENANCE_CACHE.values())
        segmentation_api._PROVENANCE_CACHE.clear()
    for entry in entries:
        if "directory" in entry:
            shutil.rmtree(entry["directory"], ignore_errors=True)


def _request_headers(filename="volume.nii.gz"):
    return {"Content-Type": "application/octet-stream", "X-Volume-Filename": filename}


def _review_manifest(file_sha256=None, geometry=None, **overrides):
    geometry = geometry or GEOMETRY
    now = datetime.now(timezone.utc)
    review = {
        "schema": segmentation_api.ORIENTATION_REVIEW_SCHEMA,
        "volume_sha256": file_sha256 or hashlib.sha256(SYNTHETIC_NIFTI_BYTES).hexdigest(),
        "geometry_sha256": segmentation_api._geometry_sha256(geometry),
        "series_instance_uid": "1.2.840.10008.1.2.3",
        "frame_of_reference_uid": "1.2.840.10008.1.2.4",
        "source_metadata_sha256": "b" * 64,
        "conversion_record_sha256": "c" * 64,
        "image_orientation_patient": [1, 0, 0, 0, 1, 0],
        "first_image_position_patient": [0, 0, 0],
        "last_image_position_patient": [0, 0, 10],
        "slice_order": "ascending",
        "nifti_axis_codes": geometry["axis_codes"],
        "conversion_software": "ExampleConverter",
        "conversion_version": "1.0",
        "reviewer_id": "reviewer-test-id",
        "laterality_verified": True,
        "slice_direction_verified": True,
        "reviewed_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=2)).isoformat(),
        "review_notes": "Synthetic test record only; values exercise schema validation and are not anatomical evidence.",
    }
    review.update(overrides)
    signature = hmac.new(
        TEST_REVIEW_KEY.encode(), segmentation_api._review_signing_bytes(review), hashlib.sha256
    ).hexdigest()
    review["signature_hmac_sha256"] = signature
    return review


def _review_header(review):
    return base64.urlsafe_b64encode(json.dumps(review).encode()).decode().rstrip("=")


def _review_headers(token, review=None, **overrides):
    headers = {
        **_request_headers(),
        "X-Geometry-Token": token,
    }
    if review is not None:
        headers["X-Orientation-Review"] = _review_header(review)
    headers.update(overrides)
    return headers


def _provenance_fixture(geometry=PROVENANCE_GEOMETRY, metadata_change=None, conversion_change=None):
    metadata = {
        "schema": "medical-imaging-source-dicom-series/v1",
        "patient_coordinate_system": "LPS",
        "series_instance_uid": "1.2.840.10008.1.2.3",
        "frame_of_reference_uid": "1.2.840.10008.1.2.4",
        "slices": [],
    }
    for index in range(2):
        metadata["slices"].append({
            "sop_instance_uid": f"1.2.840.10008.1.2.3.{index + 10}",
            "series_instance_uid": metadata["series_instance_uid"],
            "frame_of_reference_uid": metadata["frame_of_reference_uid"],
            "modality": "CT", "patient_position": "HFS",
            "image_orientation_patient": [1, 0, 0, 0, 1, 0],
            "image_position_patient": [0, 0, index * 2],
            "rows": 3, "columns": 4, "pixel_spacing_mm": [1, 1],
            "spacing_between_slices_mm": 2,
        })
    if metadata_change:
        metadata_change(metadata)
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    conversion = {
        "schema": "medical-imaging-nifti-conversion/v1",
        "nifti_affine_coordinate_system": "RAS",
        "nifti_sha256": hashlib.sha256(SYNTHETIC_NIFTI_BYTES).hexdigest(),
        "source_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "nifti_shape": geometry["shape"],
        "source_sop_instance_uids_by_slice_index": [item["sop_instance_uid"] for item in metadata["slices"]],
        "nifti_voxel_to_source_indices": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "conversion_software": "SyntheticFixtureConverter", "conversion_version": "0.0-test",
    }
    if conversion_change:
        conversion_change(conversion)
    conversion_bytes = json.dumps(conversion, sort_keys=True, separators=(",", ":")).encode()
    review = _review_manifest(
        geometry=geometry,
        source_metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        conversion_record_sha256=hashlib.sha256(conversion_bytes).hexdigest(),
        conversion_software="SyntheticFixtureConverter",
        conversion_version="0.0-test",
        first_image_position_patient=[0, 0, 0],
        last_image_position_patient=[0, 0, 2],
    )
    body = json.dumps({
        "source_metadata_base64": base64.b64encode(metadata_bytes).decode(),
        "conversion_record_base64": base64.b64encode(conversion_bytes).decode(),
    }).encode()
    return metadata_bytes, conversion_bytes, review, body


def _nifti_origin_metadata_bytes():
    return json.dumps({
        "name": "Spleen",
        "description": "Spleen Segmentation",
        "reference": "Memorial Sloan Kettering Cancer Center",
        "release": "1.0 06/08/2018",
        "licence": "CC-BY-SA 4.0",
        "modality": {"0": "CT"},
        "labels": {"0": "background", "1": "spleen"},
        "tensorImageSize": "3D",
        "numTraining": 41,
        "numTest": 20,
        "training": [{} for _ in range(41)],
        "test": ["./imagesTs/spleen_1.nii.gz"] + [f"./imagesTs/test_{index}.nii.gz" for index in range(19)],
    }).encode("utf-8")


def _nifti_origin_body(metadata_bytes=None, image_path="./imagesTs/spleen_1.nii.gz"):
    return json.dumps({
        "dataset_json_base64": base64.b64encode(metadata_bytes or _nifti_origin_metadata_bytes()).decode(),
        "image_path": image_path,
    })


def _nifti_origin_review(inspection, origin_record_sha256, **overrides):
    now = datetime.now(timezone.utc)
    review = {
        "schema": segmentation_api.NIFTI_ORIGIN_ORIENTATION_REVIEW_SCHEMA,
        "origin_record_sha256": origin_record_sha256,
        "volume_sha256": inspection["file_sha256"],
        "geometry_sha256": inspection["geometry_sha256"],
        "nifti_axis_codes": inspection["geometry"]["axis_codes"],
        "reviewer_id": "independent-reviewer-test",
        "reviewer_independent": True,
        "laterality_verified": True,
        "slice_direction_verified": True,
        "physical_slice_direction": "inferior_to_superior",
        "orientation_reference": "Synthetic test fixture; not anatomical evidence.",
        "orientation_method": "Synthetic API contract test.",
        "reviewed_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=2)).isoformat(),
        "review_notes": "Synthetic test only; no human review or clinical orientation is represented.",
    }
    review.update(overrides)
    review["signature_hmac_sha256"] = hmac.new(
        TEST_REVIEW_KEY.encode(),
        segmentation_api._review_signing_bytes(review),
        hashlib.sha256,
    ).hexdigest()
    return review


def _create_nifti_origin_record(client, inspection):
    response = client.post(
        "/ct/nifti-origin/record",
        content=_nifti_origin_body(),
        headers={
            "Content-Type": "application/json",
            "X-Geometry-Token": inspection["attestation_token"],
        },
    )
    return response


def _install_inspector(monkeypatch, geometry=GEOMETRY):
    monkeypatch.setattr(segmentation_api, "_inspect_path", lambda _path: geometry)


def test_inspection_returns_geometry_and_file_bound_token_without_inference(monkeypatch):
    _install_inspector(monkeypatch)
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args: calls.append(True))
    response = TestClient(segmentation_api.app).post(
        "/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers()
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "inspection_only"
    assert payload["model_loaded"] is False
    assert payload["file_sha256"] == hashlib.sha256(SYNTHETIC_NIFTI_BYTES).hexdigest()
    assert payload["geometry"] == GEOMETRY
    assert payload["attestation_required"] is True
    assert payload["orientation_verified"] is False
    assert payload["geometry_sha256"] == segmentation_api._geometry_sha256(GEOMETRY)
    assert payload["attestation_token"]
    cookie = response.headers["set-cookie"].lower()
    assert segmentation_api.CT_SESSION_COOKIE in cookie
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert calls == []


def test_ct_cors_allows_browser_session_credentials_for_configured_frontend():
    response = TestClient(segmentation_api.app).options(
        "/ct/evidence",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-geometry-token,x-orientation-review,x-provenance-token",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert response.headers["access-control-allow-credentials"] == "true"
    assert "x-provenance-token" in response.headers["access-control-allow-headers"].lower()


def _upload_synthetic_provenance(client, monkeypatch, geometry=PROVENANCE_GEOMETRY,
                                metadata_change=None, conversion_change=None, invalid_signature=False):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    _install_inspector(monkeypatch, geometry)
    fixture = _provenance_fixture(geometry, metadata_change, conversion_change)
    _metadata_bytes, _conversion_bytes, review, body = fixture
    if invalid_signature:
        review["signature_hmac_sha256"] = "0" * 64
    inspection = client.post("/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers()).json()
    response = client.post(
        "/ct/evidence", content=body,
        headers={
            "Content-Type": "application/json",
            "X-Geometry-Token": inspection["attestation_token"],
            "X-Orientation-Review": _review_header(review),
        },
    )
    return response, inspection, review, fixture


def test_evidence_upload_contract_verifies_synthetic_records_and_returns_short_lived_token(monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    _install_inspector(monkeypatch, PROVENANCE_GEOMETRY)
    client = TestClient(segmentation_api.app)
    _metadata, _conversion, review, body = _provenance_fixture()
    inspection = client.post("/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers()).json()
    response = client.post(
        "/ct/evidence", content=body,
        headers={"Content-Type": "application/json", "X-Geometry-Token": inspection["attestation_token"],
                 "X-Orientation-Review": _review_header(review)},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "provenance_checked"
    assert response.json()["provenance_token"]
    assert "does not establish source authenticity" in response.json()["warning"]


@pytest.mark.parametrize(
    "metadata_change,conversion_change,geometry,expected",
    [
        (lambda m: m.update(series_instance_uid="1.2.999.1", slices=[{**s, "series_instance_uid": "1.2.999.1"} for s in m["slices"]]),
         None, PROVENANCE_GEOMETRY, "series UID differs"),
        (lambda m: m.update(frame_of_reference_uid="1.2.999.2", slices=[{**s, "frame_of_reference_uid": "1.2.999.2"} for s in m["slices"]]),
         None, PROVENANCE_GEOMETRY, "FrameOfReferenceUID differs"),
        (None, lambda c: c.update(nifti_sha256="f" * 64), PROVENANCE_GEOMETRY, "not bound to these exact NIfTI bytes"),
        (None, lambda c: c.update(source_sop_instance_uids_by_slice_index=list(reversed(c["source_sop_instance_uids_by_slice_index"]))),
         PROVENANCE_GEOMETRY, "slice inventory/order"),
        (lambda m: m["slices"][1].update(image_position_patient=[0, 0, 4]), None,
         PROVENANCE_GEOMETRY, "gap or inconsistent spacing"),
        (lambda m: m["slices"][1].update(sop_instance_uid=m["slices"][0]["sop_instance_uid"]), None,
         PROVENANCE_GEOMETRY, "duplicate slice instances"),
        (None, None, {**PROVENANCE_GEOMETRY, "affine": [[-1.0, 0, 0, 9], [0, -1.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1.0]]},
         "disagrees with the NIfTI affine"),
    ],
)
def test_evidence_upload_rejects_mismatched_provenance_before_model(
    monkeypatch, metadata_change, conversion_change, geometry, expected
):
    client = TestClient(segmentation_api.app)
    calls = []
    monkeypatch.setattr(segmentation, "load_pretrained_unet", lambda *_args: calls.append(True))
    response, _inspection, _review, _fixture = _upload_synthetic_provenance(
        client, monkeypatch, geometry, metadata_change, conversion_change
    )
    assert response.status_code == 422
    assert expected in response.json()["detail"]
    assert calls == []


@pytest.mark.parametrize(
    "record_field,detail",
    [
        ("source_metadata_base64", "Source metadata file hash"),
        ("conversion_record_base64", "Conversion record file hash"),
    ],
)
def test_evidence_upload_rejects_altered_record_bytes_and_bad_review_signature(monkeypatch, record_field, detail):
    client = TestClient(segmentation_api.app)
    response, inspection, review, fixture = _upload_synthetic_provenance(client, monkeypatch)
    assert response.status_code == 200
    token = inspection["attestation_token"]
    _metadata, _conversion, review, body = fixture
    payload = json.loads(body)
    altered = base64.b64decode(payload[record_field]) + b" "
    payload[record_field] = base64.b64encode(altered).decode()
    changed = client.post(
        "/ct/evidence", json=payload,
        headers={"X-Geometry-Token": token, "X-Orientation-Review": _review_header(review)},
    )
    assert changed.status_code == 422
    assert detail in changed.json()["detail"]

    invalid = dict(payload)
    invalid["source_metadata_base64"] = json.loads(body)["source_metadata_base64"]
    review["signature_hmac_sha256"] = "0" * 64
    bad_signature = client.post(
        "/ct/evidence", json=invalid,
        headers={"X-Geometry-Token": token, "X-Orientation-Review": _review_header(review)},
    )
    assert bad_signature.status_code == 422
    assert "signature verification failed" in bad_signature.json()["detail"]


def test_evidence_upload_rejects_missing_malformed_and_oversized_records(monkeypatch):
    client = TestClient(segmentation_api.app)
    missing = client.post("/ct/evidence", json={"source_metadata_base64": "e30="})
    assert missing.status_code == 422
    malformed = client.post("/ct/evidence", content=b"not json", headers={"Content-Type": "application/json"})
    assert malformed.status_code == 422
    monkeypatch.setattr(segmentation_api, "MAX_PROVENANCE_BODY_BYTES", 4)
    oversized = client.post("/ct/evidence", content=b"12345", headers={"Content-Type": "application/json"})
    assert oversized.status_code == 413

    monkeypatch.setattr(segmentation_api, "MAX_PROVENANCE_BODY_BYTES", 3 * 1024 * 1024)
    monkeypatch.setattr(segmentation_api, "MAX_PROVENANCE_RECORD_BYTES", 4)
    response, _inspection, _review, _fixture = _upload_synthetic_provenance(client, monkeypatch)
    assert response.status_code == 413
    assert "configured decoded file limit" in response.json()["detail"]


@pytest.mark.parametrize(
    "record_name,record_bytes,detail",
    [
        ("source", b"{", "must contain UTF-8 JSON"),
        ("source", b"[]", "must contain a JSON object"),
        ("source", b'{"schema":"unknown"}', "schema is unsupported"),
        ("conversion", b'{"schema":"unknown"}', "schema is unsupported"),
    ],
)
def test_evidence_upload_rejects_invalid_json_and_record_schema_before_model(
    monkeypatch, record_name, record_bytes, detail
):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    _install_inspector(monkeypatch, PROVENANCE_GEOMETRY)
    client = TestClient(segmentation_api.app)
    calls = []
    monkeypatch.setattr(segmentation, "load_pretrained_unet", lambda *_args: calls.append(True))
    inspection = client.post(
        "/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers()
    ).json()
    metadata_bytes, conversion_bytes, review, _body = _provenance_fixture()
    if record_name == "source":
        metadata_bytes = record_bytes
    else:
        conversion_bytes = record_bytes
    if record_name == "source" and detail == "schema is unsupported":
        metadata = json.loads(metadata_bytes)
        metadata["schema"] = "unknown"
        metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    if record_name == "source" and detail != "schema is unsupported":
        conversion = json.loads(conversion_bytes)
        conversion["source_metadata_sha256"] = hashlib.sha256(metadata_bytes).hexdigest()
        conversion_bytes = json.dumps(conversion, sort_keys=True, separators=(",", ":")).encode()
    review["source_metadata_sha256"] = hashlib.sha256(metadata_bytes).hexdigest()
    review["conversion_record_sha256"] = hashlib.sha256(conversion_bytes).hexdigest()
    review["signature_hmac_sha256"] = hmac.new(
        TEST_REVIEW_KEY.encode(),
        segmentation_api._review_signing_bytes(review),
        hashlib.sha256,
    ).hexdigest()
    payload = {
        "source_metadata_base64": base64.b64encode(metadata_bytes).decode(),
        "conversion_record_base64": base64.b64encode(conversion_bytes).decode(),
    }
    response = client.post(
        "/ct/evidence",
        json=payload,
        headers={
            "X-Geometry-Token": inspection["attestation_token"],
            "X-Orientation-Review": _review_header(review),
        },
    )
    assert response.status_code == 422
    assert detail in response.json()["detail"]
    assert calls == []


def test_evidence_body_limit_rejects_streamed_bytes_without_content_length():
    chunks = iter([b"123", b"456"])

    async def receive():
        try:
            chunk = next(chunks)
        except StopIteration:
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": chunk, "more_body": True}

    request = Request({"type": "http", "headers": []}, receive)

    async def read():
        return await segmentation_api._read_bounded_body(request, 4)

    with pytest.raises(HTTPException) as error:
        asyncio.run(read())
    assert error.value.status_code == 413


def test_segment_route_requires_evidence_token_before_pretrained_runner(monkeypatch):
    client = TestClient(segmentation_api.app)
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args: calls.append(True))
    response, inspection, review, _fixture = _upload_synthetic_provenance(client, monkeypatch)
    assert response.status_code == 200
    blocked = client.post(
        "/ct/segment", content=SYNTHETIC_NIFTI_BYTES,
        headers=_review_headers(inspection["attestation_token"], review),
    )
    assert blocked.status_code == 422
    assert "Upload and verify source metadata" in blocked.json()["detail"]
    assert calls == []


def test_synthetic_evidence_token_is_bound_and_forwarded_to_mocked_segmenter(monkeypatch):
    """Contract test only: the mocked callback returns no real segmentation result."""
    client = TestClient(segmentation_api.app)
    calls = []

    def fake_segmenter(input_path, output_path, review, metadata_path, conversion_path):
        calls.append((
            Path(input_path).read_bytes(),
            Path(metadata_path).read_bytes(),
            Path(conversion_path).read_bytes(),
            review["volume_sha256"],
        ))
        mask_bytes = bytes([0, 1] * 12)
        volume_bytes = bytes(range(24))
        return {
            "volume_preview_gzip_base64": base64.b64encode(gzip.compress(volume_bytes, mtime=0)).decode(),
            "mask_gzip_base64": base64.b64encode(gzip.compress(mask_bytes, mtime=0)).decode(),
            "mask_sha256": hashlib.sha256(gzip.compress(mask_bytes, mtime=0)).hexdigest(),
            "shape": PROVENANCE_GEOMETRY["shape"], "affine": PROVENANCE_GEOMETRY["affine"],
            "model_status": "pretrained", "labels": [0, 1], "window": {"center": 53.5, "width": 221},
        }

    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", fake_segmenter)
    evidence_response, inspection, review, fixture = _upload_synthetic_provenance(client, monkeypatch)
    assert evidence_response.status_code == 200
    metadata_bytes, conversion_bytes, _review, _body = fixture
    response = client.post(
        "/ct/segment", content=SYNTHETIC_NIFTI_BYTES,
        headers={
            **_review_headers(inspection["attestation_token"], review),
            "X-Provenance-Token": evidence_response.json()["provenance_token"],
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert calls == [(SYNTHETIC_NIFTI_BYTES, metadata_bytes, conversion_bytes, inspection["file_sha256"])]
    replay = client.post(
        "/ct/segment", content=SYNTHETIC_NIFTI_BYTES,
        headers={
            **_review_headers(inspection["attestation_token"], review),
            "X-Provenance-Token": evidence_response.json()["provenance_token"],
        },
    )
    assert replay.status_code == 422


def test_provenance_token_cannot_cross_browser_sessions(monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    first_client = TestClient(segmentation_api.app)
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args: calls.append(True))
    evidence_response, inspection, review, _fixture = _upload_synthetic_provenance(first_client, monkeypatch)
    assert evidence_response.status_code == 200

    unrelated_client = TestClient(segmentation_api.app)
    unrelated_client.post(
        "/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers()
    )
    response = unrelated_client.post(
        "/ct/segment",
        content=SYNTHETIC_NIFTI_BYTES,
        headers={
            **_review_headers(inspection["attestation_token"], review),
            "X-Provenance-Token": evidence_response.json()["provenance_token"],
        },
    )
    assert response.status_code == 403
    assert "different browser session" in response.json()["detail"]
    assert calls == []


def test_provenance_token_cannot_cross_repeated_inspections_in_same_session(monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    client = TestClient(segmentation_api.app)
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args: calls.append(True))
    evidence_response, inspection, review, _fixture = _upload_synthetic_provenance(client, monkeypatch)
    repeated_inspection = client.post(
        "/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers()
    ).json()
    assert repeated_inspection["attestation_token"] != inspection["attestation_token"]

    response = client.post(
        "/ct/segment",
        content=SYNTHETIC_NIFTI_BYTES,
        headers={
            **_review_headers(repeated_inspection["attestation_token"], review),
            "X-Provenance-Token": evidence_response.json()["provenance_token"],
        },
    )
    assert response.status_code == 409
    assert "different session, inspection" in response.json()["detail"]
    assert calls == []


def test_provenance_token_cannot_be_reused_for_another_volume(monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    _install_inspector(monkeypatch, PROVENANCE_GEOMETRY)
    client = TestClient(segmentation_api.app)
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args: calls.append(True))
    evidence_response, _inspection, _review, _fixture = _upload_synthetic_provenance(client, monkeypatch)
    assert evidence_response.status_code == 200

    other_volume = b"a different synthetic NIfTI volume"
    other_inspection = client.post(
        "/ct/inspect", content=other_volume, headers=_request_headers()
    ).json()
    other_review = _review_manifest(
        file_sha256=hashlib.sha256(other_volume).hexdigest(),
        geometry=PROVENANCE_GEOMETRY,
    )
    response = client.post(
        "/ct/segment",
        content=other_volume,
        headers={
            **_review_headers(other_inspection["attestation_token"], other_review),
            "X-Provenance-Token": evidence_response.json()["provenance_token"],
        },
    )
    assert response.status_code == 409
    assert "different session, inspection, file" in response.json()["detail"]
    assert calls == []


def test_expired_provenance_token_is_rejected_before_model(monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    client = TestClient(segmentation_api.app)
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args: calls.append(True))
    response, inspection, review, _fixture = _upload_synthetic_provenance(client, monkeypatch)
    assert response.status_code == 200
    token = response.json()["provenance_token"]
    with segmentation_api._PROVENANCE_CACHE_LOCK:
        segmentation_api._PROVENANCE_CACHE[token]["expires"] = 0

    expired = client.post(
        "/ct/segment",
        content=SYNTHETIC_NIFTI_BYTES,
        headers={
            **_review_headers(inspection["attestation_token"], review),
            "X-Provenance-Token": token,
        },
    )
    assert expired.status_code == 422
    assert "missing, expired, or already used" in expired.json()["detail"]
    assert calls == []


def test_token_expiring_after_cleanup_is_rejected_inside_consumption_lock(monkeypatch):
    token = "boundary-expiry-token"
    review = {"signature_hmac_sha256": "a" * 64}
    entry = {
        "file_sha256": "b" * 64,
        "geometry_sha256": "c" * 64,
        "review_sha256": segmentation_api._review_digest(review),
        "session_sha256": "d" * 64,
        "inspection_token_sha256": hashlib.sha256(b"inspection").hexdigest(),
        "expires": 100.5,
    }
    with segmentation_api._PROVENANCE_CACHE_LOCK:
        segmentation_api._PROVENANCE_CACHE[token] = entry
    times = iter([100.0, 101.0])
    monkeypatch.setattr(segmentation_api.time, "time", lambda: next(times))

    with pytest.raises(HTTPException) as error:
        segmentation_api._take_provenance_evidence(
            token, "b" * 64, "c" * 64, review, "d" * 64, "inspection"
        )
    assert error.value.status_code == 422
    assert token not in segmentation_api._PROVENANCE_CACHE


def test_concurrent_provenance_token_consumption_has_exactly_one_winner():
    token = "concurrent-one-use-token"
    review = {"signature_hmac_sha256": "a" * 64}
    entry = {
        "file_sha256": "b" * 64,
        "geometry_sha256": "c" * 64,
        "review_sha256": segmentation_api._review_digest(review),
        "session_sha256": "d" * 64,
        "inspection_token_sha256": hashlib.sha256(b"inspection").hexdigest(),
        "expires": time.time() + 60,
        "evidence": b"synthetic test payload",
    }
    with segmentation_api._PROVENANCE_CACHE_LOCK:
        segmentation_api._PROVENANCE_CACHE[token] = entry
    barrier = threading.Barrier(8)

    def consume():
        barrier.wait()
        try:
            return segmentation_api._take_provenance_evidence(
                token, "b" * 64, "c" * 64, review, "d" * 64, "inspection"
            )
        except HTTPException as error:
            return error.status_code

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(lambda _index: consume(), range(8)))
    assert sum(isinstance(outcome, dict) for outcome in outcomes) == 1
    assert sum(outcome == 422 for outcome in outcomes) == 7
    assert token not in segmentation_api._PROVENANCE_CACHE


def test_full_pending_cache_rejects_new_evidence_without_evicting_existing_tokens():
    now = time.time()
    with segmentation_api._PROVENANCE_CACHE_LOCK:
        for index in range(segmentation_api.MAX_PENDING_PROVENANCE_RECORDS):
            segmentation_api._PROVENANCE_CACHE[f"existing-{index}"] = {"expires": now + 60}
    original = dict(segmentation_api._PROVENANCE_CACHE)

    with pytest.raises(HTTPException) as error:
        segmentation_api._store_provenance_evidence({"expires": now + 60})
    assert error.value.status_code == 429
    assert segmentation_api._PROVENANCE_CACHE == original


def test_api_blocks_hmac_only_review_until_source_record_transport_exists(monkeypatch):
    """A synthetic signed-schema fixture is not source evidence or inference authorization."""
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    _install_inspector(monkeypatch)
    calls = []
    monkeypatch.setattr(segmentation, "load_pretrained_unet", lambda *_args: calls.append(True))
    client = TestClient(segmentation_api.app)
    inspection = client.post("/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers()).json()
    response = client.post(
        "/ct/segment",
        content=SYNTHETIC_NIFTI_BYTES,
        headers=_review_headers(inspection["attestation_token"], _review_manifest()),
    )
    assert response.status_code == 422
    assert "Upload and verify source metadata" in response.json()["detail"]
    assert calls == []


def test_missing_review_manifest_never_reaches_model(monkeypatch):
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args: calls.append(True))
    response = TestClient(segmentation_api.app).post(
        "/ct/segment", content=SYNTHETIC_NIFTI_BYTES,
        headers={**_request_headers(), "X-Geometry-Token": "some-token"},
    )
    assert response.status_code == 422
    assert "reviewer-signed" in response.json()["detail"]
    assert set(response.json()) == {"detail"}
    assert calls == []


@pytest.mark.parametrize(
    "review_factory,expected_status,detail",
    [
        (lambda: "not-base64-json", 422, "malformed or unreadable"),
        (lambda: {**_review_manifest(), "signature_hmac_sha256": "0" * 64}, 422, "signature verification failed"),
        (lambda: _review_manifest(expires_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()), 422, "stale"),
        (lambda: _review_manifest(volume_sha256="d" * 64), 409, "different NIfTI file"),
        (lambda: _review_manifest(geometry_sha256="e" * 64), 409, "different NIfTI geometry"),
        (lambda: _review_manifest(image_orientation_patient=[2, 0, 0, 0, 1, 0]), 422, "not orthonormal"),
    ],
)
def test_invalid_stale_or_mismatched_signed_evidence_never_reaches_model(
    monkeypatch, review_factory, expected_status, detail
):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    _install_inspector(monkeypatch)
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args: calls.append(True))
    client = TestClient(segmentation_api.app)
    inspection = client.post("/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers()).json()
    review = review_factory()
    headers = _review_headers(inspection["attestation_token"], review) if isinstance(review, dict) else {
        **_request_headers(), "X-Geometry-Token": inspection["attestation_token"], "X-Orientation-Review": review
    }
    response = client.post("/ct/segment", content=SYNTHETIC_NIFTI_BYTES, headers=headers)
    assert response.status_code == expected_status
    assert detail in response.json()["detail"]
    assert calls == []


@pytest.mark.parametrize("field", ["source_metadata_sha256", "conversion_record_sha256"])
def test_missing_source_record_digest_rejects_before_model(monkeypatch, field):
    """Schema presence is enforced; this synthetic test cannot verify record provenance."""
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    _install_inspector(monkeypatch)
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args: calls.append(True))
    client = TestClient(segmentation_api.app)
    inspection = client.post("/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers()).json()
    review = _review_manifest(**{field: None})

    response = client.post(
        "/ct/segment",
        content=SYNTHETIC_NIFTI_BYTES,
        headers=_review_headers(inspection["attestation_token"], review),
    )

    assert response.status_code == 422
    assert field in response.json()["detail"]
    assert calls == []


def test_missing_trusted_reviewer_key_rejects_before_model(monkeypatch):
    monkeypatch.delenv("CT_ORIENTATION_REVIEW_HMAC_KEY", raising=False)
    _install_inspector(monkeypatch)
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args: calls.append(True))
    client = TestClient(segmentation_api.app)
    inspection = client.post("/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers()).json()
    response = client.post(
        "/ct/segment", content=SYNTHETIC_NIFTI_BYTES,
        headers=_review_headers(inspection["attestation_token"], _review_manifest()),
    )
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]
    assert calls == []


def test_api_pretrained_runner_blocks_without_source_record_files_before_checkpoint_access(monkeypatch, tmp_path):
    monkeypatch.setattr(segmentation_api, "CHECKPOINT_PATH", tmp_path / "must-not-be-read.pt")
    with pytest.raises(segmentation.GeometryError, match="source metadata and conversion-record files are required"):
        segmentation_api.run_pretrained_segmentation(
            tmp_path / "synthetic.nii.gz", tmp_path / "mask.nii.gz", {"synthetic": True}
        )


def test_changed_file_bytes_and_changed_geometry_reject_attestation(monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    _install_inspector(monkeypatch)
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args: calls.append(True))
    client = TestClient(segmentation_api.app)
    token = client.post("/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers()).json()["attestation_token"]
    changed = client.post(
        "/ct/segment", content=SYNTHETIC_NIFTI_BYTES + b"x",
        headers=_review_headers(token, _review_manifest()),
    )
    assert changed.status_code == 409
    assert "differ from the inspected file" in changed.json()["detail"]

    geometry_calls = iter([GEOMETRY, {**GEOMETRY, "shape": [4, 5, 7], "voxel_count": 140}])
    monkeypatch.setattr(segmentation_api, "_inspect_path", lambda _path: next(geometry_calls))
    new_token = client.post("/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers()).json()["attestation_token"]
    changed_geometry = client.post(
        "/ct/segment", content=SYNTHETIC_NIFTI_BYTES,
        headers=_review_headers(new_token, _review_manifest(geometry={**GEOMETRY, "shape": [4, 5, 7], "voxel_count": 140})),
    )
    assert changed_geometry.status_code == 409
    assert "geometry changed after inspection" in changed_geometry.json()["detail"]
    assert calls == []


def test_invalid_token_extension_and_nifti_geometry_are_rejected(monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    client = TestClient(segmentation_api.app)
    invalid_token = client.post(
        "/ct/segment", content=SYNTHETIC_NIFTI_BYTES,
        headers=_review_headers("tampered.token", _review_manifest()),
    )
    assert invalid_token.status_code == 422
    assert "invalid or expired" in invalid_token.json()["detail"]

    invalid_extension = client.post(
        "/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers("scan.dcm")
    )
    assert invalid_extension.status_code == 415

    monkeypatch.setattr(segmentation, "inspect_nifti_geometry", lambda _path: (_ for _ in ()).throw(
        segmentation.GeometryError("NIfTI qform and sform disagree; physical geometry is ambiguous.")
    ))
    bad_geometry = client.post("/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers())
    assert bad_geometry.status_code == 422
    assert "qform and sform disagree" in bad_geometry.json()["detail"]


def test_inspection_rejects_empty_and_oversized_requests(monkeypatch):
    client = TestClient(segmentation_api.app)
    empty = client.post("/ct/inspect", content=b"", headers=_request_headers())
    assert empty.status_code == 400
    monkeypatch.setattr(segmentation_api, "MAX_REQUEST_BYTES", 3)
    oversized = client.post("/ct/inspect", content=b"four", headers=_request_headers())
    assert oversized.status_code == 413


def test_inspection_accepts_generated_nifti_and_reports_actual_header_geometry(tmp_path):
    nib = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")
    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.int16), np.diag([1.0, 1.0, 2.0, 1.0]))
    source = tmp_path / "generated.nii.gz"
    nib.save(image, str(source))

    response = TestClient(segmentation_api.app).post(
        "/ct/inspect", content=source.read_bytes(), headers=_request_headers()
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["geometry"]["shape"] == [4, 5, 6]
    assert payload["geometry"]["spacing_mm"] == [1.0, 1.0, 2.0]


def test_nifti_origin_record_is_generated_from_task09_manifest_and_inspection(monkeypatch):
    _install_inspector(monkeypatch)
    client = TestClient(segmentation_api.app)
    inspection = client.post(
        "/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers("spleen_1.nii.gz")
    ).json()
    response = _create_nifti_origin_record(client, inspection)
    assert response.status_code == 200
    result = response.json()
    record = result["origin_record"]
    assert result["status"] == "nifti_origin_record_created"
    assert result["provenance_type"] == "nifti_origin"
    assert record["dataset"]["name"] == "Medical Segmentation Decathlon"
    assert record["dataset"]["task"] == "Task09_Spleen"
    assert record["dataset"]["release"] == "1.0 06/08/2018"
    assert record["dataset"]["split"] == "test"
    assert record["dataset"]["image_path"] == "./imagesTs/spleen_1.nii.gz"
    assert record["volume"]["sha256"] == inspection["file_sha256"]
    assert record["volume"]["geometry"] == inspection["geometry"]
    assert record["original_dicom_metadata"] == "not_supplied_in_the_dataset_distribution"
    assert record["conversion_history"] == "unavailable"
    assert "series_instance_uid" not in json.dumps(record)
    assert "conversion_software" not in json.dumps(record)
    assert len(result["origin_record_sha256"]) == 64


@pytest.mark.parametrize(
    "filename,body,expected",
    [
        ("volume.nii.gz", None, "requires the inspected file spleen_1"),
        ("spleen_1.nii.gz", _nifti_origin_body(image_path="./imagesTr/spleen_10.nii.gz"), "accepts only the listed"),
        ("spleen_1.nii.gz", _nifti_origin_body(json.dumps({
            "name": "Spleen", "release": "wrong", "modality": {"0": "CT"},
            "test": ["./imagesTs/spleen_1.nii.gz"],
        }).encode()), "expected Task09 Spleen release"),
    ],
)
def test_nifti_origin_record_rejects_nonselected_or_invalid_dataset(monkeypatch, filename, body, expected):
    _install_inspector(monkeypatch)
    client = TestClient(segmentation_api.app)
    inspection = client.post(
        "/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers(filename)
    ).json()
    response = client.post(
        "/ct/nifti-origin/record",
        content=body or _nifti_origin_body(),
        headers={"Content-Type": "application/json", "X-Geometry-Token": inspection["attestation_token"]},
    )
    assert response.status_code == 422
    assert expected in response.json()["detail"]


def test_nifti_origin_evidence_requires_valid_signed_independent_review(monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    _install_inspector(monkeypatch)
    client = TestClient(segmentation_api.app)
    inspection = client.post(
        "/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers("spleen_1.nii.gz")
    ).json()
    origin_response = _create_nifti_origin_record(client, inspection)
    origin_record_sha256 = origin_response.json()["origin_record_sha256"]
    valid_review = _nifti_origin_review(inspection, origin_record_sha256)
    headers = {
        "Content-Type": "application/json",
        "X-Geometry-Token": inspection["attestation_token"],
        "X-Orientation-Review": _review_header(valid_review),
    }
    valid = client.post("/ct/nifti-origin/evidence", content=_nifti_origin_body(), headers=headers)
    assert valid.status_code == 200
    assert valid.json()["provenance_type"] == "nifti_origin"
    assert valid.json()["origin_record_sha256"] == origin_record_sha256
    assert valid.json()["provenance_token"]

    invalid_signature = {**valid_review, "signature_hmac_sha256": "0" * 64}
    bad_signature = client.post(
        "/ct/nifti-origin/evidence",
        content=_nifti_origin_body(),
        headers={**headers, "X-Orientation-Review": _review_header(invalid_signature)},
    )
    assert bad_signature.status_code == 422
    assert "signature verification failed" in bad_signature.json()["detail"]

    mismatched_volume = _nifti_origin_review(
        inspection, origin_record_sha256, volume_sha256="b" * 64
    )
    mismatch = client.post(
        "/ct/nifti-origin/evidence",
        content=_nifti_origin_body(),
        headers={**headers, "X-Orientation-Review": _review_header(mismatched_volume)},
    )
    assert mismatch.status_code == 409
    assert "different NIfTI file" in mismatch.json()["detail"]

    mismatched_geometry = _nifti_origin_review(
        inspection, origin_record_sha256, geometry_sha256="0" * 64
    )
    geometry_mismatch = client.post(
        "/ct/nifti-origin/evidence",
        content=_nifti_origin_body(),
        headers={**headers, "X-Orientation-Review": _review_header(mismatched_geometry)},
    )
    assert geometry_mismatch.status_code == 409
    assert "different NIfTI geometry" in geometry_mismatch.json()["detail"]

    mismatched_record = _nifti_origin_review(
        inspection, "0" * 64
    )
    record_mismatch = client.post(
        "/ct/nifti-origin/evidence",
        content=_nifti_origin_body(),
        headers={**headers, "X-Orientation-Review": _review_header(mismatched_record)},
    )
    assert record_mismatch.status_code == 409
    assert "different provenance record" in record_mismatch.json()["detail"]

    expired_review = _nifti_origin_review(
        inspection,
        origin_record_sha256,
        reviewed_at=(datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
        expires_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
    )
    expired = client.post(
        "/ct/nifti-origin/evidence",
        content=_nifti_origin_body(),
        headers={**headers, "X-Orientation-Review": _review_header(expired_review)},
    )
    assert expired.status_code == 422
    assert "stale" in expired.json()["detail"]

    invented_dicom = _nifti_origin_review(
        inspection, origin_record_sha256, series_instance_uid="1.2.3.4"
    )
    rejected_dicom_fields = client.post(
        "/ct/nifti-origin/evidence",
        content=_nifti_origin_body(),
        headers={**headers, "X-Orientation-Review": _review_header(invented_dicom)},
    )
    assert rejected_dicom_fields.status_code == 422
    assert "unsupported or DICOM-origin fields" in rejected_dicom_fields.json()["detail"]

    monkeypatch.delenv("CT_ORIENTATION_REVIEW_HMAC_KEY")
    unconfigured = client.post(
        "/ct/nifti-origin/evidence",
        content=_nifti_origin_body(),
        headers=headers,
    )
    assert unconfigured.status_code == 503
    assert "not configured" in unconfigured.json()["detail"]


def test_nifti_origin_token_reaches_only_the_nifti_origin_runner_and_is_one_use(monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    _install_inspector(monkeypatch)
    client = TestClient(segmentation_api.app)
    inspection = client.post(
        "/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers("spleen_1.nii.gz")
    ).json()
    generated = _create_nifti_origin_record(client, inspection).json()
    review = _nifti_origin_review(inspection, generated["origin_record_sha256"])
    evidence = client.post(
        "/ct/nifti-origin/evidence",
        content=_nifti_origin_body(),
        headers={
            "Content-Type": "application/json",
            "X-Geometry-Token": inspection["attestation_token"],
            "X-Orientation-Review": _review_header(review),
        },
    )
    assert evidence.status_code == 200
    calls = []

    def fake_runner(input_path, output_path, received_review, **kwargs):
        record_path = kwargs["nifti_origin_record_path"]
        calls.append((
            Path(input_path).read_bytes(),
            json.loads(Path(record_path).read_text(encoding="utf-8")),
            received_review,
            kwargs.get("source_metadata_path"),
            kwargs.get("conversion_record_path"),
        ))
        preview = bytes(range(120))
        mask = bytes([0, 1] * 60)
        return {
            "volume_preview_gzip_base64": base64.b64encode(gzip.compress(preview, mtime=0)).decode(),
            "mask_gzip_base64": base64.b64encode(gzip.compress(mask, mtime=0)).decode(),
            "mask_sha256": hashlib.sha256(gzip.compress(mask, mtime=0)).hexdigest(),
            "shape": GEOMETRY["shape"],
            "affine": GEOMETRY["affine"],
            "model_status": "pretrained",
            "labels": [0, 1],
            "window": {"center": 53.5, "width": 221},
        }

    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", fake_runner)
    headers = {
        **_request_headers("spleen_1.nii.gz"),
        "X-Geometry-Token": inspection["attestation_token"],
        "X-Orientation-Review": _review_header(review),
        "X-Provenance-Token": evidence.json()["provenance_token"],
    }
    response = client.post("/ct/segment", content=SYNTHETIC_NIFTI_BYTES, headers=headers)
    assert response.status_code == 200
    assert response.json()["provenance_type"] == "nifti_origin"
    assert calls == [(
        SYNTHETIC_NIFTI_BYTES,
        generated["origin_record"],
        review,
        None,
        None,
    )]
    replay = client.post("/ct/segment", content=SYNTHETIC_NIFTI_BYTES, headers=headers)
    assert replay.status_code == 422
    assert "missing, expired, or already used" in replay.json()["detail"]


def test_nifti_origin_workflow_rejects_unauthorized_session_and_expired_one_use_token(monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    _install_inspector(monkeypatch)
    owner = TestClient(segmentation_api.app)
    inspection = owner.post(
        "/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers("spleen_1.nii.gz")
    ).json()
    other = TestClient(segmentation_api.app)
    unauthorized = other.post(
        "/ct/nifti-origin/record",
        content=_nifti_origin_body(),
        headers={"Content-Type": "application/json", "X-Geometry-Token": inspection["attestation_token"]},
    )
    assert unauthorized.status_code == 403
    assert "session" in unauthorized.json()["detail"]

    origin_record = _create_nifti_origin_record(owner, inspection).json()
    review = _nifti_origin_review(inspection, origin_record["origin_record_sha256"])
    evidence = owner.post(
        "/ct/nifti-origin/evidence",
        content=_nifti_origin_body(),
        headers={
            "Content-Type": "application/json",
            "X-Geometry-Token": inspection["attestation_token"],
            "X-Orientation-Review": _review_header(review),
        },
    )
    assert evidence.status_code == 200
    token = evidence.json()["provenance_token"]
    with segmentation_api._PROVENANCE_CACHE_LOCK:
        segmentation_api._PROVENANCE_CACHE[token]["expires"] = 0
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args, **_kwargs: calls.append(True))
    expired = owner.post(
        "/ct/segment",
        content=SYNTHETIC_NIFTI_BYTES,
        headers={
            **_request_headers("spleen_1.nii.gz"),
            "X-Geometry-Token": inspection["attestation_token"],
            "X-Orientation-Review": _review_header(review),
            "X-Provenance-Token": token,
        },
    )
    assert expired.status_code == 422
    assert "missing, expired, or already used" in expired.json()["detail"]
    assert calls == []


def test_nifti_origin_review_without_evidence_never_reaches_pretrained_runner(monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", TEST_REVIEW_KEY)
    _install_inspector(monkeypatch)
    client = TestClient(segmentation_api.app)
    inspection = client.post(
        "/ct/inspect", content=SYNTHETIC_NIFTI_BYTES, headers=_request_headers("spleen_1.nii.gz")
    ).json()
    record = _create_nifti_origin_record(client, inspection).json()
    review = _nifti_origin_review(inspection, record["origin_record_sha256"])
    calls = []
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args, **_kwargs: calls.append(True))
    response = client.post(
        "/ct/segment",
        content=SYNTHETIC_NIFTI_BYTES,
        headers={
            **_request_headers("spleen_1.nii.gz"),
            "X-Geometry-Token": inspection["attestation_token"],
            "X-Orientation-Review": _review_header(review),
        },
    )
    assert response.status_code == 422
    assert "Generate and verify NIfTI-origin evidence" in response.json()["detail"]
    assert calls == []


def test_inspection_rejects_corrupt_nifti_bytes_before_model_use():
    calls = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(segmentation_api, "run_pretrained_segmentation", lambda *_args: calls.append(True))
    try:
        response = TestClient(segmentation_api.app).post(
            "/ct/inspect", content=b"not a NIfTI file", headers=_request_headers()
        )
        if importlib.util.find_spec("nibabel") is None:
            assert response.status_code == 503
            assert "optional dependencies" in response.json()["detail"]
        else:
            assert response.status_code == 422
            assert "readable, valid 3D NIfTI" in response.json()["detail"]
        assert calls == []
    finally:
        monkeypatch.undo()


def test_inspection_rejects_conflicting_qform_and_sform_from_real_nifti(tmp_path):
    nib = pytest.importorskip("nibabel")
    np = pytest.importorskip("numpy")
    image = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.int16), np.eye(4))
    image.set_qform(np.eye(4), code=1)
    conflicting_sform = np.eye(4)
    conflicting_sform[0, 3] = 10
    image.set_sform(conflicting_sform, code=1)
    source = tmp_path / "conflicting.nii.gz"
    nib.save(image, str(source))

    response = TestClient(segmentation_api.app).post(
        "/ct/inspect", content=source.read_bytes(), headers=_request_headers()
    )
    assert response.status_code == 422
    assert "qform and sform disagree" in response.json()["detail"]
