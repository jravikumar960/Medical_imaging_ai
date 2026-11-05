"""Synthetic MRI multipart API tests; no medical inference or real MRI data."""
import base64
import tempfile
from pathlib import Path

import gzip

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

import mri_api


def nifti_bytes(shape=(8, 8, 8), affine=None, compressed=True):
    affine = np.eye(4) if affine is None else np.asarray(affine, dtype=np.float64)
    image = nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / ("volume.nii.gz" if compressed else "volume.nii")
        nib.save(image, str(path))
        return path.read_bytes()


def multipart_files(contents, filenames=None):
    filenames = filenames or {key: f"{key}.nii.gz" for key in contents}
    return [
        (key, (filenames[key], contents[key], "application/octet-stream"))
        for key in contents
    ]


def test_health_reports_task_specific_local_workflow_without_loading_weights():
    response = TestClient(mri_api.app).get("/health")

    assert response.status_code == 200
    assert response.json()["model"] == "MONAI/brats_mri_segmentation"
    assert response.json()["required_sequences"] == ["t1c", "t1", "t2", "flair"]
    assert response.json()["model_loaded"] is False


def test_missing_sequence_is_rejected_without_loading_weights(monkeypatch):
    calls = []
    monkeypatch.setattr(mri_api, "_model", lambda: calls.append(True))
    files = {key: b"synthetic" for key in ("t1c", "t1", "flair")}

    response = TestClient(mri_api.app).post("/mri/segment", files=multipart_files(files))

    assert response.status_code == 422
    assert "Exactly four" in response.json()["detail"]
    assert calls == []


def test_invalid_extension_is_rejected_before_inference(monkeypatch):
    calls = []
    monkeypatch.setattr(mri_api, "_run_inference", lambda *_args: calls.append(True))
    contents = {key: b"synthetic" for key in ("t1c", "t1", "t2", "flair")}
    names = {key: f"{key}.png" for key in contents}

    response = TestClient(mri_api.app).post(
        "/mri/segment",
        files=multipart_files(contents, names),
    )

    assert response.status_code == 415
    assert calls == []


def test_empty_upload_is_rejected_before_inference(monkeypatch):
    calls = []
    monkeypatch.setattr(mri_api, "_run_inference", lambda *_args: calls.append(True))
    contents = {key: b"synthetic" for key in ("t1c", "t1", "t2", "flair")}
    contents["flair"] = b""

    response = TestClient(mri_api.app).post("/mri/segment", files=multipart_files(contents))

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]
    assert calls == []


def test_oversized_request_is_rejected_before_multipart_parsing(monkeypatch):
    calls = []
    monkeypatch.setattr(mri_api, "_run_inference", lambda *_args: calls.append(True))
    files = multipart_files({key: b"small" for key in ("t1c", "t1", "t2", "flair")})

    response = TestClient(mri_api.app).post(
        "/mri/segment",
        files=files,
        headers={"Content-Length": str(mri_api.MAX_MRI_REQUEST_BYTES + 1)},
    )

    assert response.status_code == 413
    assert calls == []


def test_unreadable_nifti_is_rejected_before_loading_weights(monkeypatch):
    calls = []
    monkeypatch.setattr(mri_api, "_model", lambda: calls.append(True))
    contents = {key: b"not a NIfTI volume" for key in ("t1c", "t1", "t2", "flair")}

    response = TestClient(mri_api.app).post("/mri/segment", files=multipart_files(contents))

    assert response.status_code == 422
    assert "Could not read" in response.json()["detail"]
    assert calls == []


def test_nifti_affine_mismatch_is_rejected_before_loading_weights(monkeypatch):
    calls = []
    monkeypatch.setattr(mri_api, "_model", lambda: calls.append(True))
    content = nifti_bytes()
    mismatched_affine = np.eye(4)
    mismatched_affine[0, 3] = 2
    contents = {
        "t1c": content,
        "t1": content,
        "t2": nifti_bytes(affine=mismatched_affine),
        "flair": content,
    }

    response = TestClient(mri_api.app).post("/mri/segment", files=multipart_files(contents))

    assert response.status_code == 422
    assert "co-registered" in response.json()["detail"]
    assert calls == []


def test_valid_multipart_contract_returns_geometry_arrays_and_mask(monkeypatch):
    def fake_inference(paths: dict[str, Path], output_path: Path):
        assert set(paths) == {"t1c", "t1", "t2", "flair"}
        assert paths["t1c"].suffix == ".nii"
        assert nib.load(str(paths["t1c"])).shape == (8, 8, 8)
        output_path.write_bytes(gzip.compress(b"synthetic-output-mask", mtime=0))
        return {
            "mask": np.zeros((2, 3, 4), dtype=np.uint8),
            "preview": np.full((2, 3, 4), 127, dtype=np.uint8),
            "shape": (2, 3, 4),
            "affine": np.eye(4),
            "spacing_mm": (1.0, 1.0, 1.0),
            "roi_size": (8, 8, 8),
            "label_counts": {"0": 24, "1": 0, "2": 0, "4": 0},
        }

    monkeypatch.setattr(mri_api, "_run_inference", fake_inference)
    response = TestClient(mri_api.app).post(
        "/mri/segment",
        files=multipart_files({
            "t1c": nifti_bytes(compressed=False),
            "t1": nifti_bytes(),
            "t2": nifti_bytes(),
            "flair": nifti_bytes(),
        }, {"t1c": "t1c.nii", "t1": "t1.nii.gz", "t2": "t2.nii.gz", "flair": "flair.nii.gz"}),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["geometry"]["shape"] == [2, 3, 4]
    assert len(payload["study_id"]) == 64
    assert gzip.decompress(base64.b64decode(payload["mask_array_gzip_base64"])) == bytes(24)
