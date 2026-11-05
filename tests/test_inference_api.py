import base64
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    DigitalXRayImageStorageForPresentation,
    ExplicitVRLittleEndian,
    generate_uid,
)
import torch

import inference_api
import modality_verifier
from inference_api import api


class _Model:
    pathologies = ["Atelectasis", "Pneumonia"]

    def eval(self):
        return self

    def __call__(self, _tensor):
        return torch.tensor([[0.25, -0.5]])


def _png_bytes():
    stream = BytesIO()
    Image.new("L", (24, 24), 120).save(stream, format="PNG")
    return stream.getvalue()


def _png_16bit_bytes():
    stream = BytesIO()
    pixels = np.linspace(1000, 50000, 24 * 24, dtype=np.uint16).reshape(24, 24)
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def _synthetic_car_png_bytes():
    """Generated non-medical car illustration; not a photograph or medical fixture."""
    image = Image.new("RGB", (128, 96), "skyblue")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 68, 127, 95), fill="gray")
    draw.polygon([(20, 64), (37, 43), (83, 43), (103, 64)], fill="red")
    draw.rectangle((17, 62, 108, 78), fill="red")
    draw.ellipse((28, 72, 48, 92), fill="black")
    draw.ellipse((78, 72, 98, 92), fill="black")
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _synthetic_car_jpeg_bytes():
    image = Image.open(BytesIO(_synthetic_car_png_bytes())).convert("RGB")
    stream = BytesIO()
    image.save(stream, format="JPEG")
    return stream.getvalue()


def _dicom_bytes():
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DigitalXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Rows = dataset.Columns = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.BurnedInAnnotation = "NO"
    dataset.Modality = "DX"
    dataset.BodyPartExamined = "CHEST"
    dataset.PixelData = np.asarray([[0, 1], [2, 3]], dtype="<u2").tobytes()
    stream = BytesIO()
    dataset.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


@pytest.fixture(autouse=True)
def _reset_model_state(monkeypatch):
    monkeypatch.setattr(inference_api, "_MODEL", None)
    monkeypatch.setattr(inference_api, "_MODEL_STATUS", "not_loaded")
    monkeypatch.setattr(inference_api, "_MODEL_ERROR", None)
    monkeypatch.setattr(modality_verifier, "load_modality_verifier", lambda _path=None: None)


def test_api_health_does_not_load_or_download_model(monkeypatch):
    def should_not_load():
        raise AssertionError("health check must not load the model")

    monkeypatch.setattr(inference_api, "load_model", should_not_load)
    response = TestClient(api).get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": False,
        "model_status": "not_loaded",
        "model_error": None,
    }


def test_load_model_caches_a_successful_initialization(monkeypatch):
    model = _Model()
    calls = []
    monkeypatch.setattr(inference_api.xrv.models, "DenseNet", lambda **kwargs: calls.append(kwargs) or model)

    assert inference_api.load_model() is model
    assert inference_api.load_model() is model
    assert calls == [{"weights": "densenet121-res224-all"}]
    assert inference_api.model_status() == {
        "model_loaded": True,
        "model_status": "loaded",
        "model_error": None,
    }


def test_model_initialization_failure_is_reported_and_logged(monkeypatch, caplog):
    def fail_load(**_kwargs):
        raise OSError("synthetic missing checkpoint")

    monkeypatch.setattr(inference_api.xrv.models, "DenseNet", fail_load)
    with pytest.raises(inference_api.ModelInitializationError, match="could not be initialized"):
        inference_api.load_model()
    assert "synthetic missing checkpoint" in caplog.text
    assert inference_api.model_status() == {
        "model_loaded": False,
        "model_status": "failed",
        "model_error": "OSError",
    }


@pytest.mark.parametrize(
    "payload,content_type",
    [
        (_synthetic_car_png_bytes(), "image/png"),
        (_synthetic_car_jpeg_bytes(), "image/jpeg"),
        (_png_bytes(), "image/png"),
        (_png_16bit_bytes(), "image/png"),
        (_dicom_bytes(), "application/dicom"),
    ],
)
def test_api_rejects_unverified_images_before_model_and_gradcam(monkeypatch, payload, content_type):
    model_calls, gradcam_calls = [], []
    monkeypatch.setattr(inference_api, "load_model", lambda: model_calls.append(True) or _Model())
    monkeypatch.setattr(inference_api, "gradcam", lambda *_args: gradcam_calls.append(True))
    response = TestClient(api).post(
        "/infer", content=payload, headers={"Content-Type": content_type}
    )
    assert response.status_code == 422
    assert "no validated image-modality classifier" in response.json()["detail"].lower()
    if content_type == "application/dicom":
        assert "metadata cannot prove" in response.json()["detail"].lower()
    assert set(response.json()) == {"detail"}
    assert model_calls == []
    assert gradcam_calls == []


def test_api_rejects_direct_request_that_labels_car_png_as_dicom(monkeypatch):
    model_calls = []
    monkeypatch.setattr(inference_api, "load_model", lambda: model_calls.append(True) or _Model())
    response = TestClient(api).post(
        "/infer", content=_synthetic_car_png_bytes(), headers={"Content-Type": "application/dicom"}
    )
    assert response.status_code == 400
    assert "readable dicom" in response.json()["detail"].lower()
    assert set(response.json()) == {"detail"}
    assert model_calls == []


def test_infer_bytes_rejects_even_when_model_is_injected():
    class CountingModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, _tensor):
            self.calls += 1
            return torch.tensor([[0.25]])

    model = CountingModel()
    with pytest.raises(inference_api.InputSuitabilityError, match="no validated image-modality classifier"):
        inference_api.infer_bytes(_synthetic_car_png_bytes(), "image/png", model=model)
    assert model.calls == 0


def test_api_rejects_unsupported_content_and_unverified_raster(monkeypatch):
    model_calls = []
    monkeypatch.setattr(inference_api, "load_model", lambda: model_calls.append(True) or _Model())
    client = TestClient(api)
    assert client.post("/infer", content=b"", headers={"Content-Type": "application/octet-stream"}).status_code == 415
    response = client.post("/infer", content=b"broken", headers={"Content-Type": "image/png"})
    assert response.status_code == 400
    assert "readable png" in response.json()["detail"].lower()
    assert model_calls == []


def test_api_rejects_mismatched_raster_content_type_before_model_loading(monkeypatch):
    model_calls = []
    monkeypatch.setattr(inference_api, "load_model", lambda: model_calls.append(True) or _Model())
    response = TestClient(api).post(
        "/infer", content=_png_bytes(), headers={"Content-Type": "image/jpeg"}
    )
    assert response.status_code == 400
    assert "expected a jpeg image" in response.json()["detail"].lower()
    assert model_calls == []


def test_api_rejects_unsupported_16bit_raster_before_model_loading(monkeypatch):
    model_calls = []
    monkeypatch.setattr(inference_api, "load_model", lambda: model_calls.append(True) or _Model())
    response = TestClient(api).post(
        "/infer", content=_png_16bit_bytes(), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 422
    assert "no validated image-modality classifier" in response.json()["detail"].lower()
    assert model_calls == []


def test_api_enforces_request_limit(monkeypatch):
    monkeypatch.setattr(inference_api, "MAX_REQUEST_BYTES", 4)
    monkeypatch.setattr(inference_api, "load_model", lambda: _Model())
    response = TestClient(api).post(
        "/infer", content=b"12345", headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 413


def test_api_cors_allows_only_local_vite_origins():
    client = TestClient(api)
    allowed = client.options(
        "/infer",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    rejected = client.options(
        "/infer",
        headers={"Origin": "https://attacker.example", "Access-Control-Request-Method": "POST"},
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert rejected.headers.get("access-control-allow-origin") is None


class _MultiFindingModel:
    pathologies = ["Atelectasis", "Cardiomegaly", "Infiltration", "Pneumonia"]

    def eval(self):
        return self

    def __call__(self, _tensor):
        # Cardiomegaly (0.9, rank 1), Infiltration (0.5, rank 2), Atelectasis (0.2, rank 3), Pneumonia (-0.1, rank 4)
        return torch.tensor([[0.2, 0.9, 0.5, -0.1]])


class _EighteenFindingModel:
    pathologies = [f"Finding {index + 1}" for index in range(18)]

    def eval(self):
        return self

    def __call__(self, _tensor):
        return torch.linspace(0.9, 0.05, 18).reshape(1, 18)


class _VerificationStub:
    score = 0.985
    threshold = 0.5


def _setup_mock_pipeline(monkeypatch):
    gradcam_targets = []
    model = _MultiFindingModel()
    monkeypatch.setattr(inference_api, "require_verified_cxr_modality_image", lambda _img, _ct: _VerificationStub())
    monkeypatch.setattr(inference_api, "prepare_image", lambda _img: (torch.zeros((1, 1, 224, 224)), Image.new("L", (224, 224), 128)))
    monkeypatch.setattr(
        inference_api,
        "gradcam",
        lambda _m, _t, idx: gradcam_targets.append(idx) or (np.zeros((224, 224), dtype=np.float32), float(model(_t)[0, idx])),
    )
    monkeypatch.setattr(inference_api, "load_model", lambda: model)
    return model, gradcam_targets


def test_infer_bytes_default_argmax_fallback(monkeypatch):
    model, gradcam_targets = _setup_mock_pipeline(monkeypatch)
    result = inference_api.infer_bytes(_png_bytes(), "image/png", model=model, target_finding=None)

    assert result["target_finding"] == "Cardiomegaly"
    assert result["target_index"] == 1
    assert result["target_score"] == pytest.approx(0.9)
    assert result["target_rank"] == 1
    assert result["gradcam"]["finding"] == "Cardiomegaly"
    assert result["gradcam"]["index"] == 1
    assert result["gradcam"]["score"] == pytest.approx(0.9)
    assert result["gradcam"]["rank"] == 1
    assert gradcam_targets == [1]
    assert len(result["predictions"]) == 4


def test_infer_bytes_valid_target_finding_selection(monkeypatch):
    model, gradcam_targets = _setup_mock_pipeline(monkeypatch)
    result = inference_api.infer_bytes(_png_bytes(), "image/png", model=model, target_finding="Infiltration")

    assert result["target_finding"] == "Infiltration"
    assert result["target_index"] == 2
    assert result["target_score"] == pytest.approx(0.5)
    assert result["target_rank"] == 2
    assert result["gradcam"]["finding"] == "Infiltration"
    assert result["gradcam"]["index"] == 2
    assert result["gradcam"]["score"] == pytest.approx(0.5)
    assert result["gradcam"]["rank"] == 2
    assert gradcam_targets == [2]

    # Verify scores and predictions remain identical to default
    default_result = inference_api.infer_bytes(_png_bytes(), "image/png", model=model, target_finding=None)
    assert result["predictions"] == default_result["predictions"]


def test_infer_bytes_lowest_rank_target_finding(monkeypatch):
    model, gradcam_targets = _setup_mock_pipeline(monkeypatch)
    result = inference_api.infer_bytes(_png_bytes(), "image/png", model=model, target_finding="Pneumonia")

    assert result["target_finding"] == "Pneumonia"
    assert result["target_index"] == 3
    assert result["target_score"] == pytest.approx(-0.1)
    assert result["target_rank"] == 4
    assert result["gradcam"]["rank"] == 4
    assert gradcam_targets == [3]


def test_infer_bytes_invalid_target_finding_raises_value_error(monkeypatch):
    model, _ = _setup_mock_pipeline(monkeypatch)
    with pytest.raises(ValueError, match="Unknown target finding 'Tuberculosis'"):
        inference_api.infer_bytes(_png_bytes(), "image/png", model=model, target_finding="Tuberculosis")


def test_infer_bytes_empty_string_target_finding_raises_value_error(monkeypatch):
    model, _ = _setup_mock_pipeline(monkeypatch)
    with pytest.raises(ValueError, match="Target finding must be a non-empty string"):
        inference_api.infer_bytes(_png_bytes(), "image/png", model=model, target_finding="   ")


@pytest.mark.parametrize(
    "payload,content_type",
    [
        (_png_bytes(), "image/png"),
        (_png_16bit_bytes(), "image/png"),
        (_synthetic_car_jpeg_bytes(), "image/jpeg"),
        (_dicom_bytes(), "application/dicom"),
    ],
)
def test_inference_returns_all_18_outputs_and_processed_image_preview(
    monkeypatch, payload, content_type
):
    model = _EighteenFindingModel()
    monkeypatch.setattr(
        inference_api,
        "require_verified_cxr_modality_image",
        lambda _img, _content_type: _VerificationStub(),
    )
    monkeypatch.setattr(
        inference_api,
        "gradcam",
        lambda _model, _tensor, index: (
            np.zeros((224, 224), dtype=np.float32),
            float(model(_tensor)[0, index]),
        ),
    )

    result = inference_api.infer_bytes(payload, content_type, model=model)

    assert len(result["predictions"]) == 18
    assert result["predictions"][0]["finding"] == "Finding 1"
    preview = Image.open(BytesIO(base64.b64decode(result["input_preview_png_base64"])))
    assert preview.mode == "L"
    assert preview.size == (224, 224)
    assert result["gradcam"]["overlay_png_base64"]


def test_api_infer_endpoint_valid_target_finding_query_param(monkeypatch):
    _setup_mock_pipeline(monkeypatch)
    client = TestClient(api)
    response = client.post(
        "/infer?target_finding=Infiltration",
        content=_png_bytes(),
        headers={"Content-Type": "image/png"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["target_finding"] == "Infiltration"
    assert data["target_index"] == 2
    assert data["target_score"] == pytest.approx(0.5)
    assert data["target_rank"] == 2
    assert data["gradcam"]["finding"] == "Infiltration"
    assert data["gradcam"]["index"] == 2
    assert data["gradcam"]["rank"] == 2
    assert "overlay_png_base64" in data["gradcam"]


def test_api_infer_endpoint_valid_target_finding_header(monkeypatch):
    _setup_mock_pipeline(monkeypatch)
    client = TestClient(api)
    response = client.post(
        "/infer",
        content=_png_bytes(),
        headers={"Content-Type": "image/png", "X-Target-Finding": "Atelectasis"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["target_finding"] == "Atelectasis"
    assert data["target_index"] == 0
    assert data["target_score"] == pytest.approx(0.2)
    assert data["target_rank"] == 3
    assert data["gradcam"]["finding"] == "Atelectasis"


def test_api_infer_endpoint_invalid_target_finding_query_param(monkeypatch):
    _setup_mock_pipeline(monkeypatch)
    client = TestClient(api)
    response = client.post(
        "/infer?target_finding=InvalidFinding",
        content=_png_bytes(),
        headers={"Content-Type": "image/png"},
    )
    assert response.status_code == 400
    assert "Unknown target finding 'InvalidFinding'" in response.json()["detail"]


def test_api_infer_endpoint_missing_target_finding_falls_back_to_argmax(monkeypatch):
    _setup_mock_pipeline(monkeypatch)
    client = TestClient(api)
    response = client.post(
        "/infer",
        content=_png_bytes(),
        headers={"Content-Type": "image/png"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["target_finding"] == "Cardiomegaly"
    assert data["target_rank"] == 1
    assert data["gradcam"]["finding"] == "Cardiomegaly"


def test_api_infer_endpoint_empty_target_finding_query_param_falls_back(monkeypatch):
    _setup_mock_pipeline(monkeypatch)
    client = TestClient(api)
    response = client.post(
        "/infer?target_finding=",
        content=_png_bytes(),
        headers={"Content-Type": "image/png"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["target_finding"] == "Cardiomegaly"
    assert data["target_rank"] == 1
