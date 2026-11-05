"""Headless Streamlit review-widget smoke using a generated DICOM and tiny model."""
from io import BytesIO
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import DigitalXRayImageStorageForPresentation, ExplicitVRLittleEndian, generate_uid
from PIL import Image
from streamlit.testing.v1 import AppTest
import torchxrayvision as xrv


def _synthetic_dicom():
    pixels = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DigitalXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = "DX"
    dataset.BodyPartExamined = "CHEST"
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.BurnedInAnnotation = "NO"
    dataset.PixelData = pixels.tobytes()
    target = BytesIO()
    dataset.save_as(target, enforce_file_format=True)
    return target.getvalue()


def _synthetic_png():
    image = Image.new("RGB", (48, 32), "gray")
    target = BytesIO()
    image.save(target, format="PNG")
    return target.getvalue()


def test_dicom_metadata_alone_does_not_allow_streamlit_model_inference(monkeypatch):
    model_calls = []
    monkeypatch.setattr(xrv.models, "DenseNet", lambda **_kwargs: model_calls.append(True))
    monkeypatch.setattr("modality_verifier.load_modality_verifier", lambda *args, **kwargs: None)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15)
    app.run()
    app.file_uploader(key="cxr_upload").set_value(
        ("generated-synthetic.dcm", _synthetic_dicom(), "application/dicom")
    ).run()

    assert not app.exception
    assert any("metadata cannot prove" in node.value.lower() for node in app.error)
    assert model_calls == []


def test_nonmedical_png_is_blocked_in_streamlit_before_model_loading(monkeypatch):
    model_calls = []
    monkeypatch.setattr(xrv.models, "DenseNet", lambda **_kwargs: model_calls.append(True))
    monkeypatch.setattr("modality_verifier.load_modality_verifier", lambda *args, **kwargs: None)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15)
    app.run()
    app.file_uploader(key="cxr_upload").set_value(
        ("candidate.png", _synthetic_png(), "image/png")
    ).run()

    assert not app.exception
    assert any("inference is disabled" in node.value.lower() for node in app.error)
    assert model_calls == []
    assert len(app.dataframe) == 0
