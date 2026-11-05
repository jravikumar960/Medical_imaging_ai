from io import BytesIO
import numpy as np
import pytest
from PIL import Image
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import DigitalXRayImageStorageForPresentation, ExplicitVRLittleEndian, generate_uid

from vindr_adapter import (
    VinDrAdapterError,
    preprocess_vindr_dicom_bytes,
    load_label_crosswalk,
    validate_vindr_evaluation_readiness,
)


def _make_dicom_bytes(pixels, photometric="MONOCHROME2", bits=16):
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
    arr = np.asarray(pixels, dtype=np.uint16 if bits == 16 else np.uint8)
    dataset.Rows = arr.shape[0]
    dataset.Columns = arr.shape[1]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = bits
    dataset.BitsStored = bits
    dataset.HighBit = bits - 1
    dataset.PixelRepresentation = 0
    dataset.PixelData = arr.tobytes()

    stream = BytesIO()
    dataset.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


def test_preprocess_vindr_dicom_monochrome2_16bit():
    raw = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    data = _make_dicom_bytes(raw, photometric="MONOCHROME2", bits=16)
    tensor, img = preprocess_vindr_dicom_bytes(data)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 224, 224)
    assert tensor.dtype == torch.float32
    assert isinstance(img, Image.Image)
    assert img.size == (224, 224)
    assert img.mode == "L"


def test_preprocess_vindr_dicom_monochrome1_inversion():
    raw = np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)
    data_m2 = _make_dicom_bytes(raw, photometric="MONOCHROME2", bits=16)
    data_m1 = _make_dicom_bytes(raw, photometric="MONOCHROME1", bits=16)

    t2, _ = preprocess_vindr_dicom_bytes(data_m2)
    t1, _ = preprocess_vindr_dicom_bytes(data_m1)

    # MONOCHROME1 should invert pixel representation relative to MONOCHROME2
    assert not torch.allclose(t2, t1)


def test_preprocess_vindr_dicom_invalid_inputs():
    with pytest.raises(VinDrAdapterError, match="empty"):
        preprocess_vindr_dicom_bytes(b"")

    with pytest.raises(VinDrAdapterError, match="readable DICOM"):
        preprocess_vindr_dicom_bytes(b"not a dicom file")


def test_load_label_crosswalk():
    crosswalk = load_label_crosswalk("outputs/xray_vindr_evaluation/vindr_label_crosswalk.json")
    assert isinstance(crosswalk, list)
    assert len(crosswalk) == 18
    assert crosswalk[0]["model_label"] == "Effusion"


def test_validation_gate_raises_when_missing_files(tmp_path):
    with pytest.raises(VinDrAdapterError, match="Validation gate failed"):
        validate_vindr_evaluation_readiness(tmp_path)
