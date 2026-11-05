from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
    DigitalXRayImageStorageForProcessing,
    ExplicitVRLittleEndian,
    generate_uid,
)

from dicom_io import DicomInputError, dicom_bytes_to_cxr_candidate, dicom_bytes_to_image
from input_suitability import InputSuitabilityError


def _dicom_bytes(pixels=None, **overrides):
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
    pixel_array = np.asarray(pixels, dtype="<u2") if pixels is not None else None
    dataset.Rows, dataset.Columns = pixel_array.shape if pixel_array is not None else (2, 2)
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.BurnedInAnnotation = "NO"
    dataset.PatientName = "Sensitive^Name"
    if pixels is not None:
        dataset.PixelData = pixel_array.tobytes()
    for key, value in overrides.items():
        if key == "FileMetaMismatch":
            continue
        setattr(dataset, key, value)
        if key == "SOPClassUID":
            meta.MediaStorageSOPClassUID = value
    stream = BytesIO()
    dataset.save_as(stream, enforce_file_format=True)
    data = stream.getvalue()
    if overrides.get("FileMetaMismatch"):
        expected = str(DigitalXRayImageStorageForPresentation).encode("ascii")
        conflicting = b"1.2.840.10008.5.1.4.1.1.1.2"
        assert len(expected) == len(conflicting)
        data = data.replace(expected, conflicting, 1)
    return data


def test_dicom_decoder_returns_only_8bit_image_not_patient_dataset():
    image = dicom_bytes_to_image(_dicom_bytes([[0, 1], [2, 3]]))
    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    assert image.size == (2, 2)
    assert image.getextrema() == (0, 255)
    assert not hasattr(image, "PatientName")


def test_dicom_monochrome1_is_inverted_for_display():
    image = dicom_bytes_to_image(
        _dicom_bytes([[0, 1], [2, 3]], PhotometricInterpretation="MONOCHROME1")
    )
    assert np.asarray(image)[0, 0] == 255
    assert np.asarray(image)[1, 1] == 0


def test_dicom_percentile_display_scaling_preserves_typical_contrast():
    pixels = np.linspace(1000, 2000, 10000, dtype=np.uint16).reshape(100, 100)
    pixels[0, 0] = 0
    pixels[-1, -1] = 65535

    image = dicom_bytes_to_image(_dicom_bytes(pixels))

    assert image.getextrema() == (0, 255)
    assert 100 <= np.asarray(image)[50, 50] <= 155


def test_cxr_candidate_requires_explicit_modality_and_body_region_tags():
    image = dicom_bytes_to_cxr_candidate(_dicom_bytes([[0, 1], [2, 3]]))
    assert image.size == (2, 2)

    for changes in (
        {"Modality": "CT"},
        {"Modality": ""},
        {"BodyPartExamined": ""},
        {"BodyPartExamined": "ABDOMEN"},
    ):
        with pytest.raises(InputSuitabilityError, match="suitability cannot be established"):
            dicom_bytes_to_cxr_candidate(_dicom_bytes([[0, 1], [2, 3]], **changes))

    cr_uid = str(ComputedRadiographyImageStorage)
    assert dicom_bytes_to_cxr_candidate(
        _dicom_bytes([[0, 1], [2, 3]], Modality="CR", SOPClassUID=cr_uid)
    ).size == (2, 2)
    assert dicom_bytes_to_cxr_candidate(
        _dicom_bytes(
            [[0, 1], [2, 3]],
            SOPClassUID=str(DigitalXRayImageStorageForProcessing),
            BodyPartExamined="THORAX",
        )
    ).size == (2, 2)

    with pytest.raises(InputSuitabilityError, match="SOP class"):
        dicom_bytes_to_cxr_candidate(
            _dicom_bytes([[0, 1], [2, 3]], SOPClassUID="1.2.840.10008.5.1.4.1.1.7")
        )
    with pytest.raises(InputSuitabilityError, match="SOP class"):
        dicom_bytes_to_cxr_candidate(
            _dicom_bytes([[0, 1], [2, 3]], FileMetaMismatch=True)
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"BurnedInAnnotation": "YES"}, "burned-in"),
        ({"NumberOfFrames": 2}, "single-frame"),
        ({"PhotometricInterpretation": "RGB", "SamplesPerPixel": 3}, "single-channel"),
        ({"SamplesPerPixel": 65535}, "single-channel"),
    ],
)
def test_unsafe_or_unsupported_dicom_is_rejected(kwargs, message):
    with pytest.raises(DicomInputError, match=message):
        dicom_bytes_to_image(_dicom_bytes([[0, 1], [2, 3]], **kwargs))


def test_empty_non_dicom_missing_pixels_and_flat_image_are_rejected():
    for payload in (b"", b"not DICOM"):
        with pytest.raises(DicomInputError):
            dicom_bytes_to_image(payload)
    with pytest.raises(DicomInputError, match="no image pixel"):
        dicom_bytes_to_image(_dicom_bytes())
    with pytest.raises(DicomInputError, match="no displayable"):
        dicom_bytes_to_image(_dicom_bytes([[1, 1], [1, 1]]))
