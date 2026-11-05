"""Synthetic DICOM-to-NIfTI conversion tests; not anatomical evidence."""

import os
from pathlib import Path

import numpy as np
import pydicom
import pytest

pytest.importorskip("nibabel")
pytest.importorskip("segmentation")

from ct_dicom_conversion import DicomConversionError, convert_dicom_directory
from ct_provenance import verify_source_provenance
from orientation_review import (
    OrientationReviewError,
    prepare_orientation_review,
    verify_orientation_review,
)
import segmentation


def _write_series(tmp_path, subdir="series", n_slices=4, mutator=None):
    directory = tmp_path / subdir
    directory.mkdir()
    datasets = []
    for index in range(n_slices):
        file_meta = pydicom.Dataset()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dataset = pydicom.Dataset()
        dataset.file_meta = file_meta
        dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        dataset.SOPInstanceUID = f"1.2.3.4.{index + 1}"
        dataset.Modality = "CT"
        dataset.SeriesInstanceUID = "1.2.3.4"
        dataset.FrameOfReferenceUID = "1.2.3.5"
        dataset.PatientPosition = "HFS"
        dataset.Rows = 4
        dataset.Columns = 3
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.SpacingBetweenSlices = 2.0
        dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        dataset.ImagePositionPatient = [0, 0, index * 2.0]
        dataset.RescaleSlope = 1
        dataset.RescaleIntercept = 0
        array = (np.arange(12, dtype=np.uint16) + index).reshape(4, 3)
        dataset.PixelData = array.tobytes()
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        datasets.append(dataset)
    if mutator is not None:
        mutator(datasets)
    for index, dataset in enumerate(datasets):
        dataset.save_as(directory / f"slice_{index}.dcm", enforce_file_format=True)
    return directory


def test_valid_conversion_passes_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", "0" * 40)
    series_dir = _write_series(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    result = convert_dicom_directory(
        series_dir,
        output_dir / "volume.nii.gz",
        output_dir / "source.json",
        output_dir / "conversion.json",
    )
    geometry = segmentation.inspect_nifti_geometry(result["nifti_path"])
    review = prepare_orientation_review(
        nifti_path=result["nifti_path"],
        geometry=geometry,
        source_metadata_path=result["source_metadata_path"],
        conversion_record_path=result["conversion_record_path"],
        reviewer_id="synthetic reviewer",
        laterality_verified=True,
        slice_direction_verified=True,
        review_notes="synthetic verification",
        sign=True,
    )
    verify_source_provenance(
        result["nifti_path"],
        geometry,
        review,
        result["source_metadata_path"],
        result["conversion_record_path"],
    )
    assert geometry["axis_codes"] == ["L", "P", "S"]
    assert result["ordered_sop_instance_uids"] == [f"1.2.3.4.{i}" for i in range(1, 5)]
@pytest.mark.parametrize(
    "mutator",
    [
        lambda datasets: setattr(datasets[0], "Modality", "MR"),
        lambda datasets: setattr(datasets[0], "SpacingBetweenSlices", None),
        lambda datasets: setattr(datasets[1], "SOPInstanceUID", datasets[0].SOPInstanceUID),
        lambda datasets: setattr(datasets[1], "ImageOrientationPatient", [0, 1, 0, -1, 0, 0]),
        lambda datasets: setattr(datasets[1], "ImagePositionPatient", [0, 0, 100.0]),
        lambda datasets: setattr(datasets[0], "PixelSpacing", [0.0, 1.0]),
        lambda datasets: setattr(datasets[1], "Rows", 5),
        lambda datasets: setattr(datasets[0], "SeriesInstanceUID", "9.9.9.9"),
    ],
    ids=[
        "non_ct_modality",
        "missing_slice_spacing",
        "duplicate_sop_uid",
        "inconsistent_orientation",
        "position_gap",
        "bad_pixel_spacing",
        "inconsistent_dimensions",
        "mixed_series",
    ],
)
def test_malformed_series_rejected(tmp_path, mutator):
    series_dir = _write_series(tmp_path, mutator=mutator)
    with pytest.raises(DicomConversionError):
        convert_dicom_directory(
            series_dir,
            tmp_path / "volume.nii.gz",
            tmp_path / "source.json",
            tmp_path / "conversion.json",
        )


def test_single_slice_series_rejected(tmp_path):
    series_dir = _write_series(tmp_path, n_slices=1)
    with pytest.raises(DicomConversionError):
        convert_dicom_directory(
            series_dir,
            tmp_path / "volume.nii.gz",
            tmp_path / "source.json",
            tmp_path / "conversion.json",
        )

def test_orientation_review_requires_explicit_verification(tmp_path, monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", "0" * 40)
    series_dir = _write_series(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    result = convert_dicom_directory(
        series_dir,
        output_dir / "volume.nii.gz",
        output_dir / "source.json",
        output_dir / "conversion.json",
    )
    geometry = segmentation.inspect_nifti_geometry(result["nifti_path"])
    with pytest.raises(OrientationReviewError):
        prepare_orientation_review(
            nifti_path=result["nifti_path"],
            geometry=geometry,
            source_metadata_path=result["source_metadata_path"],
            conversion_record_path=result["conversion_record_path"],
            reviewer_id="synthetic reviewer",
            laterality_verified=False,
            slice_direction_verified=True,
            review_notes="missing laterality",
            sign=False,
        )
    with pytest.raises(OrientationReviewError):
        prepare_orientation_review(
            nifti_path=result["nifti_path"],
            geometry=geometry,
            source_metadata_path=result["source_metadata_path"],
            conversion_record_path=result["conversion_record_path"],
            reviewer_id="synthetic reviewer",
            laterality_verified=True,
            slice_direction_verified=True,
            review_notes="",
            sign=False,
        )


def test_unsigned_review_is_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("CT_ORIENTATION_REVIEW_HMAC_KEY", "0" * 40)
    series_dir = _write_series(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    result = convert_dicom_directory(
        series_dir,
        output_dir / "volume.nii.gz",
        output_dir / "source.json",
        output_dir / "conversion.json",
    )
    geometry = segmentation.inspect_nifti_geometry(result["nifti_path"])
    review = prepare_orientation_review(
        nifti_path=result["nifti_path"],
        geometry=geometry,
        source_metadata_path=result["source_metadata_path"],
        conversion_record_path=result["conversion_record_path"],
        reviewer_id="synthetic reviewer",
        laterality_verified=True,
        slice_direction_verified=True,
        review_notes="unsigned",
        sign=False,
    )
    import hashlib
    volume_sha = hashlib.sha256(Path(result["nifti_path"]).read_bytes()).hexdigest()
    with pytest.raises(OrientationReviewError):
        verify_orientation_review(review, volume_sha, geometry)
