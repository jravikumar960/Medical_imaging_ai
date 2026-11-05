"""Synthetic source-metadata/conversion consistency tests; not anatomical evidence."""
import hashlib
import json

import pytest

from ct_provenance import (
    ProvenanceError,
    _validate_index_map,
    build_nifti_origin_record,
    verify_nifti_origin_provenance,
    verify_source_provenance,
)


SERIES_UID = "1.2.840.10008.1.2.3"
FRAME_UID = "1.2.840.10008.1.2.4"
AFFINE = [[-1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1.0]]
SHAPE = [4, 3, 2]


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fixture(tmp_path, metadata_changes=None, conversion_changes=None, geometry=None):
    volume_path = tmp_path / "synthetic_ct.nii.gz"
    volume_path.write_bytes(b"synthetic NIfTI fixture bytes; no patient data")
    metadata = {
        "schema": "medical-imaging-source-dicom-series/v1",
        "patient_coordinate_system": "LPS",
        "series_instance_uid": SERIES_UID,
        "frame_of_reference_uid": FRAME_UID,
        "slices": [
            {"sop_instance_uid": "1.2.3.1", "series_instance_uid": SERIES_UID,
             "frame_of_reference_uid": FRAME_UID, "modality": "CT", "patient_position": "HFS",
             "image_orientation_patient": [1, 0, 0, 0, 1, 0],
             "image_position_patient": [0, 0, 0], "rows": 3, "columns": 4,
             "pixel_spacing_mm": [1, 1], "spacing_between_slices_mm": 2.0},
            {"sop_instance_uid": "1.2.3.2", "series_instance_uid": SERIES_UID,
             "frame_of_reference_uid": FRAME_UID, "modality": "CT", "patient_position": "HFS",
             "image_orientation_patient": [1, 0, 0, 0, 1, 0],
             "image_position_patient": [0, 0, 2], "rows": 3, "columns": 4,
             "pixel_spacing_mm": [1, 1], "spacing_between_slices_mm": 2.0},
        ],
    }
    if metadata_changes:
        metadata_changes(metadata)
    metadata_path = tmp_path / "source_metadata.json"
    metadata_path.write_bytes(_json_bytes(metadata))
    conversion = {
        "schema": "medical-imaging-nifti-conversion/v1",
        "nifti_affine_coordinate_system": "RAS",
        "nifti_sha256": hashlib.sha256(volume_path.read_bytes()).hexdigest(),
        "source_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "nifti_shape": SHAPE,
        "source_sop_instance_uids_by_slice_index": ["1.2.3.1", "1.2.3.2"],
        "conversion_software": "SyntheticFixtureConverter",
        "conversion_version": "0.0-test",
        # NIfTI [i,j,k] -> DICOM [column,row,slice-index].
        "nifti_voxel_to_source_indices": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }
    if conversion_changes:
        conversion_changes(conversion)
    conversion_path = tmp_path / "conversion_record.json"
    conversion_path.write_bytes(_json_bytes(conversion))
    actual_geometry = {"shape": SHAPE, "affine": AFFINE}
    if geometry:
        actual_geometry = geometry
    review = {
        "volume_sha256": hashlib.sha256(volume_path.read_bytes()).hexdigest(),
        "geometry_sha256": "synthetic-geometry-digest",
        "source_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "conversion_record_sha256": hashlib.sha256(conversion_path.read_bytes()).hexdigest(),
        "series_instance_uid": SERIES_UID,
        "frame_of_reference_uid": FRAME_UID,
        "image_orientation_patient": [1, 0, 0, 0, 1, 0],
        "first_image_position_patient": [0, 0, 0],
        "last_image_position_patient": [0, 0, 2],
        "slice_order": "ascending",
        "conversion_software": "SyntheticFixtureConverter",
        "conversion_version": "0.0-test",
    }
    return volume_path, metadata_path, conversion_path, review, actual_geometry


def _refresh_digests(volume, metadata_path, conversion_path, review):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    conversion["nifti_sha256"] = hashlib.sha256(volume.read_bytes()).hexdigest()
    conversion["source_metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    conversion_path.write_bytes(_json_bytes(conversion))
    review["volume_sha256"] = hashlib.sha256(volume.read_bytes()).hexdigest()
    review["source_metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    review["conversion_record_sha256"] = hashlib.sha256(conversion_path.read_bytes()).hexdigest()


def _verify(fixture):
    volume, metadata, conversion, review, geometry = fixture
    verify_source_provenance(volume, geometry, review, metadata, conversion)


@pytest.mark.parametrize(
    "shape,source_shape,index_map",
    [
        ([4, 3, 2], [4, 3, 2],
         [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        ([2, 3, 4], [4, 3, 2],
         [[0, 0, -1, 3], [0, 1, 0, 0], [-1, 0, 0, 1], [0, 0, 0, 1]]),
    ],
)
def test_full_grid_signed_axis_permutations_and_flips_are_accepted(shape, source_shape, index_map):
    result = _validate_index_map(index_map, shape, source_shape)
    assert result == [[float(value) for value in row] for row in index_map]


@pytest.mark.parametrize(
    "shape,source_shape,index_map,expected",
    [
        ([3, 3, 2], [4, 3, 2],
         [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
         "axis extents differ"),
        ([3, 3, 2], [4, 3, 2],
         [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
         "axis extents differ"),
        ([4, 3, 2], [4, 3, 2],
         [[-1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
         "shifted or cropped"),
        ([4, 3, 2], [4, 3, 2],
         [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
         "shifted or cropped"),
        ([4, 3, 2], [4, 3, 2],
         [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
         "shifted or cropped"),
    ],
)
def test_cropped_shifted_incomplete_or_out_of_bounds_index_maps_are_rejected(
    shape, source_shape, index_map, expected
):
    with pytest.raises(ProvenanceError, match=expected):
        _validate_index_map(index_map, shape, source_shape)


def test_synthetic_records_bind_dicom_series_conversion_and_nifti_affine(tmp_path):
    """A passing synthetic consistency check is not an orientation attestation."""
    _verify(_fixture(tmp_path))


def test_synthetic_full_provenance_accepts_permuted_and_flipped_voxel_grid(tmp_path):
    shape = [2, 3, 4]
    index_map = [
        [0, 0, -1, 3],
        [0, 1, 0, 0],
        [-1, 0, 0, 1],
        [0, 0, 0, 1],
    ]
    geometry = {
        "shape": shape,
        "affine": [[0, 0, 1, -3], [0, -1, 0, 0], [-2, 0, 0, 2], [0, 0, 0, 1]],
    }
    fixture = list(_fixture(
        tmp_path,
        conversion_changes=lambda conversion: conversion.update(
            nifti_shape=shape,
            nifti_voxel_to_source_indices=index_map,
        ),
        geometry=geometry,
    ))
    _refresh_digests(fixture[0], fixture[1], fixture[2], fixture[3])
    _verify(tuple(fixture))


@pytest.mark.parametrize("missing", ["metadata", "conversion"])
def test_missing_source_record_file_is_rejected(tmp_path, missing):
    fixture = _fixture(tmp_path)
    volume, metadata, conversion, review, geometry = fixture
    with pytest.raises(ProvenanceError, match="source DICOM metadata and conversion-record files are required"):
        verify_source_provenance(volume, geometry, review,
                                 None if missing == "metadata" else metadata,
                                 None if missing == "conversion" else conversion)


@pytest.mark.parametrize("field", ["series_instance_uid", "frame_of_reference_uid"])
def test_unrelated_series_or_frame_uid_is_rejected(tmp_path, field):
    def change(metadata):
        metadata[field] = "1.2.999.1"
        for item in metadata["slices"]:
            item[field] = "1.2.999.1"
    fixture = list(_fixture(tmp_path, metadata_changes=change))
    _refresh_digests(fixture[0], fixture[1], fixture[2], fixture[3])
    with pytest.raises(ProvenanceError, match="differs from the signed review"):
        _verify(tuple(fixture))


def test_incorrect_conversion_slice_order_is_rejected(tmp_path):
    fixture = list(_fixture(tmp_path, conversion_changes=lambda record: record.update(
        source_sop_instance_uids_by_slice_index=["1.2.3.2", "1.2.3.1"])))
    _refresh_digests(fixture[0], fixture[1], fixture[2], fixture[3])
    with pytest.raises(ProvenanceError, match="slice inventory/order"):
        _verify(tuple(fixture))


def test_missing_or_duplicate_slices_are_rejected(tmp_path):
    def gap(metadata):
        metadata["slices"][1]["image_position_patient"] = [0, 0, 4]
    fixture = list(_fixture(tmp_path, metadata_changes=gap))
    _refresh_digests(fixture[0], fixture[1], fixture[2], fixture[3])
    with pytest.raises(ProvenanceError, match="gap or inconsistent spacing"):
        _verify(tuple(fixture))

    other = tmp_path / "duplicate"
    other.mkdir()
    def duplicate(metadata):
        metadata["slices"][1]["sop_instance_uid"] = metadata["slices"][0]["sop_instance_uid"]
    fixture = list(_fixture(other, metadata_changes=duplicate))
    _refresh_digests(fixture[0], fixture[1], fixture[2], fixture[3])
    with pytest.raises(ProvenanceError, match="duplicate slice instances"):
        _verify(tuple(fixture))


def test_affine_disagreement_is_rejected(tmp_path):
    shifted = [row[:] for row in AFFINE]
    shifted[0][3] = 5
    with pytest.raises(ProvenanceError, match="disagrees with the NIfTI affine"):
        _verify(_fixture(tmp_path, geometry={"shape": SHAPE, "affine": shifted}))


@pytest.mark.parametrize(
    "metadata_change,conversion_change,expected",
    [
        (lambda metadata: metadata["slices"][1].update(
            image_orientation_patient=[0, 1, 0, -1, 0, 0]
        ), None, "inconsistent across slices"),
        (lambda metadata: metadata["slices"][1].update(columns=5), None,
         "dimensions or PixelSpacing are inconsistent"),
        (None, lambda conversion: conversion.update(nifti_shape=[4, 3, 3]),
         "dimensions differ from the inspected NIfTI"),
        (None, lambda conversion: conversion.update(
            nifti_voxel_to_source_indices=[[2, 0, 0, 0], [0, 1, 0, 0],
                                           [0, 0, 1, 0], [0, 0, 0, 1]]
        ), "signed axis permutation"),
    ],
)
def test_inconsistent_source_geometry_and_coordinate_mapping_are_rejected(
    tmp_path, metadata_change, conversion_change, expected
):
    fixture = list(_fixture(
        tmp_path,
        metadata_changes=metadata_change,
        conversion_changes=conversion_change,
    ))
    _refresh_digests(fixture[0], fixture[1], fixture[2], fixture[3])
    with pytest.raises(ProvenanceError, match=expected):
        _verify(tuple(fixture))


@pytest.mark.parametrize("record", ["metadata", "conversion"])
def test_altered_source_record_bytes_are_rejected(tmp_path, record):
    fixture = _fixture(tmp_path)
    target = fixture[1] if record == "metadata" else fixture[2]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ProvenanceError, match="file hash differs from the signed review"):
        _verify(fixture)


NIFTI_ORIGIN_METADATA = {
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
    "training": [
        {"image": f"./imagesTr/spleen_{index}.nii.gz", "label": f"./labelsTr/spleen_{index}.nii.gz"}
        for index in range(41)
    ],
    "test": [f"./imagesTs/spleen_{index}.nii.gz" for index in range(1, 21)],
}
NIFTI_ORIGIN_GEOMETRY = {
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


def _nifti_origin_fixture(metadata=None, geometry=None, volume_sha256=None):
    metadata_bytes = json.dumps(metadata or NIFTI_ORIGIN_METADATA).encode("utf-8")
    geometry = geometry or NIFTI_ORIGIN_GEOMETRY
    volume_sha256 = volume_sha256 or "a" * 64
    record = build_nifti_origin_record(
        metadata_bytes,
        "./imagesTs/spleen_1.nii.gz",
        volume_sha256,
        geometry,
    )
    record_bytes = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    record_sha256 = hashlib.sha256(record_bytes).hexdigest()
    review = {
        "origin_record_sha256": record_sha256,
        "volume_sha256": volume_sha256,
        "geometry_sha256": hashlib.sha256(
            json.dumps(geometry, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
    }
    return record, record_sha256, review, geometry


def test_nifti_origin_record_uses_task09_manifest_and_exact_inspection():
    record, record_sha256, review, geometry = _nifti_origin_fixture()
    assert record["dataset"] == {
        "name": "Medical Segmentation Decathlon",
        "task": "Task09_Spleen",
        "dataset_name": "Spleen",
        "description": "Spleen Segmentation",
        "reference": "Memorial Sloan Kettering Cancer Center",
        "release": "1.0 06/08/2018",
        "license": "CC-BY-SA 4.0",
        "modality": "CT",
        "labels": {"0": "background", "1": "spleen"},
        "tensor_image_size": "3D",
        "declared_num_training": 41,
        "declared_num_test": 20,
        "split": "test",
        "image_path": "./imagesTs/spleen_1.nii.gz",
        "label_path": None,
        "dataset_json_sha256": hashlib.sha256(json.dumps(NIFTI_ORIGIN_METADATA).encode()).hexdigest(),
    }
    assert record["volume_source_authenticity"] == "not_independently_authenticated"
    assert record["original_dicom_metadata"] == "not_supplied_in_the_dataset_distribution"
    assert record["conversion_history"] == "unavailable"
    assert "series_instance_uid" not in json.dumps(record)
    assert "conversion_software" not in json.dumps(record)
    verify_nifti_origin_provenance(
        record, geometry, review,
        expected_volume_sha256="a" * 64,
        expected_record_sha256=record_sha256,
    )


@pytest.mark.parametrize(
    "metadata,image_path,expected",
    [
        ({**NIFTI_ORIGIN_METADATA, "release": "unknown"}, "./imagesTs/spleen_1.nii.gz", "expected Task09 Spleen release"),
        ({**NIFTI_ORIGIN_METADATA, "test": []}, "./imagesTs/spleen_1.nii.gz", "does not list spleen_1"),
        (NIFTI_ORIGIN_METADATA, "./imagesTr/spleen_10.nii.gz", "accepts only the listed Task09 test image"),
        ({**NIFTI_ORIGIN_METADATA, "modality": {"0": "MR"}}, "./imagesTs/spleen_1.nii.gz", "does not identify CT"),
    ],
)
def test_nifti_origin_builder_rejects_wrong_dataset_release_split_or_path(metadata, image_path, expected):
    with pytest.raises(ProvenanceError, match=expected):
        build_nifti_origin_record(
            json.dumps(metadata).encode(),
            image_path,
            "a" * 64,
            NIFTI_ORIGIN_GEOMETRY,
        )


@pytest.mark.parametrize("tamper", ["volume", "geometry", "review_record", "record_hash", "history", "dicom_field"])
def test_nifti_origin_verifier_rejects_hash_geometry_or_history_tampering(tamper):
    record, record_sha256, review, geometry = _nifti_origin_fixture()
    if tamper == "volume":
        with pytest.raises(ProvenanceError, match="exact NIfTI bytes"):
            verify_nifti_origin_provenance(
                record, geometry, review,
                expected_volume_sha256="b" * 64,
                expected_record_sha256=record_sha256,
            )
    elif tamper == "geometry":
        changed_geometry = {**geometry, "shape": [4, 5, 7]}
        with pytest.raises(ProvenanceError, match="geometry differs"):
            verify_nifti_origin_provenance(
                record, changed_geometry, review,
                expected_volume_sha256="a" * 64,
                expected_record_sha256=record_sha256,
            )
    elif tamper == "review_record":
        review["origin_record_sha256"] = "0" * 64
        with pytest.raises(ProvenanceError, match="not bound to this exact NIfTI-origin record"):
            verify_nifti_origin_provenance(
                record, geometry, review,
                expected_volume_sha256="a" * 64,
                expected_record_sha256=record_sha256,
            )
    elif tamper == "record_hash":
        record["dataset"]["license"] = "tampered"
        with pytest.raises(ProvenanceError, match="record bytes differ"):
            verify_nifti_origin_provenance(
                record, geometry, review,
                expected_volume_sha256="a" * 64,
                expected_record_sha256=record_sha256,
            )
    elif tamper == "dicom_field":
        record["series_instance_uid"] = "1.2.3.4"
        changed_record_bytes = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        record_sha256 = hashlib.sha256(changed_record_bytes).hexdigest()
        review["origin_record_sha256"] = record_sha256
        with pytest.raises(ProvenanceError, match="unsupported or DICOM-origin fields"):
            verify_nifti_origin_provenance(
                record, geometry, review,
                expected_volume_sha256="a" * 64,
                expected_record_sha256=record_sha256,
            )
    else:
        record["conversion_history"] = "converted by guessed tool"
        changed_record_bytes = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        record_sha256 = hashlib.sha256(changed_record_bytes).hexdigest()
        review["origin_record_sha256"] = record_sha256
        with pytest.raises(ProvenanceError, match="must accurately state unavailable"):
            verify_nifti_origin_provenance(
                record, geometry, review,
                expected_volume_sha256="a" * 64,
                expected_record_sha256=record_sha256,
            )
